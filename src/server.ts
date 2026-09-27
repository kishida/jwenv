// Jev サーバー（TypeSafe System One 互換 API、Node.js + WebGPU、llama.cpp 非依存）
//   node src/server.ts --model <model.gguf> [--port 8080] [--weights q8|f32] [--temperature T]
//   node src/server.ts --hf kishida/jwenv-4b-poc-gguf:q4_k_m   Hugging Face から取って ~/.cache/jwenv に置く
//                      [--kv-cache-tokens 8192]  プレフィックスKVキャッシュ容量（0でリクエストをまたぐキャッシュを無効）
//                      [--api-key KEY]           指定すると Authorization: Bearer KEY を要求（未指定なら認証なし）
//                      [--max-queue 256]         待ち行列の上限（超えたら 529 Overloaded）
// エンドポイント:
//   POST /v1/systemone   https://docs.typesafe.ai/api と同じリクエスト/レスポンス
//   GET  /health
//   GET  /               Web UI（web/index.html と web/dist/。事前に `npm run build:web` が必要）
// 同時に届いたリクエストは動的バッチングで1回の forward にまとめる。1リクエスト内の質問は同じ state を共有するので、
// state 部分の K/V はバッチ内で共有され、次のリクエスト以降もプレフィックスKVキャッシュから再利用される。
import { existsSync, readFileSync, statSync } from "node:fs";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { basename, extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { JevClassifier, JevError, type PreparedRequest, type SystemOneResponse } from "./jev.ts";
import { ensureModel } from "./node/hf.ts";

const { values: argv } = parseArgs({
  options: {
    model: { type: "string" },
    hf: { type: "string" },
    port: { type: "string", default: "8080" },
    host: { type: "string", default: "127.0.0.1" },
    weights: { type: "string", default: "q8" },
    temperature: { type: "string" },
    "model-name": { type: "string" },
    "max-batch-tokens": { type: "string", default: "4096" },
    "max-seq-len": { type: "string", default: "2048" },
    "kv-cache-tokens": { type: "string", default: "8192" },
    "api-key": { type: "string" },
    "max-queue": { type: "string", default: "256" },
  },
});
// モデルの在り処を決める。--hf なら Hugging Face から落としてキャッシュする。
// ここで失敗したときは process.exit を呼ばない。Windows では fetch が開いたハンドルを
// 残したまま exit すると libuv が abort してしまうので、終了コードだけ立てて自然に終わらせる。
async function resolveModel(): Promise<string | null> {
  if (argv.model) return argv.model;
  if (!argv.hf) {
    console.error("usage: node src/server.ts --model <model.gguf> [--port 8080]");
    console.error("       node src/server.ts --hf <org/repo>[:<quant>] [--port 8080]");
    return null;
  }
  try {
    return await ensureModel(argv.hf, (m) => console.log(m));
  } catch (e) {
    console.error(String((e as Error).message ?? e));
    return null;
  }
}

const modelPath = await resolveModel();
if (modelPath === null) {
  process.exitCode = 1;
} else {
await start(modelPath);
}

async function start(modelPath: string) {
// WebGPU のネイティブ addon はここまで読み込まない。引数の誤りで終了するときに、
// 余計な後片付けを走らせないため。
const { loadModelNode } = await import("./node/gpu_node.ts");

const maxBatchTokens = Number(argv["max-batch-tokens"]);
const maxQueue = Number(argv["max-queue"]);
const { model, adapterInfo, loadMs } = await loadModelNode(modelPath, {
  weights: argv.weights as "q8" | "f32",
  maxTokens: maxBatchTokens,
  maxSeqLen: Number(argv["max-seq-len"]),
  maxSeqs: 128,
  kvCacheTokens: Number(argv["kv-cache-tokens"]),
});
const modelName = argv["model-name"] ?? basename(modelPath).replace(/\.gguf$/i, "");
const jev = new JevClassifier(model, modelName, argv.temperature ? Number(argv.temperature) : undefined);
console.log(`GPU: ${adapterInfo}`);
console.log(`model ${modelName} loaded in ${(loadMs / 1000).toFixed(1)}s (GPU ${(model.gpuBytes / 2 ** 20).toFixed(0)}MB, T=${jev.temperature.toFixed(4)})`);
console.log(`accepted model names: ${jev.acceptedModels.join(", ")}${argv["api-key"] ? " (API key required)" : ""}`);

// ---- 動的バッチング ----
interface Job {
  prep: PreparedRequest;
  tokens: number;
  resolve: (r: SystemOneResponse) => void;
  reject: (e: Error) => void;
}
const queue: Job[] = [];
let running = false;
const stats = { requests: 0, questions: 0, batches: 0, seqs: 0, tokens: 0, gpuMs: 0 };

async function drain() {
  if (running) return;
  running = true;
  try {
    while (queue.length) {
      const batch: Job[] = [];
      let tokens = 0, seqs = 0;
      while (queue.length) {
        const j = queue[0];
        if (batch.length && (tokens + j.tokens > maxBatchTokens || seqs + j.prep.seqs.length > model.maxSeqs)) break;
        batch.push(queue.shift()!);
        tokens += j.tokens;
        seqs += j.prep.seqs.length;
      }
      const tg = performance.now();
      try {
        const logits = await model.forward(batch.flatMap((j) => j.prep.seqs), jev.candIds);
        stats.batches++; stats.seqs += seqs; stats.tokens += tokens; stats.gpuMs += performance.now() - tg;
        let k = 0;
        for (const j of batch) {
          j.resolve(jev.finish(j.prep, logits.slice(k, k + j.prep.seqs.length)));
          k += j.prep.seqs.length;
        }
      } catch (e) {
        for (const j of batch) j.reject(e as Error);
      }
    }
  } finally {
    running = false;
  }
}

function submit(raw: unknown): Promise<SystemOneResponse> {
  const prep = jev.prepare(raw); // 422 はここで投げる
  // バッチ内で state を共有するぶん実際の計算量は小さいが、上限判定は素のトークン数で行う（安全側）
  const tokens = prep.seqs.reduce((a, s) => a + s.length, 0);
  if (tokens > maxBatchTokens || prep.seqs.length > model.maxSeqs)
    throw new JevError("request_too_large", `request needs ${tokens} tokens / ${prep.seqs.length} sequences (max ${maxBatchTokens} / ${model.maxSeqs}); reduce questions or state size`, "questions");
  if (queue.length >= maxQueue) throw new JevError("overloaded", "server is overloaded; retry after a short delay", undefined, 529);
  stats.requests++;
  stats.questions += prep.questions.length;
  return new Promise((resolve, reject) => {
    queue.push({ prep, tokens, resolve, reject });
    queueMicrotask(drain);
  });
}

// ---- Web UI（web/ をそのまま配信。ページ側は /health を見てサーバーモードで動く） ----
const webRoot = fileURLToPath(new URL("../web/", import.meta.url));
const webReady = existsSync(join(webRoot, "dist"));
if (!webReady) console.warn("web UI is not built: run `npm run build:web` to enable http://host:port/");
const MIME: Record<string, string> = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".wgsl": "text/plain; charset=utf-8", ".css": "text/css" };

function serveStatic(url: string, res: ServerResponse): boolean {
  const path = decodeURIComponent(url.split("?")[0]);
  const rel = path === "/" ? "index.html" : path.slice(1);
  if (rel !== "index.html" && !rel.startsWith("dist/")) return false;
  const file = normalize(join(webRoot, rel));
  if (!file.startsWith(normalize(webRoot)) || !existsSync(file) || !statSync(file).isFile()) return false;
  res.writeHead(200, { "content-type": MIME[extname(file)] ?? "application/octet-stream", "cache-control": "no-cache" });
  res.end(readFileSync(file));
  return true;
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (c: Buffer) => {
      size += c.length;
      if (size > 4 << 20) reject(new JevError("request_too_large", "body too large (max 4MB)"));
      chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf-8")));
    req.on("error", reject);
  });
}

function send(res: ServerResponse, status: number, body: unknown) {
  res.writeHead(status, { "content-type": "application/json; charset=utf-8", "access-control-allow-origin": "*" });
  res.end(JSON.stringify(body));
}

function sendError(res: ServerResponse, e: unknown) {
  if (e instanceof JevError) return send(res, e.status, e.body());
  console.error(e);
  send(res, 500, { error: { code: "internal_error", message: (e as Error).message } });
}

const server = createServer(async (req, res) => {
  try {
    if (req.method === "OPTIONS") {
      res.writeHead(204, { "access-control-allow-origin": "*", "access-control-allow-headers": "content-type, authorization", "access-control-allow-methods": "POST, GET" });
      return res.end();
    }
    if (req.method === "GET" && webReady && serveStatic(req.url ?? "/", res)) return;
    if (req.method === "GET" && req.url === "/health") {
      return send(res, 200, { status: "ok", model: modelName, models: jev.acceptedModels, temperature: jev.temperature,
        auth: !!argv["api-key"], queue: queue.length, stats,
        kv_cache: { capacity: model.kvCache.capacity, entries: model.kvCache.entries.length, ...model.kvCache.stats } });
    }
    if (req.method === "POST" && req.url === "/v1/systemone") {
      if (argv["api-key"] && req.headers.authorization !== `Bearer ${argv["api-key"]}`)
        throw new JevError("unauthorized", "missing or invalid API key", "Authorization", 401);
      let body: unknown;
      try {
        body = JSON.parse(await readBody(req));
      } catch (e) {
        if (e instanceof JevError) throw e;
        throw new JevError("invalid_json", "request body is not valid JSON");
      }
      return send(res, 200, await submit(body));
    }
    send(res, 404, { error: { code: "not_found", message: `${req.method} ${req.url}` } });
  } catch (e) {
    sendError(res, e);
  }
});
server.listen(Number(argv.port), argv.host, () => console.log(`listening on http://${argv.host}:${argv.port}`));
}

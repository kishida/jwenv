// HTTPサーバー（POST /v1/systemone）のスループット計測（同時接続数ごとの req/s と レイテンシ分布）
//   node test/bench_server.ts [http://127.0.0.1:8080] [--n 200] [--eval ../data/eval/test.jsonl]
// 評価セットの問題を1問ずつ choice の質問として投げる（プロンプト長の分布が実データに近くなる）
import { readFileSync } from "node:fs";

const args = process.argv.slice(2);
const base = args.find((a) => a.startsWith("http")) ?? "http://127.0.0.1:8080";
const opt = (k: string, d: string) => (args.includes(k) ? args[args.indexOf(k) + 1] : d);
const N = Number(opt("--n", "200"));
const evalPath = opt("--eval", "../data/eval/test.jsonl");

// eval行はプロンプト済みなので、元の質問・背景・選択肢を復元する
function parse(prompt: string) {
  const user = prompt.split("<|im_start|>user\n")[1].split("<|im_end|>")[0];
  const lines = user.split("\n");
  const instr = lines.findIndex((l) => l.startsWith("Answer the question with only"));
  const context = lines.slice(1, instr - 1).join("\n"); // "Context:" の次行から空行の手前まで
  const question = lines[instr + 1].slice("Question: ".length);
  const categories = lines.slice(instr + 3).map((l) => l.slice(3)); // "Options:" の次行から
  return { question, context, categories };
}
// 選択肢に改行を含むなど、プロンプトから正しく復元できない行は除外
const reqs = readFileSync(evalPath, "utf-8").split("\n").filter(Boolean)
  .map((l) => ({ row: JSON.parse(l), req: parse(JSON.parse(l).prompt) }))
  .filter(({ row, req }) => req.categories.length === row.n && req.categories.every((c) => c.trim()) && new Set(req.categories).size === row.n)
  .map(({ req }) => ({
    state: req.context ?? "（なし）",
    model: "jev-latest",
    questions: { q: { type: "choice", instructions: req.question, criteria: Object.fromEntries(req.categories.map((c) => [c, null])) } },
  }));

async function one(body: unknown) {
  const t0 = performance.now();
  const r = await fetch(`${base}/v1/systemone`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json();
  if (j.error) throw new Error(JSON.stringify(j.error));
  return performance.now() - t0;
}

console.log(`server ${base}, ${N} requests per setting (prompts from ${evalPath})`);
console.log("concurrency  req/s   p50ms   p95ms   max ms");
const results = [];
for (const conc of [1, 4, 16, 64]) {
  const lat: number[] = [];
  let next = 0;
  const t0 = performance.now();
  await Promise.all(Array.from({ length: conc }, async () => {
    while (next < N) {
      const i = next++;
      lat.push(await one(reqs[i % reqs.length]));
    }
  }));
  const sec = (performance.now() - t0) / 1000;
  lat.sort((a, b) => a - b);
  const q = (p: number) => lat[Math.min(lat.length - 1, Math.floor(p * lat.length))];
  const row = { concurrency: conc, rps: N / sec, p50: q(0.5), p95: q(0.95), max: lat[lat.length - 1] };
  results.push(row);
  console.log(`${String(conc).padStart(11)}  ${row.rps.toFixed(1).padStart(5)}  ${row.p50.toFixed(0).padStart(6)}  ${row.p95.toFixed(0).padStart(6)}  ${row.max.toFixed(0).padStart(7)}`);
}
const health = await (await fetch(`${base}/health`)).json();
console.log(JSON.stringify({ results, server_stats: health.stats }));

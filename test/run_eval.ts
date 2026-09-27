// 評価セット(data/eval/*.jsonl)の候補ラベルlogitを自作エンジンで計算し .npy に書き出す。
// Python側 scripts/evaluate.py --logits_dir で読み込んで同じ指標を計算する（Q8_0変換後のECE再計測、指示書7節）。
//   node test/run_eval.ts <model.gguf> <out_dir> [--weights q8|f32] [--batch-tokens 4096]
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { loadModelNode } from "../src/node/gpu_node.ts";

const args = process.argv.slice(2);
const [ggufPath, outDir] = args;
const opt = (k: string, d: string) => (args.includes(k) ? args[args.indexOf(k) + 1] : d);
const weights = opt("--weights", "q8") as "q8" | "f32";
const batchTokens = Number(opt("--batch-tokens", "4096"));
const evalDir = opt("--eval-dir", "../data/eval");

function writeNpy(path: string, data: Float32Array, shape: number[]) {
  let header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${shape.join(", ")}${shape.length === 1 ? "," : ""}), }`;
  const total = 10 + header.length + 1;
  header = header + " ".repeat((64 - (total % 64)) % 64) + "\n";
  const h = Buffer.alloc(10);
  h.write("\x93NUMPY", 0, "latin1");
  h[6] = 1; h[7] = 0;
  h.writeUInt16LE(header.length, 8);
  writeFileSync(path, Buffer.concat([h, Buffer.from(header, "latin1"), Buffer.from(data.buffer, data.byteOffset, data.byteLength)]));
}

const { model, adapterInfo } = await loadModelNode(ggufPath, { weights, maxTokens: batchTokens, maxSeqs: 128 });
console.log(`GPU: ${adapterInfo}, model ${ggufPath}`);
const CAND = [32, 33, 34, 35, 36, 37, 38, 39];
mkdirSync(outDir, { recursive: true });
const timing: Record<string, unknown> = {};
for (const set of ["val", "test", "posbias"]) {
  const rows = readFileSync(join(evalDir, `${set}.jsonl`), "utf-8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
  const toks = rows.map((r) => model.tokenizer.encode(r.prompt));
  const order = toks.map((_, i) => i).sort((a, b) => toks[a].length - toks[b].length);
  const out = new Float32Array(rows.length * 8).fill(-1e4);
  const t0 = performance.now();
  let i = 0, nTok = 0, nBatches = 0;
  while (i < order.length) {
    const idx: number[] = [];
    let t = 0;
    while (i < order.length && idx.length < 128 && (idx.length === 0 || t + toks[order[i]].length <= batchTokens)) {
      t += toks[order[i]].length;
      idx.push(order[i++]);
    }
    const lg = await model.forward(idx.map((j) => toks[j]), CAND);
    idx.forEach((j, k) => { for (let c = 0; c < rows[j].n; c++) out[j * 8 + c] = lg[k][c]; });
    nTok += t;
    nBatches++;
  }
  const sec = (performance.now() - t0) / 1000;
  timing[set] = { items: rows.length, tokens: nTok, batches: nBatches, seconds: sec, items_per_sec: rows.length / sec, tokens_per_sec: nTok / sec };
  console.log(set, JSON.stringify(timing[set]));
  writeNpy(join(outDir, `${set}.npy`), out, [rows.length, 8]);
}
writeFileSync(join(outDir, "engine_timing.json"), JSON.stringify({ model: ggufPath, weights, adapterInfo, timing }, null, 1));
model.destroy();
process.exit(0);

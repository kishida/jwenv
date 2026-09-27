// HF transformers (fp32) の中間出力と、自作WebGPUエンジンの中間出力を層ごとに比較する（指示書8-4節）。
//   node test/compare_hf.ts <model.gguf> <hf_dump_dir> [--weights q8|f32] [--min-cos 0.999]
// 各層で コサイン類似度 / 相対L2誤差 / 最大絶対誤差 を出し、最後に候補トークンのsoftmax確率を比較する。
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { loadModelNode } from "../src/node/gpu_node.ts";
import { readNpy } from "./npy.ts";

const args = process.argv.slice(2);
const ggufPath = args[0];
const dumpDir = args[1];
const weights = (args.includes("--weights") ? args[args.indexOf("--weights") + 1] : "q8") as "q8" | "f32";
const minCos = args.includes("--min-cos") ? Number(args[args.indexOf("--min-cos") + 1]) : 0.999;

function stats(a: Float32Array, b: Float32Array) {
  let dot = 0, na = 0, nb = 0, diff = 0, maxAbs = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
    const d = a[i] - b[i];
    diff += d * d;
    maxAbs = Math.max(maxAbs, Math.abs(d));
  }
  return { cos: dot / Math.sqrt(na * nb), relL2: Math.sqrt(diff / nb), maxAbs };
}
function softmax(x: ArrayLike<number>, n: number) {
  const a = Array.from(x).slice(0, n);
  const m = Math.max(...a);
  const e = a.map((v) => Math.exp(v - m));
  const s = e.reduce((p, c) => p + c, 0);
  return e.map((v) => v / s);
}

const { model, adapterInfo, loadMs } = await loadModelNode(ggufPath, { weights, maxTokens: 1024 });
console.log(`GPU: ${adapterInfo}\nmodel: ${ggufPath} (weights=${weights}) load ${(loadMs / 1000).toFixed(1)}s, GPU mem ${(model.gpuBytes / 2 ** 20).toFixed(0)}MB`);
const meta = JSON.parse(readFileSync(join(dumpDir, "meta.json"), "utf-8"));
const CAND = [32, 33, 34, 35, 36, 37, 38, 39];
let worst = 1;
let maxProbDiff = 0;
for (let pi = 0; pi < meta.length; pi++) {
  const dir = join(dumpDir, `p${pi}`);
  const hfTokens = Array.from(readNpy(join(dir, "tokens.npy")).data);
  const myTokens = model.tokenizer.encode(meta[pi].prompt);
  if (JSON.stringify(hfTokens) !== JSON.stringify(myTokens)) throw new Error(`prompt ${pi}: tokenization mismatch`);
  const names: string[] = meta[pi].names.filter((n: string) => n !== "label_logits");
  const order = (n: string) => (n === "final_norm" ? 1e9 : n.startsWith("L0.") && n !== "L0.out" ? -1 : Number(/^L(\d+)/.exec(n)?.[1] ?? 0));
  names.sort((a, b) => order(a) - order(b));
  const debug = { dump: new Set(names) } as { dump: Set<string>; dumps?: Map<string, { data: Float32Array; cols: number }> };
  const [logits] = await model.forward([myTokens], CAND, debug);
  console.log(`\n== prompt ${pi} (${myTokens.length} tokens) ==`);
  console.log("name".padEnd(16), "cos".padStart(10), "relL2".padStart(10), "maxAbs".padStart(10));
  for (const n of names) {
    const ref = readNpy(join(dir, `${n}.npy`)).data as Float32Array;
    const got = debug.dumps!.get(n);
    if (!got) { console.log(n.padEnd(16), "(not produced by engine)"); continue; }
    const s = stats(got.data.subarray(0, ref.length), ref);
    worst = Math.min(worst, s.cos);
    const flag = s.cos < minCos ? "  <-- MISMATCH" : "";
    if (n.startsWith("L0.") || n === "final_norm" || /^L\d+\.out$/.test(n) && (Number(n.slice(1, n.indexOf("."))) % 4 === 3 || flag))
      console.log(n.padEnd(16), s.cos.toFixed(7).padStart(10), s.relL2.toExponential(2).padStart(10), s.maxAbs.toExponential(2).padStart(10), flag);
  }
  const refLogits = readNpy(join(dir, "label_logits.npy")).data as Float32Array;
  const n = 5;
  const pr = softmax(refLogits, n), pg = softmax(logits, n);
  const d = Math.max(...pr.map((v, i) => Math.abs(v - pg[i])));
  maxProbDiff = Math.max(maxProbDiff, d);
  console.log("label logits HF    ", Array.from(refLogits).map((v) => v.toFixed(3)).join(" "));
  console.log("label logits engine", Array.from(logits).map((v) => v.toFixed(3)).join(" "));
  console.log(`softmax(A..E) HF     ${pr.map((v) => v.toFixed(4)).join(" ")}`);
  console.log(`softmax(A..E) engine ${pg.map((v) => v.toFixed(4)).join(" ")}   max|diff|=${d.toExponential(2)}`);
}
console.log(`\nworst cosine ${worst.toFixed(7)} (threshold ${minCos}), max prob diff ${maxProbDiff.toExponential(2)}`);
model.destroy();
process.exit(worst >= minCos ? 0 : 1);

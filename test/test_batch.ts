// パッキング（複数リクエストを1系列にまとめる）の正しさ: まとめて計算した結果 = 個別に計算した結果
//   node test/test_batch.ts <model.gguf>
import assert from "node:assert/strict";
import { loadModelNode } from "../src/node/gpu_node.ts";
import { buildPrompt } from "../src/jev.ts";

const { model } = await loadModelNode(process.argv[2], { maxTokens: 2048 });
const CAND = [32, 33, 34, 35, 36, 37, 38, 39];
const prompts = [
  buildPrompt("この顧客は怒っているように見えますか？", [["A", "yes"], ["B", "no"]], "お客様: 3回目の問い合わせです。まだ配送されていません。"),
  buildPrompt("Which is a mammal?", [["A", "shark"], ["B", "whale"], ["C", "trout"], ["D", "eel"]]),
  buildPrompt("日本の首都は？", [["A", "大阪"], ["B", "京都"], ["C", "東京"]], null),
  buildPrompt("x", [["A", "1"], ["B", "2"]]),
];
const seqs = prompts.map((p) => model.tokenizer.encode(p));
const single = [];
for (const s of seqs) single.push((await model.forward([s], CAND))[0]);
const packed = await model.forward(seqs, CAND);
const packedRev = await model.forward([...seqs].reverse(), CAND);
let maxd = 0;
for (let i = 0; i < seqs.length; i++) {
  for (let c = 0; c < 8; c++) {
    maxd = Math.max(maxd, Math.abs(single[i][c] - packed[i][c]), Math.abs(single[i][c] - packedRev[seqs.length - 1 - i][c]));
  }
}
console.log(`lengths ${seqs.map((s) => s.length)}; max |single - packed| logit diff = ${maxd.toExponential(2)}`);
assert.ok(maxd < 1e-3, "packed batch differs from single-sequence forward");
console.log("batch packing test passed");
model.destroy();
process.exit(0);

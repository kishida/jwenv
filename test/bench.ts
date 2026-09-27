// 推論速度: 単発レイテンシとパッキングによるスループット
//   node test/bench.ts <model.gguf> [--weights q8|f32]
import { loadModelNode } from "../src/node/gpu_node.ts";
import { buildPrompt } from "../src/jev.ts";

const args = process.argv.slice(2);
const weights = (args.includes("--weights") ? args[args.indexOf("--weights") + 1] : "q8") as "q8" | "f32";
const { model, adapterInfo } = await loadModelNode(args[0], { weights, maxTokens: 8192, maxSeqs: 128 });
console.log(`GPU: ${adapterInfo}, ${args[0]} (weights=${weights}, GPU ${(model.gpuBytes / 2 ** 20).toFixed(0)}MB)`);
const CAND = [32, 33, 34, 35, 36, 37, 38, 39];
// 背景の先頭を変えた別々のリクエスト（同じ文を並べるとバッチ内の接頭辞共有が効いて速く見えてしまうため）
const mk = (i: number) => model.tokenizer.encode(buildPrompt("この顧客の感情はどれに近いですか？",
  [["A", "annoyed"], ["B", "irritated"], ["C", "offended"], ["D", "furious"], ["E", "enraged"]],
  `受付${i}番 お客様: 3回目の問い合わせです。まだ配送されていません。もう限界です。`));
const seq = mk(0);
for (let i = 0; i < 3; i++) await model.forward([seq], CAND); // warmup
const N = 20;
let t0 = performance.now();
for (let i = 0; i < N; i++) await model.forward([seq], CAND);
const single = (performance.now() - t0) / N;
console.log(`single request (${seq.length} tokens): ${single.toFixed(1)} ms`);
for (const bs of [4, 16, 32, 64]) {
  const seqs = Array.from({ length: bs }, (_, i) => mk(i + 1));
  await model.forward(seqs, CAND);
  t0 = performance.now();
  const R = 5;
  for (let i = 0; i < R; i++) await model.forward(seqs, CAND);
  const ms = (performance.now() - t0) / R;
  console.log(`batch ${bs} x ${seq.length} tokens: ${ms.toFixed(1)} ms/batch, ${(bs / ms * 1000).toFixed(1)} req/s, ${(bs * seq.length / ms * 1000).toFixed(0)} tok/s`);
}
model.destroy();
process.exit(0);

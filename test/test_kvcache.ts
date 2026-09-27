// プレフィックスKVキャッシュの正しさ: キャッシュあり/なしで候補logitが一致すること
//   node test/test_kvcache.ts <model.gguf>
import assert from "node:assert/strict";
import { loadModelNode } from "../src/node/gpu_node.ts";
import { buildPrompt } from "../src/jev.ts";

const path = process.argv[2];
const CAND = [32, 33, 34, 35, 36, 37, 38, 39];
const ctx = "お客様: 3回目の問い合わせです。まだ配送されていません。もう限界です。返金してください。";
const P = (q: string, cats: string[], c: string | null = ctx) =>
  buildPrompt(q, cats.map((x, i) => ["ABCDEFGH"[i], x] as [string, string]), c);
const prompts = [
  P("この顧客は怒っているように見えますか？", ["yes", "no"]),
  P("この顧客は怒っているように見えますか？", ["no", "yes"]),                 // 選択肢だけ違う
  P("この顧客の感情はどれに近いですか？", ["annoyed", "irritated", "offended", "furious", "enraged"]), // 質問から違う
  P("この顧客は怒っているように見えますか？", ["yes", "no"]),                 // 完全一致の再送
  P("Which is a mammal?", ["shark", "whale", "trout"], null),
  P("この顧客は怒っているように見えますか？", ["yes", "no", "わからない"]),   // 既存系列の延長に近い
];

// 容量0（リクエストをまたぐキャッシュなし）: 同じバッチ内で接頭辞（同じ背景）を共有しても結果が変わらないこと
{
  const { model } = await loadModelNode(path, { maxTokens: 2048, kvCacheTokens: 0 });
  const toks = prompts.map((p) => model.tokenizer.encode(p));
  const ref: Float32Array[] = [];
  for (const t of toks) ref.push((await model.forward([t], CAND))[0]);
  const got = await model.forward(toks, CAND);
  let maxd = 0;
  got.forEach((g, i) => { for (let c = 0; c < 8; c++) maxd = Math.max(maxd, Math.abs(g[c] - ref[i][c])); });
  const st = model.kvCache.stats;
  console.log(`in-batch sharing: max diff = ${maxd.toExponential(2)}; shared ${st.batchShared} seqs, reused ${st.reusedTokens} / computed ${st.computedTokens} tokens`);
  assert.ok(maxd < 1e-3, "in-batch prefix sharing changed the logits");
  assert.ok(st.batchShared > 0);
  model.destroy();
}

// 容量あり: 新しい系列だけのバッチ（バッチ内共有 + 保存）→ その後の再利用でも結果が変わらないこと
{
  const { model } = await loadModelNode(path, { maxTokens: 2048, kvCacheTokens: 4096 });
  const toks = prompts.map((p) => model.tokenizer.encode(p));
  const ref: Float32Array[] = [];
  for (const t of toks) ref.push((await model.forward([t], CAND, undefined, { useCache: false }))[0]);
  model.kvCache.clear();
  let maxd = 0;
  const check = (g: Float32Array, i: number) => { for (let c = 0; c < 8; c++) maxd = Math.max(maxd, Math.abs(g[c] - ref[i][c])); };
  (await model.forward(toks, CAND)).forEach(check);
  for (let r = 0; r < 2; r++) for (let i = toks.length - 1; i >= 0; i--) check((await model.forward([toks[i]], CAND))[0], i);
  const st = model.kvCache.stats;
  console.log(`new batch then reuse: max diff = ${maxd.toExponential(2)}; shared ${st.batchShared}, hits ${st.hits}, reused ${st.reusedTokens}`);
  assert.ok(maxd < 1e-3, "cache entries written in a shared batch are wrong");
  model.destroy();
}

for (const cap of [4096, 160]) {  // 160: 追い出しが頻発する容量
  const { model } = await loadModelNode(path, { maxTokens: 2048, kvCacheTokens: cap });
  const toks = prompts.map((p) => model.tokenizer.encode(p));
  const ref: Float32Array[] = [];
  for (const t of toks) ref.push((await model.forward([t], CAND, undefined, { useCache: false }))[0]);
  let maxd = 0;
  const check = (got: Float32Array, i: number) => { for (let c = 0; c < 8; c++) maxd = Math.max(maxd, Math.abs(got[c] - ref[i][c])); };
  // 1件ずつ（前のリクエストのKVを再利用）
  for (let i = 0; i < toks.length; i++) check((await model.forward([toks[i]], CAND))[0], i);
  // バッチ（キャッシュ済みと新規が混在）
  const order = [5, 0, 2, 4, 1, 3];
  const got = await model.forward(order.map((i) => toks[i]), CAND);
  order.forEach((i, k) => check(got[k], i));
  const st = model.kvCache.stats;
  console.log(`cap=${cap}: max |cached - uncached| = ${maxd.toExponential(2)}; lookups ${st.lookups}, hits ${st.hits}, reused ${st.reusedTokens} / computed ${st.computedTokens} tokens, evictions ${st.evictions}`);
  assert.ok(maxd < 1e-3, "prefix KV cache changed the logits");
  assert.ok(st.reusedTokens > 0);
  model.destroy();
}
console.log("prefix KV cache test passed");
process.exit(0);

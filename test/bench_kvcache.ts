// プレフィックスKVキャッシュの効果: 同じ長い背景で、質問・選択肢だけ変えたリクエストを連続で投げる
//   node test/bench_kvcache.ts <model.gguf>
import { loadModelNode } from "../src/node/gpu_node.ts";
import { buildPrompt } from "../src/jev.ts";

const { model } = await loadModelNode(process.argv[2], { maxTokens: 4096, kvCacheTokens: 8192 });
const CAND = [32, 33, 34, 35, 36, 37, 38, 39];
const ctx = "お客様: 先月購入したノートパソコンについてです。届いた時点で画面に線が入っており、サポートに電話したところ" +
  "交換対応すると言われましたが、2週間たっても何の連絡もありません。問い合わせ番号も伝えましたし、メールも3回送りました。" +
  "仕事で使うので本当に困っています。このまま連絡がないなら消費者センターに相談することも考えています。至急対応してください。";
const tasks: [string, string[]][] = [
  ["この顧客は怒っているように見えますか？", ["yes", "no"]],
  ["この顧客の感情はどれに近いですか？", ["annoyed", "irritated", "offended", "furious", "enraged"]],
  ["この問い合わせの緊急度はどれですか？", ["低", "中", "高", "緊急"]],
  ["サポート担当者が次に取るべき対応はどれですか？", ["謝罪して返金", "交換品を送る", "使い方を案内", "上長にエスカレーション", "追加情報を依頼", "対応不要"]],
  ["このクレームの主な原因はどれですか？", ["配送", "品質", "価格", "接客", "説明不足", "システム障害"]],
  ["この顧客は解約しそうですか？", ["はい", "いいえ"]],
];
const seqs = tasks.map(([q, c]) => model.tokenizer.encode(buildPrompt(q, c.map((x, i) => ["ABCDEFGH"[i], x] as [string, string]), ctx)));
for (const s of seqs) await model.forward([s], CAND, undefined, { useCache: false }); // warmup
for (const useCache of [false, true]) {
  model.kvCache.clear();
  const times: number[] = [];
  for (let r = 0; r < 3; r++) for (const s of seqs) {
    const t0 = performance.now();
    await model.forward([s], CAND, undefined, { useCache });
    times.push(performance.now() - t0);
  }
  const first = times[0];
  const rest = times.slice(1).reduce((a, b) => a + b, 0) / (times.length - 1);
  console.log(`cache=${useCache}: prompt ~${seqs[0].length} tokens, first ${first.toFixed(1)} ms, subsequent avg ${rest.toFixed(1)} ms`);
}
console.log(JSON.stringify(model.kvCache.stats));
process.exit(0);

// HF transformers tokenizerとの完全一致テスト
import { readFileSync } from "node:fs";
import { parseGGUF } from "../src/gguf/parser.ts";
import { fileSource } from "../src/node/file_source.ts";
import { BPETokenizer } from "../src/tokenizer/bpe.ts";

// 任意の Qwen3 GGUF でよい（トークナイザだけを見るので量子化は問わない）
const model = process.argv[2] ?? process.env.JWENV_MODEL;
if (!model) {
  console.error("usage: node test/test_tokenizer.ts <model.gguf>   (or set JWENV_MODEL)");
  process.exit(1);
}
const g = await parseGGUF(fileSource(model));
const t0 = performance.now();
const tok = new BPETokenizer(g.metadata);
console.log(`tokenizer init ${(performance.now() - t0).toFixed(0)}ms`);
const cases = JSON.parse(readFileSync(new URL("./fixtures/tokenizer_cases.json", import.meta.url), "utf-8"));
let fail = 0;
const t1 = performance.now();
for (const c of cases) {
  const got = tok.encode(c.text);
  if (JSON.stringify(got) !== JSON.stringify(c.ids)) {
    if (fail < 5) console.log("MISMATCH", JSON.stringify(c.text).slice(0, 200), "\n got", got.slice(0, 40), "\n exp", c.ids.slice(0, 40));
    fail++;
  } else if (tok.decode(got) !== c.text.normalize("NFC") && tok.decode(got) !== c.text) {
    console.log("DECODE MISMATCH", JSON.stringify(c.text).slice(0, 100));
  }
}
console.log(`${cases.length - fail}/${cases.length} match (${(performance.now() - t1).toFixed(0)}ms)`);
if (fail) process.exit(1);

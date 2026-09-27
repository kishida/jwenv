// K-quant（Q2_K〜Q6_K）の逆量子化・GPU用再配置の単体テスト（正解は gguf-py の参照実装）
//   python test/make_kquant_fixture.py <dir>/qwen3-0.6b-jev-{Q2_K,Q3_K_M,Q4_K_M,Q5_K_M}.gguf
//   node test/test_kquant.ts <dir>        （または JWENV_MODELS=<dir>）
// cases.json が指す GGUF が要る。持っていなければ、手元の K-quant で作り直す。
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { parseGGUF, GGML_TYPE_NAME, type GGUFFile, type ByteSource } from "../src/gguf/parser.ts";
import { dequantRow, repackKQuant, KQUANT, QK_K } from "../src/gguf/dequant.ts";
import { fileSource } from "../src/node/file_source.ts";

const dir = new URL("./fixtures/kquant/", import.meta.url);
const { cases } = JSON.parse(readFileSync(new URL("cases.json", dir), "utf-8"));
// cases.json はファイル名だけを持つ。GGUF の置き場所はここで与える。
const models = process.argv[2] ?? process.env.JWENV_MODELS ?? ".";
const ggufPath = (name: string) => join(models, name);
const missing = [...new Set(cases.map((c: { gguf: string }) => c.gguf))]
  .filter((n) => !existsSync(ggufPath(n as string)));
if (missing.length) {
  console.error("K-quant のテストには、記録したときと同じ GGUF が要ります。見つからないもの:");
  for (const n of missing) console.error("  " + n);
  console.error("置き場所を渡してください: node test/test_kquant.ts <dir>  （または JWENV_MODELS=<dir>）");
  console.error("手元の K-quant で記録を作り直すこともできます: python test/make_kquant_fixture.py <*.gguf>");
  process.exit(1);
}
const files = new Map<string, { src: ByteSource; g: GGUFFile }>();
const count: Record<string, number> = {};
for (const c of cases) {
  if (!files.has(c.gguf)) {
    const src = fileSource(ggufPath(c.gguf));
    files.set(c.gguf, { src, g: await parseGGUF(src) });
  }
  const { src, g } = files.get(c.gguf)!;
  const t = g.tensors.get(c.tensor)!;
  assert.equal(GGML_TYPE_NAME[t.type], c.type, c.tensor);
  const bytes = await src.read(t.absOffset, t.nBytes);
  const got = dequantRow(t, bytes, c.row);
  const b = readFileSync(new URL(c.file, dir));
  const exp = new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
  let maxd = 0;
  for (let i = 0; i < exp.length; i++) maxd = Math.max(maxd, Math.abs(got[i] - exp[i]));
  assert.equal(maxd, 0, `${c.type} ${c.tensor} row ${c.row}: max diff ${maxd}`);

  // GPU 用の再配置から、シェーダ（main_kq）と同じ式で復元した値も一致すること
  const info = KQUANT[t.type];
  const cols = t.dims[0];
  const rb = (cols / QK_K) * info.bytes;
  const { quants, scales } = repackKQuant(t.type, bytes.subarray(c.row * rb, (c.row + 1) * rb), cols);
  const per = 32 / info.bits;
  for (let i = 0; i < cols; i++) {
    const code = (quants[Math.floor(i / per)] >>> (info.bits * (i % per))) & ((1 << info.bits) - 1);
    const gi = Math.floor(i / info.group);
    const v = Math.fround(scales[2 * gi] * (code - info.zero)) - scales[2 * gi + 1];
    assert.equal(v, got[i], `repack ${c.type} mismatch at ${i}`);
  }
  count[c.type] = (count[c.type] ?? 0) + 1;
}
console.log("rows checked per type:", count);
console.log("K-quant tests passed");

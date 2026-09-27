// GGUFパーサとQ8_0逆量子化の単体テスト（期待値は gguf-py の参照実装）
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { parseGGUF, GGMLType } from "../src/gguf/parser.ts";
import { dequantize, dequantRow, repackQ8_0, dequantQ8_0 } from "../src/gguf/dequant.ts";
import { fileSource } from "../src/node/file_source.ts";

const dir = new URL("./fixtures/", import.meta.url);
const src = fileSource(new URL("tiny.gguf", dir).pathname.replace(/^\/([A-Za-z]:)/, "$1"));
const g = await parseGGUF(src);

assert.equal(g.metadata.get("general.architecture"), "qwen3");
assert.equal(g.metadata.get("test.u32"), 123456);
assert.ok(Math.abs((g.metadata.get("test.f32") as number) - 1e-6) < 1e-12);
assert.equal(g.metadata.get("test.str"), "こんにちは");
assert.deepEqual(g.metadata.get("test.arr_str"), ["a", "bc", "日本"]);
assert.deepEqual([...(g.metadata.get("test.arr_i32") as Int32Array)], [1, -2, 3]);
assert.equal(g.metadata.get("test.bool"), true);

function expected(name: string) {
  const b = readFileSync(new URL(`${name}.f32`, dir));
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
}
function maxAbsDiff(a: Float32Array, b: Float32Array) {
  assert.equal(a.length, b.length);
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

for (const [name, type, dims] of [["t_f32", GGMLType.F32, [64, 3]], ["t_f16", GGMLType.F16, [96, 4]], ["t_q8", GGMLType.Q8_0, [128, 5]]] as const) {
  const t = g.tensors.get(name)!;
  assert.equal(t.type, type, name);
  assert.deepEqual(t.dims, dims, name);
  const bytes = await src.read(t.absOffset, t.nBytes);
  const got = dequantize(t, bytes);
  const d = maxAbsDiff(got, expected(name));
  assert.equal(d, 0, `${name} maxAbsDiff=${d}`);
  // 行単位
  const cols = t.dims[0];
  for (let r = 0; r < t.dims[1]; r++) {
    assert.equal(maxAbsDiff(dequantRow(t, bytes, r), expected(name).subarray(r * cols, (r + 1) * cols)), 0);
  }
  console.log(`ok ${name} (${t.nElements} elems, ${t.nBytes} bytes)`);
}

// Q8_0: ブロック構造の手計算テスト（f16スケール 0.5 = 0x3800、q = -128..127）
{
  const blk = new Uint8Array(34);
  blk[0] = 0x00; blk[1] = 0x38;
  for (let i = 0; i < 32; i++) blk[2 + i] = (i * 8 - 128) & 0xff;
  const x = dequantQ8_0(blk, 32);
  for (let i = 0; i < 32; i++) assert.equal(x[i], 0.5 * (i * 8 - 128));
  // GPU用の再配置の整合性
  const { scales, quants } = repackQ8_0(blk, 32);
  assert.equal(scales[0], 0.5);
  const q = new Int8Array(quants.buffer);
  for (let i = 0; i < 32; i++) assert.equal(q[i] * scales[0], x[i]);
  console.log("ok Q8_0 hand-made block + repack");
}
console.log("all GGUF tests passed");

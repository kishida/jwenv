// 最小限の .npy (float32 / int32, C-order) リーダ
import { readFileSync } from "node:fs";
export function readNpy(path: string): { shape: number[]; data: Float32Array | Int32Array } {
  const b = readFileSync(path);
  const hlen = b.readUInt16LE(8);
  const header = b.subarray(10, 10 + hlen).toString("latin1");
  const descr = /'descr':\s*'([^']+)'/.exec(header)![1];
  const shape = /'shape':\s*\(([^)]*)\)/.exec(header)![1].split(",").map((s) => s.trim()).filter(Boolean).map(Number);
  const raw = b.subarray(10 + hlen);
  const ab = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  if (descr === "<f4") return { shape, data: new Float32Array(ab) };
  if (descr === "<i4") return { shape, data: new Int32Array(ab) };
  throw new Error(`unsupported npy dtype ${descr}`);
}

// ブラウザ用 ByteSource: ローカルファイル(File)からGGUFを必要な範囲だけ読み込む
import type { ByteSource } from "../../src/gguf/parser.ts";

export function blobSource(blob: Blob): ByteSource {
  return {
    size: blob.size,
    async read(offset, length) {
      return new Uint8Array(await blob.slice(offset, offset + length).arrayBuffer());
    },
  };
}

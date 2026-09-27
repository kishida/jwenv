// Node用 ByteSource（fsでランダムアクセス読み込み）
import { openSync, readSync, fstatSync } from "node:fs";
import type { ByteSource } from "../gguf/parser.ts";

export function fileSource(path: string): ByteSource {
  const fd = openSync(path, "r");
  const size = fstatSync(fd).size;
  return {
    size,
    async read(offset: number, length: number) {
      const buf = new Uint8Array(length);
      let done = 0;
      while (done < length) {
        // readSyncは一度に2GB未満しか読めないので分割
        const n = readSync(fd, buf, done, Math.min(length - done, 1 << 30), offset + done);
        if (n <= 0) throw new Error("unexpected EOF");
        done += n;
      }
      return buf;
    },
  };
}

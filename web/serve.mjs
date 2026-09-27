// 静的ファイルサーバー（HTTP Range対応）。デモ確認用。
//   node serve.mjs [port]
import { createServer } from "node:http";
import { createReadStream, statSync } from "node:fs";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL(".", import.meta.url));
const port = Number(process.argv[2] ?? 8000);
const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".wgsl": "text/plain; charset=utf-8", ".json": "application/json", ".gguf": "application/octet-stream", ".css": "text/css" };

createServer((req, res) => {
  const url = decodeURIComponent(new URL(req.url, "http://x").pathname);
  const [base, rel] = [root, url === "/" ? "index.html" : url.slice(1)];
  const path = normalize(join(base, rel));
  if (!path.startsWith(base)) { res.writeHead(403).end(); return; }
  let st;
  try { st = statSync(path); } catch { res.writeHead(404).end("not found"); return; }
  if (!st.isFile()) { res.writeHead(404).end(); return; }
  const headers = { "content-type": types[extname(path)] ?? "application/octet-stream", "accept-ranges": "bytes", "cache-control": "no-cache" };
  const m = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range ?? "");
  if (m) {
    const start = Number(m[1]);
    const end = m[2] ? Math.min(Number(m[2]), st.size - 1) : st.size - 1;
    res.writeHead(206, { ...headers, "content-range": `bytes ${start}-${end}/${st.size}`, "content-length": end - start + 1 });
    createReadStream(path, { start, end }).pipe(res);
  } else {
    res.writeHead(200, { ...headers, "content-length": st.size });
    createReadStream(path).pipe(res);
  }
}).listen(port, "127.0.0.1", () => console.log(`http://127.0.0.1:${port}/`));

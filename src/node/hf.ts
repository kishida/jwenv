// Hugging Face から GGUF を取ってきてキャッシュする（Node 用）。
//   kishida/jwenv-4b-poc-gguf            リポジトリに .gguf が1つだけならそれ
//   kishida/jwenv-4b-poc-gguf:q4_k_m     量子化で絞る（大文字小文字は問わない）
//   kishida/jwenv-4b-poc-gguf/foo.gguf   ファイル名を直に指定
// 保存先は $JWENV_CACHE、既定で ~/.cache/jwenv/<org>/<repo>/<file>。
// 同じ大きさのファイルが既にあれば何もしない。非公開・ゲート付きなら $HF_TOKEN を使う。
import { createWriteStream } from "node:fs";
import { mkdir, rename, stat, unlink } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

const API = "https://huggingface.co/api/models";
const RESOLVE = "https://huggingface.co";

export interface HFSpec {
  repo: string;
  file?: string;
  quant?: string;
}

export function parseHFSpec(spec: string): HFSpec {
  const [left, quant] = spec.split(":");
  const parts = left.split("/").filter(Boolean);
  if (parts.length < 2) throw new Error(`not a Hugging Face repo: ${spec}`);
  const repo = parts.slice(0, 2).join("/");
  const file = parts.length > 2 ? parts.slice(2).join("/") : undefined;
  return { repo, file, quant };
}

function headers() {
  const token = process.env.HF_TOKEN || process.env.HUGGING_FACE_HUB_TOKEN;
  return token ? { authorization: `Bearer ${token}` } : {};
}

export function cacheDir() {
  return process.env.JWENV_CACHE || join(homedir(), ".cache", "jwenv");
}

async function listGGUF(repo: string): Promise<string[]> {
  const r = await fetch(`${API}/${repo}`, { headers: headers() });
  if (!r.ok) {
    // 本文を捨てておかないと、keep-alive の接続が残って process.exit で libuv が落ちる
    await r.body?.cancel();
    const hint = r.status === 401 || r.status === 403
      ? " — it may not exist, or be private or gated (then set HF_TOKEN)"
      : "";
    throw new Error(`Hugging Face: ${repo} returned ${r.status}${hint}`);
  }
  const j = (await r.json()) as { siblings?: { rfilename: string }[] };
  return (j.siblings ?? []).map((s) => s.rfilename).filter((n) => n.toLowerCase().endsWith(".gguf"));
}

/** spec からダウンロードすべきファイル名を1つ決める。決められなければ候補を並べて投げる。 */
export function pickFile(spec: HFSpec, files: string[]): string {
  if (spec.file) {
    if (!files.includes(spec.file)) throw new Error(`${spec.repo} has no ${spec.file}`);
    return spec.file;
  }
  let cand = files;
  if (spec.quant) {
    const q = spec.quant.toLowerCase();
    cand = files.filter((f) => f.toLowerCase().includes(q));
    if (!cand.length) throw new Error(`${spec.repo}: nothing matches "${spec.quant}"\n  ` + files.join("\n  "));
  }
  // 分割された GGUF（-00001-of-0000N）はまだ扱えない
  const split = cand.filter((f) => /-\d{5}-of-\d{5}\.gguf$/i.test(f));
  if (split.length) throw new Error(`${spec.repo}: split GGUF files are not supported yet\n  ` + split.join("\n  "));
  if (cand.length === 1) return cand[0];
  throw new Error(`${spec.repo}: several files match; add :quant or the file name\n  ` + cand.join("\n  "));
}

function human(n: number) {
  return n >= 1 << 30 ? `${(n / (1 << 30)).toFixed(2)} GB` : `${(n / (1 << 20)).toFixed(0)} MB`;
}

async function download(url: string, dest: string, onProgress?: (got: number, total: number) => void) {
  const r = await fetch(url, { headers: headers() });
  if (!r.ok || !r.body) {
    await r.body?.cancel();
    throw new Error(`download failed: ${r.status} ${url}`);
  }
  const total = Number(r.headers.get("content-length") ?? 0);
  await mkdir(dirname(dest), { recursive: true });
  const tmp = `${dest}.part`;
  let got = 0;
  const out = createWriteStream(tmp);
  await r.body.pipeTo(
    new WritableStream({
      write(chunk) {
        got += chunk.byteLength;
        onProgress?.(got, total);
        return new Promise<void>((res, rej) => out.write(chunk, (e) => (e ? rej(e) : res())));
      },
      close() {
        return new Promise<void>((res) => out.end(res));
      },
      abort(e) {
        out.destroy();
        return Promise.reject(e);
      },
    }),
  );
  await rename(tmp, dest);
}

/** spec のモデルをローカルに用意して、そのパスを返す。既にあれば何もしない。 */
export async function ensureModel(spec: string, log: (s: string) => void = () => {}): Promise<string> {
  const s = parseHFSpec(spec);
  const files = await listGGUF(s.repo);
  if (!files.length) throw new Error(`${s.repo} has no .gguf files`);
  const file = pickFile(s, files);
  const dest = join(cacheDir(), ...s.repo.split("/"), file);

  const head = await fetch(`${RESOLVE}/${s.repo}/resolve/main/${file}`, {
    method: "HEAD",
    headers: headers(),
    redirect: "follow",
  });
  const size = Number(head.headers.get("x-linked-size") ?? head.headers.get("content-length") ?? 0);

  try {
    const st = await stat(dest);
    if (!size || st.size === size) {
      log(`using cached ${dest}`);
      return dest;
    }
    log(`cached copy is ${human(st.size)}, expected ${human(size)} — downloading again`);
    await unlink(dest);
  } catch {
    // まだ無い
  }

  log(`downloading ${s.repo}/${file} (${human(size)}) to ${dest}`);
  let lastPct = -1;
  await download(`${RESOLVE}/${s.repo}/resolve/main/${file}`, dest, (got, total) => {
    const pct = total ? Math.floor((got / total) * 100) : -1;
    if (pct !== lastPct && pct % 5 === 0) {
      lastPct = pct;
      log(`  ${pct}%  ${human(got)}`);
    }
  });
  log("downloaded");
  return dest;
}

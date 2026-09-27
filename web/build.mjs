// Web UI のビルド: エンジン本体(../src)とブラウザ側(src)の TypeScript を ES モジュールにし、WGSL シェーダをコピーする
//   node web/build.mjs（npm run build:web）→ web/dist/
import { execFileSync } from "node:child_process";
import { cpSync, rmSync, mkdirSync } from "node:fs";
const here = new URL(".", import.meta.url);
rmSync(new URL("dist", here), { recursive: true, force: true });
execFileSync(process.execPath, [new URL("../node_modules/typescript/bin/tsc", here).pathname.replace(/^\/([A-Za-z]:)/, "$1"), "-p", "tsconfig.json"], { cwd: here, stdio: "inherit" });
mkdirSync(new URL("dist/web/src/shaders", here), { recursive: true });
cpSync(new URL("../src/gpu/shaders", here), new URL("dist/web/src/shaders", here), { recursive: true });
console.log("built web/dist");

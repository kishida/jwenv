// matmulシェーダ単体: CPU参照との一致 + 速度（GFLOPS）
//   node test/bench_matmul.ts [shader=matmul]
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createNodeDevice } from "../src/node/gpu_node.ts";
import { BufferPool, readBuffer, UNIFORM } from "../src/gpu/buffers.ts";
import { repackQ8_0 } from "../src/gguf/dequant.ts";

const shaderName = process.argv[2] ?? "matmul";
const { device } = await createNodeDevice();
const code = readFileSync(new URL(`../src/gpu/shaders/${shaderName}.wgsl`, import.meta.url), "utf-8");
const mod = device.createShaderModule({ code });
const pipes = { f32: device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main_f32" } }),
                q8: device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main_q8" } }) };
const pool = new BufferPool(device);

function makeQ8(N: number, K: number) {
  // ランダムなQ8_0ブロック列（f16スケール 2^-7 固定）
  const nb = (N * K) / 32;
  const bytes = new Uint8Array(nb * 34);
  const deq = new Float32Array(N * K);
  for (let b = 0; b < nb; b++) {
    bytes[b * 34] = 0x00; bytes[b * 34 + 1] = 0x20; // f16 0x2000 = 2^-7
    for (let i = 0; i < 32; i++) {
      const q = Math.floor(Math.random() * 255) - 127;
      bytes[b * 34 + 2 + i] = q & 0xff;
      deq[b * 32 + i] = q / 128;
    }
  }
  return { bytes, deq };
}

async function run(M: number, N: number, K: number, check: boolean) {
  const A = Float32Array.from({ length: M * K }, () => Math.random() * 2 - 1);
  const { bytes, deq } = makeQ8(N, K);
  const { scales, quants } = repackQ8_0(bytes, N * K);
  const bA = pool.upload(A), bC = pool.storage(M * N * 4), bW = pool.upload(deq), bQ = pool.upload(quants), bS = pool.upload(scales);
  const u = device.createBuffer({ size: 32, usage: UNIFORM });
  device.queue.writeBuffer(u, 0, new Uint32Array([M, N, K, 0, 0, 0, 0, 0]));
  const bg = {
    f32: device.createBindGroup({ layout: pipes.f32.getBindGroupLayout(0), entries: [0, 1, 2, 3].map((b) => ({ binding: b, resource: { buffer: [u, bA, bC, bW][b] } })) }),
    q8: device.createBindGroup({ layout: pipes.q8.getBindGroupLayout(0), entries: [[0, u], [1, bA], [2, bC], [4, bQ], [5, bS]].map(([b, x]) => ({ binding: b as number, resource: { buffer: x as GPUBuffer } })) }),
  };
  const res: string[] = [];
  for (const mode of ["f32", "q8"] as const) {
    const go = async (reps: number) => {
      const enc = device.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(pipes[mode]);
      p.setBindGroup(0, bg[mode]);
      for (let r = 0; r < reps; r++) p.dispatchWorkgroups(N / 64, Math.ceil(M / 64));
      p.end();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    };
    await go(1);
    if (check) {
      const C = await readBuffer(device, bC, 0, M * N * 4);
      let maxErr = 0;
      for (let m = 0; m < M; m += Math.max(1, Math.floor(M / 7))) for (let n = 0; n < N; n += 13) {
        let s = 0;
        for (let k = 0; k < K; k++) s += A[m * K + k] * deq[n * K + k];
        maxErr = Math.max(maxErr, Math.abs(s - C[m * N + n]));
      }
      assert.ok(maxErr < 1e-3, `${mode} mismatch ${maxErr}`);
    }
    const reps = 20;
    const t0 = performance.now();
    await go(reps);
    const ms = (performance.now() - t0) / reps;
    res.push(`${mode} ${ms.toFixed(3)}ms ${((2 * M * N * K) / ms / 1e6).toFixed(0)} GFLOPS`);
  }
  console.log(`M=${M} N=${N} K=${K}: ${res.join(" | ")}`);
}
await run(77, 192, 96, true);
await run(130, 256, 64, true);
for (const [M, N, K] of [[87, 1024, 1024], [87, 3072, 1024], [87, 1024, 3072], [512, 3072, 1024], [2048, 3072, 1024], [2048, 6144, 2048]]) await run(M, N, K, false);
process.exit(0);

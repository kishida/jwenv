// Node.js 用: Dawn(webgpuパッケージ)でGPUを作り、WGSLをファイルから読む
import { readFileSync } from "node:fs";
import { create, globals } from "webgpu";
import { requestDevice } from "../gpu/device.ts";
import { Qwen3Model, type LoadOptions } from "../model/qwen3.ts";
import { fileSource } from "./file_source.ts";

Object.assign(globalThis, globals);

export const nodeShaderLoader = async (name: string) =>
  readFileSync(new URL(`../gpu/shaders/${name}.wgsl`, import.meta.url), "utf-8");

// GPUオブジェクトがGCされるとDawnがクラッシュするので参照を保持しておく
let gpuInstance: GPU | null = null;

export async function createNodeDevice() {
  gpuInstance ??= create([]) as unknown as GPU;
  const info = await requestDevice(gpuInstance);
  info.device.addEventListener("uncapturederror", (e) => console.error("WebGPU error:", (e as GPUUncapturedErrorEvent).error.message));
  return info;
}

export async function loadModelNode(path: string, opts: LoadOptions = {}) {
  const { device, adapterInfo } = await createNodeDevice();
  const t0 = performance.now();
  const model = await Qwen3Model.load(fileSource(path), device, nodeShaderLoader, opts);
  return { model, device, adapterInfo, loadMs: performance.now() - t0 };
}

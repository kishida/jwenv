// Jev API（TypeSafe System One 互換、https://docs.typesafe.ai/api）のリクエスト検証・プロンプト構築・確率化。
// Node サーバーとブラウザで共通。
//
//   POST /v1/systemone
//   { "state": string | object | array, "model": "jev-latest",
//     "questions": { "<id>": { "type": "noul" | "choice" | "score", "instructions": ..., "criteria": ... }, ... } }
//   → { "model": ..., "answers": { "<id>": {...} }, "usage": { "input_tokens": n, "output_tokens": 0 } }
//
// 1つの state に対する複数の質問を1リクエストで受け取り、質問ごとに1系列（選択肢の並べ替えを使うならK系列）を作る。
// 全系列は同じ state の接頭辞を持つので、エンジン側で state 部分の K/V をバッチ内・リクエスト間で共有する。
import type { Qwen3Model } from "./model/qwen3.ts";

export const LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"];
export const MAX_OPTIONS = LABELS.length; // このモデルが扱える選択肢数の上限（TypeSafe は Choice 255 / Score 10）
export const MAX_QUESTIONS = 64;
export const MODEL_ALIASES = ["jev-latest"];

type Structured = string | Record<string, unknown> | unknown[];

export interface NoulQuestion {
  type: "noul";
  instructions: Structured;
  criteria?: { true?: Structured; false?: Structured };
}
export interface ChoiceQuestion {
  type: "choice";
  instructions: Structured;
  criteria: Record<string, Structured | null>;
}
export interface ScoreQuestion {
  type: "score";
  instructions: Structured;
  criteria: Structured[];
}
export type Question = NoulQuestion | ChoiceQuestion | ScoreQuestion;

export interface SystemOneRequest {
  state: Structured;
  model: string;
  questions: Record<string, Question>;
  // 拡張（TypeSafe の仕様にはない）
  options?: {
    temperature_scaling?: boolean; // 事後校正済みの確率を返す（既定 true）
    permutations?: number; // 選択肢の提示順を巡回シフトして K 回推論し平均する（位置バイアス緩和、既定1）
  };
}

export type Answer =
  | { type: "noul"; noul: number }
  | { type: "choice"; choice: string; probabilities: Record<string, number>; confidence: number }
  | { type: "score"; score: number; legend: Record<string, string>; probabilities: Record<string, number>; confidence: number };

export interface SystemOneResponse {
  model: string;
  answers: Record<string, Answer>;
  usage: { input_tokens: number; output_tokens: number };
}

export interface ErrorBody {
  error: { code: string; message: string; field?: string };
}

/** HTTP ステータス付きのエラー（422: 検証エラー、401: 認証、529: 過負荷） */
export class JevError extends Error {
  code: string;
  status: number;
  field?: string;
  constructor(code: string, message: string, field?: string, status = 422) {
    super(message);
    this.code = code;
    this.field = field;
    this.status = status;
  }
  body(): ErrorBody {
    return { error: { code: this.code, message: this.message, ...(this.field ? { field: this.field } : {}) } };
  }
}

/** Python側 jev/prompt.py の build_prompt（＝llama-server の /v1/systemone）と完全一致させること */
export function buildPrompt(question: string, labelMap: [string, string][], context?: string | null, system?: string): string {
  const sysPart = system ? `<|im_start|>system\n${system}<|im_end|>\n` : "";
  const opts = labelMap.map(([l, c]) => `${l}: ${c}`).join("\n");
  return (
    `${sysPart}<|im_start|>user\nContext:\n${context || "(none)"}\n\n` +
    "Answer the question with only the label of the best option (the character before the colon), nothing else.\n" +
    `Question: ${question}\nOptions:\n${opts}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
  );
}

// ---------------------------------------------------------------- 検証

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const isStructured = (v: unknown): v is Structured =>
  (typeof v === "string" && v.trim() !== "") || Array.isArray(v) || isObj(v);

/** string はそのまま、object / array は JSON 文字列にしてプロンプトへ入れる */
function asText(v: Structured, pretty = false): string {
  return typeof v === "string" ? v : JSON.stringify(v, null, pretty ? 2 : undefined);
}

export function validate(raw: unknown, acceptedModels: string[]): SystemOneRequest {
  if (!isObj(raw)) throw new JevError("invalid_request", "request body must be a JSON object");
  const r = raw as Record<string, unknown>;
  if (!isStructured(r.state)) throw new JevError("invalid_state", "state must be a non-empty string, object, or array", "state");
  if (typeof r.model !== "string" || !r.model) throw new JevError("missing_field", "model is required", "model");
  if (!acceptedModels.includes(r.model))
    throw new JevError("unknown_model", `unknown model "${r.model}" (available: ${acceptedModels.join(", ")})`, "model");
  if (!isObj(r.questions)) throw new JevError("invalid_questions", "questions must be a map of question id to question", "questions");
  const ids = Object.keys(r.questions);
  if (ids.length === 0) throw new JevError("invalid_questions", "questions must contain at least one question", "questions");
  if (ids.length > MAX_QUESTIONS) throw new JevError("too_many_questions", `at most ${MAX_QUESTIONS} questions per request`, "questions");
  for (const id of ids) {
    const q = r.questions[id];
    const f = `questions.${id}`;
    if (!isObj(q)) throw new JevError("invalid_question", "question must be an object", f);
    if (!isStructured(q.instructions)) throw new JevError("invalid_instructions", "instructions must be a non-empty string, object, or array", `${f}.instructions`);
    if (q.type === "noul") {
      if (q.criteria !== undefined) {
        if (!isObj(q.criteria)) throw new JevError("invalid_criteria", "noul criteria must be an object with optional true / false", `${f}.criteria`);
        for (const k of Object.keys(q.criteria)) {
          if (k !== "true" && k !== "false") throw new JevError("invalid_criteria", `unknown noul criteria key "${k}"`, `${f}.criteria.${k}`);
          if (!isStructured(q.criteria[k])) throw new JevError("invalid_criteria", "noul criteria values must be non-empty", `${f}.criteria.${k}`);
        }
      }
    } else if (q.type === "choice") {
      if (!isObj(q.criteria)) throw new JevError("invalid_criteria", "choice criteria must be a map of option to description (or null)", `${f}.criteria`);
      const opts = Object.keys(q.criteria);
      if (opts.length < 2) throw new JevError("too_few_options", "choice criteria must contain at least 2 options", `${f}.criteria`);
      if (opts.length > MAX_OPTIONS) throw new JevError("too_many_options", `this model supports at most ${MAX_OPTIONS} options per choice`, `${f}.criteria`);
      for (const o of opts) {
        if (!o.trim()) throw new JevError("invalid_criteria", "option names must be non-empty", `${f}.criteria`);
        const d = q.criteria[o];
        if (d !== null && !isStructured(d)) throw new JevError("invalid_criteria", "option description must be a string, object, array, or null", `${f}.criteria.${o}`);
      }
    } else if (q.type === "score") {
      if (!Array.isArray(q.criteria)) throw new JevError("invalid_criteria", "score criteria must be an ordered array of level descriptions", `${f}.criteria`);
      if (q.criteria.length < 2) throw new JevError("too_few_levels", "score criteria must contain at least 2 levels", `${f}.criteria`);
      if (q.criteria.length > MAX_OPTIONS) throw new JevError("too_many_levels", `this model supports at most ${MAX_OPTIONS} levels per score`, `${f}.criteria`);
      q.criteria.forEach((d, i) => {
        if (!isStructured(d)) throw new JevError("invalid_criteria", "level description must be a non-empty string, object, or array", `${f}.criteria.${i}`);
      });
    } else {
      throw new JevError("invalid_type", 'type must be "noul", "choice", or "score"', `${f}.type`);
    }
  }
  const o = r.options;
  if (o !== undefined) {
    if (!isObj(o)) throw new JevError("invalid_options", "options must be an object", "options");
    if (o.temperature_scaling !== undefined && typeof o.temperature_scaling !== "boolean")
      throw new JevError("invalid_options", "options.temperature_scaling must be a boolean", "options.temperature_scaling");
    if (o.permutations !== undefined && (!Number.isInteger(o.permutations) || (o.permutations as number) < 1 || (o.permutations as number) > MAX_OPTIONS))
      throw new JevError("invalid_options", `options.permutations must be an integer in [1, ${MAX_OPTIONS}]`, "options.permutations");
  }
  return r as unknown as SystemOneRequest;
}

// ---------------------------------------------------------------- 確率 → 回答

const round = (x: number, d = 4) => Math.round(x * 10 ** d) / 10 ** d;

/** confidence = 1 - 正規化エントロピー（一様分布で0、1点集中で1） */
export function confidenceOf(p: number[]): number {
  const n = p.length;
  let h = 0;
  for (const x of p) if (x > 0) h -= x * Math.log(x);
  return Math.max(0, 1 - h / Math.log(n));
}

interface PreparedQuestion {
  id: string;
  q: Question;
  keys: string[]; // 元の並び（choice: 選択肢名 / score: "0".. / noul: ["true","false"]）
  seqStart: number;
  perms: number[][]; // perms[k][pos] = 提示位置posに置いた元の選択肢index
}

export interface PreparedRequest {
  req: SystemOneRequest;
  seqs: number[][];
  questions: PreparedQuestion[];
}

export class JevClassifier {
  readonly model: Qwen3Model;
  readonly modelName: string;
  temperature: number;
  readonly candIds: number[];

  constructor(model: Qwen3Model, modelName: string, temperature?: number) {
    this.model = model;
    this.modelName = modelName;
    this.temperature = temperature ?? (model.gguf.metadata.get("jev.temperature") as number | undefined) ?? 1.0;
    this.candIds = LABELS.map((l) => {
      const ids = model.tokenizer.encode(l);
      if (ids.length !== 1) throw new Error(`label ${l} is not a single token`);
      return ids[0];
    });
  }

  get acceptedModels() {
    return [...MODEL_ALIASES, this.modelName];
  }

  /** 検証して、質問ごとのプロンプト（トークン列）を作る */
  prepare(raw: unknown): PreparedRequest {
    const req = validate(raw, this.acceptedModels);
    const context = asText(req.state, true);
    const K = req.options?.permutations ?? 1;
    const seqs: number[][] = [];
    const questions: PreparedQuestion[] = [];
    for (const [id, q] of Object.entries(req.questions)) {
      const question = asText(q.instructions);
      let keys: string[];
      let texts: string[];
      if (q.type === "noul") {
        keys = ["true", "false"];
        const t = q.criteria?.true, f = q.criteria?.false;
        texts = [t !== undefined ? `yes: ${asText(t)}` : "yes", f !== undefined ? `no: ${asText(f)}` : "no"];
      } else if (q.type === "choice") {
        keys = Object.keys(q.criteria);
        texts = keys.map((k) => (q.criteria[k] == null ? k : `${k}: ${asText(q.criteria[k]!)}`));
      } else {
        keys = q.criteria.map((_, i) => String(i));
        texts = q.criteria.map((d) => asText(d));
      }
      const n = keys.length;
      const k = Math.min(K, n);
      const pq: PreparedQuestion = { id, q, keys, seqStart: seqs.length, perms: [] };
      for (let s = 0; s < k; s++) {
        const shift = Math.round((s * n) / k);
        const perm = Array.from({ length: n }, (_, i) => (i + shift) % n);
        const labelMap = perm.map((ci, pos) => [LABELS[pos], texts[ci]] as [string, string]);
        const ids = this.model.tokenizer.encode(buildPrompt(question, labelMap, context));
        if (ids.length > this.model.maxSeqLen)
          throw new JevError("context_too_long", `prompt for question "${id}" is ${ids.length} tokens (max ${this.model.maxSeqLen})`, "state");
        seqs.push(ids);
        pq.perms.push(perm);
      }
      questions.push(pq);
    }
    return { req, seqs, questions };
  }

  /** forward の結果（各系列の候補 logit）から回答を作る */
  finish(p: PreparedRequest, logits: Float32Array[]): SystemOneResponse {
    const T = (p.req.options?.temperature_scaling ?? true) ? this.temperature : 1.0;
    const answers: Record<string, Answer> = {};
    for (const pq of p.questions) {
      const n = pq.keys.length;
      const acc = new Array(n).fill(0);
      pq.perms.forEach((perm, s) => {
        const lg = logits[pq.seqStart + s];
        const z = Array.from(lg.subarray(0, n), (v) => v / T);
        const m = Math.max(...z);
        const e = z.map((v) => Math.exp(v - m));
        const sum = e.reduce((a, b) => a + b, 0);
        e.forEach((v, pos) => (acc[perm[pos]] += v / sum / pq.perms.length));
      });
      if (pq.q.type === "noul") {
        answers[pq.id] = { type: "noul", noul: round(acc[0]) };
      } else if (pq.q.type === "choice") {
        let best = 0;
        acc.forEach((v, i) => { if (v > acc[best]) best = i; });
        answers[pq.id] = {
          type: "choice",
          choice: pq.keys[best],
          probabilities: Object.fromEntries(pq.keys.map((k, i) => [k, round(acc[i])])),
          confidence: round(confidenceOf(acc)),
        };
      } else {
        const crit = pq.q.criteria;
        answers[pq.id] = {
          type: "score",
          score: round(acc.reduce((a, v, i) => a + i * v, 0)),
          legend: Object.fromEntries(crit.map((d, i) => [String(i), asText(d)])),
          probabilities: Object.fromEntries(acc.map((v, i) => [String(i), round(v)])),
          confidence: round(confidenceOf(acc)),
        };
      }
    }
    return {
      model: this.modelName,
      answers,
      // 生成はしないので output_tokens は常に0（候補ラベルの次トークン確率を読むだけ）
      usage: { input_tokens: p.seqs.reduce((a, s) => a + s.length, 0), output_tokens: 0 },
    };
  }

  /** 単発（バッチングなし）で評価する */
  async systemOne(raw: unknown): Promise<SystemOneResponse> {
    const p = this.prepare(raw);
    const logits = await this.model.forward(p.seqs, this.candIds);
    return this.finish(p, logits);
  }
}

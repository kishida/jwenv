"""HF transformers (fp32) の中間出力を層ごとに .npy で書き出す（自作エンジンとの数値比較用）。

python test/dump_hf.py <hf_model_or_dir> <out_dir> [--adapter DIR] [--gguf-weights MODEL.gguf]

--gguf-weights を付けると、HF モデルの重みを GGUF の値（gguf-py で逆量子化したもの）に差し替える。
量子化モデルでエンジンの計算そのものが正しいか（量子化誤差を除いて一致するか）を確かめるために使う。
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jev.prompt import LABEL_TOKEN_IDS, assign_labels, build_prompt

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("out"); ap.add_argument("--adapter"); ap.add_argument("--gguf-weights")
a = ap.parse_args()
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="eager")
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
if a.gguf_weights:
    from gguf import GGUFReader, quants
    names = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
             "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
             "ffn_down": "mlp.down_proj", "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
             "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm"}
    sd = model.state_dict()
    with torch.no_grad():
        for t in GGUFReader(a.gguf_weights).tensors:
            w = torch.from_numpy(np.ascontiguousarray(quants.dequantize(t.data, t.tensor_type), dtype=np.float32))
            if t.name == "token_embd.weight":
                key = "model.embed_tokens.weight"
            elif t.name == "output_norm.weight":
                key = "model.norm.weight"
            elif t.name == "output.weight":
                key = "lm_head.weight"
            else:
                _, i, kind, _ = t.name.split(".")
                key = f"model.layers.{i}.{names[kind]}.weight"
            sd[key].copy_(w.reshape(sd[key].shape))
    print("weights replaced from", a.gguf_weights)
model.eval()

lm, _, _ = assign_labels(["annoyed", "irritated", "offended", "furious", "enraged"], None, shuffle=False)
prompts = [
    build_prompt("この顧客の感情はどれに近いですか？", lm, "お客様: 3回目の問い合わせです。まだ配送されていません。もう限界です。"),
    build_prompt("植物が光合成を行うために必要なものはどれか？", dict(zip("ABCD", ["水", "二酸化炭素", "光", "すべて正しい"]))),
]
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
meta = []
for pi, p in enumerate(prompts):
    ids = tok.encode(p, add_special_tokens=False)
    caps = {}
    hooks = []
    L0 = model.model.layers[0]
    def save(name):
        def f(mod, inp, outp):
            o = outp[0] if isinstance(outp, tuple) else outp
            caps[name] = o.detach()[0].reshape(len(ids), -1).numpy().astype(np.float32)
        return f
    def save_in(name):
        def f(mod, inp):
            caps[name] = inp[0].detach()[0].reshape(len(ids), -1).numpy().astype(np.float32)
        return f
    hooks.append(L0.input_layernorm.register_forward_hook(save("L0.attn_norm")))
    hooks.append(L0.self_attn.q_proj.register_forward_hook(save("L0.q")))
    hooks.append(L0.self_attn.k_proj.register_forward_hook(save("L0.k")))
    hooks.append(L0.self_attn.v_proj.register_forward_hook(save("L0.v")))
    hooks.append(L0.self_attn.q_norm.register_forward_hook(save("L0.q_norm")))
    hooks.append(L0.self_attn.k_norm.register_forward_hook(save("L0.k_norm")))
    hooks.append(L0.self_attn.o_proj.register_forward_pre_hook(save_in("L0.attn")))
    hooks.append(L0.post_attention_layernorm.register_forward_pre_hook(save_in("L0.resid_attn")))
    hooks.append(L0.post_attention_layernorm.register_forward_hook(save("L0.ffn_norm")))
    hooks.append(L0.mlp.down_proj.register_forward_pre_hook(save_in("L0.mlp_act")))
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(save(f"L{i}.out")))
    hooks.append(model.model.norm.register_forward_hook(save("final_norm")))
    with torch.no_grad():
        o = model(torch.tensor([ids]))
    for h in hooks: h.remove()
    caps["label_logits"] = o.logits[0, -1, LABEL_TOKEN_IDS].numpy().astype(np.float32)[None, :]
    d = out / f"p{pi}"; d.mkdir(exist_ok=True)
    np.save(d / "tokens.npy", np.array(ids, dtype=np.int32))
    for k, v in caps.items():
        np.save(d / f"{k}.npy", v)
    meta.append({"prompt": p, "n_tokens": len(ids), "names": sorted(caps)})
    print(pi, len(ids), "tokens; label logits", caps["label_logits"][0].round(3))
json.dump(meta, open(out / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

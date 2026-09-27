"""dequant単体テスト用の小さなGGUFと期待値を作る（gguf-pyの参照実装を正とする）。"""
import sys
from pathlib import Path
import numpy as np
import gguf
from gguf import quants

out = Path(sys.argv[1] if len(sys.argv) > 1 else "test/fixtures")
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
w = gguf.GGUFWriter(str(out / "tiny.gguf"), "qwen3")
w.add_uint32("test.u32", 123456)
w.add_float32("test.f32", 1e-6)
w.add_string("test.str", "こんにちは")
w.add_array("test.arr_str", ["a", "bc", "日本"])
w.add_array("test.arr_i32", [1, -2, 3])
w.add_bool("test.bool", True)
exp = {}
a = rng.standard_normal((3, 64)).astype(np.float32)
w.add_tensor("t_f32", a); exp["t_f32"] = a
b = (rng.standard_normal((4, 96)) * 3).astype(np.float16)
w.add_tensor("t_f16", b); exp["t_f16"] = b.astype(np.float32)
c = (rng.standard_normal((5, 128)) * np.array([0.01, 1, 10, 100, 0.5])[:, None]).astype(np.float32)
c[0, :32] = 0.0  # 全ゼロブロック（スケール0）
qc = quants.quantize(c, gguf.GGMLQuantizationType.Q8_0)
w.add_tensor("t_q8", qc, raw_shape=qc.shape, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
exp["t_q8"] = quants.dequantize(qc, gguf.GGMLQuantizationType.Q8_0).astype(np.float32)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
for k, v in exp.items():
    v.astype("<f4").tofile(out / f"{k}.f32")
print({k: v.shape for k, v in exp.items()})

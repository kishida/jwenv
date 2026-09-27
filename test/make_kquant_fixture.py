"""K-quant（Q2_K〜Q6_K）の逆量子化テスト用に、gguf-py の参照実装で実ファイルの一部を逆量子化して書き出す。
型ごとに最大2テンソル、各3行（先頭・2行目・最終行）を取り出す。
python test/make_kquant_fixture.py <a.gguf> [<b.gguf> ...]
"""
import json, sys
from pathlib import Path
from gguf import GGUFReader, quants

out = Path("test/fixtures/kquant")
out.mkdir(parents=True, exist_ok=True)
for f in out.glob("*.f32"):
    f.unlink()
cases, seen = [], {}
for src in sys.argv[1:]:
    r = GGUFReader(src)
    for t in r.tensors:
        ty = t.tensor_type.name
        if not ty.endswith("_K") or seen.get(ty, 0) >= 2:
            continue
        seen[ty] = seen.get(ty, 0) + 1
        cols = int(t.shape[0])
        deq = quants.dequantize(t.data, t.tensor_type).reshape(-1, cols)
        for row in sorted({0, 1, deq.shape[0] - 1}):
            fn = f"{len(cases):03d}.f32"
            deq[row].astype("<f4").tofile(out / fn)
            cases.append({"gguf": str(Path(src).resolve()), "tensor": t.name, "type": ty, "row": row, "file": fn})
json.dump({"cases": cases}, open(out / "cases.json", "w"), indent=1)
print(len(cases), "cases:", seen)

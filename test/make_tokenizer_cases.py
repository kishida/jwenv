"""HF tokenizerの出力を正解としてトークナイザ比較ケースを作る。"""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from transformers import AutoTokenizer
from jev.prompt import assign_labels, build_prompt
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
rng = random.Random(0)
texts = [
    "Hello world", "I'm DON'T we'll they'VE", "  multiple   spaces\t\ttabs\n\n\nnewlines  \n ", "1234567890 3.14159",
    "日本語のテキスト、句読点。「かぎ括弧」（全角）ＡＢＣ１２３", "emoji 😀👍🏽 and ümlauts café", "\r\nCRLF\r\n", "a" * 50,
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", "<think>not special?</think><|endoftext|>",
    "ガ NFC test é", "x<y and y>z <|im_start", "Ｃ＋＋ と C++ と C#", "    def f(x):\n        return x**2\n",
]
for f in ["data/unified/objective_test.jsonl", "data/unified/objective_val.jsonl", "data/unified/subjective_all.jsonl"]:
    if not Path(f).exists(): continue
    for l in open(f, encoding="utf-8"):
        it = json.loads(l)
        lm, _, _ = assign_labels(it["choices"], None, shuffle=True, rng=rng)
        texts.append(build_prompt(it["question"], lm, it.get("context")))
cases = [{"text": t, "ids": tok.encode(t, add_special_tokens=False)} for t in texts]
Path("qwen3-engine/test/fixtures").mkdir(parents=True, exist_ok=True)
json.dump(cases, open("qwen3-engine/test/fixtures/tokenizer_cases.json", "w", encoding="utf-8"), ensure_ascii=False)
print(len(cases), "cases")

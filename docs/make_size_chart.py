"""README に貼る「サイズと正解率」の図を SVG で書く。

GitHub の README は HTML を描画しないので、Chart.js の図は使えない。依存を増やさず、
ライト／ダークどちらでも読めるよう、色は prefers-color-scheme で切り替える。
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "engine/jwenv-engine/docs/size-vs-accuracy.svg"

# (label, GiB, accuracy, tuned)
# 図は1モデル1点に絞る（量子化ごとの値は表のほうに載せる）。学習前後で量子化をそろえてある。
POINTS = [
    ("0.6B", 0.60, 0.695, True),
    ("1.7B", 1.71, 0.804, True),
    ("4B", 2.33, 0.868, True),
    ("0.6B ", 0.75, 0.513, False),
    ("1.7B ", 2.02, 0.713, False),
    ("4B Instruct", 2.33, 0.831, False),
]
PAIRS = [("0.6B ", "0.6B"), ("1.7B ", "1.7B"), ("4B Instruct", "4B")]
CHANCE = 0.283

W, H = 760, 420
L, R, T, B = 64, 180, 30, 54          # 右は凡例とラベルの余白
X0, X1 = 0.40, 2.75                    # GiB
Y0, Y1 = 0.45, 0.92

# ラベルの置き方 (dx, dy, anchor)
NUDGE = {                      # jwenv は点の上、素の Qwen3 は点の下に置いて、ぶつからないようにする
    "0.6B": (0, -13, "middle"),
    "1.7B": (0, -13, "middle"),
    "4B": (0, -13, "middle"),
    "0.6B ": (0, 19, "middle"),
    "1.7B ": (0, 19, "middle"),
    "4B Instruct": (0, 19, "middle"),
}


def sx(gb):
    return L + (gb - X0) / (X1 - X0) * (W - L - R)


def sy(acc):
    return H - B - (acc - Y0) / (Y1 - Y0) * (H - B - T)


def main():
    by = {p[0]: p for p in POINTS}
    e = []
    e.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
             f'height="{H}" font-family="system-ui, -apple-system, Segoe UI, sans-serif">')
    e.append("""<style>
  .bg{fill:#ffffff}.fg{fill:#1d1d1b}.mut{fill:#6b6b66}.grid{stroke:#e2e1dc}
  .base{fill:#2a78d6}.tuned{fill:#eb6834}.link{stroke:#eb6834;stroke-opacity:.45}
  .chance{stroke:#9a9993}
  @media (prefers-color-scheme: dark){
    .bg{fill:#1f1f1d}.fg{fill:#ecebe6}.mut{fill:#9a9993}.grid{stroke:#34332f}
    .base{fill:#3987e5}.tuned{fill:#f07a4a}.link{stroke:#f07a4a;stroke-opacity:.5}
  }
</style>""")
    e.append(f'<rect class="bg" width="{W}" height="{H}" rx="8"/>')

    for acc in (0.5, 0.6, 0.7, 0.8, 0.9):
        y = sy(acc)
        e.append(f'<line class="grid" x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}"/>')
        e.append(f'<text class="mut" x="{L - 10}" y="{y + 4:.1f}" font-size="12" '
                 f'text-anchor="end">{acc:.1f}</text>')
    for gb in (0.5, 1.0, 1.5, 2.0, 2.5):
        x = sx(gb)
        e.append(f'<line class="grid" x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{H - B}"/>')
        e.append(f'<text class="mut" x="{x:.1f}" y="{H - B + 18}" font-size="12" '
                 f'text-anchor="middle">{gb:g} GB</text>')

    y = sy(CHANCE) if CHANCE > Y0 else None
    e.append(f'<text class="mut" x="{L}" y="{H - 12}" font-size="12">'
             f'GGUF のファイルサイズ　／　当てずっぽうは {CHANCE}</text>')

    for a, b in PAIRS:
        pa, pb = by[a], by[b]
        e.append(f'<line class="link" x1="{sx(pa[1]):.1f}" y1="{sy(pa[2]):.1f}" '
                 f'x2="{sx(pb[1]):.1f}" y2="{sy(pb[2]):.1f}" stroke-width="2"/>')

    for label, gb, acc, tuned in POINTS:
        x, y = sx(gb), sy(acc)
        cls = "tuned" if tuned else "base"
        if tuned:
            e.append(f'<rect class="{cls}" x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" '
                     f'transform="rotate(45 {x:.1f} {y:.1f})"/>')
        else:
            e.append(f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="5"/>')
        dx, dy, anchor = NUDGE[label]
        e.append(f'<text class="fg" x="{x + dx:.1f}" y="{y + dy:.1f}" font-size="13" '
                 f'text-anchor="{anchor}">{label.strip()}</text>')

    lx, ly = W - R + 10, T + 14
    e.append(f'<rect class="tuned" x="{lx}" y="{ly - 5}" width="10" height="10" '
             f'transform="rotate(45 {lx + 5} {ly})"/>')
    e.append(f'<text class="fg" x="{lx + 16}" y="{ly + 4}" font-size="12">jwenv（学習済み）</text>')
    e.append(f'<circle class="base" cx="{lx + 5}" cy="{ly + 24}" r="5"/>')
    e.append(f'<text class="fg" x="{lx + 16}" y="{ly + 28}" font-size="12">素の Qwen3</text>')
    e.append(f'<text class="mut" x="{lx}" y="{ly + 56}" font-size="11">線は同じモデルの</text>')
    e.append(f'<text class="mut" x="{lx}" y="{ly + 72}" font-size="11">学習前後を結ぶ</text>')

    e.append(f'<text class="fg" x="{L}" y="{T - 8}" font-size="13">'
             f'jev-bench の正解率（客観 1,191問、2〜8択）</text>')
    e.append("</svg>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(e), encoding="utf-8")
    print("wrote", OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()

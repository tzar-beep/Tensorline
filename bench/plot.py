"""Render README charts as standalone SVG from bench/results/*.json (no plotting dependency).

    python bench/plot.py      -> docs/throughput.svg
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "bench", "results")
SERIES = [  # (results file, label, light color, dark color) - validated categorical slots 1-3
    ("naive.json", "Naive (Phase 1)", "#2a78d6", "#3987e5"),
    ("eager.json", "Continuous batching", "#eb6834", "#d95926"),
    ("graphs.json", "Batching + CUDA graphs", "#1baf7a", "#199e70"),
]


def throughput_chart(path):
    W, H, L, R, T, B = 720, 360, 64, 200, 48, 48
    data = []
    for fn, label, c1, c2 in SERIES:
        rows = json.load(open(os.path.join(RES, fn)))["results"]
        data.append((label, c1, c2, [(r["concurrency"], r["throughput_tok_s"]) for r in rows]))
    xs = sorted({c for *_, pts in data for c, _ in pts})
    ymax = 1400
    px = lambda c: L + xs.index(c) * (W - L - R) / (len(xs) - 1)
    py = lambda v: T + (H - T - B) * (1 - v / ymax)
    css = []
    for i, (_, c1, c2, _) in enumerate(data):
        css.append(f".s{i}{{stroke:{c1};fill:{c1}}}")
    dark = "".join(f".s{i}{{stroke:{c2};fill:{c2}}}" for i, (_, _, c2, _) in enumerate(data))
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         'role="img" aria-labelledby="t d" font-family="system-ui,-apple-system,Segoe UI,sans-serif">',
         '<title id="t">Output throughput vs concurrency</title>',
         '<desc id="d">Qwen2.5-0.5B on an RTX 3050 laptop GPU. Naive stays near 15 tok/s; continuous batching '
         'reaches 558 tok/s at 64 concurrent requests; with CUDA graphs it reaches 1259 tok/s.</desc>',
         "<style>.bg{fill:#fcfcfb}.ink{fill:#0b0b0b}.ink2{fill:#52514e}.grid{stroke:#e4e3df}"
         + "".join(css) + ".ln{fill:none;stroke-width:2}.mk{stroke:#fcfcfb;stroke-width:2}"
         "@media (prefers-color-scheme:dark){.bg{fill:#1a1a19}.ink{fill:#f0efec}.ink2{fill:#c3c2b7}"
         ".grid{stroke:#33332f}.mk{stroke:#1a1a19}" + dark + "}</style>",
         f'<rect class="bg" width="{W}" height="{H}" rx="8"/>',
         f'<text class="ink" x="{L}" y="26" font-size="15" font-weight="600">Output throughput (tok/s) vs concurrent requests</text>']
    for v in range(0, ymax + 1, 200):
        o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{py(v):.1f}" y2="{py(v):.1f}"/>')
        o.append(f'<text class="ink2" x="{L - 8}" y="{py(v) + 4:.1f}" font-size="11" text-anchor="end">{v}</text>')
    for c in xs:
        o.append(f'<text class="ink2" x="{px(c):.1f}" y="{H - B + 18}" font-size="11" text-anchor="middle">{c}</text>')
    o.append(f'<text class="ink2" x="{(L + W - R) / 2}" y="{H - 10}" font-size="11" text-anchor="middle">concurrent requests</text>')
    for i, (label, _, _, pts) in enumerate(data):
        d = " ".join(f"{'M' if j == 0 else 'L'}{px(c):.1f},{py(v):.1f}" for j, (c, v) in enumerate(pts))
        o.append(f'<path class="ln s{i}" d="{d}"/>')
        for c, v in pts:
            o.append(f'<circle class="mk s{i}" cx="{px(c):.1f}" cy="{py(v):.1f}" r="4.5">'
                     f"<title>{label}: {v:.0f} tok/s at {c} concurrent</title></circle>")
        c, v = pts[-1]
        o.append(f'<circle class="s{i}" cx="{W - R + 14}" cy="{py(v):.1f}" r="4"/>')
        o.append(f'<text class="ink" x="{W - R + 24}" y="{py(v) + 4:.1f}" font-size="12">{label} · {v:.0f}</text>')
    o.append("</svg>")
    open(path, "w", encoding="utf-8").write("\n".join(o))


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    throughput_chart(os.path.join(ROOT, "docs", "throughput.svg"))
    print("wrote docs/throughput.svg")

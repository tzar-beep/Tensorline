"""Render README charts as standalone SVG from bench/results/*.json (no plotting dependency).

    python bench/plot.py      -> docs/throughput.svg, docs/vs_ollama.svg, docs/vs_ollama_long.svg
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "bench", "results")
# Validated categorical slots (light, dark): 1 blue, 2 orange, 3 aqua.
BLUE, ORANGE, AQUA = ("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70")


def load(fn):
    rows = json.load(open(os.path.join(RES, fn)))["results"]
    return [(r["concurrency"], r["throughput_tok_s"]) for r in rows]


def nice_max(v):
    e = 10 ** math.floor(math.log10(v))
    for f in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if v <= f * e:
            return f * e
    return 10 * e


def line_chart(path, title, desc, series):
    """series: [(label, (light, dark), [(concurrency, tok/s), ...])]"""
    W, H, L, R, T, B = 720, 360, 64, 210, 48, 48
    xs = sorted({c for *_, pts in series for c, _ in pts})
    ymax = nice_max(max(v for *_, pts in series for _, v in pts) * 1.08)
    px = lambda c: L + xs.index(c) * (W - L - R) / max(1, len(xs) - 1)
    py = lambda v: T + (H - T - B) * (1 - v / ymax)
    light = "".join(f".s{i}{{stroke:{c[0]};fill:{c[0]}}}" for i, (_, c, _) in enumerate(series))
    dark = "".join(f".s{i}{{stroke:{c[1]};fill:{c[1]}}}" for i, (_, c, _) in enumerate(series))
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         'role="img" aria-labelledby="t d" font-family="system-ui,-apple-system,Segoe UI,sans-serif">',
         f'<title id="t">{title}</title>', f'<desc id="d">{desc}</desc>',
         "<style>.bg{fill:#fcfcfb}.ink{fill:#0b0b0b}.ink2{fill:#52514e}.grid{stroke:#e4e3df}" + light +
         ".ln{fill:none;stroke-width:2}.mk{stroke:#fcfcfb;stroke-width:2}"
         "@media (prefers-color-scheme:dark){.bg{fill:#1a1a19}.ink{fill:#f0efec}.ink2{fill:#c3c2b7}"
         ".grid{stroke:#33332f}.mk{stroke:#1a1a19}" + dark + "}</style>",
         f'<rect class="bg" width="{W}" height="{H}" rx="8"/>',
         f'<text class="ink" x="{L}" y="26" font-size="15" font-weight="600">{title}</text>']
    for i in range(5):
        v = ymax * i / 4
        o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{py(v):.1f}" y2="{py(v):.1f}"/>')
        o.append(f'<text class="ink2" x="{L - 8}" y="{py(v) + 4:.1f}" font-size="11" text-anchor="end">{v:,.0f}</text>')
    for c in xs:
        o.append(f'<text class="ink2" x="{px(c):.1f}" y="{H - B + 18}" font-size="11" text-anchor="middle">{c}</text>')
    o.append(f'<text class="ink2" x="{(L + W - R) / 2}" y="{H - 10}" font-size="11" text-anchor="middle">'
             'concurrent requests · output tokens/s</text>')
    ends = []
    for i, (label, _, pts) in enumerate(series):
        d = " ".join(f"{'M' if j == 0 else 'L'}{px(c):.1f},{py(v):.1f}" for j, (c, v) in enumerate(pts))
        o.append(f'<path class="ln s{i}" d="{d}"/>')
        for c, v in pts:
            o.append(f'<circle class="mk s{i}" cx="{px(c):.1f}" cy="{py(v):.1f}" r="4.5">'
                     f"<title>{label}: {v:,.0f} tok/s at {c} concurrent</title></circle>")
        ends.append([py(pts[-1][1]), i, label, pts[-1][1]])
    ends.sort()
    for k in range(1, len(ends)):                     # keep end labels at least 16 px apart
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 16)
    for y, i, label, v in ends:
        o.append(f'<circle class="s{i}" cx="{W - R + 14}" cy="{y:.1f}" r="4"/>')
        o.append(f'<text class="ink" x="{W - R + 24}" y="{y + 4:.1f}" font-size="12">{label} · {v:,.0f}</text>')
    o.append("</svg>")
    open(path, "w", encoding="utf-8").write("\n".join(o))


if __name__ == "__main__":
    docs = os.path.join(ROOT, "docs")
    os.makedirs(docs, exist_ok=True)
    line_chart(os.path.join(docs, "throughput.svg"), "Phase 1 → Phase 2: output tok/s vs concurrent requests",
               "Qwen2.5-0.5B on an RTX 3050 laptop GPU: naive one-at-a-time serving vs continuous batching vs "
               "batching with CUDA graphs (historical back-to-back run).",
               [("Naive (Phase 1)", BLUE, load("naive.json")), ("Continuous batching", ORANGE, load("eager.json")),
                ("+ CUDA graphs", AQUA, load("graphs.json"))])
    line_chart(os.path.join(docs, "vs_ollama.svg"), "Tensorline vs Ollama · short requests",
               "Same model (Qwen2.5-0.5B fp16), same GPU, back to back; ~30-token prompts, 128 output tokens.",
               [("Tensorline", BLUE, load("vs_tensorline.json")), ("Ollama (16 slots)", ORANGE, load("vs_ollama_p16.json"))])
    line_chart(os.path.join(docs, "vs_ollama_long.svg"), "Tensorline vs Ollama · long-context requests",
               "Same model and GPU, back to back; unique ~880-token prompts, 512 output tokens.",
               [("Tensorline", BLUE, load("vs_tensorline_heavy.json")),
                ("Ollama (16 slots)", ORANGE, load("vs_ollama_p16_heavy.json"))])
    print("wrote docs/throughput.svg, docs/vs_ollama.svg, docs/vs_ollama_long.svg")

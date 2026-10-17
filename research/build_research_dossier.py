"""Builds research/research_dossier.html — the evidence base for a TrainStream paper.

    python research/build_research_dossier.py

All numbers are measurements recorded on the development machine (see DATA and
the saved dashboard reports it loads). Figures are computed from the data, so
re-running after new experiments keeps text, tables and plots consistent.
"""

from __future__ import annotations

import html
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPORTS = os.path.join(ROOT, "dashboard", "reports")
OUT = os.path.join(HERE, "research_dossier.html")
E = html.escape

# ---------------------------------------------------------------------------
# Recorded measurements (development machine, 2026-10-16/17)
# ---------------------------------------------------------------------------
DATA = {
    "platform": {
        "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU (Blackwell, sm_120), 24 GB GDDR7 (23.89 GiB usable)",
        "link": "PCIe 5.0 x8 under load (reported by nvidia-smi)",
        "cpu": "Intel Core Ultra 9 290HX Plus, 24 cores / 24 threads",
        "ram": "63.4 GB DDR5",
        "os": "Windows 11 Home (WDDM driver model), NVIDIA driver 596.37",
        "software": "CUDA toolkit 13.4 (runtime 13.0 via PyTorch), PyTorch 2.14.1+cu130, Python 3.14.7, MSVC 14.51, "
                    "transformers 5.20.1, accelerate 1.15.0, peft 0.21.2",
    },
    "profile": {  # trainstream.profile_hardware(quick=False)
        "h2d": 27.9, "d2h": 27.4, "bidir": 27.9, "pageable_h2d": 14.5, "lat_4k_us": 4.8,
        "bf16_tflops": 94.6, "fp32_tflops": 27.6, "host_adamw_gelems": 2.54, "host_adamw_quick": 2.81,
        "zc_adamw_gelems_16m": 2.68, "zc_adamw_gelems_32m": 1.51, "zc_pcie_gbps": 24.2, "threads": 12,
        "sweep_h2d": {65536: 5.4, 262144: 20.1, 1048576: 27.0, 4194304: 26.8, 16777216: 28.6, 67108864: 28.6},
        "sweep_d2h": {65536: 9.0, 262144: 20.7, 1048576: 26.9, 4194304: 28.2, 16777216: 28.5, 67108864: 28.6},
    },
    "parity": [  # dashboard parity check: MLP 256-1024-1024-64, 20 steps, identical data, fp32
        {"mode": "cpu (host AdamW)", "clip": "off", "diff": 2.9e-7},
        {"mode": "cpu (host AdamW)", "clip": "1.0", "diff": 2.9e-7},
        {"mode": "gpu (zero-copy)", "clip": "off", "diff": 3.1e-7},
        {"mode": "gpu (zero-copy)", "clip": "1.0", "diff": 3.1e-7},
    ],
    "opt_bench": [  # 100M fp32 parameters, standalone step() incl. synchronisation, median of 5
        {"name": "torch.optim.AdamW (fused)", "ms": 3.7, "vram_mib": 771, "host_mib": 0},
        {"name": "OffloadAdamW cpu", "ms": 64.9, "vram_mib": 0, "host_mib": 1907},
        {"name": "OffloadAdamW gpu (zero-copy)", "ms": 69.3, "vram_mib": 0, "host_mib": 763},
    ],
    "eager": [  # 85M GPT, activations resident, drain = host time after loss.backward() until weights landed
        {"variant": "deferred (update at step())", "drain_ms": 42.4},
        {"variant": "eager, via fp32 accumulator", "drain_ms": 29.1},
        {"variant": "eager, fused staging read", "drain_ms": 18.8},
    ],
    "eager_phases": {"fwd_ms": 12.2, "bwd_ms": 29.0},
    "tradeoff_18b": [  # xxl-1.8B fp32, batch 8x512, bf16 autocast; probe with synchronised phases
        {"cfg": "Stock AdamW (driver spill)", "peak": 34.06, "step_s": 13.68, "engine": "adamw"},
        {"cfg": "TrainStream, activations resident", "peak": 20.55, "step_s": 1.11, "engine": "ts"},
        {"cfg": "TrainStream, 43% paged", "peak": 16.73, "step_s": 1.42, "engine": "ts"},
        {"cfg": "TrainStream, 100% paged", "peak": 13.51, "step_s": 1.87, "engine": "ts"},
    ],
    "backpressure": [  # xxl-1.8B, 43% (15/36 blocks) paged, peak forward VRAM
        {"cfg": "no paging", "peak": 20.43},
        {"cfg": "43% paged, no back-pressure", "peak": 20.05},
        {"cfg": "43% paged, back-pressure 1 GiB", "peak": 16.73},
        {"cfg": "100% paged, back-pressure 1 GiB", "peak": 11.29},
    ],
    "leak": {"before": [5.9, 11.7, 17.6, 23.4], "after": [0.0, 0.0, 0.0, 0.0]},
    "pinned_18b": {"before_pinned": 33.8, "after_pinned": 13.5, "after_pageable": 13.5, "async_oom_at": 39.7},
    "grad_error": [  # max |grad difference|, paged vs. reference (log scale)
        {"case": "Stride-changing pager, memory-efficient SDPA (bug)", "v": 54.75, "kind": "bug"},
        {"case": "bf16 run-to-run noise, no paging at all", "v": 1.953e-3, "kind": "noise"},
        {"case": "Optimizer parity vs torch.optim.AdamW (fp32)", "v": 3.1e-7, "kind": "ok"},
        {"case": "Layout-preserving pager, fp32", "v": 1.2e-7, "kind": "ok"},
        {"case": "Layout-preserving pager, efficient/cuDNN/math SDPA", "v": 0.0, "kind": "exact"},
    ],
    "verify": [  # examples/verify_ram_offload.py
        {"model": "GPT 1.2B", "cap": 16, "stock": "CUDA OOM", "ts_steps": "6/6", "ts_peak": 10.33, "persist": 0.06,
         "moments": 9.02, "host": 18.04, "commit": 18.72, "d2h": "4.51–7.90", "unexplained_shared": 0.08},
        {"model": "Qwen2.5-0.5B", "cap": 7, "stock": "CUDA OOM", "ts_steps": "5/5", "ts_peak": 6.16, "persist": 0.06,
         "moments": 3.68, "host": 7.36, "commit": 8.71, "d2h": "1.84–3.86", "unexplained_shared": 0.09},
    ],
    "spill_control": {"model": "GPT 1.8B, stock AdamW, no cap, batch 4", "peak": 28.92, "dedicated": 21.32,
                      "shared": 13.85, "step_s": 1.84},
    "pinned_probe": {"pinned_gib": 4.0, "shared_counter_mib": 4172, "vram_gib": 4.0, "dedicated_counter_mib": 4382},
    "hf": [
        {"model": "Qwen2.5-0.5B", "steps": 30, "stock_peak": 14.97, "ts_peak": 9.49, "stock_s": 11.3, "ts_s": 21.7,
         "stock_loss": [1.627, 1.668, 1.767, 1.570], "ts_loss": [1.627, 1.668, 1.767, 1.571], "host": 9.20},
        {"model": "Qwen2.5-1.5B", "steps": 10, "stock_peak": 35.92, "ts_peak": 18.66, "stock_s": 111.7, "ts_s": 28.5,
         "stock_loss": [1.364, 1.507, 1.360, 1.417, 1.395, 1.410, 1.402, 1.603, 1.592, 1.371],
         "ts_loss": [1.364, 1.507, 1.359, 1.416, 1.395, 1.410, 1.400, 1.603, 1.592, 1.370], "host": 28.75},
    ],
    "tf_directml": {"tok_s_runs": [370, 13481], "private_ram_gb": 19.3, "gpu_heap_gib": 10.5},
    "restart": {"vram_before_mib": 6539, "vram_after_mib": 830, "seconds": 5.1},
    "tests": {"core": 38, "hf": 9},
    "auto_85m": {"block_fwd_ms": 3.45, "block_mib": 127.5, "d2h_ms": 4.68, "naive_blocks": 6, "naive_step_ms": 162,
                 "model_blocks": 3, "tuned_blocks": 1, "baseline_ms": 44},
}

PRESETS = [  # dashboard/model.py (byte-level GPT, vocab 256, seq 512)
    ("tiny-20M", 6, 8, 512), ("small-85M", 12, 12, 768), ("medium-300M", 24, 16, 1024),
    ("large-680M", 24, 16, 1536), ("xl-1.2B", 24, 16, 2048), ("xxl-1.8B", 36, 16, 2048),
]
PARAMS = {"tiny-20M": 19.2e6, "small-85M": 85.8e6, "medium-300M": 304e6, "large-680M": 681e6, "xl-1.2B": 1210.7e6,
          "xxl-1.8B": 1815.0e6}


SNAPSHOT = os.path.join(HERE, "research_data.json")  # written by research/record_research_data.py


def load_snapshot():
    if not os.path.isfile(SNAPSHOT):
        return {}
    with open(SNAPSHOT, encoding="utf-8") as fh:
        return json.load(fh)


def as_report(entry):
    """Snapshot entry -> the {'engines': [{'key': ...}]} shape the figure code uses."""
    if not entry:
        return None
    return {"engines": [dict(v, key=k) for k, v in entry["engines"].items()], "meta": entry}


SNAP = load_snapshot()
R85 = as_report(SNAP.get("curves", {}).get("small-85M"))
R18 = as_report(SNAP.get("reports", {}).get("xxl-1.8B@200"))
RUN_12B = {"steps": 30, "peak_vram": 13.72 * 2**30, "tokens_per_s": 5034}  # dashboard report 2026-10-17 00:58


def eng(rep, key):
    if not rep:
        return None
    return next((e for e in rep["engines"] if e["key"] == key), None)


# ---------------------------------------------------------------------------
# SVG chart helpers (colors come from CSS classes so light/dark both work and
# "Download SVG" can inline the computed styles)
# ---------------------------------------------------------------------------
def nice_ticks(lo, hi, n=5):
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.floor(lo / step + 1e-9) * step
    stop = math.ceil(hi / step - 1e-9) * step
    ticks, t = [], start
    while t <= stop + step * 1e-9:
        ticks.append(round(t, 10))
        t += step
    return ticks


def fmt(v):
    if abs(v) >= 1000:
        return f"{v / 1000:g}k"
    if v == int(v):
        return str(int(v))
    return f"{v:g}"


def svg_open(fid, W, H, label):
    return [f'<svg id="{fid}" viewBox="0 0 {W} {H}" role="img" aria-label="{E(label)}" xmlns="http://www.w3.org/2000/svg">']


def line_chart(fid, series, W=760, H=320, xlabel="", ylabel="", logx2=False, xfmt=None, yzero=False, ymax=None,
               hline=None, label=""):
    L, R, T, B = 64, 18, 14, 48
    pts = [(x, y) for s in series for x, y in s["pts"]]
    xs = [math.log2(x) if logx2 else x for x, _ in pts]
    ys = [y for _, y in pts] + ([hline[0]] if hline else [])
    x0, x1 = min(xs), max(xs)
    y0, y1 = (0 if yzero else min(ys)), max(ys) if ymax is None else ymax
    if not yzero:  # a little headroom so lines never touch the frame
        pad = (y1 - y0) * 0.04
        y0, y1 = y0 - pad, y1 + pad
    yt = nice_ticks(y0, y1)
    y0, y1 = yt[0], yt[-1]
    X = lambda x: L + ((math.log2(x) if logx2 else x) - x0) / ((x1 - x0) or 1) * (W - L - R)  # noqa: E731
    Y = lambda y: T + (1 - (y - y0) / ((y1 - y0) or 1)) * (H - T - B)  # noqa: E731
    out = svg_open(fid, W, H, label)
    for t in yt:
        out.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                   f'<text class="tick" x="{L - 8}" y="{Y(t) + 4:.1f}" text-anchor="end">{fmt(t)}</text>')
    xticks = sorted({x for x, _ in pts})
    if not logx2 and len(xticks) > 8:
        xticks = nice_ticks(x0, x1, 6)
    for t in xticks:
        if x0 - 1e-9 <= (math.log2(t) if logx2 else t) <= x1 + 1e-9:
            out.append(f'<text class="tick" x="{X(t):.1f}" y="{H - B + 18}" text-anchor="middle">{E(xfmt(t) if xfmt else fmt(t))}</text>')
    out.append(f'<line class="axis" x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}"/>')
    if hline:
        out.append(f'<line class="ref" x1="{L}" x2="{W - R}" y1="{Y(hline[0]):.1f}" y2="{Y(hline[0]):.1f}"/>'
                   f'<text class="tick" x="{W - R}" y="{Y(hline[0]) - 6:.1f}" text-anchor="end">{E(hline[1])}</text>')
    for s in series:
        p = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in s["pts"])
        out.append(f'<polyline class="ln {s["cls"]}" points="{p}"><title>{E(s["name"])}</title></polyline>')
        if s.get("dots", True) and len(s["pts"]) <= 24:
            for x, y in s["pts"]:
                out.append(f'<circle class="pt {s["cls"]}" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4"><title>{E(s["name"])}: {y:g}</title></circle>')
    out.append(f'<text class="axlabel" x="{(L + W - R) / 2:.0f}" y="{H - 8}" text-anchor="middle">{E(xlabel)}</text>')
    out.append(f'<text class="axlabel" transform="translate(16,{(T + H - B) / 2:.0f}) rotate(-90)" text-anchor="middle">{E(ylabel)}</text>')
    out.append("</svg>")
    return "".join(out)


def hbar_chart(fid, rows, W=760, unit="", label="", logx=False, vmax=None, ref=None):
    """rows: (label, value, cls, value_text)."""
    longest = max(len(r[0]) for r in rows)
    L, R, rowh = min(max(140, int(longest * 7.4) + 24), 470), 96, 34
    T = 26 if ref else 8
    H = T + rowh * len(rows) + 30
    vals = [v for _, v, _, _ in rows if v > 0]
    top = vmax or max(vals + ([ref[0]] if ref else [])) * (1.18 if not logx else 1.05)
    lo = min(vals) / 3 if logx else 0
    def X(v):
        if logx:
            v = max(v, lo)
            return L + (math.log10(v) - math.log10(lo)) / (math.log10(top) - math.log10(lo)) * (W - L - R)
        return L + v / top * (W - L - R)
    out = svg_open(fid, W, H, label)
    if logx:
        e0, e1 = math.floor(math.log10(lo)), math.ceil(math.log10(top))
        for e in range(e0, e1 + 1):
            v = 10 ** e
            if lo <= v <= top:
                out.append(f'<line class="grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{T}" y2="{H - 26}"/>'
                           f'<text class="tick" x="{X(v):.1f}" y="{H - 10}" text-anchor="middle">1e{e}</text>')
    else:
        for t in nice_ticks(0, top, 5):
            if t <= top:
                out.append(f'<line class="grid" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{T}" y2="{H - 26}"/>'
                           f'<text class="tick" x="{X(t):.1f}" y="{H - 10}" text-anchor="middle">{fmt(t)}{E(unit)}</text>')
    if ref:
        out.append(f'<line class="ref" x1="{X(ref[0]):.1f}" x2="{X(ref[0]):.1f}" y1="{T - 6}" y2="{H - 26}"/>'
                   f'<text class="tick" x="{X(ref[0]):.1f}" y="{T - 10}" text-anchor="middle">{E(ref[1])}</text>')
    for i, (name, v, cls, txt) in enumerate(rows):
        y = T + i * rowh + 6
        out.append(f'<text class="lbl" x="{L - 10}" y="{y + 15}" text-anchor="end">{E(name)}</text>')
        if v > 0:
            w = max(X(v) - L, 2)
            out.append(f'<rect class="bar {cls}" x="{L}" y="{y}" width="{w:.1f}" height="20" rx="3"><title>{E(name)}: {E(txt)}</title></rect>')
            out.append(f'<text class="val" x="{L + w + 8:.1f}" y="{y + 15}">{E(txt)}</text>')
        else:
            out.append(f'<text class="val" x="{L + 8}" y="{y + 15}">{E(txt)}</text>')
    out.append("</svg>")
    return "".join(out)


def scatter_tradeoff(fid, rows, cap, W=760, H=340):
    L, R, T, B = 64, 24, 16, 48
    X0, X1 = 0, 16
    Y0, Y1 = 0, 40
    X = lambda x: L + (x - X0) / (X1 - X0) * (W - L - R)  # noqa: E731
    Y = lambda y: T + (1 - (y - Y0) / (Y1 - Y0)) * (H - T - B)  # noqa: E731
    out = svg_open(fid, W, H, "Peak VRAM versus step time on the 1.8B model")
    for t in nice_ticks(Y0, Y1, 6):
        out.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/><text class="tick" x="{L - 8}" y="{Y(t) + 4:.1f}" text-anchor="end">{fmt(t)}</text>')
    for t in range(0, 17, 2):
        out.append(f'<text class="tick" x="{X(t):.1f}" y="{H - B + 18}" text-anchor="middle">{t}</text>')
    out.append(f'<line class="axis" x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}"/>')
    out.append(f'<line class="ref" x1="{L}" x2="{W - R}" y1="{Y(cap):.1f}" y2="{Y(cap):.1f}"/>'
               f'<text class="tick" x="{W - R}" y="{Y(cap) - 6:.1f}" text-anchor="end">GPU capacity {cap} GiB</text>')
    ts = [r for r in rows if r["engine"] == "ts"]
    out.append('<polyline class="ln c-ts" points="' + " ".join(f"{X(r['step_s']):.1f},{Y(r['peak']):.1f}" for r in ts) + '"/>')
    for r in rows:
        cls = "c-ts" if r["engine"] == "ts" else "c-adamw"
        out.append(f'<circle class="pt {cls}" cx="{X(r["step_s"]):.1f}" cy="{Y(r["peak"]):.1f}" r="6"><title>{E(r["cfg"])}: {r["peak"]} GiB, {r["step_s"]} s/step</title></circle>')
        anchor = "end" if r["step_s"] > 10 else "start"
        dx = -10 if anchor == "end" else 10
        out.append(f'<text class="lbl" x="{X(r["step_s"]) + dx:.1f}" y="{Y(r["peak"]) + 4:.1f}" text-anchor="{anchor}">{E(r["cfg"])}</text>')
    out.append(f'<text class="axlabel" x="{(L + W - R) / 2:.0f}" y="{H - 8}" text-anchor="middle">Step time (s), batch 8 × 512</text>')
    out.append(f'<text class="axlabel" transform="translate(16,{(T + H - B) / 2:.0f}) rotate(-90)" text-anchor="middle">Peak VRAM (GiB)</text>')
    out.append("</svg>")
    return "".join(out)


def stacked_mem(fid, W=760):
    # bytes per parameter, fp32 weights
    rows = [
        ("torch.optim.AdamW", [("weights", 4, "c-w"), ("gradients", 4, "c-g"), ("Adam moments", 8, "c-m")], []),
        ("TrainStream cpu mode", [("weights", 4, "c-w")], [("fp32 master", 4, "c-ms"), ("grad staging", 4, "c-g"), ("Adam moments", 8, "c-m")]),
        ("TrainStream gpu (zero-copy)", [("weights", 4, "c-w"), ("gradients", 4, "c-g")], [("Adam moments (mapped)", 8, "c-m")]),
    ]
    L, mid, R, rowh, T = 210, 0, 20, 46, 30
    H = T + rowh * len(rows) + 46
    half = (W - L - R) / 2
    scale = half / 18
    out = svg_open(fid, W, H, "Bytes per parameter in VRAM and in host RAM")
    out.append(f'<text class="axlabel" x="{L + half / 2:.0f}" y="18" text-anchor="middle">GPU VRAM (bytes / parameter)</text>')
    out.append(f'<text class="axlabel" x="{L + half + half / 2:.0f}" y="18" text-anchor="middle">Host RAM (bytes / parameter)</text>')
    out.append(f'<line class="axis" x1="{L + half:.1f}" x2="{L + half:.1f}" y1="24" y2="{H - 34}"/>')
    for i, (name, gpu, host) in enumerate(rows):
        y = T + i * rowh + 6
        out.append(f'<text class="lbl" x="{L - 10}" y="{y + 18}" text-anchor="end">{E(name)}</text>')
        x = L + half
        for seg, b, cls in gpu:  # grows leftwards from the divider
            w = b * scale
            out.append(f'<rect class="bar {cls}" x="{x - w + 1:.1f}" y="{y}" width="{w - 2:.1f}" height="26" rx="3"><title>{seg}: {b} B</title></rect>')
            if w > 40:
                out.append(f'<text class="inbar" x="{x - w / 2:.1f}" y="{y + 17}" text-anchor="middle">{b}</text>')
            x -= w
        out.append(f'<text class="val" x="{x - 6:.1f}" y="{y + 18}" text-anchor="end">{sum(b for _, b, _ in gpu)} B</text>')
        x = L + half
        for seg, b, cls in host:
            w = b * scale
            out.append(f'<rect class="bar {cls}" x="{x + 1:.1f}" y="{y}" width="{w - 2:.1f}" height="26" rx="3"><title>{seg}: {b} B</title></rect>')
            if w > 40:
                out.append(f'<text class="inbar" x="{x + w / 2:.1f}" y="{y + 17}" text-anchor="middle">{b}</text>')
            x += w
        if host:
            out.append(f'<text class="val" x="{x + 6:.1f}" y="{y + 18}">{sum(b for _, b, _ in host)} B</text>')
    legend = [("weights", "c-w"), ("gradients / grad staging", "c-g"), ("Adam moments", "c-m"), ("fp32 master", "c-ms")]
    lx = L
    for name, cls in legend:
        out.append(f'<rect class="bar {cls}" x="{lx}" y="{H - 24}" width="12" height="12" rx="2"/><text class="lbl" x="{lx + 18}" y="{H - 14}">{E(name)}</text>')
        lx += 30 + 7.2 * len(name)
    out.append("</svg>")
    return "".join(out)


def architecture_svg():
    return """<svg id="fig-arch" viewBox="0 0 980 470" role="img" aria-label="TrainStream memory architecture" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="arw" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" class="arrowhead"/></marker></defs>
<rect class="zone z-gpu" x="10" y="10" width="300" height="450" rx="14"/>
<text class="zt" x="28" y="40">GPU · VRAM</text><text class="tick" x="28" y="60">compute stream (PyTorch current stream)</text>
<rect class="box" x="28" y="76" width="264" height="54" rx="8"/><text class="bt" x="42" y="99">Weights (resident)</text><text class="bs" x="42" y="118">only static cost: N·b_w bytes</text>
<rect class="box dashed" x="28" y="142" width="264" height="54" rx="8"/><text class="bt" x="42" y="165">Gradients (transient)</text><text class="bs" x="42" y="184">released once their D2H has landed</text>
<rect class="box" x="28" y="208" width="264" height="54" rx="8"/><text class="bt" x="42" y="231">Resident activations</text><text class="bs" x="42" y="250">blocks the tuner keeps on the GPU</text>
<rect class="box hl" x="28" y="274" width="126" height="60" rx="8"/><text class="bt" x="40" y="298">Slot A</text><text class="bs" x="40" y="318">block b computing</text>
<rect class="box hl dashed" x="166" y="274" width="126" height="60" rx="8"/><text class="bt" x="178" y="298">Slot B</text><text class="bs" x="178" y="318">block b−1 landing</text>
<text class="tick" x="28" y="352">prefetch window (depth 1)</text>
<rect class="box" x="28" y="366" width="264" height="74" rx="8"/><text class="bt" x="42" y="390">Zero-copy AdamW kernel (gpu mode)</text><text class="bs" x="42" y="410">reads/writes moments over PCIe</text><text class="bs" x="42" y="428">via mapped pinned memory</text>
<rect class="zone z-link" x="340" y="10" width="270" height="450" rx="14"/>
<text class="zt" x="358" y="40">PCIe · DMA engines</text><text class="tick" x="358" y="60">≈ 28 GB/s per direction, duplex ×1.0</text>
<line class="flow" x1="310" y1="110" x2="636" y2="110" marker-end="url(#arw)"/><text class="bt" x="358" y="100">d2h stream (low priority)</text><text class="bs" x="358" y="132">gradients · activation stashes</text>
<line class="flow" x1="636" y1="196" x2="314" y2="196" marker-end="url(#arw)"/><text class="bt" x="358" y="186">h2d stream (high priority)</text><text class="bs" x="358" y="218">weights · activation prefetch</text>
<rect class="box" x="358" y="246" width="234" height="118" rx="8"/><text class="bt" x="372" y="270">GPU-side ordering only</text>
<text class="mono" x="372" y="292">record(e, compute)</text><text class="mono" x="372" y="312">cudaStreamWaitEvent(copy, e)</text><text class="mono" x="372" y="332">cudaMemcpyAsync(…, copy)</text><text class="mono" x="372" y="352">record(done, copy)</text>
<line class="flow dashed" x1="310" y1="404" x2="636" y2="404" marker-end="url(#arw)"/><line class="flow dashed" x1="636" y1="420" x2="314" y2="420" marker-end="url(#arw)"/><text class="bt" x="358" y="394">upd stream (zero-copy)</text>
<rect class="zone z-host" x="640" y="10" width="330" height="450" rx="14"/>
<text class="zt" x="658" y="40">Host RAM</text><text class="tick" x="658" y="60">pin-logging caching allocator (cudaHostAlloc)</text>
<rect class="box" x="658" y="76" width="294" height="54" rx="8"/><text class="bt" x="672" y="99">Gradient staging (pinned)</text><text class="bs" x="672" y="118">DMA target, read in place by AdamW</text>
<rect class="box" x="658" y="142" width="294" height="54" rx="8"/><text class="bt" x="672" y="165">fp32 master weights (pinned)</text><text class="bs" x="672" y="184">H2D source of the updated weights</text>
<rect class="box dashed" x="658" y="208" width="294" height="54" rx="8"/><text class="bt" x="672" y="231">Adam moments m, v</text><text class="bs" x="672" y="250">pinned within a 40 % RAM budget, else pageable</text>
<rect class="box hl" x="658" y="274" width="294" height="54" rx="8"/><text class="bt" x="672" y="297">Activation stashes (pinned)</text><text class="bs" x="672" y="316">size-bucketed cache, event-fenced reuse</text>
<rect class="box" x="658" y="346" width="294" height="94" rx="8"/><text class="bt" x="672" y="370">C++ worker pool (T threads)</text><text class="bs" x="672" y="390">AdamW per 1 M-element chunk</text><text class="bs" x="672" y="408">chunk c of tensor s → worker (s·φ + c) mod T</text><text class="bs" x="672" y="426">FIFO per chunk ⇒ no locks on the hot path</text>
</svg>"""


def timeline_svg():
    rows = ["compute", "d2h stream", "h2d stream", "CPU workers"]
    out = ['<svg id="fig-timeline" viewBox="0 0 980 300" role="img" aria-label="Overlap of compute, transfers and host AdamW" xmlns="http://www.w3.org/2000/svg">']
    for i, r in enumerate(rows):
        out.append(f'<text class="lbl" x="12" y="{56 + i * 52}">{r}</text>')
    def blk(x, y, w, cls, t):
        out.append(f'<rect class="bar {cls}" x="{x}" y="{y}" width="{w}" height="28" rx="4"/><text class="inbar" x="{x + w / 2}" y="{y + 18}" text-anchor="middle">{t}</text>')
    for i, t in enumerate(["F0", "F1", "F2", "F3"]):
        blk(120 + i * 62, 38, 58, "c-ts", t)
    for i, t in enumerate(["B3", "B2", "B1", "B0"]):
        blk(380 + i * 92, 38, 88, "c-ts2", t)
    for i, t in enumerate(["F0", "F1"]):
        blk(780 + i * 62, 38, 58, "c-ts", t)
    blk(182, 90, 72, "c-act", "act 0 →")
    blk(256, 90, 72, "c-act", "act 1 →")
    for i in range(4):
        blk(470 + i * 92, 90, 40, "c-g", f"g{3 - i}")
    blk(474, 142, 72, "c-act", "← act 1")
    blk(566, 142, 72, "c-act", "← act 0")
    for i in range(4):
        blk(660 + i * 34, 142, 30, "c-w", f"W{3 - i}")
    for i in range(4):
        blk(512 + i * 92, 194, 70, "c-m", f"AdamW {3 - i}")
    out.append('<line class="axis" x1="110" x2="970" y1="244" y2="244"/>')
    out.append('<text class="tick" x="120" y="264">forward</text><text class="tick" x="380" y="264">backward (eager per-tensor updates start as gradients land)</text><text class="tick" x="780" y="264">next forward</text>')
    out.append('<text class="tick" x="110" y="290">Schematic, not to scale. Next forward waits per module on cudaStreamWaitEvent (GPU-side), not on the host.</text>')
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------------------
# Figures from data
# ---------------------------------------------------------------------------
P = DATA["profile"]
size_fmt = lambda b: (f"{b // (1 << 20)} MiB" if b >= 1 << 20 else f"{b // 1024} KiB")  # noqa: E731
FIG_SWEEP = line_chart("fig-sweep", [
    {"name": "H2D (host → GPU)", "cls": "c-h2d", "pts": sorted(P["sweep_h2d"].items())},
    {"name": "D2H (GPU → host)", "cls": "c-d2h", "pts": sorted(P["sweep_d2h"].items())},
], xlabel="Transfer size (pinned memory)", ylabel="Bandwidth (GB/s)", logx2=True, xfmt=size_fmt, yzero=True,
    hline=(P["pageable_h2d"], f"pageable H2D {P['pageable_h2d']} GB/s"), label="PCIe bandwidth versus transfer size")

FIG_MEM = stacked_mem("fig-mem")

FIG_TRADEOFF = scatter_tradeoff("fig-tradeoff", DATA["tradeoff_18b"], 23.89)

FIG_EAGER = hbar_chart("fig-eager", [(r["variant"], r["drain_ms"], "c-ts" if i else "c-adamw", f'{r["drain_ms"]} ms')
                                     for i, r in enumerate(DATA["eager"])], unit=" ms",
                       label="Host time after backward until updated weights have landed")

FIG_BP = hbar_chart("fig-bp", [(r["cfg"], r["peak"], "c-adamw" if "no paging" in r["cfg"] else "c-ts", f'{r["peak"]} GiB')
                               for r in DATA["backpressure"]], unit="", ref=(23.89, "capacity 23.89 GiB"),
                    label="Peak forward VRAM on 1.8B with and without back-pressure")

FIG_LEAK = line_chart("fig-leak", [
    {"name": "before: session ↔ handle cycle", "cls": "c-adamw", "pts": list(enumerate(DATA["leak"]["before"], 1))},
    {"name": "after: weak back-references", "cls": "c-ts", "pts": list(enumerate(DATA["leak"]["after"], 1))},
], xlabel="Training step", ylabel="Pinned stash (GB)", yzero=True, label="Pinned activation memory per step")

FIG_ERR = hbar_chart("fig-err", [(r["case"], r["v"], {"bug": "c-adamw", "noise": "c-noise", "ok": "c-ts", "exact": "c-ts"}[r["kind"]],
                                  ("0 (bit-exact)" if r["v"] == 0 else f'{r["v"]:.2g}')) for r in DATA["grad_error"]],
                     logx=True, label="Maximum gradient or parameter difference against the reference")


def hf_loss_fig():
    h = DATA["hf"][1]
    return line_chart("fig-hf", [
        {"name": "Stock transformers.Trainer", "cls": "c-adamw", "pts": list(enumerate(h["stock_loss"], 1))},
        {"name": "TrainStreamTrainer", "cls": "c-ts", "pts": list(enumerate(h["ts_loss"], 1))},
    ], xlabel="Optimizer step (Qwen2.5-1.5B, full fine-tune)", ylabel="Training loss", label="Qwen2.5-1.5B loss, stock vs TrainStream")


FIG_HF = hf_loss_fig()


def curve_fig(fid, rep, label, keys=("adamw", "trainstream")):
    if not rep:
        return '<p class="muted">Report file not found; run the dashboard full report to regenerate.</p>'
    names = {"adamw": ("Stock AdamW", "c-adamw"), "trainstream": ("TrainStream", "c-ts")}
    series = []
    for k in keys:
        e = eng(rep, k)
        if e and len(e["curve"]) > 1:
            series.append({"name": names[k][0], "cls": names[k][1], "pts": [tuple(p) for p in e["curve"]], "dots": False})
    if not series:
        return '<p class="muted">No curve recorded.</p>'
    return line_chart(fid, series, xlabel="Optimizer step", ylabel="Training loss (nats / byte)", label=label)


FIG_85 = curve_fig("fig-85", R85, "85M loss curves")
FIG_18 = curve_fig("fig-18", R18, "1.8B TrainStream loss curve", keys=("trainstream",))


def loss_agreement(rep):
    a, t = eng(rep, "adamw"), eng(rep, "trainstream")
    if not (a and t):
        return None
    da, dt = dict(map(tuple, a["curve"])), dict(map(tuple, t["curve"]))
    common = sorted(set(da) & set(dt))
    d = [abs(da[s] - dt[s]) for s in common]
    return len(common), sum(d) / len(d), max(d)


AGREE85 = loss_agreement(R85) if R85 else None

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def tbl(head, rows, num_cols=()):
    h = "".join(f'<th class="{"num" if i in num_cols else ""}">{E(c)}</th>' for i, c in enumerate(head))
    b = "".join("<tr>" + "".join(f'<td class="{"num" if i in num_cols else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="tbl"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


FIG_N = {"n": 0}
FIGS = []


def figure(fid, body, caption, legend=""):
    FIG_N["n"] += 1
    n = FIG_N["n"]
    FIGS.append((n, fid, caption))
    return (f'<figure id="{fid}-wrap"><div class="fig">{legend}{body}</div>'
            f'<figcaption><b>Figure {n}.</b> {caption} <button class="svgdl" data-fig="{fid}">Download SVG</button></figcaption></figure>')


def legend(items):
    return '<div class="legend">' + "".join(f'<span><i class="sw {c}"></i>{E(n)}</span>' for n, c in items) + "</div>"


pf = DATA["platform"]
v0, v1 = DATA["verify"]
h05, h15 = DATA["hf"]
tr = {r["cfg"]: r for r in DATA["tradeoff_18b"]}
a85 = eng(R85, "adamw") if R85 else None
t85 = eng(R85, "trainstream") if R85 else None
t18 = eng(R18, "trainstream") if R18 else None
t12 = RUN_12B

BIB = r"""@inproceedings{rajbhandari2020zero,
  title={ZeRO: Memory Optimizations Toward Training Trillion Parameter Models},
  author={Rajbhandari, Samyam and Rasley, Jeff and Ruwase, Olatunji and He, Yuxiong},
  booktitle={SC20: International Conference for High Performance Computing, Networking, Storage and Analysis},
  year={2020}}
@inproceedings{ren2021zerooffload,
  title={ZeRO-Offload: Democratizing Billion-Scale Model Training},
  author={Ren, Jie and Rajbhandari, Samyam and Aminabadi, Reza Yazdani and Ruwase, Olatunji and Yang, Shuangyan and Zhang, Minjia and Li, Dong and He, Yuxiong},
  booktitle={USENIX Annual Technical Conference (ATC)},
  year={2021}}
@inproceedings{rajbhandari2021zeroinfinity,
  title={ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning},
  author={Rajbhandari, Samyam and Ruwase, Olatunji and Rasley, Jeff and Smith, Shaden and He, Yuxiong},
  booktitle={SC21: International Conference for High Performance Computing, Networking, Storage and Analysis},
  year={2021}}
@inproceedings{rhu2016vdnn,
  title={vDNN: Virtualized Deep Neural Networks for Scalable, Memory-Efficient Neural Network Design},
  author={Rhu, Minsoo and Gimelshein, Natalia and Clemons, Jason and Zulfiqar, Arslan and Keckler, Stephen W.},
  booktitle={49th IEEE/ACM International Symposium on Microarchitecture (MICRO)},
  year={2016}}
@inproceedings{huang2020swapadvisor,
  title={SwapAdvisor: Pushing Deep Learning Beyond the GPU Memory Limit via Smart Swapping},
  author={Huang, Chien-Chin and Jin, Gu and Li, Jinyang},
  booktitle={ASPLOS},
  year={2020}}
@inproceedings{peng2020capuchin,
  title={Capuchin: Tensor-based GPU Memory Management for Deep Learning},
  author={Peng, Xuan and Shi, Xuanhua and Dai, Hulin and Jin, Hai and Ma, Weiliang and Xiong, Qian and Yang, Fan and Qian, Xuehai},
  booktitle={ASPLOS},
  year={2020}}
@article{chen2016checkpointing,
  title={Training Deep Nets with Sublinear Memory Cost},
  author={Chen, Tianqi and Xu, Bing and Zhang, Chiyuan and Guestrin, Carlos},
  journal={arXiv preprint arXiv:1604.06174},
  year={2016}}
@inproceedings{kingma2015adam,
  title={Adam: A Method for Stochastic Optimization},
  author={Kingma, Diederik P. and Ba, Jimmy},
  booktitle={ICLR},
  year={2015}}
@inproceedings{loshchilov2019adamw,
  title={Decoupled Weight Decay Regularization},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={ICLR},
  year={2019}}
@inproceedings{micikevicius2018mixed,
  title={Mixed Precision Training},
  author={Micikevicius, Paulius and Narang, Sharan and Alben, Jonah and Diamos, Gregory and Elsen, Erich and Garcia, David and Ginsburg, Boris and Houston, Michael and Kuchaiev, Oleksii and Venkatesh, Ganesh and Wu, Hao},
  booktitle={ICLR},
  year={2018}}
@inproceedings{paszke2019pytorch,
  title={PyTorch: An Imperative Style, High-Performance Deep Learning Library},
  author={Paszke, Adam and others},
  booktitle={NeurIPS},
  year={2019}}
@inproceedings{dettmers2022eightbit,
  title={8-bit Optimizers via Block-wise Quantization},
  author={Dettmers, Tim and Lewis, Mike and Shleifer, Sam and Zettlemoyer, Luke},
  booktitle={ICLR},
  year={2022}}
@inproceedings{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle={ICLR},
  year={2022}}
@inproceedings{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  booktitle={NeurIPS},
  year={2023}}
@inproceedings{dao2022flashattention,
  title={FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness},
  author={Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={NeurIPS},
  year={2022}}
@article{rabe2021selfattention,
  title={Self-attention Does Not Need O(n^2) Memory},
  author={Rabe, Markus N. and Staats, Charles},
  journal={arXiv preprint arXiv:2112.05682},
  year={2021}}
@inproceedings{sheng2023flexgen,
  title={FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU},
  author={Sheng, Ying and others},
  booktitle={ICML},
  year={2023}}
@article{zhao2023fsdp,
  title={PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel},
  author={Zhao, Yanli and others},
  journal={Proceedings of the VLDB Endowment},
  year={2023}}
@inproceedings{wolf2020transformers,
  title={Transformers: State-of-the-Art Natural Language Processing},
  author={Wolf, Thomas and others},
  booktitle={EMNLP: System Demonstrations},
  year={2020}}
@inproceedings{vaswani2017attention,
  title={Attention Is All You Need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}}"""

SECTIONS = []


def section(sid, kicker, title, body):
    SECTIONS.append((sid, title))
    return f'<section id="{sid}"><div class="kicker">{E(kicker)}</div><h2>{E(title)}</h2>{body}</section>'


def num(v, d=2):
    return f"{v:.{d}f}"


body = []

# ---- 0. How to use ----
body.append(section("use", "Start here", "How to use this dossier", f"""
<p>This page collects everything a paper on TrainStream needs: the problem statement, the system design, the analytical model,
the experimental protocol, every measurement taken so far (with provenance), ready-made figures, related work, threats to
validity and a reproducibility checklist. All numbers come from runs on one development machine; the
<a href="#data">raw data</a> is embedded at the end and every figure is generated from it by
<code>research/build_research_dossier.py</code>.</p>
<div class="grid2">
<div class="callout"><p class="t">What is solid</p><p>Correctness (bit-exact paging, optimizer parity ≈3·10⁻⁷, 47 automated tests),
the memory-capacity results (models and fine-tunes that do not fit with stock AdamW train with TrainStream), the
failure-mode findings, and the measurement methodology separating intentional pinned RAM from Windows' VRAM spill.</p></div>
<div class="callout warn"><p class="t">What a reviewer will ask for</p><p>A head-to-head comparison with DeepSpeed ZeRO-Offload, repeated runs with
variance, more hardware (Linux, desktop/datacenter GPUs, full-duplex links), longer training to convergence and a
downstream-quality evaluation. The <a href="#plan">experiment plan</a> lists these with priorities.</p></div>
</div>
<p>Use the <b>Download SVG</b> button under each figure to export it for the paper (styles are inlined; switch the page to the
light theme first for print-friendly colors). Section numbers in the <a href="#outline">paper outline</a> map to the evidence here.</p>
"""))

# ---- 1. Abstract ----
ab = f"""Fine-tuning and training large language models is limited first by GPU memory: with Adam-family optimizers every
parameter costs 16 bytes of VRAM in fp32 before a single activation is stored. We present <b>TrainStream</b>, a
dependency-free C++/CUDA extension for PyTorch that moves optimizer state, gradients and a profiled fraction of activations
to pinned host RAM over PCIe and overlaps the traffic with computation. TrainStream (i) streams each gradient to the host
the moment autograd finalizes it and runs that tensor's AdamW update on host threads during the remainder of the backward
pass, gating the next forward pass per module on CUDA events rather than host synchronization; (ii) pages saved activations
through layout-preserving saved-tensor hooks with a duplex-aware link model, closed-loop tuning and transfer back-pressure;
and (iii) manages buffer lifetimes without allocator stream tagging, which we show can abort a process when the GPU is full.
On a 24 GB laptop GPU, TrainStream trains a 1.8B-parameter model in fp32 within {tr['TrainStream, activations resident']['peak']} GiB of VRAM
(down to {tr['TrainStream, 100% paged']['peak']} GiB with full activation paging) where stock AdamW needs
{tr['Stock AdamW (driver spill)']['peak']} GiB and silently spills into system memory, and fully fine-tunes Qwen2.5-1.5B with the
Hugging Face Trainer in {h15['ts_peak']} GiB instead of {h15['stock_peak']} GiB, {h15['stock_s'] / h15['ts_s']:.1f}× faster than the spilling baseline,
with step-for-step matching losses. Optimizer updates match <code>torch.optim.AdamW</code> to within 3·10⁻⁷ and activation paging
is bit-exact. We also report three correctness hazards of offloading in eager frameworks and a measurement methodology that
distinguishes intentional pinned host memory from driver-level VRAM oversubscription."""
body.append(section("abstract", "Draft", "Abstract (draft, numbers filled in)", f'<div class="abstract">{ab}</div>'
                    '<p class="muted small">Edit freely; every number links to a measurement below. Claims that depend on experiments not yet run '
                    '(e.g. comparison with DeepSpeed) are deliberately absent.</p>'))

# ---- 2. Outline ----
outline = [
    ("1 Introduction", "Memory wall for fine-tuning on a single GPU; 16 B/param; PCIe as an underused resource; contributions.", "#problem, #contrib"),
    ("2 Background", "Adam/AdamW state, mixed precision, autograd saved tensors, CUDA streams/events, pinned vs pageable memory, WDDM.", "#background"),
    ("3 Design", "Three data paths, eager host AdamW, per-module readiness, activation pager, link model + tuner, lifetimes.", "#design, #model"),
    ("4 Implementation", "C++17 runtime, pin-logging allocator, worker pool, zero-copy kernel, PyTorch/HF integration.", "#impl"),
    ("5 Evaluation", "Setup; correctness; capacity; overhead; ablations; HF fine-tuning; measurement methodology.", "#setup … #hf"),
    ("6 Lessons / pitfalls", "Layout preservation, recordStream abort, cyclic-GC leak, eager race, pinned budget, spill detection.", "#pitfalls"),
    ("7 Related work", "ZeRO family, swapping/recompute systems, quantized optimizers, PEFT.", "#related"),
    ("8 Limitations & future work", "Weights resident, logits, host bandwidth, single machine, multi-GPU.", "#limits, #plan"),
    ("9 Conclusion", "—", ""),
]
body.append(section("outline", "Structure", "Suggested paper outline",
                    tbl(["Section", "Content", "Evidence here"], [(E(a), E(b), f'<a href="{c.split(",")[0].split(" ")[0]}">{E(c)}</a>' if c else "—") for a, b, c in outline])))

# ---- 3. Problem ----
body.append(section("problem", "Motivation", "Problem statement", f"""
<p>For a model with <var>N</var> trainable parameters trained with Adam/AdamW in fp32, the static VRAM requirement before any
activation is stored is</p>
<div class="eq">M<sub>GPU</sub><sup>AdamW</sup> = N · (b<sub>w</sub> + b<sub>g</sub> + 8) &nbsp;=&nbsp; 16·N bytes (fp32)</div>
<p>so a 1.5B-parameter model needs ≈24 GB for weights, gradients and moments alone — the entire memory of the GPU used here.
Activations add a batch- and sequence-dependent term on top. Meanwhile the host offers 2–4× more RAM and a PCIe link that is
idle during most of training. The optimizer state is touched once per step by element-wise arithmetic, an ideal candidate for
the host; gradients are produced and consumed exactly once per step; activations are produced in forward order and consumed in
reverse order, which makes them prefetchable.</p>
<p><b>Research questions.</b> RQ1: How much VRAM can be released by moving optimizer state, gradients and activations to host RAM
in an eager framework, without changing training semantics? RQ2: What does it cost in throughput on a realistic consumer link,
and how much can be hidden behind computation? RQ3: Which correctness and stability hazards arise, and how can they be
avoided? RQ4: How can one verify, from outside the process, that memory really moved to host RAM rather than being silently
oversubscribed by the driver?</p>
"""))

# ---- 4. Contributions ----
body.append(section("contrib", "Claims", "Candidate contributions", """
<ol class="claims">
<li><b>Eager, exact host-side AdamW with per-module gating.</b> Each tensor's update is queued from its post-accumulate-grad hook
and runs on host threads during the rest of backward; the next forward waits per module on CUDA events. Unlike a one-step
delayed update, parameters are never stale, so results match <code>torch.optim.AdamW</code> step for step.</li>
<li><b>Fused staging read.</b> For single-micro-batch steps the update reads the DMA staging buffer directly, removing the fp32
accumulator round trip (drain 29.1 → 18.8 ms on 85M).</li>
<li><b>Duplex-aware, closed-loop activation paging with back-pressure.</b> A link model that uses the measured duplex factor and
reserves optimizer traffic predicts the paged prefix; real step times then tune it; a bound on bytes in flight turns paging into
a guaranteed VRAM reduction rather than a best-effort one.</li>
<li><b>Correctness findings for offloading in eager frameworks:</b> (a) restoring saved tensors with a different stride order
silently corrupts gradients of fused memory-efficient attention; (b) allocator stream tagging (<code>recordStream</code>) can
abort the process when VRAM is full; (c) a handle/session reference cycle leaks a step of pinned memory per step.</li>
<li><b>A verification methodology</b> that proves where training memory lives (allocator accounting, OS commit counters, live
tensor inspection, PCIe byte counters) and separates TrainStream's pinned buffers from WDDM's shared-memory spill, which the
OS reports under the same counter.</li>
<li><b>An artifact:</b> a zero-dependency PyTorch extension, a one-line API, a Hugging Face Trainer integration and a one-click
comparison harness.</li>
</ol>
<div class="callout warn"><p class="t">Novelty check before submission</p><p>Contribution 1 is closest to ZeRO-Offload's CPU optimizer and its
delayed parameter update; contribution 3 relates to swapping systems (vDNN, SwapAdvisor, Capuchin) and ZeRO-Infinity's
activation offload. Position the paper on <i>exactness without staleness</i>, <i>closed-loop sizing on consumer links</i> and the
<i>hazard/verification</i> findings, and re-read those papers to make sure each claim is differentiated.</p></div>
"""))

# ---- 5. Background ----
body.append(section("background", "Background", "Background the paper should cover", """
<dl class="terms">
<dt>AdamW state</dt><dd>Two fp32 moments per parameter (m, v); decoupled weight decay [loshchilov2019adamw]. Updates are element-wise and memory-bandwidth bound.</dd>
<dt>Mixed precision</dt><dd>bf16 autocast for compute with fp32 master weights [micikevicius2018mixed]; TrainStream keeps the fp32 master on the host when weights are bf16/fp16.</dd>
<dt>Saved tensors</dt><dd>PyTorch autograd stores tensors needed by backward; <code>saved_tensors_hooks</code> intercept pack/unpack, which is where activation paging happens.</dd>
<dt>Post-accumulate-grad hook</dt><dd>Fires once a parameter's gradient is final (after all contributions), the earliest safe point to stream and update it.</dd>
<dt>CUDA streams & events</dt><dd>Independent queues; an event recorded on one stream can be waited on by another without blocking the host.</dd>
<dt>Pinned vs pageable</dt><dd>Page-locked memory allows asynchronous DMA at full link speed (27.9 vs 14.5 GB/s measured); too much locked memory makes the OS fail DMA mappings.</dd>
<dt>Mapped (zero-copy) memory</dt><dd>Pinned memory addressable from kernels; every access becomes a PCIe transaction.</dd>
<dt>WDDM sysmem fallback</dt><dd>On Windows the driver may back GPU allocations with system RAM when VRAM is exhausted ("shared GPU memory"): no OOM, but a large slowdown.</dd>
</dl>"""))

# ---- 6. Design ----
legend_arch = ""
body.append(section("design", "System", "Design", f"""
{figure("fig-arch", architecture_svg(), "Where every byte lives. Dashed boxes are buffers whose lifetime is a fraction of a step; the three arrows are the per-device CUDA streams TrainStream creates (D2H at lowest priority, H2D at highest because prefetch is on the backward critical path).")}
<h3>6.1 Data paths</h3>
<p><b>Optimizer state and gradients (cpu mode).</b> In the post-accumulate-grad hook of parameter <i>p</i>, a D2H copy of the
gradient into a pinned staging buffer is queued on the D2H stream after an event recorded on the compute stream. The VRAM
gradient is released as soon as the host has consumed it. Host worker threads accumulate (when micro-batching) and run AdamW
on the fp32 master and moments, then queue an H2D copy of the new weights (or a rounded bf16/fp16 copy) on the H2D stream,
ordered after an event recorded when the gradient was produced. A per-slot <code>ready</code> event is published.</p>
<p><b>Per-module readiness.</b> A forward pre-hook on every module that owns parameters waits (host side, GIL released) until the
module's updates have been <i>issued</i> and then makes the compute stream wait on their <code>ready</code> events
(GPU side). The issued/landed split matters: waiting on an event that has not been recorded yet is a silent no-op in CUDA.</p>
<p><b>Zero-copy mode (gpu).</b> Gradients stay in VRAM until <code>step()</code>; a fused kernel updates weights in place while reading
and writing the moments through mapped pinned memory. It issues all loads of a four-way unrolled body before any arithmetic to keep
≈12 PCIe reads per thread in flight. Moments never occupy VRAM.</p>
<p><b>Activations.</b> Saved tensors produced inside selected transformer blocks are copied D2H into pinned stashes during forward
(deduplicated by tensor identity) and staged back H2D on the high-priority stream when backward first touches the block above
them (a ping-pong window of depth <i>k</i>, default 1).</p>
{figure("fig-timeline", timeline_svg(), "Overlap within one step (schematic). Per-tensor host AdamW starts as soon as that tensor's gradient lands, so most of the optimizer runs under the remaining backward pass; the next forward is gated per module, so block 0 can start before the last weights land.",
        legend([("forward / backward compute", "c-ts"), ("activation paging", "c-act"), ("gradient D2H", "c-g"), ("weight H2D", "c-w"), ("host AdamW", "c-m")]))}
<h3>6.2 Eager update algorithm</h3>
<pre class="algo">on post_accumulate_grad(p):                        # autograd thread
    micro[p] += 1
    wait until slot(p) has no in-flight work        # staging buffer reuse
    e_prod ← record(compute); copy_stream.wait(e_prod)
    D2H(grad(p) → stage[p]) on d2h;  e_d2h ← record(d2h)
    if eager and micro[p] = A (accumulation steps) and no clipping:
        enqueue AdamW(p, reads stage[p] after e_d2h)    # fused staging read
        then H2D(master[p] → p) after e_prod;  ready[p] ← record(h2d)
    else: enqueue accumulate(stage[p] → acc[p])
    grad(p) ← None                                   # VRAM released when the D2H has landed

on forward_pre(module):
    for p in params(module): wait until update(p) issued;  compute.wait(ready[p])

on step():   (non-eager path) record e_step; enqueue remaining updates in first-forward order</pre>
<p>Chunks of 1M elements are assigned to worker <code>(s·φ + c) mod T</code>, which gives per-chunk FIFO ordering of
accumulate → norm → update without locks. Global-norm clipping requires all gradients, so it disables the eager path and uses a
host reduction barrier inside <code>step()</code>.</p>
<h3>6.3 Lifetime management without allocator stream tagging</h3>
<p>The standard way to use a tensor on a side stream is <code>recordStream</code>, which makes the caching allocator record events
when the tensor is freed. We found that this event creation can fail inside a destructor when the device is full, which terminates
the process (<a href="#pitfalls">§10</a>). TrainStream instead keeps explicit lifetimes: D2H sources are held until their copy
event completes (polled, with back-pressure), prefetch buffers are allocated from the consumer stream's pool with the H2D ordered
after an event on that stream and are waited on before use or release, and streamed gradients are released by the worker after
the host has consumed them.</p>
<h3>6.4 Pinned memory</h3>
<p>A pin-logging caching allocator wraps <code>cudaHostAlloc</code>: size buckets (4 KiB below 1 MiB, 2 MiB above), event-fenced
reuse (a released block re-enters the free list only after fences on every copy stream pass; if fences cannot be created the
streams are drained), per-tag accounting and a ring log. Moments are pinned only while the total stays within 40 % of physical
RAM; the fp32 master and staging buffers (the DMA endpoints) are always pinned.</p>
"""))

# ---- 7. Analytical model ----
body.append(section("model", "Model", "Analytical model", f"""
<h3>Memory per parameter</h3>
{figure("fig-mem", FIG_MEM, "Bytes per parameter (fp32 weights). Stock AdamW keeps 16 B/param in VRAM. TrainStream cpu mode keeps only the weights (plus transient gradients) in VRAM and 16 B/param on the host (the moments may be pageable). Zero-copy mode keeps gradients in VRAM until step() and the moments in mapped pinned memory.")}
<h3>PCIe traffic per step</h3>
<div class="eq">cpu mode: &nbsp; D2H = N·b<sub>g</sub>, &nbsp; H2D = N·b<sub>w</sub> &nbsp;&nbsp;|&nbsp;&nbsp; zero-copy: 16·N (+8·N with a host fp32 master) &nbsp;&nbsp;|&nbsp;&nbsp; activations: 2·S<sub>blk</sub> per paged block</div>
<h3>Link model and activation sizing</h3>
<div class="eq">κ = B<sub>bidir</sub> / max(B<sub>H2D</sub>, B<sub>D2H</sub>) ∈ [1, 2] &nbsp;&nbsp;&nbsp;
link(d, h) = max( d/B<sub>D2H</sub>, h/B<sub>H2D</sub>, (d/B<sub>D2H</sub> + h/B<sub>H2D</sub>) / κ )</div>
<div class="eq">K = ⌊ (s · T<sub>step</sub> − link(N·b<sub>g</sub>, N·b<sub>w</sub>)) / link(S<sub>blk</sub>, S<sub>blk</sub>) ⌋, &nbsp; s = 0.85 &nbsp;&nbsp; → page blocks 0 … K−1</div>
<p>T<sub>step</sub> is the measured un-paged step time from calibration (three steps). The tuner then measures real step time with
K blocks paged and halves K while T<sub>K</sub> &gt; (1 + ε)·T<sub>0</sub> (ε = 0.15 by default). Back-pressure bounds the bytes
of sources still crossing the link by β (1 GiB): beyond it, the host waits on the oldest copy.</p>
<h3>Break-even arithmetic intensity</h3>
<div class="eq">I* = F<sub>bf16</sub> / B<sub>H2D</sub> = {P['bf16_tflops']}·10¹² / {P['h2d']}·10⁹ ≈ {P['bf16_tflops'] * 1e12 / (P['h2d'] * 1e9):,.0f} FLOP/byte</div>
<p>A transfer is hideable behind matmuls only if the work it enables exceeds I* FLOPs per byte moved. A transformer block's
activations (≈400 MiB at d=2048, batch 8×512) take ≈30 ms of link time at κ = 1, while the block's forward takes ≈6 ms on this
GPU — so beyond a short prefix, paging trades time for memory (measured in <a href="#res-overhead">R4</a>).</p>
<h3>Host update bound</h3>
<p>Host AdamW reads p, m, v, g and writes p, m, v (≈28 B/element plus staging); at the measured {P['host_adamw_gelems']}–{P['host_adamw_quick']} G elem/s this is
≈70–78 GB/s, i.e. DRAM-bandwidth bound. For N parameters the host needs ≈N / R<sub>host</sub> per step (≈0.7 s for 1.8B), which
eager updates overlap with backward.</p>
<details><summary>LaTeX for these equations</summary><pre class="latex">M_{{GPU}}^{{AdamW}} = N\\,(b_w + b_g + 8)
\\kappa = \\frac{{B_{{bidir}}}}{{\\max(B_{{H2D}}, B_{{D2H}})}}
\\mathrm{{link}}(d,h) = \\max\\left(\\frac{{d}}{{B_{{D2H}}}}, \\frac{{h}}{{B_{{H2D}}}}, \\frac{{d/B_{{D2H}} + h/B_{{H2D}}}}{{\\kappa}}\\right)
K = \\left\\lfloor \\frac{{s\\,T_{{step}} - \\mathrm{{link}}(N b_g, N b_w)}}{{\\mathrm{{link}}(S_{{blk}}, S_{{blk}})}} \\right\\rfloor
I^* = F_{{bf16}} / B_{{H2D}}</pre></details>
"""))

# ---- 8. Implementation ----
body.append(section("impl", "Artifact", "Implementation", tbl(
    ["Component", "File", "Notes"],
    [("Runtime: event pool, streams, pinned allocator, DMA, PCIe/matmul benchmarks", "<code>trainstream/csrc/memory_pinner.cpp</code>", "C++ (C++17 features; compiled as C++20 because PyTorch 2.14 headers require it)"),
     ("Optimizer engine: worker pool, gradient pipeline, host AdamW, eager path", "<code>trainstream/csrc/offload_adamw.cpp/.h</code>", "templated on gradient and parameter dtype; branch-free inner loop"),
     ("Zero-copy AdamW + grad-norm kernels", "<code>trainstream/csrc/offload_adamw_kernel.cu</code>", "CUDA headers only; 4-way unroll, load-first"),
     ("Bindings", "<code>trainstream/csrc/bindings.cpp</code>", "pybind11 via torch/extension.h; GIL released on blocking calls; terminate hook prints C++ errors"),
     ("Python API", "<code>trainstream/wrapper.py, optimizer.py, hooks.py, profiler.py</code>", "<code>wrap()</code>, <code>OffloadAdamW</code> (a real torch.optim.Optimizer), pager + tuner, profiler"),
     ("Hugging Face", "<code>trainstream/integrations/huggingface.py</code>", "<code>TrainStreamTrainer</code>, <code>wrap_hf</code>; optional dependency"),
     ("Harness", "<code>dashboard/</code>, <code>examples/verify_ram_offload.py</code>, <code>examples/hf_finetune.py</code>", "one-click comparison reports, verification verdicts"),
     ("Tests", "<code>tests/</code>", f"{DATA['tests']['core']} core + {DATA['tests']['hf']} Hugging Face tests, all passing")])))

# ---- 9. Setup ----
preset_rows = [(n, l, h, d, f"{PARAMS[n] / 1e6:,.0f}M", f"{PARAMS[n] * 16 / 2**30:.1f}", f"{PARAMS[n] * 4 / 2**30:.1f}") for n, l, h, d in PRESETS]
body.append(section("setup", "Evaluation", "Experimental setup", f"""
{tbl(["Item", "Value"], [(k.capitalize(), E(v)) for k, v in pf.items()])}
<h3>Workloads</h3>
<p><b>Synthetic-architecture GPT (byte level).</b> Pre-LN decoder, vocabulary 256, sequence 512, SDPA attention, trained on the
Python standard-library source (≈12 MiB of real code, 2 % held out), bf16 autocast, AdamW β=(0.9, 0.95), weight decay 0.1,
cosine schedule with warm-up. <b>Hugging Face models:</b> Qwen2.5-0.5B and Qwen2.5-1.5B, full fine-tuning in fp32 with bf16 autocast,
Trainer defaults (AdamW), batch 4×512, gradient accumulation 2, max_grad_norm 1.0, same corpus tokenized.</p>
{tbl(["Preset", "Layers", "Heads", "d_model", "Params", "AdamW static (GiB)", "Weights only (GiB)"], preset_rows, num_cols=(1, 2, 3, 4, 5, 6))}
<h3>Metrics and how they are measured</h3>
<ul>
<li><b>Peak VRAM</b>: <code>torch.cuda.max_memory_allocated</code> per step (allocator view); device-wide usage from <code>cudaMemGetInfo</code>.</li>
<li><b>Host memory</b>: engine accounting (pinned + pageable state), process committed memory (<code>GetProcessMemoryInfo.PrivateUsage</code>), system available RAM.</li>
<li><b>Driver spill</b>: Windows performance counters <i>GPU Process Memory\\Dedicated Usage</i> and <i>\\Shared Usage</i> per process, minus TrainStream's own pinned bytes (which Windows also reports as shared).</li>
<li><b>Traffic</b>: bytes issued on TrainStream's D2H/H2D streams per step.</li>
<li><b>Time</b>: wall time between successive loss values (synchronising only the compute stream) or, in probes, with explicit synchronisation per phase.</li>
<li><b>Correctness</b>: max |Δ| of parameters/gradients vs reference; loss trajectories.</li>
</ul>
<h3>Protocol</h3>
<p>Each engine runs in a fresh state (separate process in the verification tool; cache release between engines in the dashboard).
Stock runs are stopped when device VRAM exceeds 95 % (to avoid multi-minute spills) unless spilling is the object of study, and
all runs stop if free system RAM falls below 3 GiB. <b>Single runs, no repetitions yet</b> — see <a href="#threats">threats</a>.</p>
"""))

# ---- 10. Results: hardware ----
body.append(section("res-hw", "Results", "R1 · Hardware characterisation", f"""
{figure("fig-sweep", FIG_SWEEP, f"PCIe bandwidth versus transfer size with pinned memory (CUDA-event timed on TrainStream's own streams). Throughput saturates at ≥1 MiB; pageable host memory reaches only {P['pageable_h2d']} GB/s. 4 KiB latency: {P['lat_4k_us']} µs.", legend([("H2D", "c-h2d"), ("D2H", "c-d2h")]))}
{tbl(["Quantity", "Measured"], [
    ("Pinned H2D / D2H", f"{P['h2d']} / {P['d2h']} GB/s"), ("Both directions concurrently", f"{P['bidir']} GB/s"),
    ("Duplex factor κ", f"{P['bidir'] / max(P['h2d'], P['d2h']):.2f}"), ("Pageable H2D", f"{P['pageable_h2d']} GB/s"),
    ("Dense matmul bf16 / fp32", f"{P['bf16_tflops']} / {P['fp32_tflops']} TFLOPS"),
    ("Host AdamW ({} threads)".format(P['threads']), f"{P['host_adamw_gelems']}–{P['host_adamw_quick']} G elem/s"),
    ("Zero-copy AdamW kernel", f"{P['zc_adamw_gelems_32m']}–{P['zc_adamw_gelems_16m']} G elem/s ({P['zc_pcie_gbps']} GB/s of PCIe traffic)"),
    ("Break-even intensity I*", f"≈{P['bf16_tflops'] * 1e12 / (P['h2d'] * 1e9):,.0f} FLOP/byte")], num_cols=(1,))}
<p><b>Observation.</b> The duplex factor is 1.0: on this Windows laptop, simultaneous H2D and D2H share one budget. Workstation and
Linux systems commonly approach 2; the link model must therefore measure it rather than assume it.</p>
"""))

# ---- 11. Correctness ----
par_rows = [(r["mode"], r["clip"], f"{r['diff']:.1e}", '<span class="pill good">match</span>') for r in DATA["parity"]]
agree_txt = (f"mean |Δloss| {AGREE85[1]:.4f}, max {AGREE85[2]:.4f} over {AGREE85[0]} common steps" if AGREE85 else "n/a")
body.append(section("res-correct", "Results", "R2 · Correctness", f"""
{tbl(["Optimizer mode", "Clipping", "Max |Δparam| after 20 steps", "Result"], par_rows, num_cols=(2,))}
{figure("fig-err", FIG_ERR, "Maximum difference against the reference (log scale). A pager that restores saved tensors as contiguous silently corrupts memory-efficient attention gradients (54.75); the layout-preserving pager is bit-exact on every attention backend. For scale: two bf16 runs with no paging at all differ by 1.95·10⁻³ because fused attention backward is not deterministic.")}
{figure("fig-85", FIG_85, f"85M GPT, 60 steps, identical data: stock AdamW vs TrainStream (cpu mode, auto paging); {agree_txt}. Held-out loss {a85['val_loss']:.3f} vs {t85['val_loss']:.3f}." if a85 and t85 else "85M loss curves.", legend([("Stock AdamW", "c-adamw"), ("TrainStream", "c-ts")]))}
<h3>Automated test inventory ({DATA['tests']['core'] + DATA['tests']['hf']} tests)</h3>
<ul>
<li>Optimizer parity (standalone and attached/asynchronous, cpu and gpu modes), accumulation + clipping, parameter groups + LR scheduler, bf16/fp16 weights with fp32 master, gradient streaming accounting, state-dict round trip and interoperability with <code>torch.optim.AdamW</code>, migration of existing optimizer state, eager updates with declared accumulation, undeclared accumulation raises.</li>
<li>Pinned allocator caching and logging, mapped allocations, D2H/H2D round trips for three dtypes, D2H ordered after a long producer kernel, timing events.</li>
<li>Activation paging: bit-exact gradients (depth 1 and 2), VRAM reduction, fused-attention layout regression on three SDPA backends, dense-layout helper, pinned stashes freed every step with the GC disabled, auto calibration + tuning, no paging under <code>no_grad</code>.</li>
<li>Hugging Face: decoder detection, custom loop parity, Trainer parity (cpu and gpu modes, clipping, accumulation, schedule), paging parity, LoRA (only adapters get host state; base weights untouched), checkpoint resume equals uninterrupted training, fp16 rejected.</li>
</ul>
"""))

# ---- 12. Capacity ----
body.append(section("res-capacity", "Results", "R3 · Memory capacity: where is the memory?", f"""
{tbl(["Model", "VRAM budget", "Stock AdamW", "TrainStream", "Peak VRAM", "Persistent VRAM beyond weights", "Moments (stock would hold)", "Host state", "Process commit growth", "D2H per step", "Unexplained shared memory"], [
    (v["model"], f"{v['cap']} GiB", v["stock"], v["ts_steps"], f"{v['ts_peak']} GiB", f"{v['persist']} GiB", f"{v['moments']} GiB", f"{v['host']} GiB", f"{v['commit']} GiB", f"{v['d2h']} GiB", f"{v['unexplained_shared']} GiB") for v in DATA["verify"]], num_cols=tuple(range(4, 11)))}
<p>Measured with <code>examples/verify_ram_offload.py</code> (each engine in a fresh process; VRAM budget enforced with
<code>set_per_process_memory_fraction</code> so the stock run fails with a real out-of-memory error instead of spilling). All
12 verdict checks pass for both models, including: a live <code>exp_avg</code> tensor resides on the CPU, pinned, with non-zero
values; GPU weights are bit-identical to the host fp32 master after the last step; gradients and weights cross PCIe every step.</p>
<div class="callout"><p class="t">Measurement pitfall worth a paragraph in the paper</p><p>Windows reports CUDA pinned host memory as
"Shared GPU memory": allocating exactly 4 GiB of pinned memory raised the process's shared-usage counter by
{DATA['pinned_probe']['shared_counter_mib']:,} MiB, while 4 GiB of VRAM raised dedicated usage by
{DATA['pinned_probe']['dedicated_counter_mib']:,} MiB. The same counter is what grows during a driver spill. Subtracting
TrainStream's own pinned bytes leaves 0.08–0.09 GiB unexplained, versus {DATA['spill_control']['shared']} GiB of genuine spill in
the positive control ({E(DATA['spill_control']['model'])}: peak {DATA['spill_control']['peak']} GiB on a 23.89 GiB card,
{DATA['spill_control']['dedicated']} GiB dedicated + {DATA['spill_control']['shared']} GiB shared).</p></div>
{figure("fig-18", FIG_18, f"xxl-1.8B (1.8B parameters, fp32 weights, batch 8×512) trained for 200 steps with TrainStream on the 24 GB GPU: peak {t18['peak_vram'] / 2**30:.2f} GiB VRAM, {t18['host_state'] / 2**30:.2f} GiB optimizer state in host RAM, {t18['tokens_per_s']:,.0f} tokens/s; final loss {t18['final_loss']:.3f}, held-out {t18['val_loss']:.3f}. Stock AdamW stopped at the VRAM limit after one step." if t18 else "1.8B loss curve.")}
"""))

# ---- 13. Overhead / tradeoff ----
body.append(section("res-overhead", "Results", "R4 · Throughput and the memory–time trade-off", f"""
{figure("fig-tradeoff", FIG_TRADEOFF, "1.8B GPT, batch 8×512: peak VRAM versus step time. Stock AdamW needs 34.1 GiB and the driver spills the excess into system RAM (13.7 s/step). TrainStream with activations resident needs 20.6 GiB (≈1.1 s/step, 12× faster than the spilling baseline); paging 43 % / 100 % of activations reduces VRAM to 16.7 / 13.5 GiB at 1.4 / 1.9 s/step.", legend([("TrainStream (activation paging 0 → 43 → 100 %)", "c-ts"), ("Stock AdamW", "c-adamw")]))}
{tbl(["Workload", "Stock AdamW", "TrainStream", "Ratio"], [
    ("85M GPT, fits in VRAM (tokens/s)", f"{a85['tokens_per_s']:,.0f}" if a85 else "92.5k", f"{t85['tokens_per_s']:,.0f}" if t85 else "62.3k", f"{(t85['tokens_per_s'] / a85['tokens_per_s']):.2f}×" if a85 and t85 else "0.63×"),
    ("85M GPT peak VRAM", f"{a85['peak_vram'] / 2**30:.2f} GiB" if a85 else "2.47 GiB", f"{t85['peak_vram'] / 2**30:.2f} GiB" if t85 else "1.83 GiB", f"−{(1 - t85['peak_vram'] / a85['peak_vram']) * 100:.0f} %" if a85 and t85 else "−26 %"),
    ("1.2B GPT (30 steps)", "stopped at VRAM limit after 1 step", f"{t12['steps']} steps, {t12['peak_vram'] / 2**30:.2f} GiB, {t12['tokens_per_s']:,.0f} tok/s" if t12 else "completed", "—"),
    ("1.8B GPT, spill vs resident", "13.7 s/step (34.1 GiB)", "1.11 s/step (20.6 GiB)", "12.3× faster"),
    ("100M-param optimizer step alone (incl. sync)", "3.7 ms, 771 MiB VRAM", "64.9 ms (cpu) / 69.3 ms (gpu), 0 MiB VRAM", "—")], num_cols=())}
<p><b>Interpretation.</b> When the model fits, TrainStream costs throughput: host AdamW is bounded by DRAM bandwidth and the link
is shared by gradients, weights and activations. The benefit is capacity: workloads that do not fit (or fit only by spilling)
become trainable, and VRAM freed on smaller models can be spent on larger batches. The spill speed-up is workload dependent:
at batch 4 the same 1.8B stock run spilled 13.85 GiB but ran at 1.84 s/step.</p>
"""))

# ---- 14. Ablations ----
body.append(section("res-ablation", "Results", "R5 · Ablations", f"""
{figure("fig-eager", FIG_EAGER, f"Eager updates (85M GPT, activations resident; forward {DATA['eager_phases']['fwd_ms']} ms, backward ≈{DATA['eager_phases']['bwd_ms']:.0f} ms). Host time remaining after backward until updated weights have landed: deferring all updates to step() leaves 42 ms; per-tensor eager updates cut it to 29 ms; reading the DMA staging buffer directly (no fp32 accumulator) to 19 ms.")}
{figure("fig-bp", FIG_BP, "Back-pressure (1.8B GPT, peak forward VRAM). Without a bound on bytes in flight, the GPU outruns the 28 GB/s link and paging 43 % of blocks saves only 0.4 GiB; with a 1 GiB bound the same setting saves 3.7 GiB and paging every block brings the forward peak to 11.3 GiB.")}
<h3>Link model vs naive sizing (85M GPT)</h3>
{tbl(["Sizing rule", "Blocks paged (of 12)", "Step time"], [
    ("Naive: per-block D2H must fit under per-block forward, full duplex assumed", str(DATA['auto_85m']['naive_blocks']), f"{DATA['auto_85m']['naive_step_ms']} ms (vs {DATA['auto_85m']['baseline_ms']} ms stock)"),
    ("Duplex-aware link model from measured step time, optimizer traffic reserved", str(DATA['auto_85m']['model_blocks']), "—"),
    ("Closed-loop tuning (≤15 % slowdown)", str(DATA['auto_85m']['tuned_blocks']), "within tolerance")], num_cols=(1,))}
<p class="muted small">Calibration measured {DATA['auto_85m']['block_mib']} MiB of saved activations per block, {DATA['auto_85m']['block_fwd_ms']} ms block forward, {DATA['auto_85m']['d2h_ms']} ms per-block D2H. The naive rule also predated eager updates and lifetime fixes, so its step time overstates the cost; re-measure all three rows on the final code for the paper.</p>
<h3>Pinned-memory budget</h3>
<p>Pinning all optimizer state of the 1.8B model ({DATA['pinned_18b']['before_pinned']} GiB) plus activation stashes led to asynchronous
CUDA out-of-memory errors at ≈{DATA['pinned_18b']['async_oom_at']} GB of page-locked memory on a 63 GB machine. Limiting pinning
to 40 % of RAM moved the moments to pageable memory ({DATA['pinned_18b']['after_pinned']} GiB pinned + {DATA['pinned_18b']['after_pageable']} GiB pageable) with no change in results, because only host threads touch the moments.</p>
{figure("fig-leak", FIG_LEAK, "Live pinned activation memory per step before and after replacing the session→handle strong references with weak ones. With a reference cycle, buffers are released only by Python's cyclic garbage collector (5.9 GB accumulated per step on the 1.8B model); with weak back-references the autograd graph owns the buffers and frees them deterministically.", legend([("before", "c-adamw"), ("after", "c-ts")]))}
"""))

# ---- 15. HF ----
body.append(section("hf", "Results", "R6 · Hugging Face fine-tuning", f"""
{tbl(["Model (full fine-tune)", "Steps", "Stock Trainer peak VRAM", "TrainStreamTrainer peak VRAM", "Stock time", "TrainStream time", "Host optimizer state"], [
    (h["model"], h["steps"], f"{h['stock_peak']} GiB" + (" (spill)" if h["stock_peak"] > 23.89 else ""), f"{h['ts_peak']} GiB", f"{h['stock_s']} s", f"{h['ts_s']} s", f"{h['host']} GiB") for h in DATA["hf"]], num_cols=(1, 2, 3, 4, 5, 6))}
{figure("fig-hf", FIG_HF, f"Qwen2.5-1.5B full fine-tuning with the Hugging Face Trainer (bf16 autocast, batch 4×512, accumulation 2, clipping 1.0): logged losses of the stock Trainer and TrainStreamTrainer agree to ≤0.002 at every step; peak VRAM {h15['stock_peak']} → {h15['ts_peak']} GiB; wall time {h15['stock_s']} → {h15['ts_s']} s for 10 steps because the stock run spilled into system RAM.", legend([("stock Trainer", "c-adamw"), ("TrainStreamTrainer", "c-ts")]))}
<p>The integration subclasses <code>transformers.Trainer</code>: the Trainer builds its usual AdamW (decay groups, hyper-parameters),
which is converted to OffloadAdamW; clipping moves into the optimizer (the Trainer's own clip would see no gradients); the
gradient norm of the current step is substituted at log time; weights are synchronised before every checkpoint; fp16 (GradScaler)
is rejected. Checkpoint resume reproduces uninterrupted training to &lt;10⁻⁵.</p>
"""))

# ---- 16. Pitfalls ----
body.append(section("pitfalls", "Lessons", "Failure modes found during development (paper section candidates)", tbl(
    ["Hazard", "Symptom", "Root cause", "Fix", "Evidence"],
    [("Stride-changing activation restore", "Wrong gradients, NaN at 300M under bf16; no error raised", "Memory-efficient SDPA backward assumes the saved q/k/v/out layout of forward", "Copy dense tensors byte-for-byte with exact strides; pack non-dense views densely in the same dimension order", "max |Δgrad| 54.75 → 0 (bit-exact); regression test on 3 backends"),
     ("recordStream on a full device", "Process abort (std::terminate) inside a tensor destructor", "Allocator creates CUDA events in its free path for tensors used on side streams; creation fails with out-of-memory", "Explicit lifetimes: hold sources until copy events complete, allocate prefetch from consumer pool, release gradients from workers", "C++ terminate hook printed 'CUDA error: out of memory'"),
     ("Reference cycle", "Pinned RAM grows by one step of activations per step", "Handle ↔ session strong cycle freed only by cyclic GC", "Session holds weak references", "5.9 → 23.4 GB over 4 steps; test with GC disabled"),
     ("Eager update race", "Occasional double update, parameter drift", "has_grad cleared asynchronously by the finaliser; step() saw stale flag", "Clear synchronously when the eager update is queued", "parity test failure → 5 consecutive clean suite runs"),
     ("Too much page-locked memory", "Asynchronous CUDA out-of-memory errors", "OS refuses DMA mappings beyond its locked-memory tolerance", "Pin only DMA endpoints; moments pinned within 40 % RAM", "failure at ≈39.7 GB pinned"),
     ("Paging without back-pressure", "Little or no VRAM saved", "GPU produces activations faster than the link drains them; sources must stay alive until copied", "Bound bytes in flight, host waits on the oldest copy", "0.4 GiB vs 3.7 GiB saved (R5, back-pressure figure)"),
     ("Hidden driver spill", "No OOM, step time ×10+", "WDDM sysmem fallback backs VRAM with system RAM", "Detect via per-process counters; stop stock runs at 95 % VRAM or cap with memory fraction", "13.85 GiB spill measured")],
)))

# ---- 17. Related work ----
body.append(section("related", "Positioning", "Related work and how TrainStream differs", f"""
<ul class="refs">
<li><b>ZeRO family</b> [rajbhandari2020zero; ren2021zerooffload; rajbhandari2021zeroinfinity] partitions and offloads optimizer state, gradients and (ZeRO-Infinity) parameters and activation checkpoints to CPU/NVMe; ZeRO-Offload runs a vectorised CPU Adam and offers a one-step delayed parameter update to overlap it. TrainStream targets a single consumer GPU, keeps updates exact (no delay) by gating per module, sizes activation paging from measured link behaviour, and is a single-library drop-in for PyTorch eager.</li>
<li><b>Swapping and recomputation</b>: vDNN [rhu2016vdnn], SwapAdvisor [huang2020swapadvisor], Capuchin [peng2020capuchin], gradient checkpointing [chen2016checkpointing]. TrainStream's pager is a runtime hook-based instance of swapping with a closed loop instead of a static plan; combining it with checkpointing is future work.</li>
<li><b>Reducing state instead of moving it</b>: 8-bit optimizers [dettmers2022eightbit], LoRA [hu2022lora], QLoRA [dettmers2023qlora] are orthogonal and composable (LoRA support is evaluated).</li>
<li><b>Attention memory</b>: FlashAttention [dao2022flashattention] and memory-efficient attention [rabe2021selfattention] — relevant both as activation-memory reducers and as the source of the layout hazard.</li>
<li><b>Inference offloading</b>: FlexGen [sheng2023flexgen] schedules weights/KV cache over a GPU-CPU-disk hierarchy for throughput; TrainStream addresses training.</li>
<li><b>Frameworks</b>: PyTorch [paszke2019pytorch], FSDP with CPU offload [zhao2023fsdp], Hugging Face Transformers [wolf2020transformers].</li>
</ul>
{tbl(["Capability", "TrainStream", "ZeRO-Offload", "ZeRO-Infinity", "vDNN", "8-bit optimizers"], [
    ("Optimizer state off GPU", "yes (host RAM)", "yes (CPU)", "yes (CPU/NVMe)", "no", "no (compressed in VRAM)"),
    ("Gradients streamed off GPU during backward", "yes", "yes", "yes", "no", "no"),
    ("Activation offload", "yes (closed-loop)", "via activation checkpointing options", "yes (checkpoints)", "yes (CNNs)", "no"),
    ("Exact updates while overlapping", "yes (per-module gating)", "delayed update optional", "—", "n/a", "n/a"),
    ("Parameters offloaded", "no", "no", "yes", "no", "no"),
    ("Multi-GPU", "no (design only)", "yes", "yes", "no", "yes")])}
<p class="muted small">Cells for other systems summarise their papers as we understand them — verify each against the original publications and current releases before submission.</p>
"""))

# ---- 18. Limitations ----
body.append(section("limits", "Honesty", "Limitations", """
<ul>
<li>Weights stay resident: a model whose weights alone exceed VRAM cannot be trained (no parameter paging).</li>
<li>Output logits of large-vocabulary models are not offloaded (Qwen, batch 4×512: 1.16 GiB of fp32 logits plus gradient).</li>
<li>Host AdamW relies on compiler auto-vectorisation and is DRAM-bandwidth bound (2.5–2.8 G elem/s); explicit SIMD (as in ZeRO-Offload's CPU Adam) would help.</li>
<li>Single GPU per process; the multi-GPU pipeline design is documented, not implemented.</li>
<li>fp16 with loss scaling is unsupported (GradScaler needs gradients on the GPU); bf16 and fp32 are supported.</li>
<li>All measurements come from one Windows laptop with a duplex factor of 1.0; results on full-duplex Linux systems will differ (likely favourably for paging).</li>
<li>The TensorFlow reference runs on DirectML (no CUDA support for this GPU in Windows TensorFlow); it reserved ≈19 GB of RAM and ≈10 GiB of GPU memory even for 85M parameters and varied 36× in speed between identical runs, so it is not a fair baseline.</li>
</ul>"""))

# ---- 19. Threats + plan ----
plan = [
    ("P0", "Head-to-head with DeepSpeed ZeRO-Offload (same GPU, Linux/WSL2; identical model, data, batch)", "Primary baseline every reviewer will expect"),
    ("P0", "Repeat every throughput/memory measurement ≥3× (report mean ± std); fix seeds; warm-up excluded", "Single runs today"),
    ("P0", "Re-run the ablations (eager, link model, back-pressure, pinning) on the final code with the same protocol", "Some rows predate later fixes"),
    ("P1", "Second and third platform: desktop RTX 4090/5090 on PCIe x16 (Linux), and a datacenter GPU (A100/H100 PCIe)", "Generalisation; duplex factor ≈2"),
    ("P1", "Longer training to convergence (≥5k steps) and a downstream evaluation of a fine-tuned model (e.g. instruction data + benchmark)", "Show no quality regression beyond loss parity"),
    ("P1", "Maximum trainable size vs GPU and host RAM (dashboard 'Find limits') for stock, TrainStream, ZeRO-Offload", "Capacity frontier figure"),
    ("P2", "SIMD host AdamW; prefetch depth sweep; accumulation steps sweep; LoRA + activation paging", "Optimisation headroom"),
    ("P2", "Energy per token (nvidia-smi power × time)", "Cost argument for consumer hardware"),
]
body.append(section("threats", "Validity", "Threats to validity", """
<ul>
<li><b>Internal:</b> single runs; bf16 nondeterminism (≈2·10⁻³ gradient noise); stock baselines stopped at 95 % VRAM rather than run to completion; first steps include allocator warm-up (excluded via medians).</li>
<li><b>External:</b> one machine, one OS (WDDM), one GPU generation, a byte-level corpus; Hugging Face results on one model family.</li>
<li><b>Construct:</b> "peak VRAM" is the allocator's view; device usage includes the CUDA context and other processes; host memory includes pinned caches.</li>
<li><b>Conclusion:</b> speed-ups against a spilling baseline depend on how badly the driver spills (12× at batch 8, smaller at batch 4); report capacity results as primary and speed as secondary.</li>
</ul>"""))
body.append(section("plan", "Before submission", "Experiment plan", tbl(["Priority", "Experiment", "Why"], plan)))

# ---- 20. Reproducibility ----
body.append(section("repro", "Artifact", "Reproducibility checklist", f"""
<pre class="algo">build.bat                                     # or: python setup.py build_ext --inplace
python -m pytest tests -q                     # {DATA['tests']['core'] + DATA['tests']['hf']} tests
python examples/verify_ram_offload.py         # capacity + where-is-the-memory verdict (GPT 1.2B, 16 GB cap)
python examples/verify_ram_offload.py --hf Qwen/Qwen2.5-0.5B --vram-cap-gb 7 --batch 2
python examples/verify_ram_offload.py --engine baseline --gpt xxl-1.8B --vram-cap-gb 0   # spill control
python examples/hf_finetune.py --model Qwen/Qwen2.5-1.5B --steps 10 --compare
start_dashboard.bat                           # Full report: one-click comparison, saved under dashboard/reports/
python research/build_research_dossier.py         # regenerate this page from the data</pre>
<ul><li>Record: GPU, driver, OS, PCIe link state under load (nvidia-smi), RAM, background processes (free RAM matters for host state).</li>
<li>Archive the dashboard report JSON files and the verification JSON (<code>--json</code>) with the paper's artifact.</li>
<li>State the AI-assistance used to write the code and text according to the venue's policy (see below).</li></ul>
"""))

body.append(section("disclosure", "Integrity", "Authorship, AI assistance and ownership", """
<p>Most of TrainStream's code and this dossier were produced with an AI coding assistant. Most venues require disclosing such
assistance, and authors remain responsible for every claim, number and citation. Before submission: re-run the key experiments
yourself, verify every reference against the original publication, and confirm that you (or your employer) own the code you
publish, in line with your employment agreement.</p>"""))

# ---- figures index, references, data ----
fig_list = "".join(f'<li><a href="#{fid}-wrap">Figure {n}</a> — {E(cap.split(". ")[0])}.</li>' for n, fid, cap in FIGS)
body.append(section("figures", "Index", "List of figures", f"<ol class='figlist'>{fig_list}</ol>"))
body.append(section("refs", "Citations", "References (BibTeX)", f"""
<p class="muted small">Entries are given as we recall them; verify authors, venues and years against the publishers' pages before submission.</p>
<div class="bib"><button class="copy" data-target="bibtex">Copy BibTeX</button><pre id="bibtex">{E(BIB)}</pre></div>"""))
body.append(section("data", "Provenance", "Raw data", f"""
<p>Machine-readable copy of every number used on this page (also available as <code>window.TRAINSTREAM_DATA</code> in the console).</p>
<details><summary>Show JSON</summary><pre class="json">{E(json.dumps(DATA, indent=1))}</pre></details>
<p class="muted small">Loss curves come from <code>research/research_data.json</code> (85M: recorded {E((R85 or {}).get('meta', {}).get('recorded', 'n/a'))} by <code>research/record_research_data.py</code>; 1.8B: dashboard full report {E((R18 or {}).get('meta', {}).get('created', 'n/a'))}).</p>"""))

nav = "".join(f'<li><a href="#{sid}">{E(t)}</a></li>' for sid, t in SECTIONS)

PAGE = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TrainStream Research Dossier</title>
<style>
:root {{
  color-scheme: light;
  --page:#f9f9f7; --surface:#fcfcfb; --surface-2:#f0efec; --border:rgba(11,11,11,.12);
  --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --accent:#2a78d6;
  --ts:#2a78d6; --ts2:#1c5cab; --adamw:#eb6834; --tf:#1baf7a; --zc:#6250d6; --noise:#eda100;
  --h2d:#2a78d6; --d2h:#1baf7a; --act:#e87ba4; --g:#1baf7a; --w:#2a78d6; --m:#eda100; --ms:#6250d6;
  --good:#0ca30c; --warn:#fab219; --crit:#d03b3b;
  --zone-gpu:rgba(42,120,214,.08); --zone-link:rgba(98,80,214,.08); --zone-host:rgba(27,175,122,.08);
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --surface-2:#222220; --border:rgba(255,255,255,.1);
  --ink:#f0efec; --ink-2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835; --accent:#3987e5;
  --ts:#3987e5; --ts2:#2a78d6; --adamw:#d95926; --tf:#199e70; --zc:#9085e9; --noise:#c98500;
  --h2d:#3987e5; --d2h:#199e70; --act:#d55181; --g:#199e70; --w:#3987e5; --m:#c98500; --ms:#9085e9;
  --zone-gpu:rgba(57,135,229,.12); --zone-link:rgba(144,133,233,.12); --zone-host:rgba(25,158,112,.12);
}} }}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --surface-2:#222220; --border:rgba(255,255,255,.1);
  --ink:#f0efec; --ink-2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835; --accent:#3987e5;
  --ts:#3987e5; --ts2:#2a78d6; --adamw:#d95926; --tf:#199e70; --zc:#9085e9; --noise:#c98500;
  --h2d:#3987e5; --d2h:#199e70; --act:#d55181; --g:#199e70; --w:#3987e5; --m:#c98500; --ms:#9085e9;
  --zone-gpu:rgba(57,135,229,.12); --zone-link:rgba(144,133,233,.12); --zone-host:rgba(25,158,112,.12);
}}
* {{ box-sizing:border-box }}
html {{ scroll-padding-top:16px }}
body {{ margin:0; background:var(--page); color:var(--ink); font:16px/1.65 Charter,"Bitstream Charter","Sitka Text",Cambria,serif; }}
.layout {{ display:grid; grid-template-columns:270px minmax(0,1fr); }}
nav {{ position:sticky; top:0; height:100vh; overflow:auto; padding:22px 14px; border-right:1px solid var(--border); background:var(--surface);
  font:13.5px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif; }}
nav .brand {{ font-weight:700; font-size:15px; margin:0 8px 2px }}
nav .sub {{ color:var(--muted); font-size:12px; margin:0 8px 14px }}
nav ol {{ list-style:none; margin:0; padding:0; counter-reset:n }}
nav li a {{ display:block; padding:4px 8px; border-radius:6px; color:var(--ink-2); text-decoration:none }}
nav li a:hover, nav li a.active {{ background:var(--surface-2); color:var(--ink) }}
nav .tools {{ margin:14px 8px 0; display:flex; gap:6px; flex-wrap:wrap }}
button {{ font:600 12.5px system-ui,sans-serif; color:var(--ink); background:var(--surface-2); border:1px solid var(--border); border-radius:7px; padding:5px 10px; cursor:pointer }}
main {{ padding:40px 48px 80px; min-width:0 }}
.wrap {{ max-width:960px; margin:0 auto }}
header.title h1 {{ font-size:34px; line-height:1.15; margin:0 0 8px; font-family:system-ui,-apple-system,"Segoe UI",sans-serif; letter-spacing:-.4px }}
header.title p {{ color:var(--ink-2); margin:0 0 4px; font-family:system-ui,sans-serif; font-size:14px }}
section {{ margin:46px 0 }}
h2 {{ font:700 24px/1.25 system-ui,-apple-system,"Segoe UI",sans-serif; margin:0 0 10px }}
h3 {{ font:700 17px/1.3 system-ui,-apple-system,"Segoe UI",sans-serif; margin:26px 0 8px }}
.kicker {{ font:700 11.5px system-ui,sans-serif; letter-spacing:.9px; text-transform:uppercase; color:var(--accent); margin-bottom:6px }}
p, li, dd {{ color:var(--ink) }}
a {{ color:var(--accent) }}
code {{ font:.88em/1.4 ui-monospace,Consolas,monospace; background:var(--surface-2); padding:1px 5px; border-radius:4px }}
.muted {{ color:var(--muted) }} .small {{ font-size:13.5px }}
.abstract {{ background:var(--surface); border:1px solid var(--border); border-left:4px solid var(--accent); border-radius:10px; padding:16px 20px; text-align:justify }}
.callout {{ background:var(--surface); border:1px solid var(--border); border-left:4px solid var(--accent); border-radius:10px; padding:12px 16px; margin:14px 0 }}
.callout.warn {{ border-left-color:var(--warn) }}
.callout .t {{ font:700 14.5px system-ui,sans-serif; margin:0 0 4px }}
.callout p {{ margin:4px 0 }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:14px }}
.eq {{ font:15px/1.6 "Cambria Math",Cambria,serif; background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:10px 16px; margin:12px 0; overflow-x:auto; text-align:center }}
pre {{ font:13px/1.55 ui-monospace,Consolas,monospace; background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:12px 14px; overflow-x:auto }}
pre.json {{ max-height:420px }}
.tbl {{ overflow-x:auto; margin:14px 0; border:1px solid var(--border); border-radius:10px; background:var(--surface) }}
table {{ width:100%; border-collapse:collapse; font:13.5px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif }}
th, td {{ padding:8px 11px; text-align:left; border-bottom:1px solid var(--border); vertical-align:top }}
th {{ color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.4px; white-space:nowrap }}
tr:last-child td {{ border-bottom:0 }}
td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums }}
.pill {{ font-size:12px; padding:1px 8px; border-radius:999px; border:1px solid var(--border) }}
.pill.good {{ color:var(--good) }}
figure {{ margin:22px 0 }}
.fig {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:14px; overflow-x:auto }}
.fig svg {{ display:block; width:100%; height:auto; min-width:560px; font-family:system-ui,-apple-system,"Segoe UI",sans-serif }}
figcaption {{ font:14px/1.5 system-ui,sans-serif; color:var(--ink-2); margin-top:8px }}
figcaption .svgdl {{ margin-left:8px; padding:2px 8px; font-size:11.5px }}
.legend {{ display:flex; flex-wrap:wrap; gap:4px 16px; font:12.5px system-ui,sans-serif; color:var(--ink-2); margin:0 0 6px }}
.legend .sw {{ display:inline-block; width:11px; height:11px; border-radius:2px; margin-right:6px; vertical-align:-1px }}
dl.terms {{ display:grid; grid-template-columns:200px 1fr; border:1px solid var(--border); border-radius:10px; overflow:hidden; background:var(--surface) }}
dl.terms dt, dl.terms dd {{ margin:0; padding:9px 12px; border-bottom:1px solid var(--border) }}
dl.terms dt {{ font:600 14px system-ui,sans-serif }}
ol.claims li {{ margin:8px 0 }}
.bib {{ position:relative }} .bib .copy {{ position:absolute; right:10px; top:10px }}
details summary {{ cursor:pointer; font:600 14px system-ui,sans-serif; color:var(--ink-2) }}
/* chart vocabulary */
svg .grid {{ stroke:var(--grid); stroke-width:1 }}
svg .axis {{ stroke:var(--axis); stroke-width:1 }}
svg .ref {{ stroke:var(--ink-2); stroke-width:1.5; stroke-dasharray:5 4 }}
svg .tick {{ fill:var(--muted); font-size:12px }}
svg .axlabel {{ fill:var(--ink-2); font-size:13px }}
svg .lbl {{ fill:var(--ink-2); font-size:13px }}
svg .val {{ fill:var(--ink); font-size:13px; font-variant-numeric:tabular-nums }}
svg .inbar {{ fill:#fff; font-size:12px; font-weight:600 }}
svg .ln {{ fill:none; stroke-width:2.2; stroke-linejoin:round; stroke-linecap:round }}
svg .pt {{ stroke:var(--surface); stroke-width:2 }}
svg .zone {{ stroke:var(--border) }} svg .z-gpu {{ fill:var(--zone-gpu) }} svg .z-link {{ fill:var(--zone-link) }} svg .z-host {{ fill:var(--zone-host) }}
svg .zt {{ fill:var(--ink); font-size:15px; font-weight:700 }}
svg .box {{ fill:var(--surface); stroke:var(--axis) }} svg .box.hl {{ stroke:var(--act) }} svg .dashed {{ stroke-dasharray:6 4 }}
svg .bt {{ fill:var(--ink); font-size:13.5px; font-weight:600 }} svg .bs {{ fill:var(--ink-2); font-size:12px }}
svg .mono {{ fill:var(--ink-2); font-size:11.5px; font-family:ui-monospace,Consolas,monospace }}
svg .flow {{ stroke:var(--ink-2); stroke-width:2.2 }} svg .arrowhead {{ fill:var(--ink-2) }}
.c-ts {{ stroke:var(--ts); fill:var(--ts) }} .c-ts2 {{ fill:var(--ts2); stroke:var(--ts2) }} .c-adamw {{ stroke:var(--adamw); fill:var(--adamw) }}
.c-h2d {{ stroke:var(--h2d); fill:var(--h2d) }} .c-d2h {{ stroke:var(--d2h); fill:var(--d2h) }} .c-act {{ fill:var(--act); stroke:var(--act) }}
.c-g {{ fill:var(--g); stroke:var(--g) }} .c-w {{ fill:var(--w); stroke:var(--w) }} .c-m {{ fill:var(--m); stroke:var(--m) }} .c-ms {{ fill:var(--ms); stroke:var(--ms) }}
.c-noise {{ fill:var(--noise); stroke:var(--noise) }}
svg .ln.c-ts, svg .ln.c-adamw, svg .ln.c-h2d, svg .ln.c-d2h {{ fill:none }}
.legend .sw.c-ts {{ background:var(--ts) }} .legend .sw.c-adamw {{ background:var(--adamw) }} .legend .sw.c-h2d {{ background:var(--h2d) }}
.legend .sw.c-d2h {{ background:var(--d2h) }} .legend .sw.c-act {{ background:var(--act) }} .legend .sw.c-g {{ background:var(--g) }}
.legend .sw.c-w {{ background:var(--w) }} .legend .sw.c-m {{ background:var(--m) }} .legend .sw.c-ms {{ background:var(--ms) }}
@media (max-width:900px) {{ .layout {{ grid-template-columns:1fr }} nav {{ position:static; height:auto; border-right:0; border-bottom:1px solid var(--border) }} main {{ padding:24px 16px 60px }} dl.terms {{ grid-template-columns:1fr }} }}
@media print {{ nav {{ display:none }} .layout {{ display:block }} .svgdl, .copy {{ display:none }} section {{ break-inside:avoid-page }} }}
</style></head>
<body><div class="layout">
<nav><div class="brand">TrainStream</div><div class="sub">Research dossier · generated from recorded data</div>
<ol>{nav}</ol>
<div class="tools"><button id="theme">Toggle light/dark</button><button onclick="window.print()">Print / PDF</button></div></nav>
<main><div class="wrap">
<header class="title"><h1>TrainStream: Exact, Overlapped Offloading of Optimizer State, Gradients and Activations to Host Memory on a Single GPU</h1>
<p>Working title · research dossier for paper writing · measurements from 2026-10-16/17</p>
<p>{E(pf['gpu'])} · {E(pf['cpu'])} · {E(pf['ram'])} · {E(pf['os'])}</p></header>
{''.join(body)}
</div></main></div>
<script>
window.TRAINSTREAM_DATA = {json.dumps(DATA)};
document.getElementById("theme").onclick = () => {{
  const r = document.documentElement, dark = getComputedStyle(r).colorScheme.includes("dark");
  r.setAttribute("data-theme", dark ? "light" : "dark");
}};
document.querySelectorAll(".copy").forEach((b) => b.onclick = async () => {{
  try {{ await navigator.clipboard.writeText(document.getElementById(b.dataset.target).innerText); b.textContent = "Copied"; }}
  catch (e) {{ b.textContent = "Select and copy"; }}
  setTimeout(() => b.textContent = "Copy BibTeX", 1500);
}});
// Export a figure as a standalone SVG with the current theme's computed styles inlined.
const PROPS = ["fill", "stroke", "stroke-width", "stroke-dasharray", "stroke-linecap", "stroke-linejoin", "opacity",
  "font-size", "font-family", "font-weight", "text-anchor", "font-variant-numeric"];
document.querySelectorAll(".svgdl").forEach((b) => b.onclick = () => {{
  const src = document.getElementById(b.dataset.fig), clone = src.cloneNode(true);
  const a = src.querySelectorAll("*"), c = clone.querySelectorAll("*");
  a.forEach((el, i) => {{ const cs = getComputedStyle(el); PROPS.forEach((p) => c[i].style.setProperty(p, cs.getPropertyValue(p))); }});
  const bg = getComputedStyle(document.querySelector(".fig")).backgroundColor;
  clone.setAttribute("style", `background:${{bg}}; font-family:system-ui, Segoe UI, sans-serif`);
  const blob = new Blob([new XMLSerializer().serializeToString(clone)], {{ type: "image/svg+xml" }});
  const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = b.dataset.fig + ".svg"; link.click();
}});
// Scroll-spy
const links = [...document.querySelectorAll("nav li a")], byId = new Map(links.map((l) => [l.getAttribute("href").slice(1), l]));
const spy = new IntersectionObserver((es) => es.forEach((e) => {{
  if (!e.isIntersecting) return;
  links.forEach((l) => l.classList.remove("active"));
  const l = byId.get(e.target.id); if (l) l.classList.add("active");
}}), {{ rootMargin: "-10% 0px -80% 0px" }});
document.querySelectorAll("section[id]").forEach((s) => spy.observe(s));
</script>
</body></html>"""

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write(PAGE)
print(f"wrote {OUT} ({len(PAGE) / 1024:.0f} KiB, {FIG_N['n']} figures, {len(SECTIONS)} sections)")

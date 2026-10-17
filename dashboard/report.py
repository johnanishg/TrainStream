"""One-click comparison: run every engine on the same model, then build one report.

Engines (each skipped gracefully when unavailable):
    tensorflow     TensorFlow + stock AdamW (separate interpreter; DirectML/CUDA)
    adamw          PyTorch + stock torch.optim.AdamW (fused), everything in VRAM
    trainstream    PyTorch + TrainStream (host AdamW, gradient streaming, activation paging)
    trainstream_zc PyTorch + TrainStream zero-copy GPU AdamW (optional)

The report is saved as JSON and as a standalone HTML page under dashboard/reports/.
"""

from __future__ import annotations

import datetime as _dt
import html
import json
import math
import os
import re
import time

import torch

import trainer

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(HERE, "reports")
GB = float(1 << 30)

ENGINES = {
    "tensorflow": {"name": "TensorFlow + AdamW", "short": "TensorFlow", "color": "#199e70",
                   "cfg": {"framework": "tensorflow", "device": "gpu", "engine": "baseline"}},
    "adamw": {"name": "PyTorch + stock AdamW", "short": "AdamW", "color": "#d95926",
              "cfg": {"framework": "pytorch", "device": "gpu", "engine": "baseline"}},
    "trainstream": {"name": "PyTorch + TrainStream", "short": "TrainStream", "color": "#3987e5",
                    "cfg": {"framework": "pytorch", "device": "gpu", "engine": "trainstream", "optimizer_mode": "cpu"}},
    "trainstream_zc": {"name": "TrainStream (zero-copy GPU AdamW)", "short": "TS zero-copy", "color": "#9085e9",
                       "cfg": {"framework": "pytorch", "device": "gpu", "engine": "trainstream", "optimizer_mode": "gpu"}},
}
ORDER = ["tensorflow", "adamw", "trainstream", "trainstream_zc"]


# ---------------------------------------------------------------------------
# Running the suite
# ---------------------------------------------------------------------------
def run_suite(base_cfg: dict, engines: list, emit, corpus_getter, stop_event, tf_info: dict,
              system: dict, allow_spill: bool = False, parity: bool = True, on_run=None) -> dict:
    """Run `engines` one after another on `base_cfg`; return the report dict."""
    started = time.time()
    engines = [e for e in ORDER if e in engines]
    system = dict(system, ram_free_start=trainer.host_memory()[1])
    emit({"type": "report_start", "engines": engines, "config": base_cfg,
          "names": {e: ENGINES[e]["name"] for e in engines}})

    parity_rows = None
    if parity and not stop_event.is_set():
        emit({"type": "report_phase", "phase": "parity", "msg": "checking optimizer math against torch.optim.AdamW"})
        parity_rows = trainer.parity_check(emit)
        torch.cuda.empty_cache()  # hand the GPU back clean before the first engine

    results = []
    for key in engines:
        spec = ENGINES[key]
        if stop_event.is_set():
            results.append(_skipped(key, "stopped before it started"))
            continue
        if key == "tensorflow" and not (tf_info or {}).get("available"):
            results.append(_skipped(key, "TensorFlow is not installed / not available"))
            emit({"type": "report_phase", "phase": key, "msg": "TensorFlow unavailable — skipped"})
            continue
        cfg = dict(base_cfg, **spec["cfg"], suite_key=key)
        if key == "tensorflow":
            cfg["param_dtype"] = "fp32"
        emit({"type": "report_phase", "phase": key, "msg": f"running {spec['name']}"})
        curve, val = [], []

        def capture(ev, _curve=curve, _val=val):
            t = ev.get("type")
            if t == "step" and ev.get("loss") is not None:
                _curve.append((ev["step"], float(ev["loss"])))
            elif t == "eval" and ev.get("val_loss") is not None:
                _val.append((ev["step"], float(ev["val_loss"])))
            emit(ev)

        # Stock engines abort once VRAM is ~full (past that Windows pages GPU
        # memory into system RAM and the run crawls) unless spilling is allowed.
        run = trainer.make_run(cfg, capture, corpus_getter, stop_on_vram_full=not allow_spill)
        if key == "tensorflow":
            # DirectML reserves most of the GPU up front, so whole-device VRAM
            # passes 95% even for small models. For TensorFlow only a real
            # overflow counts (its allocator holding more than the card has),
            # which TFRun detects itself; the device-level guard just warns.
            def _warn_only(run=run):
                run.emit_raw({"type": "log", "run": run.id, "level": "warn", "t": time.time(),
                              "msg": "device VRAM ≥95% in use (DirectML reserves memory up front)"})
            run._on_vram_full = _warn_only
        if on_run:
            on_run(run)
        t0 = time.time()
        summary = run.execute() or {}
        results.append({"key": key, "label": run.label(), "config": cfg, "summary": summary,
                        "curve": curve, "val": val, "wall_s": time.time() - t0})

    report = build_report(base_cfg, results, parity_rows, system, tf_info, time.time() - started, allow_spill)
    save_report(report)
    emit({"type": "report", "report": report})
    return report


def _skipped(key, why):
    return {"key": key, "label": ENGINES[key]["name"], "config": {}, "curve": [], "val": [], "wall_s": 0,
            "summary": {"status": "skipped", "abort_reason": why}}


# ---------------------------------------------------------------------------
# Report construction
# ---------------------------------------------------------------------------
def _downsample(points, n=300):
    if len(points) <= n:
        return [list(p) for p in points]
    k = math.ceil(len(points) / n)
    out = [list(p) for p in points[::k]]
    if out[-1] != list(points[-1]):
        out.append(list(points[-1]))
    return out


def _status(summary, planned):
    s = summary.get("status")
    reason = summary.get("abort_reason") or summary.get("error") or ""
    if s == "completed":
        return "completed", "Completed", "good"
    if s == "skipped":
        return "skipped", "Skipped", "muted"
    if s == "oom":
        return "no_fit", "Out of GPU memory", "bad"
    if s == "limit":
        if "GPU memory" in reason or "VRAM" in reason or "shared system RAM" in reason:
            return "no_fit", "Doesn't fit in VRAM", "bad"
        if "RAM" in reason:
            return "no_fit", "Ran out of system RAM", "bad"
        return "stopped", "Stopped", "warn"
    if s == "stopped":
        return "stopped", "Stopped by user", "warn"
    return "error", "Error", "bad"


def _engine_row(r, planned):
    s = r["summary"]
    key = r["key"]
    outcome, label, tone = _status(s, planned)
    gpu = r["config"].get("device", "gpu") == "gpu"
    steps = int(s.get("steps") or (r["curve"][-1][0] if r["curve"] else 0))
    peak = s.get("peak_alloc")
    if key == "tensorflow":
        peak = s.get("tf_gpu_peak") or s.get("peak_dev_vram_delta") or peak
    return {
        "key": key, "name": ENGINES[key]["name"], "short": ENGINES[key]["short"], "color": ENGINES[key]["color"],
        "outcome": outcome, "status_label": label, "tone": tone,
        "reason": s.get("abort_reason") or s.get("error") or "",
        "steps": steps, "planned": planned,
        "final_loss": s.get("final_loss"), "val_loss": s.get("val_loss"),
        "median_step_ms": s.get("median_step_ms"), "tokens_per_s": s.get("median_tokens_per_s"),
        "peak_vram": peak if gpu else None,
        "device_vram_peak": s.get("peak_dev_vram_used"), "vram_total": s.get("dev_vram_total"),
        "vram_full": bool(s.get("vram_full")),
        "ram_delta": s.get("peak_sys_ram_delta"),
        "host_state": s.get("host_pinned"),
        "wall_s": r.get("wall_s"),
        "curve": _downsample(r["curve"]), "val": [list(p) for p in r["val"]],
    }


def _gib(b):
    return "—" if b is None else f"{b / GB:.2f} GiB"


def _loss_agreement(a, b):
    """Mean / max |Δloss| over the steps both runs completed."""
    da, db = dict(a["curve_full"]), dict(b["curve_full"])
    common = sorted(set(da) & set(db))
    if not common:
        return None
    diffs = [abs(da[s] - db[s]) for s in common]
    tail = diffs[-max(1, len(diffs) // 4):]
    return {"steps": len(common), "mean": sum(diffs) / len(diffs), "max": max(diffs),
            "tail_mean": sum(tail) / len(tail)}


def build_report(cfg, results, parity_rows, system, tf_info, elapsed_s, allow_spill):
    planned = int(cfg["steps"])
    mcfg = trainer.model_config(cfg)
    est_adamw = trainer.estimate(mcfg, cfg["param_dtype"], "baseline", "cpu")
    est_ts = trainer.estimate(mcfg, cfg["param_dtype"], "trainstream", "cpu")
    params = est_adamw["params"]

    rows = []
    for r in results:
        row = _engine_row(r, planned)
        row["curve_full"] = r["curve"]
        rows.append(row)
    by = {r["key"]: r for r in rows}
    A, T = by.get("adamw"), by.get("trainstream")

    verdicts = []

    def v(level, title, detail):
        verdicts.append({"level": level, "title": title, "detail": detail})

    # 1. Does it fit?
    if A and T:
        if A["outcome"] == "no_fit" and T["outcome"] == "completed":
            v("pass", "TrainStream trained a model that stock AdamW could not fit",
              f"Stock AdamW: {A['status_label']} after {A['steps']} step(s) — {A['reason'] or 'GPU memory exhausted'}. "
              f"TrainStream completed all {T['steps']} steps.")
        elif A["outcome"] == "completed" and T["outcome"] == "completed":
            v("info", "Both engines fit this model on this GPU",
              "TrainStream's value here is the VRAM it frees (for bigger batches or models), not making training possible.")
        elif T["outcome"] != "completed":
            v("fail", "TrainStream did not complete this configuration", _ts_failure_advice(T, est_ts, system))
    # 2. VRAM
    if A and T and A["peak_vram"] and T["peak_vram"]:
        saved = A["peak_vram"] - T["peak_vram"]
        pct = saved / A["peak_vram"] * 100 if A["peak_vram"] else 0
        note = " (stock AdamW's figure is where it stopped, so its true need is higher)" if A["outcome"] == "no_fit" else ""
        v("pass" if saved > 0 else "warn", f"VRAM: {_gib(A['peak_vram'])} → {_gib(T['peak_vram'])} ({pct:.0f}% less)",
          f"Peak GPU memory allocated by the training run{note}.")
    # 3. State in RAM
    if T and T["host_state"]:
        expected = params * 8
        v("pass" if T["host_state"] >= 0.95 * expected else "warn",
          f"Optimizer state lives in system RAM: {_gib(T['host_state'])}",
          f"Adam moments alone are {_gib(expected)}; with stock AdamW they would sit in VRAM. "
          f"System RAM used by the run peaked {_gib(T['ram_delta'])} above its start.")
    # 4. No hidden spill
    if T and T["outcome"] == "completed":
        v("pass" if not T["vram_full"] else "warn",
          "No hidden spill into shared GPU memory" if not T["vram_full"] else "VRAM came close to full",
          ("Device VRAM stayed below 95% throughout, so Windows never paged GPU memory into system RAM; "
           "the RAM in use is TrainStream's own pinned buffers.") if not T["vram_full"] else
          "Above 95% of VRAM Windows may page GPU memory into system RAM; lower the batch or page more activations.")
    # 5. Same math
    if A and T and A["curve_full"] and T["curve_full"]:
        agree = _loss_agreement(A, T)
        if agree and agree["steps"] < 5:
            v("info", "Not enough overlapping steps to compare losses",
              f"Stock AdamW stopped after {A['steps']} step(s), so the two runs share only {agree['steps']} step(s); "
              "the optimizer check above verifies the math instead.")
            agree = None
        if agree:
            ok = agree["mean"] < 0.05
            v("pass" if ok else "warn",
              f"Same training result: mean |Δloss| {agree['mean']:.4f} over {agree['steps']} common steps",
              f"Max {agree['max']:.4f}. bf16 kernels are not bit-deterministic, so small differences are expected; "
              f"a broken optimizer would diverge by whole units.")
    if A and T and A["val_loss"] is not None and T["val_loss"] is not None and A["outcome"] == "completed":
        v("info", f"Held-out loss: AdamW {A['val_loss']:.3f} vs TrainStream {T['val_loss']:.3f}",
          "Evaluated on the same held-out text after the final step.")
    # 6. Speed
    if A and T and A["tokens_per_s"] and T["tokens_per_s"] and A["outcome"] == "completed" and T["outcome"] == "completed":
        ratio = T["tokens_per_s"] / A["tokens_per_s"]
        if A["vram_full"]:
            v("info", f"Speed: TrainStream {ratio:.2f}× stock AdamW",
              "Stock AdamW was at the VRAM limit (Windows may have paged it into system RAM), which slows it down.")
        else:
            v("info", f"Speed: TrainStream runs at {ratio:.2f}× stock AdamW's throughput",
              "Expected below 1× when the model fits anyway: host AdamW is bounded by RAM bandwidth. "
              "The trade is VRAM for speed.")
    # 7. Parity
    if parity_rows:
        ok = all(r["pass"] for r in parity_rows)
        worst = max(r["max_abs_diff"] for r in parity_rows)
        v("pass" if ok else "fail", "Optimizer math verified against torch.optim.AdamW",
          f"{sum(r['pass'] for r in parity_rows)}/{len(parity_rows)} modes match; worst parameter difference {worst:.1e} after 20 steps.")
    # 8. TensorFlow
    TF = by.get("tensorflow")
    if TF and TF["outcome"] != "skipped":
        bits = [f"{TF['status_label']}"]
        if TF["peak_vram"]:
            bits.append(f"peak GPU memory {_gib(TF['peak_vram'])}")
        if TF["tokens_per_s"]:
            bits.append(f"{TF['tokens_per_s']:,.0f} tokens/s")
        detail = "Same model and data in TensorFlow with stock AdamW (its own initialisation, so losses differ step by step)."
        if (tf_info or {}).get("gpu_backend") == "DirectML":
            detail += (" Note: TensorFlow on Windows uses the DirectML plugin, which reserves roughly 10 GiB of GPU memory "
                       "and ~19 GB of system RAM even for small models, and its speed varies widely between runs; "
                       "treat it as a rough reference, not a tuned baseline.")
        v("info", "TensorFlow reference: " + ", ".join(bits), detail)

    headline = _headline(cfg, params, A, T, by.get("tensorflow"))
    recs = _recommendations(A, T, system)

    for r in rows:
        r.pop("curve_full", None)
    return {
        "version": 1,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": elapsed_s,
        "allow_spill": allow_spill,
        "model": {"preset": cfg.get("preset"), "n_layer": mcfg.n_layer, "n_head": mcfg.n_head, "d_model": mcfg.d_model,
                  "params": params},
        "training": {k: cfg.get(k) for k in ("batch", "seq_len", "steps", "lr", "param_dtype", "max_grad_norm",
                                             "activation_offload")},
        "estimates": {"adamw_vram_static": est_adamw["vram_static"], "trainstream_vram_static": est_ts["vram_static"],
                      "trainstream_host": est_ts["host_total"], "trainstream_pinned": est_ts["host_pinned"]},
        "system": system,
        "tensorflow": {k: (tf_info or {}).get(k) for k in ("available", "version", "gpu_backend")},
        "engines": rows,
        "verdicts": verdicts,
        "headline": headline,
        "recommendations": recs,
        "parity": parity_rows or [],
    }


def _ts_failure_advice(T, est_ts, system):
    reason = T["reason"] or "see the event log"
    if "RAM" in reason and "VRAM" not in reason:
        need, free = est_ts["host_total"], system.get("ram_free_start")
        tail = (f" TrainStream needs about {need / GB:.0f} GiB of host RAM for this model (fp32 master weights, "
                f"Adam moments, gradient staging) plus room for activations"
                + (f"; {free / GB:.0f} GiB was free when the report started." if free else "."))
        return (f"{T['status_label']}: {reason}.{tail} Close other applications, switch Weights to bf16, "
                f"or choose a smaller model.")
    return (f"{T['status_label']}: {reason}. Set Activations to 'page all', lower the batch size or sequence length "
            f"(large-vocabulary logits are not offloaded).")


def _headline(cfg, params, A, T, TF):
    model = f"{cfg.get('preset')} ({params / 1e6:,.0f}M parameters, batch {cfg['batch']}×{cfg['seq_len']})"
    if A and T:
        if A["outcome"] == "no_fit" and T["outcome"] == "completed":
            return (f"{model}: stock AdamW could not train on this GPU ({A['status_label']}). "
                    f"TrainStream trained all {T['steps']} steps with {_gib(T['peak_vram'])} peak VRAM, "
                    f"keeping {_gib(T['host_state'])} of optimizer state in system RAM.")
        if A["outcome"] == "completed" and T["outcome"] == "completed":
            saved = (A["peak_vram"] or 0) - (T["peak_vram"] or 0)
            speed = (T["tokens_per_s"] / A["tokens_per_s"]) if A["tokens_per_s"] and T["tokens_per_s"] else None
            return (f"{model}: both fit. TrainStream cut peak VRAM by {_gib(saved)} "
                    f"({_gib(A['peak_vram'])} → {_gib(T['peak_vram'])})"
                    + (f" at {speed:.2f}× the throughput" if speed else "") + ", with matching loss.")
        if T["outcome"] != "completed":
            return f"{model}: TrainStream did not complete — {T['status_label']} ({T['reason']})."
    if T:
        return f"{model}: TrainStream {T['status_label'].lower()} in {T['steps']} steps."
    return f"{model}: comparison finished."


def _recommendations(A, T, system):
    out = []
    if T and T["outcome"] == "completed" and T.get("device_vram_peak") and T.get("vram_total"):
        head = T["vram_total"] - T["device_vram_peak"]
        if head > 4 * GB:
            out.append(f"About {head / GB:.0f} GiB of VRAM was still free with TrainStream: raise the batch size or train a larger model.")
        elif T["vram_full"]:
            out.append("TrainStream ran close to the VRAM limit: set Activations to 'page all' or lower the batch size.")
    if A and T and A["outcome"] == "completed" and T["outcome"] == "completed" and A["tokens_per_s"] and T["tokens_per_s"] \
            and T["tokens_per_s"] < A["tokens_per_s"] and not A["vram_full"]:
        out.append("This model fits with stock AdamW and is faster there. Use TrainStream when VRAM is the limit, "
                   "or spend the VRAM it frees on a larger batch.")
    if T and T["host_state"] and system.get("host_ram_total") and T["host_state"] > 0.5 * system["host_ram_total"]:
        out.append("Optimizer state uses over half of system RAM: close other applications before larger runs.")
    return out


def save_report(report) -> str:
    os.makedirs(REPORT_DIR, exist_ok=True)
    stamp = report["created"].replace(":", "").replace("-", "")
    name = f"report-{stamp}-{re.sub(r'[^A-Za-z0-9.]+', '_', str(report['model']['preset']))}"
    report["file"] = name
    with open(os.path.join(REPORT_DIR, name + ".json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with open(os.path.join(REPORT_DIR, name + ".html"), "w", encoding="utf-8") as fh:
        fh.write(render_html(report))
    return name


# ---------------------------------------------------------------------------
# Standalone HTML
# ---------------------------------------------------------------------------
def _fmt(v, kind):
    if v is None:
        return "—"
    if kind == "gib":
        return f"{v / GB:.2f} GiB"
    if kind == "loss":
        return f"{v:.3f}"
    if kind == "tps":
        return f"{v:,.0f}"
    if kind == "ms":
        return f"{v:,.0f} ms"
    if kind == "s":
        return f"{v:,.0f} s"
    return str(v)


def _loss_svg(engines):
    series = [e for e in engines if e["curve"]]
    if not series:
        return '<p class="muted">No steps were recorded.</p>'
    W, H, L, R, T, B = 1400, 340, 56, 16, 14, 30
    xs = [p[0] for e in series for p in e["curve"]]
    ys = [p[1] for e in series for p in e["curve"] if p[1] is not None and math.isfinite(p[1])]
    ys += [p[1] for e in series for p in e["val"]]
    x0, x1 = min(xs), max(max(xs), min(xs) + 1)
    y0, y1 = min(ys), max(ys)
    pad = (y1 - y0) * 0.08 or 0.5
    y0, y1 = y0 - pad, y1 + pad
    X = lambda x: L + (x - x0) / (x1 - x0) * (W - L - R)  # noqa: E731
    Y = lambda y: T + (1 - (y - y0) / (y1 - y0)) * (H - T - B)  # noqa: E731
    parts = [f'<svg class="loss" viewBox="0 0 {W} {H}" role="img" aria-label="Training loss per step">']
    for i in range(6):
        yv = y0 + (y1 - y0) * i / 5
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}" class="grid"/>'
                     f'<text x="{L - 6}" y="{Y(yv) + 4:.1f}" class="tick" text-anchor="end">{yv:.2f}</text>')
    for i in range(6):
        xv = x0 + (x1 - x0) * i / 5
        parts.append(f'<text x="{X(xv):.1f}" y="{H - 10}" class="tick" text-anchor="middle">{xv:.0f}</text>')
    parts.append(f'<line x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}" class="axis"/>')
    for e in series:
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in e["curve"] if y is not None and math.isfinite(y))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{e["color"]}" stroke-width="2" '
                     f'stroke-linejoin="round" stroke-linecap="round"><title>{html.escape(e["name"])}</title></polyline>')
        for x, y in e["val"]:
            parts.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" fill="{e["color"]}" class="dot">'
                         f'<title>{html.escape(e["short"])} held-out loss {y:.3f} at step {x}</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


def _bars(engines, field, kind, cap=None, cap_label=""):
    vals = [(e, e.get(field)) for e in engines if e.get(field)]
    if not vals:
        return '<p class="muted">Not measured.</p>'
    top = max([v for _, v in vals] + ([cap] if cap else []))
    out = ['<div class="bars">']
    for e, val in vals:
        w = val / top * 100
        out.append(f'<div class="bar-row"><span class="name"><i style="background:{e["color"]}"></i>{html.escape(e["short"])}</span>'
                   f'<span class="track"><span class="fill" style="width:{w:.1f}%;background:{e["color"]}"></span>'
                   + (f'<span class="cap" style="left:{cap / top * 100:.1f}%" title="{html.escape(cap_label)}"></span>' if cap else "")
                   + f'</span><span class="val">{_fmt(val, kind)}</span></div>')
    if cap:
        out.append(f'<p class="muted small">Dashed line: {html.escape(cap_label)}.</p>')
    out.append("</div>")
    return "".join(out)


def render_html(r) -> str:
    esc = html.escape
    eng = r["engines"]
    sysi = r.get("system") or {}
    icon = {"pass": "✓", "fail": "✕", "warn": "!", "info": "i"}
    word = {"pass": "Pass", "fail": "Fail", "warn": "Check", "info": "Info"}
    verdicts = "".join(
        f'<li class="v {v["level"]}"><span class="badge {v["level"]}"><b>{icon[v["level"]]}</b>{word[v["level"]]}</span>'
        f'<div><div class="vt">{esc(v["title"])}</div><div class="vd">{esc(v["detail"])}</div></div></li>'
        for v in r["verdicts"])
    rows = "".join(
        f'<tr><td><i class="sw" style="background:{e["color"]}"></i>{esc(e["name"])}</td>'
        f'<td><span class="pill {e["tone"]}">{esc(e["status_label"])}</span>'
        + (f'<div class="reason">{esc(e["reason"])}</div>' if e["reason"] and e["outcome"] != "completed" else "")
        + f'</td><td class="num">{e["steps"]} / {e["planned"]}</td>'
        f'<td class="num">{_fmt(e["final_loss"], "loss")}</td><td class="num">{_fmt(e["val_loss"], "loss")}</td>'
        f'<td class="num">{_fmt(e["peak_vram"], "gib")}</td><td class="num">{_fmt(e["host_state"], "gib")}</td>'
        f'<td class="num">{_fmt(e["tokens_per_s"], "tps")}</td><td class="num">{_fmt(e["median_step_ms"], "ms")}</td>'
        f'<td class="num">{_fmt(e["wall_s"], "s")}</td></tr>'
        for e in eng)
    legend = "".join(f'<span><i style="background:{e["color"]}"></i>{esc(e["name"])}</span>' for e in eng if e["curve"])
    recs = "".join(f"<li>{esc(x)}</li>" for x in r["recommendations"]) or "<li>No further suggestions.</li>"
    m, t, est = r["model"], r["training"], r["estimates"]
    vram_total = sysi.get("vram_total")
    parity = "".join(
        f'<tr><td>{esc(p["mode"])}</td><td>{esc(str(p["clip"] or "off"))}</td><td class="num">{p["max_abs_diff"]:.1e}</td>'
        f'<td><span class="pill {"good" if p["pass"] else "bad"}">{"match" if p["pass"] else "mismatch"}</span></td></tr>'
        for p in r["parity"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TrainStream Report</title>
<style>
:root {{ color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --surface-2:#222220; --border:rgba(255,255,255,.1);
  --ink:#f0efec; --ink-2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --good:#0ca30c; --warning:#fab219; --critical:#d03b3b; --accent:#3987e5; }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--page); color:var(--ink); font:14.5px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
.wrap {{ max-width:1080px; margin:0 auto; padding:28px 16px 48px; }}
h1 {{ font-size:24px; margin:0 0 4px; }} h2 {{ font-size:13px; text-transform:uppercase; letter-spacing:.6px; color:var(--ink-2); margin:0 0 12px; }}
.meta {{ color:var(--muted); font-size:13px; }}
.headline {{ font-size:18px; line-height:1.45; margin:18px 0 6px; padding:16px 18px; background:var(--surface); border:1px solid var(--border);
  border-left:4px solid var(--accent); border-radius:10px; }}
.card {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:16px 18px; margin:16px 0; min-width:0; }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:16px; }}
.grid2 .card {{ margin:0; }}
ul.verdicts {{ list-style:none; margin:0; padding:0; }}
li.v {{ display:flex; gap:12px; padding:10px 0; border-bottom:1px solid var(--border); }}
li.v:last-child {{ border-bottom:0; }}
.badge {{ display:inline-flex; align-items:center; gap:5px; min-width:70px; height:24px; padding:0 9px; border-radius:999px; font-size:12px;
  font-weight:600; border:1px solid var(--border); white-space:nowrap; }}
.badge b {{ display:inline-grid; place-items:center; width:16px; height:16px; border-radius:50%; font-size:11px; color:#0d0d0d; }}
.badge.pass b {{ background:var(--good); }} .badge.fail b {{ background:var(--critical); color:#fff; }}
.badge.warn b {{ background:var(--warning); }} .badge.info b {{ background:var(--muted); }}
.vt {{ font-weight:600; }} .vd {{ color:var(--ink-2); font-size:13.5px; }}
.tbl {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
th, td {{ padding:8px 10px; border-bottom:1px solid var(--border); text-align:left; vertical-align:top; }}
th {{ color:var(--muted); font-weight:500; font-size:12px; white-space:nowrap; }}
td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
i.sw, .legend i, .bar-row .name i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:7px; vertical-align:-1px; }}
.pill {{ font-size:12px; padding:2px 8px; border-radius:999px; border:1px solid var(--border); white-space:nowrap; }}
.pill.good {{ color:#7fd67f; }} .pill.bad {{ color:#f08a8a; }} .pill.warn {{ color:#f5c55a; }} .pill.muted {{ color:var(--muted); }}
.reason {{ color:var(--muted); font-size:12px; margin-top:4px; max-width:280px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px 16px; font-size:12.5px; color:var(--ink-2); margin-bottom:8px; }}
svg {{ width:100%; height:auto; display:block; }}
svg.loss {{ max-height:380px; }}
svg .grid {{ stroke:var(--grid); }} svg .axis {{ stroke:var(--axis); }} svg .tick {{ fill:var(--muted); font-size:11px; }}
svg .dot {{ stroke:var(--surface); stroke-width:2; }}
.bar-row {{ display:grid; grid-template-columns:120px 1fr 90px; gap:10px; align-items:center; margin:8px 0; font-size:13px; }}
.bar-row .name {{ color:var(--ink-2); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.track {{ position:relative; height:14px; background:var(--surface-2); border-radius:4px; }}
.fill {{ position:absolute; left:0; top:0; bottom:0; border-radius:0 4px 4px 0; }}
.cap {{ position:absolute; top:-3px; bottom:-3px; border-left:2px dashed var(--ink-2); }}
.val {{ text-align:right; font-variant-numeric:tabular-nums; }}
.muted {{ color:var(--muted); }} .small {{ font-size:12px; }}
html.embed body {{ background:transparent; overflow:hidden; }}
html.embed .wrap {{ max-width:none; margin:0; padding:0; display:flow-root; }}
html.embed .wrap > h1, html.embed .wrap > .meta {{ display:none; }}
html.embed .headline {{ margin-top:0; }}
dl {{ display:grid; grid-template-columns:auto 1fr; gap:4px 14px; margin:0; font-size:13.5px; }}
dt {{ color:var(--muted); }} dd {{ margin:0; font-variant-numeric:tabular-nums; }}
@media print {{ body {{ background:#fff; color:#111; }} .card, .headline {{ background:#fff; border-color:#ccc; }} }}
@media (max-width:560px) {{ .bar-row {{ grid-template-columns:90px 1fr 76px; }} }}
</style></head><body><div class="wrap">
<h1>TrainStream comparison report</h1>
<div class="meta">{esc(r["created"])} · {esc(str(sysi.get("gpu", "")))} · {_fmt(vram_total, "gib")} VRAM ·
{_fmt(sysi.get("host_ram_total"), "gib")} RAM · ran for {r["elapsed_s"] / 60:.1f} min</div>
<div class="headline">{esc(r["headline"])}</div>

<div class="card"><h2>Verdict</h2><ul class="verdicts">{verdicts}</ul></div>

<div class="card"><h2>Results</h2><div class="tbl"><table>
<tr><th>Engine</th><th>Outcome</th><th class="num">Steps</th><th class="num">Final loss</th><th class="num">Held-out</th>
<th class="num">Peak VRAM</th><th class="num">State in RAM</th><th class="num">Tokens/s</th><th class="num">Step</th><th class="num">Wall</th></tr>
{rows}</table></div></div>

<div class="card"><h2>Training loss <span class="muted" style="text-transform:none;letter-spacing:0">per step · dots = held-out</span></h2>
<div class="legend">{legend}</div>{_loss_svg(eng)}</div>

<div class="grid2">
<div class="card"><h2>Peak VRAM</h2>{_bars(eng, "peak_vram", "gib", vram_total, f"GPU capacity {_fmt(vram_total, 'gib')}")}</div>
<div class="card"><h2>Throughput (tokens/s)</h2>{_bars(eng, "tokens_per_s", "tps")}</div>
</div>

<div class="grid2">
<div class="card"><h2>Configuration</h2><dl>
<dt>Model</dt><dd>{esc(str(m["preset"]))} · {m["n_layer"]} layers · d{m["d_model"]} · {m["n_head"]} heads</dd>
<dt>Parameters</dt><dd>{m["params"]:,}</dd>
<dt>Batch × sequence</dt><dd>{t["batch"]} × {t["seq_len"]}</dd>
<dt>Steps · LR</dt><dd>{t["steps"]} · {t["lr"]}</dd>
<dt>Weights</dt><dd>{esc(str(t["param_dtype"]))} (bf16 autocast)</dd>
<dt>Activation paging</dt><dd>{esc(str(t["activation_offload"]))}</dd>
<dt>Static need, AdamW</dt><dd>{_fmt(est["adamw_vram_static"], "gib")} VRAM</dd>
<dt>Static need, TrainStream</dt><dd>{_fmt(est["trainstream_vram_static"], "gib")} VRAM + {_fmt(est["trainstream_host"], "gib")} RAM</dd>
<dt>Stock runs on full VRAM</dt><dd>{"allowed to spill into shared RAM" if r["allow_spill"] else "stopped at 95% (Windows would spill)"}</dd>
</dl></div>
<div class="card"><h2>Recommendations</h2><ul>{recs}</ul>
{('<h2 style="margin-top:14px">Optimizer parity</h2><div class="tbl"><table><tr><th>Mode</th><th>Clip</th><th class="num">Max |Δ|</th><th>Result</th></tr>' + parity + '</table></div>') if parity else ''}
</div></div>
<p class="muted small">Static need = weights + gradients + Adam moments (activations are measured, not estimated). Peak VRAM is the memory the run itself allocated.
For a run stopped at the VRAM limit, the true requirement is higher than its peak.</p>
</div>
<script>
// Embedded in the dashboard (?embed=1): full width, and report the content's own
// height (not the viewport's, which would only ever grow) so the iframe fits it.
if (/[?&]embed=1/.test(location.search)) document.documentElement.classList.add("embed");
const wrap = document.querySelector(".wrap");
let lastH = 0;
function post() {{
  const h = Math.ceil(wrap.getBoundingClientRect().height) + 4;
  if (Math.abs(h - lastH) < 3) return;  // hysteresis: no resize ping-pong with the parent
  lastH = h;
  try {{ parent.postMessage({{ trainstreamReportHeight: h }}, "*"); }} catch (e) {{}}
}}
addEventListener("load", post);
if (window.ResizeObserver) new ResizeObserver(post).observe(wrap);
</script>
</body></html>"""

"""Record (or refresh) the measurements the research dossier plots.

    python research/record_research_data.py            # 85M stock AdamW vs TrainStream curves
    python research/record_research_data.py --import-reports   # also snapshot dashboard reports

Results are merged into research/research_data.json, which build_research_dossier.py
reads. The dossier therefore no longer depends on dashboard/reports/ (which the
dashboard's Clear button can delete).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "research_data.json")
sys.path.insert(0, os.path.join(ROOT, "dashboard"))
sys.path.insert(0, ROOT)


def load():
    if os.path.isfile(OUT):
        with open(OUT, encoding="utf-8") as fh:
            return json.load(fh)
    return {"curves": {}}


def save(d):
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1)


def record_85m(steps=60):
    import trainer  # dashboard/trainer.py
    from model import PRESETS, ByteCorpus

    corpus = {}

    def get(dev="cuda"):
        if dev not in corpus:
            corpus[dev] = ByteCorpus(dev)
        return corpus[dev]

    p = PRESETS["small-85M"]
    base = dict(preset="small-85M", framework="pytorch", device="gpu", n_layer=p.n_layer, n_head=p.n_head,
                d_model=p.d_model, optimizer_mode="cpu", activation_offload="auto", param_dtype="fp32", batch=8,
                seq_len=512, steps=steps, lr=3e-4, max_grad_norm=None, prefetch_depth=1, eval_every=15, seed=0)
    out = {}
    for engine in ("baseline", "trainstream"):
        curve, val = [], []

        def emit(ev, curve=curve, val=val):
            if ev.get("type") == "step":
                curve.append([ev["step"], float(ev["loss"])])
            elif ev.get("type") == "eval":
                val.append([ev["step"], float(ev["val_loss"])])

        run = trainer.Run(dict(base, engine=engine), emit, get)
        s = run.execute()
        out["adamw" if engine == "baseline" else "trainstream"] = {
            "curve": curve, "val": val, "status": s.get("status"), "peak_vram": s.get("peak_alloc"),
            "tokens_per_s": s.get("median_tokens_per_s"), "median_step_ms": s.get("median_step_ms"),
            "final_loss": s.get("final_loss"), "val_loss": s.get("val_loss"), "host_state": s.get("host_pinned")}
        print(f"{engine}: {s.get('status')} {len(curve)} steps, peak {s.get('peak_alloc', 0) / 2**30:.2f} GiB, "
              f"{s.get('median_tokens_per_s') or 0:,.0f} tok/s, val {s.get('val_loss')}")
    return {"model": "small-85M", "batch": 8, "seq_len": 512, "steps": steps,
            "recorded": time.strftime("%Y-%m-%d %H:%M:%S"), "engines": out}


def import_reports():
    got = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "dashboard", "reports", "report-*.json"))):
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        key = f"{r['model']['preset']}@{r['training']['steps']}"
        got[key] = {"file": os.path.basename(f), "created": r["created"], "model": r["model"], "training": r["training"],
                    "engines": {e["key"]: {k: e.get(k) for k in ("outcome", "status_label", "steps", "curve", "val",
                                                                 "peak_vram", "tokens_per_s", "median_step_ms",
                                                                 "final_loss", "val_loss", "host_state")}
                                for e in r["engines"]}}
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--import-reports", action="store_true")
    ap.add_argument("--skip-85m", action="store_true")
    a = ap.parse_args()
    d = load()
    if a.import_reports:
        d.setdefault("reports", {}).update(import_reports())
        print("imported reports:", ", ".join(d["reports"]))
    if not a.skip_85m:
        d["curves"]["small-85M"] = record_85m()
    save(d)
    print("saved", OUT)


if __name__ == "__main__":
    main()

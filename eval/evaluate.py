"""Run the 50-case evaluation and write results.json + report.md.

    python eval/evaluate.py --backend real                 # IndicXlit + IndicTrans2 (needs models)
    python eval/evaluate.py --backend real --polish        # also run Param1 between the two steps
    python eval/evaluate.py --backend oracle               # metric sanity check (should score perfect)
    python eval/evaluate.py --backend identity             # floor: returns the input unchanged
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import chrf, cer_counts, corpus_scores, normalize_devanagari  # noqa: E402


def load_cases(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["transliteration"], data["translation"]


def make_functions(backend: str, polish: bool, xlit_cases, trans_cases):
    """Return (xlit_fn, translate_pipeline_fn) taking a Hinglish string."""
    if backend == "identity":
        return (lambda t: t), (lambda t: t)
    if backend == "oracle":
        xl = {c["input"]: c["reference"] for c in xlit_cases}
        tr = {c["input"]: c["reference"] for c in trans_cases}
        return (lambda t: xl.get(t, t)), (lambda t: tr.get(t, t))

    from src.backends import build_backends

    xlit, polisher, translate = build_backends("real")

    def translate_pipeline(text: str) -> str:
        native = xlit(text, "hi")
        if polish:
            native = polisher(native, "hi")
        return translate(native, "hi")

    return (lambda t: xlit(t, "hi")), translate_pipeline


def per_tag(rows, key_pred, key_ref, metric):
    groups = defaultdict(list)
    for r in rows:
        for tag in r["tags"]:
            groups[tag].append(r)
    return {t: metric([g[key_pred] for g in rs], [g[key_ref] for g in rs]) for t, rs in sorted(groups.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["real", "oracle", "identity"], default="real")
    ap.add_argument("--polish", action="store_true", help="insert the Param1 step before translation")
    ap.add_argument("--cases", default=str(ROOT / "eval" / "test_cases.json"))
    ap.add_argument("--out", default=str(ROOT / "eval" / "results"))
    args = ap.parse_args()

    xlit_cases, trans_cases = load_cases(Path(args.cases))
    xlit_fn, trans_fn = make_functions(args.backend, args.polish, xlit_cases, trans_cases)

    # ---- transliteration -------------------------------------------------
    xrows = []
    for c in xlit_cases:
        pred = xlit_fn(c["input"])
        ce, cl = cer_counts(normalize_devanagari(pred), normalize_devanagari(c["reference"]))
        xrows.append({**c, "prediction": pred, "cer": ce / max(cl, 1)})
    xp, xr = [r["prediction"] for r in xrows], [r["reference"] for r in xrows]
    xlit_scores = {
        "normalized": corpus_scores(xp, xr, normalize=True),
        "raw": corpus_scores(xp, xr, normalize=False),
        "cer_by_tag": per_tag(xrows, "prediction", "reference", lambda p, r: corpus_scores(p, r)["cer"]),
    }

    # ---- hinglish -> english (end to end) -------------------------------
    trows = []
    for c in trans_cases:
        trows.append({**c, "prediction": trans_fn(c["input"])})
    tp, tr_ = [r["prediction"] for r in trows], [r["reference"] for r in trows]
    trans_scores = {
        "chrF": chrf(tp, tr_),
        "chrF_by_tag": per_tag(trows, "prediction", "reference", chrf),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(
            {"backend": args.backend, "polish": args.polish,
             "transliteration": {"scores": xlit_scores, "cases": xrows},
             "translation": {"scores": trans_scores, "cases": trows}},
            ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ---- report ---------------------------------------------------------
    n, m = xlit_scores["normalized"], xlit_scores["raw"]
    lines = [
        f"# Evaluation report (backend=`{args.backend}`, polish={args.polish})",
        "",
        f"## Transliteration: {len(xrows)} cases",
        "",
        "| Metric | Normalized | Raw |",
        "|---|---|---|",
        f"| CER (lower is better) | {n['cer']:.3f} | {m['cer']:.3f} |",
        f"| Word accuracy | {n['word_accuracy']:.3f} | {m['word_accuracy']:.3f} |",
        f"| Sentence exact match | {n['exact_match']:.3f} | {m['exact_match']:.3f} |",
        "",
        "CER by category (normalized): "
        + ", ".join(f"{k}={v:.3f}" for k, v in xlit_scores["cer_by_tag"].items()),
        "",
        "### Ten worst transliteration cases",
        "",
        "| ID | Input | Prediction | Reference | CER |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(xrows, key=lambda r: -r["cer"])[:10]:
        lines.append(f"| {r['id']} | {r['input']} | {r['prediction']} | {r['reference']} | {r['cer']:.2f} |")
    lines += [
        "",
        f"## Hinglish → English: {len(trows)} cases",
        "",
        f"Corpus chrF: **{trans_scores['chrF']:.1f}**  ",
        "By category: " + ", ".join(f"{k}={v:.1f}" for k, v in trans_scores["chrF_by_tag"].items()),
        "",
        "| ID | Input | Prediction | Reference |",
        "|---|---|---|---|",
    ]
    for r in trows:
        lines.append(f"| {r['id']} | {r['input']} | {r['prediction']} | {r['reference']} |")
    lines += [
        "",
        "## Failure analysis (fill in after a real run)",
        "",
        "Group the worst cases above by cause, e.g. loanword spelling (ऑफिस vs ओफिस), nukta/chandrabindu,",
        "gender agreement, colloquial forms, and translation errors caused by an earlier transliteration error.",
        "State how many cases fall in each group and what you would change.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:16]))
    print(f"\nWrote {out/'results.json'} and {out/'report.md'}")


if __name__ == "__main__":
    main()

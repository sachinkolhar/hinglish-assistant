"""Evaluation metrics: CER, word accuracy, exact match, chrF."""
from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[।॥.,?!;:\"'()\-]")


def normalize_devanagari(s: str) -> str:
    """Collapse spellings that are all acceptable in practice.

    Removes the nukta (ज़ -> ज), maps chandrabindu to anusvara (हूँ -> हूं) and strips
    punctuation / extra whitespace. Reported alongside raw scores so the effect is visible.
    """
    s = unicodedata.normalize("NFC", s)  # decomposes precomposed nukta letters (U+0958-095F)
    s = s.replace("़", "").replace("ँ", "ं")
    s = _PUNCT.sub("", s)
    return " ".join(s.split())


def edit_distance(a, b) -> int:
    """Levenshtein distance between two sequences (strings or token lists)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer_counts(pred: str, ref: str):
    """(char edits, reference length), spaces ignored so word-spacing is scored separately."""
    p, r = pred.replace(" ", ""), ref.replace(" ", "")
    return edit_distance(p, r), len(r)


def word_counts(pred: str, ref: str):
    """(word edits, reference word count)."""
    return edit_distance(pred.split(), ref.split()), len(ref.split())


def corpus_scores(preds, refs, normalize=True):
    norm = normalize_devanagari if normalize else (lambda x: x)
    ce = cl = we = wl = exact = 0
    for p, r in zip(preds, refs):
        p, r = norm(p), norm(r)
        a, b = cer_counts(p, r)
        c, d = word_counts(p, r)
        ce, cl, we, wl = ce + a, cl + b, we + c, wl + d
        exact += p == r
    n = max(len(refs), 1)
    return {
        "cer": ce / max(cl, 1),
        "word_accuracy": max(0.0, 1 - we / max(wl, 1)),
        "exact_match": exact / n,
    }


def chrf(preds, refs) -> float:
    """Corpus chrF via sacrebleu (0-100)."""
    from sacrebleu.metrics import CHRF

    return CHRF().corpus_score(list(preds), [list(refs)]).score

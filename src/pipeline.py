"""Hinglish -> native script -> (optional polish) -> (optional English)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .backends import build_backends


@dataclass
class Result:
    source: str
    native: str = ""
    polished: str = ""
    english: str = ""
    timings: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


class HinglishAssistant:
    def __init__(self, mode: str | None = None):
        self.xlit, self.polish, self.translate = build_backends(mode)

    def run(self, text: str, lang: str = "hi", polish: bool = False, translate: bool = False) -> Result:
        res = Result(source=text)
        text = (text or "").strip()
        if not text:
            res.warnings.append("Empty input.")
            return res

        t = time.perf_counter()
        res.native = self.xlit(text, lang)
        res.timings["transliterate_s"] = round(time.perf_counter() - t, 2)
        current = res.native

        if polish:
            t = time.perf_counter()
            try:
                res.polished = self.polish(current, lang)
                current = res.polished or current
            except Exception as exc:  # LLM is optional; never lose the transliteration
                res.warnings.append(f"Polish step failed ({type(exc).__name__}); showing transliteration.")
            res.timings["polish_s"] = round(time.perf_counter() - t, 2)

        if translate:
            t = time.perf_counter()
            try:
                res.english = self.translate(current, lang)
            except Exception as exc:
                res.warnings.append(f"Translation failed ({type(exc).__name__}: {exc}).")
            res.timings["translate_s"] = round(time.perf_counter() - t, 2)
        return res

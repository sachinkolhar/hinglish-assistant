"""Model backends for the Hinglish Smart Writing Assistant.

Each backend is a small callable wrapper around one pretrained model. Heavy
imports happen lazily so the evaluation code and unit tests run without a GPU
or any model downloads.

    IndicXlitBackend    AI4Bharat IndicXlit      Roman -> native script
    IndicTrans2Backend  AI4Bharat IndicTrans2    native <-> English
    Param1Backend       BharatGen Param1         phrasing improvement (LLM)
    MockBackend         no model                 UI development / CI
"""
from __future__ import annotations

import os
from functools import lru_cache

# ISO code -> FLORES code used by IndicTrans2
FLORES = {
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "ta": "tam_Taml",
    "mr": "mar_Deva",
    "bn": "ben_Beng",
    "en": "eng_Latn",
}

LANG_NAMES = {
    "hi": "Hindi",
    "kn": "Kannada",
    "ta": "Tamil",
    "mr": "Marathi",
    "bn": "Bengali",
}

# Checkpoint ids are configurable: confirm availability / licence / GPU needs
# on Hugging Face before running (see README "Model notes").
INDICTRANS2_INDIC_EN = os.getenv("INDICTRANS2_INDIC_EN", "ai4bharat/indictrans2-indic-en-dist-200M")
PARAM1_CHECKPOINT = os.getenv("PARAM1_CHECKPOINT", "bharatgenai/Param-1-2.9B-Instruct")


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# IndicXlit: Roman -> native script
# --------------------------------------------------------------------------
class IndicXlitBackend:
    def __init__(self, beam_width: int = 10):
        self.beam_width = beam_width

    @lru_cache(maxsize=8)
    def _engine(self, lang: str):
        from ai4bharat.transliteration import XlitEngine

        return XlitEngine(lang, beam_width=self.beam_width, src_script_type="en")

    def __call__(self, text: str, lang: str = "hi") -> str:
        if not text.strip():
            return ""
        out = self._engine(lang).translit_sentence(text)
        return out[lang] if isinstance(out, dict) else out


# --------------------------------------------------------------------------
# IndicTrans2: native script -> English
# --------------------------------------------------------------------------
class IndicTrans2Backend:
    def __init__(self, checkpoint: str = INDICTRANS2_INDIC_EN):
        self.checkpoint = checkpoint
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        from IndicTransToolkit.processor import IndicProcessor
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = _device()
        self.tok = AutoTokenizer.from_pretrained(self.checkpoint, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.checkpoint, trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        self.proc = IndicProcessor(inference=True)
        self._loaded = True

    def __call__(self, text: str, lang: str = "hi") -> str:
        if not text.strip():
            return ""
        import torch

        self._load()
        src, tgt = FLORES[lang], FLORES["en"]
        batch = self.proc.preprocess_batch([text], src_lang=src, tgt_lang=tgt)
        inputs = self.tok(
            batch, truncation=True, padding="longest", return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, num_beams=5, num_return_sequences=1, max_length=256, use_cache=False
            )
        dec = self.tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return self.proc.postprocess_batch(dec, lang=tgt)[0]


# --------------------------------------------------------------------------
# Param1: LLM phrasing improvement
# --------------------------------------------------------------------------
POLISH_PROMPT = (
    "You are a careful {language} writing assistant. Rewrite the sentence below so it "
    "reads clearly and naturally in standard {language}. Keep the meaning exactly the "
    "same, do not add information, and fix spelling and grammar only where needed. "
    "Reply with the rewritten sentence only.\n\nSentence: {text}"
)


class Param1Backend:
    def __init__(self, checkpoint: str = PARAM1_CHECKPOINT):
        self.checkpoint = checkpoint
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = _device()
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(self.checkpoint, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint, torch_dtype=dtype, trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        self._loaded = True

    def __call__(self, text: str, lang: str = "hi") -> str:
        if not text.strip():
            return ""
        import torch

        self._load()
        prompt = POLISH_PROMPT.format(language=LANG_NAMES.get(lang, "Hindi"), text=text)
        if getattr(self.tok, "chat_template", None):
            ids = self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.device)
        else:
            ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=128, do_sample=False)
        reply = self.tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
        # Guard against the model echoing the instruction or returning nothing.
        reply = reply.splitlines()[0].strip() if reply else ""
        return reply or text


# --------------------------------------------------------------------------
# Mock: lets the UI and tests run with no models
# --------------------------------------------------------------------------
class MockBackend:
    def __init__(self, tag: str):
        self.tag = tag

    def __call__(self, text: str, lang: str = "hi") -> str:
        return f"[{self.tag}] {text}" if text.strip() else ""


def build_backends(mode: str | None = None):
    """Return (transliterate, polish, translate) callables."""
    mode = mode or os.getenv("HINGLISH_BACKEND", "real")
    if mode == "mock":
        return MockBackend("xlit"), MockBackend("polish"), MockBackend("en")
    return IndicXlitBackend(), Param1Backend(), IndicTrans2Backend()

# Hinglish Smart Writing Assistant

CS3235 mini-project (Project 4). Converts Roman-script Hinglish into native script, optionally improves the phrasing, and optionally translates to English.

```
Hinglish text ──► IndicXlit ──► native script ──► (Param1 polish) ──► IndicTrans2 ──► English
                  AI4Bharat                         BharatGen          AI4Bharat
```

| Step | Model | Role |
|---|---|---|
| Transliteration | AI4Bharat IndicXlit (trained on Aksharantar) | Roman → Devanagari / Kannada / Tamil / Marathi / Bengali |
| Phrasing | BharatGen Param1 (prompted, optional) | Clean up grammar and wording |
| Translation | AI4Bharat IndicTrans2 (Indic→En, 200M distilled) | Native script → English |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py                      # opens the Gradio UI
HINGLISH_BACKEND=mock python app.py   # UI demo with no models (for development)
```

A GPU is recommended for Param1 and IndicTrans2; IndicXlit and the 200M IndicTrans2 checkpoint run on CPU.

### Model notes (confirm before running)

Checkpoint ids can be overridden with environment variables:

- `INDICTRANS2_INDIC_EN` (default `ai4bharat/indictrans2-indic-en-dist-200M`)
- `PARAM1_CHECKPOINT` (default `bharatgenai/Param-1-2.9B-Instruct`)

Check the `bharatgenai` Hugging Face organisation for the exact Param1 checkpoint name, licence, language coverage and GPU memory needs, and accept any gated-model terms. If Param1 is unavailable, leave "Improve phrasing" off; the rest of the pipeline works without it.

## Evaluation

`eval/test_cases.json` holds **50 cases**: 40 transliteration (Hindi) and 10 Hinglish→English. Cases are tagged (`plain`, `loanword`, `colloquial`, `gender`, `spelling_variant`) so errors can be broken down by category.

```bash
python eval/evaluate.py --backend real              # IndicXlit + IndicTrans2
python eval/evaluate.py --backend real --polish     # with Param1 in the chain
python -m pytest tests                              # metric and pipeline tests
```

Outputs go to `eval/results/` (`results.json`, `report.md`).

Metrics:

- **CER** and **word accuracy** and **sentence exact match** for transliteration, computed on normalized text (nukta removed, chandrabindu = anusvara, punctuation stripped) and on raw text. Normalization matters because ज़रूर / जरूर are both acceptable spellings.
- **chrF** for translation.

Sanity checks of the harness: `--backend oracle` scores perfectly, `--backend identity` gives the worst-case floor.

**Caveat:** the references were written by hand with one accepted spelling per sentence. Loanwords and informal spellings have several valid forms, so treat CER as a lower bound on quality and review the worst cases by eye. For a stronger report, add a second native-speaker reference or extend the set to Kannada/Tamil.

## Failure analysis

After a real run, open `eval/results/report.md`, take the ten worst cases, and group them by cause (loanword spelling, nukta/chandrabindu, gender agreement, colloquial forms, errors propagated into translation). Paste the table and your conclusions into the project report.

## Layout

```
app.py              Gradio frontend
src/backends.py     IndicXlit, IndicTrans2, Param1 wrappers (+ mock)
src/pipeline.py     orchestration with graceful fallback if an optional step fails
eval/               test cases, metrics, evaluation script
tests/              unit tests
```

## Known limitations

- IndicXlit transliterates every word, so English words in a mixed sentence are also converted to native script (e.g. "office" → ऑफिस). That matches the usual Hinglish-to-Devanagari convention, but not every user will want it.
- Param1 is prompted rather than fine-tuned; it may rephrase more than asked. The pipeline falls back to the transliteration if the step fails or returns nothing.
- Translation quality depends on transliteration quality, since IndicTrans2 sees the transliterated text.

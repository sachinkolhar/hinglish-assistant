"""Gradio frontend.  Run:  python app.py   (HINGLISH_BACKEND=mock python app.py for a model-free UI demo)"""
import gradio as gr

from src.backends import LANG_NAMES
from src.pipeline import HinglishAssistant

assistant = HinglishAssistant()
LANG_CHOICES = [(f"{name} ({code})", code) for code, name in LANG_NAMES.items()]

EXAMPLES = [
    ["mera naam rahul hai aur main bengaluru mein rehta hoon", "hi", False, True],
    ["kal assignment submit karna hai, bahut tension hai", "hi", True, True],
    ["mujhe chai pasand hai", "hi", False, False],
]


def process(text, lang, polish, translate):
    r = assistant.run(text, lang=lang, polish=polish, translate=translate)
    notes = " | ".join(r.warnings) if r.warnings else ""
    timing = ", ".join(f"{k.replace('_s', '')}: {v}s" for k, v in r.timings.items())
    return r.native, r.polished, r.english, (notes + ("  " if notes and timing else "") + timing).strip()


with gr.Blocks(title="Hinglish Smart Writing Assistant") as demo:
    gr.Markdown(
        "# Hinglish Smart Writing Assistant\n"
        "Type Roman-script Hinglish, get native script, optionally improve the phrasing "
        "(BharatGen Param1) and translate to English (AI4Bharat IndicTrans2)."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Hinglish input", lines=4, placeholder="mera naam rahul hai")
            lang = gr.Dropdown(LANG_CHOICES, value="hi", label="Target script / language")
            polish = gr.Checkbox(label="Improve phrasing (Param1)", value=False)
            translate = gr.Checkbox(label="Translate to English (IndicTrans2)", value=True)
            btn = gr.Button("Convert", variant="primary")
        with gr.Column():
            native = gr.Textbox(label="Native script (IndicXlit)", lines=3, show_copy_button=True)
            polished = gr.Textbox(label="Improved phrasing (Param1)", lines=3, show_copy_button=True)
            english = gr.Textbox(label="English translation", lines=3, show_copy_button=True)
            info = gr.Textbox(label="Notes / timings", lines=1)
    btn.click(process, [inp, lang, polish, translate], [native, polished, english, info])
    gr.Examples(EXAMPLES, [inp, lang, polish, translate])

if __name__ == "__main__":
    demo.launch()

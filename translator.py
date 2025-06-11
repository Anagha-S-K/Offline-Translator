import tkinter as tk
from tkinter import ttk
from transformers import MarianMTModel, MarianTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
import torch


# === EN to HI ===
def translate_en_hi(text: str) -> str:
    try:
        model_name = "Helsinki-NLP/opus-mt-en-hi"
        
        # Load model/tokenizer only once (good for GUI apps)
        if not hasattr(translate_en_hi, "tokenizer"):
            translate_en_hi.tokenizer = MarianTokenizer.from_pretrained(model_name)
            translate_en_hi.model = MarianMTModel.from_pretrained(model_name)

        tokenizer = translate_en_hi.tokenizer
        model = translate_en_hi.model

        # Preprocess and tokenize
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            translated_tokens = model.generate(
                **inputs,
                max_length=512,
                num_beams=5,
                early_stopping=True
            )

        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text

    except Exception as e:
        return f"❌ Error in translation: {str(e)}"
# === EN to KN ===
def translate_en_kn(text):
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    src_lang = "eng_Latn"
    tgt_lang = "kan_Knda"

    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt")
    bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    translated = model.generate(**inputs, forced_bos_token_id=bos_token_id)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# === On Translate Click ===
def perform_translation():
    input_text = text_input.get("1.0", tk.END).strip()
    language = language_var.get()

    if not input_text:
        result_label.config(text="❗ Please enter text to translate.")
        return

    try:
        if language == "Hindi":
            translated = translate_en_hi(input_text)
            translated_output.config(font=("Nirmala UI", 14))
        elif language == "Kannada":
            translated = translate_en_kn(input_text)
            translated_output.config(font=("Noto Sans Kannada", 14))
        else:
            translated = "⚠️ Invalid language selected."

        translated_output.delete("1.0", tk.END)
        translated_output.insert(tk.END, translated)

    except Exception as e:
        translated_output.delete("1.0", tk.END)
        translated_output.insert(tk.END, f"❌ Error: {str(e)}")

# === GUI Setup ===
root = tk.Tk()
root.title("Offline Translator 🇮🇳")
root.geometry("600x500")

tk.Label(root, text="Enter English text:", font=("Segoe UI", 12)).pack(pady=10)

text_input = tk.Text(root, height=5, width=70, font=("Segoe UI", 12))
text_input.pack()

tk.Label(root, text="Translate to:", font=("Segoe UI", 12)).pack(pady=5)

language_var = tk.StringVar(value="Hindi")
language_dropdown = ttk.Combobox(root, textvariable=language_var, values=["Hindi", "Kannada"], font=("Segoe UI", 11))
language_dropdown.pack()

tk.Button(root, text="Translate", command=perform_translation, font=("Segoe UI", 12), bg="#4CAF50", fg="white").pack(pady=10)

tk.Label(root, text="Translated Text:", font=("Segoe UI", 12)).pack(pady=5)

translated_output = tk.Text(root, height=5, width=70, font=("Nirmala UI", 14))
translated_output.pack()

result_label = tk.Label(root, text="", font=("Segoe UI", 10), fg="red")
result_label.pack()

root.mainloop()

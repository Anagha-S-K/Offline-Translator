
# 🈯 Offline Live Translator | English ↔ Hindi/Kannada

A simple and effective **offline translation tool** built with **Tkinter GUI** that translates English text into **Hindi** or **Kannada** using pretrained NLP models — perfect for users who need a lightweight and local language assistant without internet dependence for translation.

---

## 🛠 Technologies Used

- **Python**
- **Tkinter** – for GUI
- **Helsinki-NLP MarianMT** – for English ↔ Hindi translation
- **Meta NLLB-200 distilled model** – for English ↔ Kannada translation
- **Transformers (HuggingFace)** – for model and tokenizer integration
- **PyTorch** – for efficient model inference

---

## 💡 Features

- Translate **English to Hindi** using `Helsinki-NLP/opus-mt-en-hi`
- Translate **English to Kannada** using `facebook/nllb-200-distilled-600M`
- **Completely Offline**: Once the models are downloaded, no internet is required
- **Easy-to-use GUI** with language dropdown
- Unicode font compatibility for native script rendering

---

## 📷 Demo

![demo](https://github.com/your-username/your-repo-name/assets/demo-gif.gif) <!-- Optional: Include screenshot or GIF -->

---

## 🚀 Installation & Usage

1. **Clone this repository:**

```bash
git clone https://github.com/your-username/offline-translator.git
cd offline-translator

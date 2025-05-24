from langid import classify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load NLLB model
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ISO language code to NLLB language code mapping
ISO_TO_NLLB = {
    "hi": "hin_Deva",     # Hindi
    "bn": "ben_Beng",     # Bengali
    "ta": "tam_Taml",     # Tamil
    "te": "tel_Telu",     # Telugu
    "kn": "kan_Knda",     # Kannada
    "ml": "mal_Mlym",     # Malayalam
    "gu": "guj_Gujr",     # Gujarati
    "mr": "mar_Deva",     # Marathi
    "pa": "pan_Guru",     # Punjabi
    "or": "ory_Orya",     # Odia
    "as": "asm_Beng",     # Assamese
    "ur": "urd_Arab",     # Urdu
    "en": "eng_Latn"      # English
}

# Auto-detect translation function
def translate_auto(text, tgt_lang):
    detected_iso, _ = classify(text)
    src = ISO_TO_NLLB.get(detected_iso)
    print("Detected language:", detected_iso)

    if src is None:
        raise ValueError(f"Unsupported source language: {detected_iso}")

    tgt = ISO_TO_NLLB.get(tgt_lang.lower()[:2])  # e.g., 'english' -> 'en' -> 'eng_Latn'
    if tgt is None:
        raise ValueError(f"Unsupported target language: {tgt_lang}")

    tokenizer.src_lang = src
    encoded = tokenizer(text, return_tensors="pt")
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt)

    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=forced_bos_token_id
    )

    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# Main block to test translation
if __name__ == "__main__":
    text = "मैं बाजार जा रहा हूँ।"
    target_lang = "English"
    translation = translate_auto(text, target_lang)
    print("Translation:", translation)

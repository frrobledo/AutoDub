import whisper
# from transformers import MarianMTModel, MarianTokenizer
import nltk
from nltk.tokenize import sent_tokenize
import deepl
import math
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch


model = whisper.load_model('turbo')
# Mapping from ISO 639-1 codes to MBART50 language codes
language_code_mapping = {
    'ar': 'ar_AR',
    'cs': 'cs_CZ',
    'de': 'de_DE',
    'en': 'en_XX',
    'es': 'es_XX',
    'et': 'et_EE',
    'fi': 'fi_FI',
    'fr': 'fr_XX',
    'gu': 'gu_IN',
    'hi': 'hi_IN',
    'it': 'it_IT',
    'ja': 'ja_XX',
    'kk': 'kk_KZ',
    'ko': 'ko_KR',
    'lt': 'lt_LT',
    'lv': 'lv_LV',
    'my': 'my_MM',
    'ne': 'ne_NP',
    'nl': 'nl_XX',
    'ro': 'ro_RO',
    'ru': 'ru_RU',
    'si': 'si_LK',
    'tr': 'tr_TR',
    'vi': 'vi_VN',
    'zh': 'zh_CN',
    # Add more mappings as needed
}


def transcribe(audio):
    """
    Transcribe an audio file using the Whisper model.

    Parameters
    ----------
    audio : str
        The path to the audio file to be transcribed.

    Returns
    -------
    transcribed_text : str
        The transcribed text from the audio file.
    detected_language : str
        The detected language of the audio file in ISO 639-1 format.
    """
    result = model.transcribe(audio)
    
    transcribed_text = result['text']
    detected_language = result['language']
    return transcribed_text, detected_language

def translate_deepl(text, target_language_code, source_language_code=None):
    """
    Translate a given text from one language to another using the DeepL API.

    Parameters
    ----------
    text : str
        The text to be translated.
    target_language_code : str
        The ISO 639-1 language code of the target language.
    source_language_code : str, optional
        The ISO 639-1 language code of the source language. If not provided, the DeepL API will detect the source language.

    Returns
    -------
    translated_text : str
        The translated text.

    Raises
    ------
    deepl.DeepLException
        If an error occurs during translation.
    """

    auth_key = 'd48d4cbe-5d75-f41e-cf66-e74ac26e5d99:fx'
    translator = deepl.Translator(auth_key)

    # DeepL API limit per request
    max_chars_per_request = 30000

    # Split text into chunks if necessary
    if len(text) > max_chars_per_request:
        num_chunks = math.ceil(len(text) / max_chars_per_request)
        chunk_size = len(text) // num_chunks
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    else:
        chunks = [text]

    translated_chunks = []
    try:
        for chunk in chunks:
            result = translator.translate_text(
                chunk,
                source_lang=source_language_code.upper() if source_language_code else None,
                target_lang=target_language_code.upper()
            )
            translated_chunks.append(result.text)
        translated_text = ''.join(translated_chunks)
        return translated_text
    except deepl.DeepLException as e:
        print(f"An error occurred during translation: {e}")
        return None



def translate(text, source_lang_code, target_lang_code, max_chunk_length=500):
    """
    Translate a given text from one language to another using the MBart50 model.

    Parameters
    ----------
    text : str
        The text to be translated.
    source_lang_code : str
        The ISO 639-1 language code of the source language.
    target_lang_code : str
        The ISO 639-1 language code of the target language.
    max_chunk_length : int, optional
        The maximum length of each chunk of text to be translated. Chunks longer than this will be split into multiple chunks.
        Defaults to 500.

    Returns
    -------
    full_translation : str
        The translated text.
    """

    nltk.download('punkt_tab', quiet=True)


    source_lang = language_code_mapping.get(source_lang_code)
    target_lang = language_code_mapping.get(target_lang_code)

    if source_lang is None or target_lang is None:
        raise ValueError(f"Unsupported language code. Source: {source_lang_code}, Target: {target_lang_code}")

    # Load tokenizer and model
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to('cuda')

    # Split text into sentences using nltk
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_length:
            current_chunk += ' ' + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Translate each chunk
    translations = []
    for chunk in chunks:
        tokenizer.src_lang = source_lang
        encoded_input = tokenizer(chunk, return_tensors="pt", truncation=True).to('cuda')
        with torch.no_grad():
            generated_tokens = model.generate(
                **encoded_input,
                forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
                max_length=1024
            )
        translated_chunk = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]
        translations.append(translated_chunk)

    # Combine translations
    full_translation = ' '.join(translations)
    return full_translation




if __name__ == '__main__':
    audio = 'original_audios/C6RvwUsCFfw.wav'

    # transcribe audio
    transcribed_text, detected_language = transcribe(audio)
    print(f"Detected language: {detected_language}")
    print(f"Transcribed text: {transcribed_text[:500]}...")

    # target language
    target_lang_code = 'es'

    # translate text
    # transcribed_text = 'Hello, how are you?'
    # detected_language = 'en'
    # target_lang_code = 'es'
    # translated_text = translate(transcribed_text, detected_language, target_lang_code)
    translated_text = translate_deepl(transcribed_text, target_lang_code, detected_language)
    print(f"Translated text: {translated_text[:500]}...")
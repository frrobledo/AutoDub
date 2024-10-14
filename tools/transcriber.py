import whisper
from transformers import MarianMTModel, MarianTokenizer

model = whisper.load_model('turbo')

def transcribe(audio):
    result = model.transcribe(audio)
    
    transcribed_text = result['text']
    return transcribed_text


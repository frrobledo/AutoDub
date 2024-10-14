from tools.audio_synthesis import *
from tools.transcriber import *
from tools.audio_splitter_ffmpeg import *


if __name__ == '__main__':
    # Set the multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn')


    ## Load audio
    original_audio_file = 'original_audios/C6RvwUsCFfw.wav'
    original_audio_name = os.path.splitext(os.path.basename(original_audio_file))[0]

    # Transcribe audio
    # segments, detected_language = transcribe(original_audio_file)
    # print(f"Audio transcribed. Detected language: {detected_language}")

    # # Split audio file
    # process_full_audio_with_spleeter(original_audio_file)
    # print("Audio split")

    # Translate each segment
    # new_segments = []
    # i = 0
    # total_segments = len(segments)
    # for segment in segments:
    #     if len(segment['text']) > 0:
    #         translated_text = translate_deepl(segment['text'], 'es', detected_language)
    #     else:
    #         translated_text = ''
    #     new_segments.append({'id': segment['id'], 
    #                         'seek': segment['seek'],
    #                         'start': segment['start'],
    #                         'end': segment['end'],
    #                         'text': translated_text
    #                         })
    #     percentage = (i+1)/total_segments*100
    #     # Print the progress 
    #     print(f'Translation progress: {percentage:.2f}%')
    #     i += 1
    # print(f"Audio translated")

    # # save new_segments as a pickle for later loading
    # import pickle

    # with open('new_segments.pkl', 'wb') as f:
    #     pickle.dump(new_segments, f)
    # print("New segments saved")

    # Load new_segments from pickle
    import pickle
    with open('new_segments.pkl', 'rb') as f:
        new_segments = pickle.load(f)

    # Synthesize audio
    target_lang_code = "es"

    synthesized_segments_paths = synthesize_segments_with_workers(
        segments=new_segments,
        speaker_wav_path=f"output_audio/{original_audio_name}_vocals.wav",
        target_language_code=target_lang_code,
        num_workers=3,
        device='cuda'
    )
    print(f"Audio synthesized")

    # Overlay audio files together
    overlay_synthesized_speech(
        segments=new_segments,
        synthesized_segments_paths=synthesized_segments_paths,
        background_audio_path=f'output_audio/{original_audio_name}_accompaniment.wav',
        output_path=f'output_audio/{original_audio_name}-{target_lang_code}.wav'
    )

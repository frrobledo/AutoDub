from tools.audio_synthesis import *
from tools.transcriber import *
from tools.audio_splitter_ffmpeg import *
from tools.video_downloader import *
from tools.video_editing import *

if __name__ == '__main__':
    # Prepare folders
    os.makedirs('downloads', exist_ok=True)
    os.makedirs('original_audios', exist_ok=True)
    os.makedirs('output_audio', exist_ok=True)
    os.makedirs('final_output', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Set the multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn')

    # Download video
    # url = input("Enter YouTube URL: ")
    # video = video_download(url)
    video = "downloads/xSh7PuWAxXU.mp4"

    # # Extract audio
    # audio = audio_extractor(video)

    ## Load audio
    audio = 'original_audios/xSh7PuWAxXU.wav'
    original_audio_name = os.path.splitext(os.path.basename(audio))[0]

    ## Transcribe audio
    segments, detected_language = transcribe(audio)
    print(f"Audio transcribed. Detected language: {detected_language}")

    ## Split audio file
    process_full_audio_with_spleeter(audio)
    print("Audio split")

    ## Translate each segment
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

    ## save new_segments as a pickle for later loading
    # import pickle

    # with open('new_segments.pkl', 'wb') as f:
    #     pickle.dump(new_segments, f)
    # print("New segments saved")

    ## Load new_segments from pickle
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
    # overlay_synthesized_speech(
    #     segments=new_segments,
    #     synthesized_segments_paths=synthesized_segments_paths,
    #     background_audio_path=f'output_audio/{original_audio_name}_accompaniment.wav',
    #     output_path=f'final_output/{original_audio_name}-{target_lang_code}.wav'
    # )
    # print(f"Audio overlaid")


    # Video editing
    video_output = f'final_output/{original_audio_name}-{target_lang_code}.mp4'
    background_audio = f'output_audio/{original_audio_name}_accompaniment.wav'

    adjust_video_to_synthesized_audio(
        segments=new_segments,
        synthesized_segments_paths=synthesized_segments_paths,
        video_path=video,
        background_audio_path=background_audio,
        output_video_path=video_output
    )

    # Delete all content in output_audio folder
    # for filename in os.listdir('output_audio'):
    #     file_path = os.path.join('output_audio', filename)
    #     if os.path.isfile(file_path):
    #         os.remove(file_path)
    

    # Mix old video file and new audio file
    # video_path = f'downloads/{original_audio_name}.mp4'
    # audio_path = f'final_output/{original_audio_name}-{target_lang_code}.wav'
    # replace_audio_in_video(video, audio_path, f'final_output/{original_audio_name}-{target_lang_code}.mp4')
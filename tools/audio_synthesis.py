from TTS.api import TTS
import nltk
import re


def synthesize_speech(text, speaker_wav_path, target_language_code):
    """
    Synthesize speech from a given text using the given speaker's voice and target language.

    Parameters
    ----------
    text : str
        The text to be synthesized.
    speaker_wav_path : str
        The path to the speaker's voice audio file.
    target_language_code : str
        The ISO 639-1 language code of the target language.

    Returns
    -------
    None
    """
    from TTS.api import TTS
    import torch
    import os

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Init TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    # Define output path
    # Original speaker wav file name
    speaker_wav_name = os.path.splitext(os.path.basename(speaker_wav_path))[0]
    # Output wav file name
    output_wav_name = f"output_audio/{speaker_wav_name}_{target_language_code}.wav"

    # Run TTS
    # Text to speech list of amplitude values as output
    # wav = tts.tts(text=text, speaker_wav=speaker_wav_path, language=target_language_code)
    # Text to speech to a file
    tts.tts_to_file(text=text, speaker_wav=speaker_wav_path, language=target_language_code, file_path=output_wav_name)

    print(f"Synthesized audio saved to {output_wav_name}")

def synthesize_speech_segment(args):
    text, speaker_wav_path, target_language_code, segment_index = args

    from TTS.api import TTS
    import torch
    import os

    # Initialize TTS once outside this function if possible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using {device} device')
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    # Output file for the segment
    original_speaker_wav_name = os.path.splitext(os.path.basename(speaker_wav_path))[0]
    output_wav_name = f"output_audio/{original_speaker_wav_name}_segment_{segment_index}.wav"

    # Synthesize speech
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav_path,
        language=target_language_code,
        file_path=output_wav_name
    )
    print(f"Synthesized segment {segment_index+1}")
    return output_wav_name


def adjust_speech_duration(synthesized_wav_path, target_duration):
    import librosa
    import soundfile as sf

    y, sr = librosa.load(synthesized_wav_path)
    original_duration = librosa.get_duration(y=y, sr=sr)

    # Calculate the time-stretch rate
    rate = original_duration / target_duration if target_duration > 0 else 1.0

    # Time-stretch the audio to match the target duration
    y_stretched = librosa.effects.time_stretch(y, rate)

    # Save the adjusted audio
    sf.write(synthesized_wav_path, y_stretched, sr)


def synthesize_segments_in_parallel(segments, speaker_wav_path, target_language_code):
    from multiprocessing import Pool, cpu_count
    from functools import partial

    args_list = []
    for index, segment in enumerate(segments):
        text = segment['text']
        args_list.append((text, speaker_wav_path, target_language_code, index))

    # Use a partial function if needed
    # synthesize_partial = partial(synthesize_speech_segment, speaker_wav_path=speaker_wav_path, target_language_code=target_language_code)

    # Set up multiprocessing pool
    num_workers = 3
    # num_workers = cpu_count()
    with Pool(processes=num_workers) as pool:
        synthesized_segments_paths = pool.map(synthesize_speech_segment, args_list)

    return synthesized_segments_paths



def overlay_synthesized_speech(segments, synthesized_segments_paths, background_audio_path, output_path):
    from pydub import AudioSegment
    import numpy as np
    import librosa
    import soundfile as sf
    import os
    import tempfile

    # Load the background audio
    background_audio = AudioSegment.from_file(background_audio_path)

    # Ensure the background audio is in the correct format
    background_audio = background_audio.set_channels(1)
    background_audio = background_audio.set_frame_rate(22050)

    # Create an empty audio segment for the output
    output_audio = background_audio

    # Overlay each synthesized segment
    for segment, synthesized_path in zip(segments, synthesized_segments_paths):
        start_time_ms = int(segment['start'] * 1000)
        end_time_ms = int(segment['end'] * 1000)

        # Load the synthesized audio
        synthesized_audio = AudioSegment.from_file(synthesized_path)

        # # Load synthesized audio using librosa
        # y_synth, sr_synth = librosa.load(synthesized_path, sr=None)

        # # Calculate the target duration in seconds
        # target_duration_sec = (end_time_ms - start_time_ms) / 1000.0

        # # Get the original duration of the synthesized audio
        # original_duration_sec = librosa.get_duration(y=y_synth, sr=sr_synth)

        # # Calculate the time-stretch rate
        # if original_duration_sec > 0 and target_duration_sec > 0:
        #     rate = original_duration_sec / target_duration_sec
        # else:
        #     rate = 1.0  # Avoid division by zero

        # # Hard code the rate as 1
        # rate = 1.0

        # # Time-stretch the synthesized audio to match the target duration
        # y_stretched = librosa.effects.time_stretch(y=y_synth, rate=rate)

        # # Ensure the stretched audio has the same sample rate as the background audio
        # y_stretched = librosa.resample(
        #     y=y_stretched,
        #     orig_sr=sr_synth,
        #     target_sr=background_audio.frame_rate
        # )
        # sr_stretched = background_audio.frame_rate

        # # Ensure y_stretched is in the range -1.0 to 1.0
        # y_stretched = np.clip(y_stretched, -1.0, 1.0)

        # # Scale to int16 range and convert to int16
        # y_stretched_int16 = (y_stretched * 32767).astype(np.int16)

        # # Convert numpy array back to AudioSegment
        # synthesized_audio = AudioSegment(
        #     y_stretched_int16.tobytes(),
        #     frame_rate=sr_stretched,
        #     sample_width=2,  # 16-bit audio
        #     channels=1
        # )

        # Adjust the synthesized audio to match the background audio format
        synthesized_audio = synthesized_audio.set_frame_rate(background_audio.frame_rate)
        synthesized_audio = synthesized_audio.set_channels(background_audio.channels)
        synthesized_audio = synthesized_audio.set_sample_width(background_audio.sample_width)

        # Overlay the synthesized audio onto the background audio at the correct position
        output_audio = output_audio.overlay(synthesized_audio, position=start_time_ms)

    # Export the final audio
    output_audio.export(output_path, format="wav")



import multiprocessing
import os
from TTS.api import TTS

class TTSWorker(multiprocessing.Process):
    def __init__(self, task_queue, result_queue, device):
        super(TTSWorker, self).__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.device = device

    def run(self):
        # Ensure NLTK punkt tokenizer is downloaded
        nltk.download('punkt', quiet=True)

        # Initialize the TTS model once
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

        while True:
            task = self.task_queue.get()
            if task is None:
                # Poison pill means shutdown
                self.task_queue.task_done()
                break
            text, speaker_wav_path, target_language_code, segment_index, target_duration = task
            try:
                output_wav_name = f"output_audio/segment_{segment_index}.wav"

                # Estimate the default duration of the synthesized speech
                estimated_duration = self.estimate_speech_duration(text)

                # Calculate the speed factor
                speed = estimated_duration / target_duration

                # Limit the speed factor to a reasonable range
                speed = max(min(speed, 1.5), 0.75)

                # Synthesize speech with adjusted speed
                tts.tts_to_file(
                    text=text,
                    speaker_wav=speaker_wav_path,
                    language=target_language_code,
                    file_path=output_wav_name,
                    speed=speed  # Adjusted speech rate
                )
                self.result_queue.put((segment_index, output_wav_name))
                print(f"Synthesized segment {segment_index+1} with speed factor {speed:.2f}")
            except Exception as e:
                print(f"Error in processing segment {segment_index}: {e}")
            self.task_queue.task_done()

    def estimate_speech_duration(self, text):
        """
        Estimate the duration of the speech for the given text based on average speech rate.

        Parameters
        ----------
        text : str
            The text to estimate the duration for.

        Returns
        -------
        float
            The estimated duration in seconds.
        """
        # Average speech rate: 150 words per minute (approx. 2.5 words per second)
        words = self.tokenize_text(text)
        num_words = len(words)
        average_words_per_second = 2.5  # You can adjust this value based on experimentation
        estimated_duration = num_words / average_words_per_second
        return estimated_duration

    def tokenize_text(self, text):
        """
        Tokenize the text into words.

        Parameters
        ----------
        text : str
            The text to tokenize.

        Returns
        -------
        list of str
            A list of words.
        """
        # Remove punctuation and split into words
        words = nltk.word_tokenize(text)
        words = [word for word in words if re.match(r'\w+', word)]
        return words


def synthesize_segments_with_workers(segments, speaker_wav_path, target_language_code, device="cpu", num_workers=2):
    import multiprocessing
    import torch

    print(f"Using {device} device")

    # Create queues
    task_queue = multiprocessing.JoinableQueue()
    result_queue = multiprocessing.Queue()

    # Start worker processes
    workers = []
    for i in range(num_workers):
        worker = TTSWorker(task_queue, result_queue, device)
        worker.start()
        workers.append(worker)

    # Enqueue tasks with target duration
    for index, segment in enumerate(segments):
        text = segment['text']
        target_duration = segment['end'] - segment['start']
        task_queue.put((text, speaker_wav_path, target_language_code, index, target_duration))

    # Add a poison pill for each worker to signal shutdown
    for _ in range(num_workers):
        task_queue.put(None)

    # Wait for all tasks to be processed
    task_queue.join()

    # Collect results
    synthesized_segments = {}
    while not result_queue.empty():
        segment_index, output_wav_name = result_queue.get()
        synthesized_segments[segment_index] = output_wav_name

    # Ensure all workers have finished
    for worker in workers:
        worker.join()
    print('All workers have finished')

    # Return paths in order
    print(f"Synthesized {len(synthesized_segments)} segments")
    synthesized_segments_paths = [synthesized_segments[i] for i in sorted(synthesized_segments.keys())]

    print(f"Synthesized audio saved to output_audio")
    return synthesized_segments_paths

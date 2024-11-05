from TTS.api import TTS
import nltk
import re
from tools.logger import log_segment_duration


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

        # Trim silence from the synthesized audio
        synthesized_audio = trim_silence(synthesized_audio, silence_thresh=-16.0)

        # Adjust speed of the synthesized audio
        synthesized_audio = adjust_speed_to_fit_duration(
            synthesized_audio, 
            start_time_ms=start_time_ms, 
            end_time_ms=end_time_ms, 
            segment_id=segment['id']
        )

        # Log duration of the synthesized audio
        log_segment_duration(
            synthesized_audio, 
            start_time_ms=start_time_ms, 
            end_time_ms=end_time_ms, 
            segment_id=segment['id'], 
            text=segment['text']
        )

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
                speed = max(min(speed, 3.0), 0.75)

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

def adjust_speed_to_fit_duration(synthesized_audio, start_time_ms, end_time_ms, segment_id):
    """
    Adjusts the speed of the synthesized audio so that it fits within the original segment duration.

    Parameters
    ----------
    synthesized_audio : AudioSegment
        The synthesized audio segment to adjust.
    start_time_ms : int
        The start time of the original segment in milliseconds.
    end_time_ms : int
        The end time of the original segment in milliseconds.
    segment_id : int or str
        Identifier for the segment (used for temporary file naming).

    Returns
    -------
    adjusted_synthesized_audio : AudioSegment
        The synthesized audio adjusted to fit the original segment duration.
    """
    import subprocess
    import os
    import tempfile
    from pydub import AudioSegment

    # Calculate the target duration in milliseconds
    target_duration_ms = end_time_ms - start_time_ms

    # Calculate the current duration of the synthesized audio in milliseconds
    current_duration_ms = len(synthesized_audio)

    # If the durations are the same, no adjustment is needed
    if current_duration_ms == target_duration_ms or current_duration_ms == 0:
        return synthesized_audio

    # Calculate the time-stretch ratio
    if current_duration_ms != 0 and target_duration_ms != 0:
        time_stretch_ratio = current_duration_ms / target_duration_ms
    else:
        time_stretch_ratio = 1.0
    # time_stretch_ratio = target_duration_ms / current_duration_ms

    # Save synthesized_audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_in_file:
        temp_in_filename = temp_in_file.name
        synthesized_audio.export(temp_in_filename, format='wav')

    # Create a temporary output file
    temp_out_filename = temp_in_filename.replace('.wav', f'_adjusted_{segment_id}.wav')

    # Build the FFmpeg command using the rubberband filter
    command = [
        'ffmpeg',
        '-y',  # Overwrite output file
        '-i', temp_in_filename,
        '-filter:a', f"rubberband=tempo={time_stretch_ratio}",
        temp_out_filename
    ]

    # Run the command
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Check if FFmpeg command was successful
    if result.returncode != 0:
        print("FFmpeg error:")
        print(result.stderr.decode('utf-8'))
        raise RuntimeError("Failed to adjust audio speed using FFmpeg.")

    # Load the adjusted audio
    adjusted_synthesized_audio = AudioSegment.from_file(temp_out_filename, format='wav')

    # Clean up temporary files
    os.remove(temp_in_filename)
    os.remove(temp_out_filename)

    # Double-check the duration
    adjusted_duration_ms = len(adjusted_synthesized_audio)
    duration_difference_ms = abs(adjusted_duration_ms - target_duration_ms)

    # If there's a minor discrepancy, adjust by trimming or padding
    # if duration_difference_ms > 10:  # Allow a small tolerance
    #     if adjusted_duration_ms > target_duration_ms:
    #         # Trim the audio
    #         adjusted_synthesized_audio = adjusted_synthesized_audio[:target_duration_ms]
    #     else:
    #         # Pad the audio with silence
    #         silence_duration_ms = target_duration_ms - adjusted_duration_ms
    #         silence = AudioSegment.silent(duration=silence_duration_ms)
    #         adjusted_synthesized_audio += silence

    return adjusted_synthesized_audio

# def adjust_speed_to_fit_duration(synthesized_audio, start_time_ms, end_time_ms, segment_id):
#     import numpy as np
#     import librosa
#     from pydub import AudioSegment

#     # Calculate target and current durations in seconds
#     target_duration_sec = (end_time_ms - start_time_ms) / 1000.0
#     current_duration_sec = len(synthesized_audio) / 1000.0

#     if current_duration_sec == 0 or current_duration_sec == target_duration_sec:
#         return synthesized_audio

#     # Calculate the time-stretch rate
#     if current_duration_sec != 0 and target_duration_sec != 0:
#         rate = current_duration_sec / target_duration_sec
#     else:
#         rate = 1.0

#     # Convert AudioSegment to numpy array
#     y = np.array(synthesized_audio.get_array_of_samples()).astype(np.float32)
#     sr = synthesized_audio.frame_rate

#     # Time-stretch using librosa
#     y_stretched = librosa.effects.time_stretch(y, rate=rate)

#     # Convert back to AudioSegment
#     adjusted_synthesized_audio = AudioSegment(
#         y_stretched.tobytes(),
#         frame_rate=sr,
#         sample_width=synthesized_audio.sample_width,
#         channels=synthesized_audio.channels
#     )

#     return adjusted_synthesized_audio




def trim_silence(audio_segment, silence_thresh=-50.0, chunk_size=10):
    from pydub import AudioSegment, silence
    trimmed_audio = silence.detect_nonsilent(
        audio_segment,
        min_silence_len=chunk_size,
        silence_thresh=silence_thresh
    )
    if trimmed_audio:
        start_trim = trimmed_audio[0][0]
        end_trim = trimmed_audio[-1][1]
        return audio_segment[start_trim:end_trim]
    else:
        return audio_segment

def adjust_background_music(background_audio, translated_audio_segments):
    """
    Adjust the background music based on the translated audio segments.

    Parameters
    ----------
    background_audio : AudioSegment
        The original background audio.
    translated_audio_segments : list of AudioSegment
        The list of translated audio segments.

    Returns
    -------
    adjusted_background_audio : AudioSegment
        The adjusted background audio.
    """
    adjusted_background_audio = background_audio

    for segment in translated_audio_segments:
        # Adjust volume, tempo, or other parameters based on the segment
        # Example: Reduce background music volume during speech segments
        adjusted_background_audio = adjusted_background_audio.overlay(segment, gain_during_overlay=-10)

    return adjusted_background_audio

def apply_seamless_transitions(audio_segments):
    """
    Apply seamless transitions between different audio segments.

    Parameters
    ----------
    audio_segments : list of AudioSegment
        The list of audio segments.

    Returns
    -------
    seamless_audio : AudioSegment
        The audio with seamless transitions.
    """
    seamless_audio = AudioSegment.empty()

    for segment in audio_segments:
        # Apply fade-in and fade-out effects for seamless transitions
        segment = segment.fade_in(100).fade_out(100)
        seamless_audio += segment

    return seamless_audio

def reduce_background_noise(audio_segment):
    """
    Reduce background noise in the given audio segment.

    Parameters
    ----------
    audio_segment : AudioSegment
        The audio segment to reduce background noise.

    Returns
    -------
    denoised_audio : AudioSegment
        The audio segment with reduced background noise.
    """
    import noisereduce as nr
    import numpy as np

    # Convert AudioSegment to numpy array
    y = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    sr = audio_segment.frame_rate

    # Apply noise reduction
    denoised_y = nr.reduce_noise(y=y, sr=sr)

    # Convert back to AudioSegment
    denoised_audio = AudioSegment(
        denoised_y.tobytes(),
        frame_rate=sr,
        sample_width=audio_segment.sample_width,
        channels=audio_segment.channels
    )

    return denoised_audio

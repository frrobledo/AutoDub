
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
    """
    Overlay the synthesized speech segments onto the background audio.

    Parameters
    ----------
    segments : list
        A list of dictionaries containing the start and end times of each segment.
    synthesized_segments_paths : list
        A list of paths to the synthesized audio files, one for each segment.
    background_audio_path : str
        The path to the background audio file.
    output_path : str
        The path to the output audio file.

    Returns
    -------
    None
    """
    from pydub import AudioSegment

    # Load the background audio
    background_audio = AudioSegment.from_file(background_audio_path)

    # Create an empty audio segment for the output
    output_audio = background_audio

    # Overlay each synthesized segment
    for segment, synthesized_path in zip(segments, synthesized_segments_paths):
        start_time_ms = int(segment['start'] * 1000)
        end_time_ms = int(segment['end'] * 1000)

        # Load synthesized audio
        synthesized_audio = AudioSegment.from_file(synthesized_path)

        # Adjust the synthesized audio to match the duration of the original segment
        segment_duration_ms = end_time_ms - start_time_ms
        synthesized_audio = synthesized_audio.set_frame_rate(background_audio.frame_rate)
        synthesized_audio = synthesized_audio.set_channels(background_audio.channels)
        synthesized_audio = synthesized_audio.set_sample_width(background_audio.sample_width)
        synthesized_audio = synthesized_audio[:segment_duration_ms]

        # Overlay the synthesized audio onto the background audio
        output_audio = output_audio.overlay(synthesized_audio, position=start_time_ms)

    # Export the final audio
    output_audio.export(output_path, format="wav")

import multiprocessing
import os
from TTS.api import TTS

class TTSWorker(multiprocessing.Process):
    def __init__(self, task_queue, result_queue, device):
        """
        Initialize the TTSWorker

        Parameters
        ----------
        task_queue : multiprocessing.JoinableQueue
            The queue to get tasks from
        result_queue : multiprocessing.Queue
            The queue to put results in
        device : str
            The device to use ('cpu' or 'cuda')
        """
        super(TTSWorker, self).__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.device = device

    def run(self):
        # Initialize the TTS model once
        """
        Process tasks from the task queue and put results in the result queue.

        This method runs an infinite loop, getting tasks from the task queue, processing them
        using the TTS model, and putting the results in the result queue. If a task is None,
        it is considered a poison pill and the loop breaks, shutting down the worker.

        Tasks are expected to be tuples of (text, speaker_wav_path, target_language_code, segment_index).
        Results are expected to be tuples of (segment_index, output_wav_name).

        If an error occurs while processing a task, it is printed to the console.
        """
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        while True:
            task = self.task_queue.get()
            if task is None:
                # Poison pill means shutdown
                self.task_queue.task_done()
                break
            text, speaker_wav_path, target_language_code, segment_index = task
            try:
                output_wav_name = f"output_audio/segment_{segment_index}.wav"
                tts.tts_to_file(
                    text=text,
                    speaker_wav=speaker_wav_path,
                    language=target_language_code,
                    file_path=output_wav_name
                )
                self.result_queue.put((segment_index, output_wav_name))
                print(f"Synthesized segment {segment_index+1}")
            except Exception as e:
                print(f"Error in processing segment {segment_index}: {e}")
            self.task_queue.task_done()

def synthesize_segments_with_workers(segments, speaker_wav_path, target_language_code, device = "cpu", num_workers=2):
    """
    Synthesize speech segments using multiple worker processes.

    This function synthesizes the given segments of text using the given speaker's voice and target language,
    using multiple worker processes to speed up the process. The segments are processed in parallel using as many
    worker processes as specified by num_workers, and the results are collected and returned in a list.

    Parameters
    ----------
    segments : list of dict
        A list of dictionaries, where each dictionary represents a segment of text to synthesize and has at least a 'text' key.
    speaker_wav_path : str
        The path to the speaker's voice audio file.
    target_language_code : str
        The ISO 639-1 language code of the target language.
    num_workers : int, optional
        The number of worker processes to use. Defaults to 2.

    Returns
    -------
    list of str
        A list of paths to the synthesized audio files, in the same order as the input segments.
    """
    import multiprocessing
    import torch

    # device = "cpu"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
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

    # Enqueue tasks
    for index, segment in enumerate(segments):
        text = segment['text']
        task_queue.put((text, speaker_wav_path, target_language_code, index))

    # Add a poison pill for each worker to signal shutdown
    for _ in range(num_workers):
        task_queue.put(None)

    # Wait for all tasks to be processed
    task_queue.join()

    # Collect results
    synthesized_segments = {}
    while not result_queue.empty():
        segment_index, output_wav_name = result_queue.get()
        print(f"Synthesized segment {segment_index+1}")
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

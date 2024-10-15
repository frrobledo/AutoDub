import os

def prepare_full_segments_list(segments, total_duration):
    """
    Prepare a list that includes both speech segments and non-speech intervals.

    Parameters
    ----------
    segments : list of dict
        The list of speech segments with 'start', 'end', and 'text' keys.
    total_duration : float
        The total duration of the video in seconds.

    Returns
    -------
    full_segments_list : list of dict
        A list of all segments (speech and non-speech intervals), sorted by start time.
    """
    full_segments_list = []
    last_end = 0.0

    for i, segment in enumerate(segments):
        start = segment['start']
        end = segment['end']

        # Non-speech interval before the current speech segment
        if start > last_end:
            non_speech_segment = {
                'type': 'non-speech',
                'start': last_end,
                'end': start
            }
            full_segments_list.append(non_speech_segment)

        # Current speech segment
        speech_segment = {
            'type': 'speech',
            'start': start,
            'end': end,
            'index': i  # Keep track of the index for synthesized audio
        }
        full_segments_list.append(speech_segment)

        last_end = end

    # Non-speech interval after the last speech segment
    if last_end < total_duration:
        non_speech_segment = {
            'type': 'non-speech',
            'start': last_end,
            'end': total_duration
        }
        full_segments_list.append(non_speech_segment)

    return full_segments_list


def process_segments(full_segments_list, synthesized_segments_paths, video_path, background_audio_path, output_dir):
    """
    Process each segment: adjust video speed and replace audio for speech segments,
    and keep original speed for non-speech intervals.

    Parameters
    ----------
    full_segments_list : list of dict
        The list of all segments (speech and non-speech intervals).
    synthesized_segments_paths : list of str
        The list of paths to the synthesized audio files for speech segments.
    video_path : str
        The path to the original video file.
    background_audio_path : str
        The path to the background audio file (e.g., accompaniment audio).
    output_dir : str
        The directory to save the processed video segments.

    Returns
    -------
    segment_video_paths : list of str
        The list of paths to the processed video segment files.
    """
    import os
    import subprocess
    from pydub import AudioSegment

    segment_video_paths = []

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for segment in full_segments_list:
        start = segment['start']
        end = segment['end']
        duration = end - start
        segment_type = segment['type']
        segment_index = full_segments_list.index(segment)

        if segment_type == 'speech':
            # Speech segment
            speech_index = segment['index']
            synthesized_audio_path = synthesized_segments_paths[speech_index]

            # Get duration of synthesized audio
            synthesized_audio = AudioSegment.from_file(synthesized_audio_path)
            synthesized_duration_ms = len(synthesized_audio)
            synthesized_duration = synthesized_duration_ms / 1000.0  # Convert to seconds

            # Calculate speed ratio
            if duration > 0:
                speed_ratio = duration / synthesized_duration
            else:
                speed_ratio = 1.0  # Default to 1.0 if duration is zero

            # Adjust video speed and replace audio
            output_segment_path = os.path.join(output_dir, f'segment_{segment_index:04d}.mp4')
            adjust_video_speed_and_replace_audio(
                video_path,
                background_audio_path,
                start,
                duration,
                speed_ratio,
                synthesized_audio_path,
                output_segment_path
            )
            segment_video_paths.append(output_segment_path)

        elif segment_type == 'non-speech':
            # Non-speech interval
            output_segment_path = os.path.join(output_dir, f'segment_{segment_index:04d}.mp4')
            extract_video_segment(
                video_path,
                background_audio_path,
                start,
                duration,
                output_segment_path
            )
            segment_video_paths.append(output_segment_path)

    return segment_video_paths

def adjust_video_speed_and_replace_audio(video_path, background_audio_path, start_time, duration, speed_ratio, synthesized_audio_path, output_path):
    """
    Adjust the video speed and replace audio for a speech segment.

    Parameters
    ----------
    video_path : str
        The path to the original video file.
    background_audio_path : str
        The path to the background audio file.
    start_time : float
        The start time of the segment in seconds.
    duration : float
        The duration of the segment in seconds.
    speed_ratio : float
        The ratio of original duration to synthesized audio duration.
    synthesized_audio_path : str
        The path to the synthesized audio file.
    output_path : str
        The path to save the processed video segment.

    Returns
    -------
    None
    """
    import subprocess
    import os

    # First, extract the video segment
    temp_video_segment = f"{output_path}_temp_video.mp4"
    command = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-ss', str(start_time),
        '-t', str(duration),
        '-c', 'copy',
        temp_video_segment
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Extract the background audio segment
    temp_audio_segment = f"{output_path}_temp_audio.wav"
    command = [
        'ffmpeg',
        '-y',
        '-i', background_audio_path,
        '-ss', str(start_time),
        '-t', str(duration),
        '-c', 'copy',
        temp_audio_segment
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Adjust the speed of the video
    adjusted_video = f"{output_path}_adjusted_video.mp4"
    command = [
        'ffmpeg',
        '-y',
        '-i', temp_video_segment,
        '-filter:v', f"setpts={1/speed_ratio}*PTS",
        '-an',
        adjusted_video
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Adjust the speed of the background audio
    adjusted_audio = f"{output_path}_adjusted_audio.wav"

    atempo_filters = generate_atempo_filters(speed_ratio)

    command = [
        'ffmpeg',
        '-y',
        '-i', temp_audio_segment,
        '-filter:a', atempo_filters,
        adjusted_audio
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Combine the adjusted video and synthesized audio
    command = [
        'ffmpeg',
        '-y',
        '-i', adjusted_video,
        '-i', synthesized_audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        output_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Clean up temporary files
    os.remove(temp_video_segment)
    os.remove(temp_audio_segment)
    os.remove(adjusted_video)
    os.remove(adjusted_audio)

def generate_atempo_filters(speed_ratio):
    """
    Generate the atempo filter string for FFmpeg based on the speed ratio.

    Parameters
    ----------
    speed_ratio : float
        The desired speed ratio.

    Returns
    -------
    atempo_filters : str
        The atempo filter string for FFmpeg.
    """
    if speed_ratio <= 0:
        speed_ratio = 0.01  # Avoid division by zero or negative speeds

    atempo_limit = 2.0
    filters = []
    remaining_ratio = speed_ratio

    while remaining_ratio > atempo_limit:
        filters.append(f"atempo={atempo_limit}")
        remaining_ratio /= atempo_limit

    while remaining_ratio < 0.5:
        filters.append(f"atempo=0.5")
        remaining_ratio /= 0.5

    filters.append(f"atempo={remaining_ratio}")
    atempo_filters = ",".join(filters)
    return atempo_filters


def extract_video_segment(video_path, background_audio_path, start_time, duration, output_path):
    """
    Extract the video segment without adjusting the speed.

    Parameters
    ----------
    video_path : str
        The path to the original video file.
    background_audio_path : str
        The path to the background audio file.
    start_time : float
        The start time of the segment in seconds.
    duration : float
        The duration of the segment in seconds.
    output_path : str
        The path to save the processed video segment.

    Returns
    -------
    None
    """
    import subprocess

    command = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-i', background_audio_path,
        '-ss', str(start_time),
        '-t', str(duration),
        '-map', '0:v',
        '-map', '1:a',
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        output_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def create_concat_file(segment_video_paths, concat_file_path):
    """
    Create a text file listing the video segments for FFmpeg concatenation.

    Parameters
    ----------
    segment_video_paths : list of str
        The list of paths to the video segments.
    concat_file_path : str
        The path to save the concat file.

    Returns
    -------
    None
    """
    with open(concat_file_path, 'w') as f:
        for segment_path in segment_video_paths:
            f.write(f"file '{os.path.abspath(segment_path)}'\n")

def concatenate_segments(concat_file_path, output_video_path):
    """
    Concatenate the video segments into a final video.

    Parameters
    ----------
    concat_file_path : str
        The path to the concat file.
    output_video_path : str
        The path to save the final video.

    Returns
    -------
    None
    """
    import subprocess

    command = [
        'ffmpeg',
        '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_file_path,
        '-c', 'copy',
        output_video_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def adjust_video_to_synthesized_audio(segments, synthesized_segments_paths, video_path, background_audio_path, output_video_path):
    """
    Adjust the video speed to match the synthesized audio segments.

    Parameters
    ----------
    segments : list of dict
        The list of speech segments with 'start', 'end', and 'text' keys.
    synthesized_segments_paths : list of str
        The list of paths to the synthesized audio files.
    video_path : str
        The path to the original video file.
    background_audio_path : str
        The path to the background audio file.
    output_video_path : str
        The path to save the final video.

    Returns
    -------
    None
    """
    import os
    from moviepy.editor import VideoFileClip

    # Get total duration of the video
    video = VideoFileClip(video_path)
    total_duration = video.duration
    video.close()

    # Prepare full segments list
    full_segments_list = prepare_full_segments_list(segments, total_duration)

    # Process segments
    output_dir = 'processed_segments'
    segment_video_paths = process_segments(
        full_segments_list,
        synthesized_segments_paths,
        video_path,
        background_audio_path,
        output_dir
    )

    # Create concat file
    concat_file_path = 'concat_list.txt'
    create_concat_file(segment_video_paths, concat_file_path)

    # Concatenate segments
    concatenate_segments(concat_file_path, output_video_path)

    # Clean up temporary files
    os.remove(concat_file_path)
    # Optionally, delete the processed_segments directory if you don't need it anymore
    # import shutil
    # shutil.rmtree(output_dir)

from yt_dlp import YoutubeDL
import os

def video_download(url):
    """
    Downloads a video from a given YouTube URL using yt-dlp.
    
    Downloads the highest quality video and audio formats available and
    merges them into a single MP4 file. The video is saved in the
    ./downloads/ directory with the title of the YouTube video as the filename.
    
    Parameters
    ----------
    url : str
        The URL of the YouTube video to be downloaded.
    
    Returns
    -------
    None
    """
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'outtmpl':  'downloads/%(id)s.%(ext)s'   
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print(f"Video downloaded from {url}")

    # Get video name
    video_name = 'downloads/' + url.split('watch?v=')[-1] + '.mp4'
    return video_name

    # Extract the audio from the video using ffmpeg
    # audio_extractor('downloads/' + url.split('watch?v=')[-1] + '.mp4')
    # print(f"Audio extracted from {url}")

def audio_extractor(video_path):
    # extract the name of the original video from the path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Extract the audio from the video using ffmpeg
    os.system(f"ffmpeg -i {video_path} -q:a 0 -map a original_audios/{video_name}.wav")
    print(f"Audio extracted from {video_path}")

    return f"original_audios/{video_name}.wav"

def replace_audio_in_video(video_path, audio_path, output_path):
    """
    Replace the audio track of a video file with a new audio track using FFmpeg.

    Parameters
    ----------
    video_path : str
        The path to the original video file.
    audio_path : str
        The path to the new audio file to be used as the audio track.
    output_path : str
        The path where the output video file will be saved.

    Returns
    -------
    None
    """
    import subprocess
    import os

    # Check if input files exist
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Build the FFmpeg command
    command = [
        'ffmpeg',
        '-y',  # Overwrite output files without asking
        '-i', video_path,  # Input video file
        '-i', audio_path,  # Input audio file
        '-c:v', 'copy',  # Copy the video stream without re-encoding
        '-map', '0:v:0',  # Select the video stream from the first input
        '-map', '1:a:0',  # Select the audio stream from the second input
        '-shortest',  # Stop writing output when the shortest input stream ends
        output_path  # Output file path
    ]

    # Run the FFmpeg command
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Check if FFmpeg command was successful
    if result.returncode != 0:
        print("FFmpeg error:")
        print(result.stderr.decode('utf-8'))
        raise RuntimeError("Failed to replace audio in video.")
    else:
        print(f"Output video saved to {output_path}")

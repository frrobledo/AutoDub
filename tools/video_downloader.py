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

    # Extract the audio from the video using ffmpeg
    # audio_extractor('downloads/' + url.split('watch?v=')[-1] + '.mp4')
    # print(f"Audio extracted from {url}")

def audio_extractor(video_path):
    # extract the name of the original video from the path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Extract the audio from the video using ffmpeg
    os.system(f"ffmpeg -i {video_path} -q:a 0 -map a original_audios/{video_name}.wav")
    print(f"Audio extracted from {video_path}")

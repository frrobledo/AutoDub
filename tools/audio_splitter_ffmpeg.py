import subprocess
import os

def split_audio_file_ffmpeg(audio_path, chunk_length_sec=600):  # 10 minutes in seconds
    # Get audio duration using ffprobe
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries',
        'format=duration', '-of',
        'default=noprint_wrappers=1:nokey=1', audio_path
    ]
    duration = float(subprocess.check_output(cmd).strip())
    chunks = []
    for i in range(0, int(duration), chunk_length_sec):
        chunk_name = f"{os.path.splitext(audio_path)[0]}_chunk_{i//chunk_length_sec}.wav"
        cmd = [
            'ffmpeg', '-y', '-i', audio_path,
            '-ss', str(i),
            '-t', str(chunk_length_sec),
            '-c', 'copy', chunk_name
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        chunks.append(chunk_name)
    return chunks

def concatenate_audio_files_ffmpeg(file_paths, output_path):
    import tempfile

    # Create a temporary file listing all the files to concatenate
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        for file_path in file_paths:
            f.write(f"file '{os.path.abspath(file_path)}'\n")
        concat_list_filename = f.name

    # Use ffmpeg to concatenate the audio files
    cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i',
        concat_list_filename, '-c', 'copy', output_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # os.remove(concat_list_filename)

def audio_separator_chunk(chunk_path):
    import subprocess
    import os
    from spleeter.separator import Separator

    base_name = os.path.splitext(os.path.basename(chunk_path))[0]
    output_dir = f'output_audio/{base_name}'
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    separator = Separator('spleeter:2stems')
    separator.separate_to_file(chunk_path, output_dir, filename_format="{instrument}.{codec}")


def process_full_audio_with_spleeter(audio_path):
    audio_file_name = os.path.splitext(os.path.basename(audio_path))[0]

    # Step 1: Split the audio file using ffmpeg
    chunk_paths = split_audio_file_ffmpeg(audio_path, chunk_length_sec=600)  # 10-minute chunks

    # Step 2: Process each chunk with Spleeter
    for chunk_path in chunk_paths:
        audio_separator_chunk(chunk_path)

    # Step 3: Collect the output files for concatenation
    vocals_files = []
    accompaniment_files = []
    for chunk_path in chunk_paths:
        base_name = os.path.splitext(os.path.basename(chunk_path))[0]
        output_dir = f'output_audio/{base_name}/'
        vocals_files.append(f"{output_dir}/vocals.wav")
        accompaniment_files.append(f"{output_dir}/accompaniment.wav")

    # Step 4: Concatenate the separated chunks using ffmpeg
    concatenate_audio_files_ffmpeg(vocals_files, f'output_audio/{audio_file_name}_vocals.wav')
    print(f'Created output_audio/{audio_file_name}_vocals.wav')
    concatenate_audio_files_ffmpeg(accompaniment_files, f'output_audio/{audio_file_name}_accompaniment.wav')
    print(f'Created output_audio/{audio_file_name}_accompaniment.wav')

    # Clean up temporary chunk files (optional)
    for chunk_path in chunk_paths:
        os.remove(chunk_path)

if __name__ == '__main__':
    audio_path = 'original_audios/xSh7PuWAxXU.wav'
    process_full_audio_with_spleeter(audio_path)
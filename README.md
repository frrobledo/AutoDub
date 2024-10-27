# AutoDub

An advanced AI-powered tool that automatically translates and dubs YouTube videos into different languages while dynamically adjusting video speed. This project combines state-of-the-art speech recognition, translation, and voice cloning technologies to create natural-sounding dubbed videos.

## Features

- **Automatic Video Processing**: Downloads YouTube videos using [yt-dlp](https://github.com/yt-dlp/yt-dlp) and extracts audio automatically
- **Speech Recognition**: Uses [Whisper AI](https://github.com/openai/whisper) for accurate speech-to-text transcription
- **Voice Separation**: Splits original audio into vocal and instrumental tracks using [Spleeter](https://github.com/deezer/spleeter)
- **Neural Translation**: Supports high-quality translation through [DeepL](https://www.deepl.com) API
- **Voice Cloning**: Uses [XTTS v2](https://huggingface.co/coqui/XTTS-v2) for natural-sounding voice synthesis that matches the original speaker
- **Intelligent Video Speed Adjustment**: Automatically adjusts video speed per speech segment to maintain lip-sync
- **Background Music Preservation**: Maintains original background music and sound effects
- **Multi-language Support**: Can translate and dub into multiple target languages

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster processing)
- FFmpeg installed and added to system PATH

## Installation

1. Clone the repository:
```bash
git clone https://github.com/frrobledo/AutoDub.git
cd AutoDub
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Install additional dependencies:
```bash
apt-get install ffmpeg  # for debian based systems
```
For other OS, refer to the [ffmpeg installation guide](https://www.ffmpeg.org/download.html)

4. Set up API keys:
   - Create a [DeepL API](https://www.deepl.com/en/pro-api) account and add your API key to the configuration

## Project Structure

```
├── tools/
│   ├── audio_synthesis.py     # Voice cloning and audio processing
│   ├── transcriber.py         # Speech recognition and translation
│   ├── video_editing.py       # Video speed adjustment and editing
│   ├── video_downloader.py    # YouTube video downloading
│   ├── audio_splitter_ffmpeg.py # Audio separation
│   └── logger.py             # Logging utilities
├── main.py                   # Main execution script
└── README.md
```

## Usage

1. Run the main script:
```bash
python main.py
```

2. Enter the YouTube URL when prompted.

3. The script will automatically:
   - Download the video
   - Extract and transcribe the audio
   - Separate speech from background audio
   - Translate the speech
   - Clone the voice in the target language
   - Adjust video speed for lip-sync
   - Combine everything into the final video

4. Find the output video in the `final_output` directory.

## How It Works

1. **Video Processing**:
   - Downloads YouTube video using yt-dlp
   - Extracts audio track
   - Separates vocals from background using Spleeter

2. **Speech Processing**:
   - Transcribes speech using Whisper AI
   - Detects spoken language automatically
   - Translates text using DeepL API

3. **Voice Synthesis**:
   - Clones original voice using XTTS v2
   - Generates speech in target language
   - Matches timing of original speech segments

4. **Video Adjustment**:
   - Analyzes duration of original vs. translated speech
   - Adjusts video speed per segment for lip-sync
   - Preserves original background audio
   - Combines all elements into final video

## Configuration

The project creates several directories for processing:
- `downloads/`: Downloaded YouTube videos
- `original_audios/`: Extracted audio files
- `output_audio/`: Processed audio segments
- `final_output/`: Final dubbed videos
- `logs/`: Processing logs

## Known Limitations

- Video quality depends on source YouTube video
- For some languages, audio generation can produce artifacts and very slow/fast segments
- Processing time varies based on video length and hardware
- Some languages may have better results than others

## Contributing

Contributions are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.

## Acknowledgments

- [Whisper AI](https://github.com/openai/whisper) for speech recognition
- [XTTS v2](https://huggingface.co/coqui/XTTS-v2/) for voice cloning
- [Spleeter](https://github.com/deezer/spleeter) for audio separation
- [DeepL](https://www.deepl.com/) for neural translation
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for video downloading

## Contact

For questions or support, please create an issue in the GitHub repository.
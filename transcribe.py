import whisper
import subprocess
import argparse
import os

def convert_to_mp3(audio_path):
    """ Converts an audio file to MP3 using FFmpeg and returns the new path """
    try:
        mp3_path = os.path.splitext(audio_path)[0] + ".mp3"
        command = ["ffmpeg", "-i", audio_path, "-q:a", "2", "-y", mp3_path]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return mp3_path
    except Exception as e:
        print(f"Error converting to MP3: {e}")
        return None

def transcribe_audio(audio_path, model_name="large"):
    try:
        # Convert to MP3 if needed (e.g., for M4A/WAV)
        if not audio_path.endswith(".mp3"):
            print("Converting audio to MP3...")
            audio_path = convert_to_mp3(audio_path)
            if not audio_path:
                return None  # Stop if conversion fails
        
        # Load model and transcribe
        model = whisper.load_model(model_name)
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", help="Path to audio file", required=True)
    parser.add_argument("--model", help="Whisper model (tiny, base, small, medium, large)", default="large")
    args = parser.parse_args()

    transcript = transcribe_audio(args.audio, args.model)
    if transcript:
        print("\nTranscript:\n", transcript)
        # Save to file
        with open("transcript.txt", "w") as f:
            f.write(transcript)
        print("\nTranscript saved to transcript.txt")
    else:
        print("Transcription failed.")

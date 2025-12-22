#
import subprocess
from pathlib import Path

# Constant bitrate to use for MP3 (e.g., "320k", "256k", "192k")
BITRATE = "320k"

# Use constant bitrate with libmp3lame
command_fstring = "ffmpeg -y -i {input} -c:a libmp3lame -b:a {bitrate} {output}"

def get_wavs_in_folder(folder_path):
    """Get a list of .wav files in the specified folder"""
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"Warning: {folder_path} is not a valid directory.")
        return []
    
    wav_files = sorted(folder.glob("*.wav"))
    return [str(wav) for wav in wav_files]

def wav_to_mp3(input_wav):
    """Convert a WAV file to MP3 format using ffmpeg"""
    input_path = Path(input_wav)
    if not input_path.is_file() or input_path.suffix.lower() != '.wav':
        print(f"Error: {input_wav} is not a valid WAV file.")
        return None
    
    output_path = input_path.with_suffix('.mp3')
    command = command_fstring.format(input=str(input_path), output=str(output_path), bitrate=BITRATE)
    
    try:
        subprocess.run(command, shell=True, check=True)
        return str(output_path)
    except Exception as e:
        print(f"Error during conversion: {e}")
        return None
    
if __name__ == "__main__":
    # men folder
    men_folder = Path(__file__).parent / "python/src/carfac/reference/men"
    women_folder = Path(__file__).parent / "python/src/carfac/reference/women"

    for f in get_wavs_in_folder(men_folder):
        print(f"Converting {f} to MP3...")
        wav_to_mp3(f)

    for f in get_wavs_in_folder(women_folder):
        print(f"Converting {f} to MP3...")
        wav_to_mp3(f)
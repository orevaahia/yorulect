import os
from pydub import AudioSegment
import argparse
from tqdm import tqdm


def resample_wav_files(root_folder, target_sample_rate=16000):
    # Collect all .wav files
    wav_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.wav'):
                file_path = os.path.join(dirpath, filename)
                wav_files.append(file_path)

    # Process files with a progress bar
    for file_path in tqdm(wav_files, desc="Resampling WAV files", unit="file"):
        # Load the audio file
        audio = AudioSegment.from_wav(file_path)

        # Resample the audio to the target sample rate
        resampled_audio = audio.set_frame_rate(target_sample_rate)

        # Export the resampled audio back to the same file
        resampled_audio.export(file_path, format="wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample WAV files in a folder to 16 kHz")
    parser.add_argument('--root_folder', type=str, help="Path to the input folder containing WAV files")

    args = parser.parse_args()

    resample_wav_files(args.root_folder)

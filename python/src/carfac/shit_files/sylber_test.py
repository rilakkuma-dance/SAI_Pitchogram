from sylber import Segmenter
import torchaudio
from python.src.carfac.modules.tone_detection_word import ToneClassifierTester

# Loading Sylber
segmenter = Segmenter(model_ckpt="sylber")

classifier = ToneClassifierTester()

# Run Sylber
wav_file = "python/src/carfac/reference/women/7_women.wav"

# outputs = segmenter(wav_file, in_second=False) # in_second can be False to output segments in frame numbers.
# outputs = {"segments": numpy array of [start, end] of segment,
#            "segment_features": numpy array of segment-averaged features,
#            "hidden_states": numpy array of raw features used for segmentation.

def get_segment_arrays_from_audio(wav):
    """Get segments and features from an audio file using Sylber"""
    results = segmenter(wav=wav, in_second=True)
    segments = results['segments']

    print(segments)
    return segments

def get_segment_arrays_from_audio_file(wav_file):
    results = segmenter(wav_file=wav_file, in_second=True)
    segments = results['segments']

    print(segments)
    return segments

def get_segment_arrays_from_audio_file(wav_file):
    wav, sr = torchaudio.load(wav_file)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        wav = resampler(wav)
    wav = (wav - wav.mean()) / wav.std()

    return get_segment_arrays_from_audio(wav)

def get_segment_wavs_from_audio_file(wav_file):
    segments = get_segment_arrays_from_audio_file(wav_file)
    wav, sr = torchaudio.load(wav_file)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        wav = resampler(wav)
    wav = (wav - wav.mean()) / wav.std()

    segment_wavs = []
    for start, end in segments:
        start_sample = int(start * 16000)
        end_sample = int(end * 16000)
        segment_wavs.append(wav[:, start_sample:end_sample])

    print(segment_wavs)

    return segment_wavs

def predict_tones_for_segments(segment_wavs):
    """Predict tones for a list of segment waveforms"""
    tones = []
    for segment_wav in segment_wavs:
        tone = classifier.predict_tone(segment_wav, sr=16000)
        tones.append(tone)
        print(tone)
    return tones

if __name__ == "__main__":
    segments = get_segment_wavs_from_audio_file(wav_file)
    tones = predict_tones_for_segments(segments)
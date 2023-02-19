import librosa
import argparse
import soundfile as sf


def convertsample(filename, extension, samplerate):
    audio, sr = librosa.load(filename+extension, sr=None)
    new_audio = librosa.resample(audio, orig_sr=sr, target_sr=samplerate)
    sf.write(filename+'.ogg', new_audio, samplerate)
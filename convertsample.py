import librosa
import argparse
import soundfile as sf


def convertsample(filename, extension, samplerate):
    audio, sr = librosa.load(filename+extension, sr=None)
    new_audio = librosa.resample(audio, orig_sr=sr, target_sr=samplerate)
    sf.write(filename+'.ogg', new_audio, samplerate)
    
    
def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", type=str, default="test")
    parser.add_argument("-e", "--extension", type=str, default=".mp3")
    parser.add_argument("-r", "--samplerate", type=int, default=44100)
    args = parser.parse_args()
    convertsample(args.filename, args.extension, args.samplerate)
    
    
if __name__ == "__main__":
    main()
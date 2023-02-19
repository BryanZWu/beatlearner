import numpy as np
from pydub import AudioSegment
import argparse


def convert2ogg(filename, extension):
    sound = AudioSegment.from_mp3(filename+extension)
    sound.export(f"{filename}.ogg", format="ogg")
    # sound.export(f"{filename}.mp4", format="mp4")
    #print('test')

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", type=str, default="test")
    parser.add_argument("-e", "--extension", type=str, default=".mp3")
    args = parser.parse_args()
    convert2ogg(args.filename, args.extension)


if __name__ == "__main__":
    main()
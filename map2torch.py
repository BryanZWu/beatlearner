"""
A file which contains functions for processing maps. Map info.dat files are
stored in the following format:

{
    "_version": "2.0.0",
    "_notes": [
        {
            "_time": 0,
            "_lineIndex": 0,
            "_lineLayer": 0,
            "_type": 0,
            "_cutDirection": 0
        },
        ...
    ],
    "_obstacles": [
        {
            "_time": 0,
            "_lineIndex": 0,
            "_type": 0,
            "_duration": 0,
            "_width": 0
        },
        ...
    ],
    "_events": [...]
}

After one hot encoding the fields in notes (minus _time), we get a tensor of
shape (21,) for each note. 

(note: 21 is obtained by 4 _lineIndex, 3 _lineLayer, 4 _type, 9 _cutDirection,
and a boolean for whether the note note exists.)

We wish take the list of notes, the bpm of the song, the sample rate of the
audio, and the hop length (128) of the audio encoding and convert this information
into a tensor of shape (song_length * sample_rate / hop_length, 21). 
"""

import torch
import json
import os
import math
import torchaudio

def map2torch(notes, bpm, song_length, sample_rate=41_000, hop_length=128):
    """
    Converts a list of notes from beats to timestamps. 
    args:
        notes: a list of notes in the format of a map's _notes field.
        bpm: the bpm of the song.
        song_length: the length of the song in samples.
        sample_rate: the sample rate of the audio.
        hop_length: the hop length of the audio encoding.
    """
    # sample_rate / hop_length is the number of samples per second.
    bps = bpm / 60
    # (samples/second) / (beats/second) = samples/beat
    factor = sample_rate / (hop_length) / bps

    # The length of the song in adjusted samples.
    song_length_samples = math.ceil(song_length / hop_length)

    # A tensor of shape (song_length_samples, 21) which will be filled with
    # the notes.
    out_tensor = torch.zeros(song_length_samples, 21)

    def note2tensor(note):
        """
        Converts a note to a tensor.
        """
        out = torch.zeros(21)
        out[note['_lineIndex']] = 1
        out[note['_lineLayer'] + 4] = 1
        out[note['_type'] + 7] = 1
        out[note['_cutDirection'] + 11] = 1
        out[20] = 1
        return out

    for note in notes:
        out_tensor[int(note['_time'] * factor)] = note2tensor(note)
    return out_tensor

bs_map = json.load(open('test_map/ExpertStandard.dat'))
bs_info = json.load(open('test_map/info.dat'))

# Find the length of the song by looking at the ogg file.
ogg_file = 'test_map/Seishun Complex.egg'
test_sample, sample_rate = torchaudio.load(ogg_file)
song_length = test_sample.shape[1]

bpm = bs_info['_beatsPerMinute']
notes = bs_map['_notes']

tensor = map2torch(notes, bpm, song_length, sample_rate=sample_rate)
# save the tensor to a file.

def mapdir2torch(map_dir):
    """
    Converts one map directory to a torch tensor.
    """
    # Anything that ends in standard.dat is a map file.
    map_files = [f for f in os.listdir(map_dir) if f.endswith('Standard.dat')]
    if len(map_files) == 0:
        raise ValueError('No map files found in directory.')
    for map_file in map_files:
        map_path = os.path.join(map_dir, map_file)
        map_data = json.load(open(map_path))

        info_path = os.path.join(map_dir, 'info.dat')
        info_data = json.load(open(info_path))

        # Audio path is just first egg/ogg file in the directory.
        audio_files = [f for f in os.listdir(map_dir) if f.endswith('.egg') or f.endswith('.ogg')]
        if len(audio_files) == 0:
            raise ValueError('No audio files found in directory.')
        elif len(audio_files) > 1:
            # warning only
            print('More than one audio file found in directory. Using first one.')
        audio_path = os.path.join(map_dir, audio_files[0])
        audio_sample, sample_rate = torchaudio.load(audio_path)
        song_length = audio_sample.shape[1]

        bpm = info_data['_beatsPerMinute']
        notes = map_data['_notes']

        tensor = map2torch(notes, bpm, song_length, sample_rate=sample_rate)
        # save the tensor to a file.
        output_path = os.path.join(map_dir, map_file.replace('.dat', '.pt'))
        torch.save(tensor, output_path)

def map_directory2torch(map_directory):
    """
    Converts a directory of map directories to torch tensors.
    """
    map_dirs = [f for f in os.listdir(map_directory) if os.path.isdir(os.path.join(map_directory, f))]
    for map_dir in map_dirs:
        mapdir2torch(os.path.join(map_directory, map_dir))
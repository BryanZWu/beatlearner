import torch
import os
import torchaudio

from google.cloud import storage
from google.oauth2 import service_account
import io
import random
import json

class MapDataset(torch.utils.data.Dataset):
    '''
    A dataset of beatsaber maps, stored in google cloud storage.

    Data is in the google cloud in the following structure:
    - bucket
        - map_directory
            - [songname].egg
            - [difficulty]Standard.pt
        - map_directory
    
    Each training point is one egg file, paired with the corresponding
    dat file. All egg files are ogg files that contain 44.1khz audio.
    '''
    def __init__(self, bucket):
        self.bucket = bucket
        self.map_set = set()
        
        for blob in self.bucket.list_blobs():
            if blob.name.endswith('.egg'):
                map_dir = os.path.dirname(blob.name)
                self.map_set.add(map_dir)
        self.map_list = list(self.map_set)
    
    def __len__(self):
        return len(self.bucket.list_blobs(prefix=self.map_directory))
    
    def __getitem__(self, idx):
        map = self.map_list[idx]

        audio = None
        pts = []
        dats = []
        
        # Get one egg file and one random dat file or pt file
        for blob in self.bucket.list_blobs(prefix=map):
            if blob.name.endswith('.egg'):
                # download the egg file and then read it into a tensor
                file_as_string = blob.download_as_string()
                data, sample_rate = torchaudio.load(io.BytesIO(file_as_string))
                audio = data.mean(0, keepdim=True).T
            
            elif blob.name.endswith('.pt'):
                # download the pt file and then read it into a tensor
                file_as_string = blob.download_as_string()
                pts.append(torch.load(io.BytesIO(file_as_string)))
            
            elif blob.name.endswith('.dat'):
                dats.append(blob)
        
        # Get info.dat and convert to json
        info = self.bucket.get_blob(os.path.join(map, 'info.dat'))
        info = json.load(io.BytesIO(info.download_as_string()))
            
        if pts:
            pt = random.choice(pts)
        else:
            dat = random.choice(dats)
            # load the dat file into json
            dat = json.load(io.BytesIO(dat.download_as_string()))
            bpm = info['_beatsPerMinute']
            notes = dat['_notes']
            song_length = audio.shape[1]
            pt = self.map2torch(notes, bpm, song_length, sample_rate=sample_rate)
    
        return audio, pt
    
    def get_pt_from_dat(self, dat):
        '''
        Convert a dat file into a pt file.
        '''
        pass
    def map2torch(self, notes, bpm, song_length, sample_rate=41_000, hop_length=128):
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


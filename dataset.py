import torch
import os
import torchaudio

from google.cloud import storage
from google.oauth2 import service_account
import io
import random

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
                dats.append(blob.download_as_string())
            
        if pts:
            pt = random.choice(pts)
        else:
            pt = self.get_pt_from_dat(random.choice(dats))
    
        return audio, pt
    
    def get_pt_from_dat(self, dat):
        '''
        Convert a dat file into a pt file.
        '''
        pass

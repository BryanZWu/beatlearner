import torch
import os

class MapDataset(torch.utils.data.Dataset):
    '''
    A dataset of beatsaber maps, stored in google cloud storage.

    Data is in the google cloud in the following structure:
    - bucket
        - map_directory
            - song.egg
            - [difficulty]Standard.pt
        - map_directory
    
    Each training point is one egg file, paired with the corresponding
    dat file. All egg files are ogg files that contain 44.1khz audio.

    Due to the size of the dataset, we do not load the entire dataset
    in memory. Instead, we will chunk the dataset into smaller pieces
    and load them as needed. This is done in the __getitem__ method.
    '''
    chunk_size = 1000
    def __init__(self, bucket):
        self.bucket = bucket
        self.map_set = set()
        self.chunk = None
        
        for blob in self.bucket.list_blobs():
            if blob.name.endswith('.egg'):
                map_dir = os.path.dirname(blob.name)
                self.map_set.add(map_dir)
        self.map_list = list(self.map_set)
    
    def __len__(self):
        return len(self.bucket.list_blobs(prefix=self.map_directory))
    
    def __getitem__(self, idx):
        # Automatical chunking of the dataset. This is done by
        # splitting the dataset into chunks of size 1000, and
        # loading the chunk that the index is in.
        
        # Get the chunk number
        chunk_num = idx // self.chunk_size
        # Get the index of the item in the chunk
        chunk_idx = idx % self.chunk_size

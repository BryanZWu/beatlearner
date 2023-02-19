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

import torch
import torchaudio 
import math
import modal
import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import os

volume2 = modal.SharedVolume().persist("vae-checkpoint")
VAE_CACHE = "/vae_cache"
stub = modal.Stub("jukebox", image=modal.Image.debian_slim().pip_install(\
    ["transformers","torchaudio","torch","google-cloud-storage"]))

class ConditionedSparseAttention(nn.Module):
    '''
    ConditionedSparseAttention is a sparse attention mechanism that takes in a conditioning sequence
    (of audio features) with shape (batch, cond_time, cond_size). It takes in an input sequence 
    of shape (batch, input_time, input_size) and returns an output sequence of shape (batch, input_time, output_size).

    It uses torch.nn.MultiheadAttention to compute the attention weights, but only attends to the last
    attention_window frames of the input sequence and the last attention_window frames of the conditioning sequence.
    '''
    def __enter__(self, embed_dim, num_heads, dropout, attention_window):
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_window = attention_window

    def forward(self, x, condition, end_inds):
        '''
        x should be (batch, input_time, input_size)
        condition should be (batch, cond_time, cond_size)
        end_ind should be (batch). It is the index of the last frame of the input sequence for each batch.

        The end indices are the indices of the last frame of both sequences. 
        They are used to compute the attention masks.
        '''
        batch_size = x.shape[0]
        input_time = x.shape[1]
        cond_time = condition.shape[1]

        # create a tensor of indices
        input_indices = torch.arange(input_time).unsqueeze(0).repeat(batch_size, 1)
        cond_indices = torch.arange(cond_time).unsqueeze(0).repeat(batch_size, 1)

        # Masks for things that are after (end - attention_window)
        input_mask_ge = torch.ge(input_indices, end_inds.unsqueeze(1)-self.attention_window).float()
        cond_mask_ge = torch.ge(cond_indices, end_inds.unsqueeze(1)-self.attention_window).float()

        # Masks for things that are before (end)
        input_mask_lt = torch.lt(input_indices, end_inds.unsqueeze(1)).float()
        cond_mask_lt = torch.lt(cond_indices, end_inds.unsqueeze(1)).float()

        input_mask = input_mask_ge * input_mask_lt
        cond_mask = cond_mask_ge * cond_mask_lt

        # Concatenate the masks and the sequences
        mask = torch.cat([input_mask, cond_mask], dim=1)
        input = torch.cat([x, condition], dim=1)

        # Run the self attention. Output is of shape (batch, input_time + cond_time, input_size)
        attended = self.attention(input, input, input, attn_mask=mask, need_weights=False)

        return attended

class MapNetDecoder(nn.Module):
    '''
    MapNetDecoder is a transformer decoder that takes in a conditioning sequence of audio features
    of shape (batch, time, num_audio_features). It then uses a sparse attention mechanism,
    conditioned on the audio features, to attend to the input sequence of shape (batch, time, input_size).

    The decoder is autoregressive, meaning that it can only attend to previous frames of the input sequence.
    '''

    def __enter__(self, ndims, num_heads, dropout, num_layers, attention_window):
        super(MapNetDecoder, self).__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.attention_window = attention_window
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            self.layers.append(ConditionedSparseAttention(ndims, num_heads, dropout, attention_window))
        self.output_layer = nn.Linear(ndims, 21)
    
    def forward(self, x, condition, end_inds):
        '''
        The forward pass. Applies conditional sparse attention to the input sequence, using 
        CONDITION as the conditioning sequence. The conditional sparse attention is applied
        to the last self.attention_window frames of x and the last self.attention_window frames
        of CONDITION.

        Args:
            x: (Batch, Time, Channels). The map we are building on
            condition: (Batch, Time, Channels). The Condition, in our case, the audio encoding.
            end_inds: (Batch). The index of the last frame of the input sequence for each batch.
        '''
        for layer in self.layers:
            x = layer(x, condition, end_inds)
        x = self.output_layer(x)
        return x
    

class MapNet(nn.Module):
    '''
    MapNet is a model that takes in a raw audio waveform and outputs a beatsaber map of
    the audio. It is composed of a Jukebox VQVAE and a MapNetDecoder.
    '''
    def __init__(self, vqvae, decoder):
        super(MapNet, self).__init__()
        self.vqvae = vqvae
        self.decoder = decoder

    def forward(self, x, end_inds):
        '''
        The forward pass. Takes in a raw audio waveform and outputs a beatsaber map of the audio.

        Args:
            x: (Batch, Time, Channels). The raw audio waveform.
            end_inds: (Batch). The index of the last frame of the input sequence for each batch.
        '''
        # Encode the audio. Output is of shape (Batch, Tokens, Num_Audio_Features)
        x, _ = self.vqvae.encode(x)
        # Decode the map from the audio
        x = self.decoder(x, x, end_inds)
        return x
    
    def positional_encoding(self, x, max_len=1_000_000):
        '''
        Adds a positional encoding to the input sequence. 
        '''
        batch_size, time, data_dim = x.shape
        
        # Create a tensor of indices
        pos = torch.arange(time).unsqueeze(1).repeat(1, data_dim)

        div_term = torch.exp(torch.arange(0, data_dim, 2) * -(math.log(10000.0) / data_dim))

        # Create a tensor of positional encodings
        pos_encoding = torch.zeros(batch_size, max_len, data_dim)

        # Even channels
        pos_encoding[:, :, 0::2] = torch.sin(pos[:, 0::2] * div_term)
        # Odd channels
        pos_encoding[:, :, 1::2] = torch.cos(pos[:, 1::2] * div_term)

        # Truncate the positional encoding to the length of the input sequence
        pos_encoding = pos_encoding[:, :time, :]

        # Add the positional encoding to the input sequence
        x = x + pos_encoding

        return x

        
# TODOS: Pos encoding, end_ind masking.

@stub.function(gpu="A100",\
            shared_volumes={VAE_CACHE: volume2},\
# Set the transformers cache directory to the volume we created above.
# For details, see https://huggingface.co/transformers/v4.0.1/installation.html#caching-models
) # this is run in the cloud
def run_model(data):
    from transformers import set_seed, JukeboxVQVAE
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # actually it was running fine without the tokens hmmm
    model = JukeboxVQVAE.from_pretrained("openai/jukebox-1b-lyrics",\
                                        cache_dir=VAE_CACHE).eval()
    set_seed(0)
    model.levels=1
    for _ in range(2):
        model.encoders.pop(0)
    model.decoders = torch.nn.ModuleList()
    model.to(device)
    
    input_audio = data.to(device)# .T.unsqueeze(0)
    results = model.encode(input_audio.to(device))
    
    print("IN_SHAPE", data.shape) # should be batched
    print("OUT_SHAPE", results[0].shape)
    return results[0].cpu() # get the single result

@stub.function(secret=modal.Secret.from_name("treehacks-gcloud")) # this is run in the cloud
def encoder():
    '''
    This returns a list of size (B, embeddings) for each audio input.
    '''
    import torchaudio
    import torch
    # from google.cloud import storage
    # from google.oauth2 import service_account
    # import io
    # service_account_info = json.loads(os.environ["CLOUD_INFO"])
    # credentials = service_account.Credentials.from_service_account_info(service_account_info)
    # client = storage.Client(credentials=credentials)
    # bucket = client.get_bucket("sabermaps")
    # i = 0 
    # datas = []
    # for blob in bucket.list_blobs(): # prefix='zipped'
    #     # TODO: IGNORE ALL ZIPS
    #     # if blob.name.endswith(".zip"):
    #     #     continue
    #     if blob.name.endswith(".egg"):
    #         file_as_string = blob.download_as_string()
    #         # convert the string to bytes and then finally to audio samples as floats 
    #         # and the audio sample rate
    #         data, sample_rate = torchaudio.load(io.BytesIO(file_as_string))
    #         datas.append(data.mean(0, keepdim=True).T) # ESSENTIAL PREPROCESSING: Average out the channel, 
    datas = torch.nn.utils.rnn.pad_sequence(datas, batch_first=True) #torch.stack(datas)
    res = run_model.call(datas)


@stub.local_entrypoint
def main():
    # data, _ = torchaudio.load('Seishun Complex.egg')
    encoder.call()

'''
A training loop for the model
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.tensorboard as tensorboard
import os

from google.cloud import storage
from google.oauth2 import service_account
import io

from dataset import MapDataset
from model import MapNet

hparams = {
    'batch_size': 1,
    'learning_rate': 1e-4,
    'num_epochs': 100,
    'num_workers': 4,
    'save_interval': 20, # In batches
    'exists_weight': 10, # Weight for the exists loss compared to each one of the other losses
}


def train():
    # set up the dataset
    credentials = service_account.Credentials.from_service_account_file('credentials.json')
    client = storage.Client(project='beat-saber-ml', credentials=credentials)
    bucket = client.get_bucket('beat-saber-ml')
    dataset = MapDataset(bucket)
    dataloader = data.DataLoader(dataset, batch_size=hparams['batch_size'], shuffle=True, num_workers=hparams['num_workers'])

    # set up the model
    model = MapNet()
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=hparams['learning_rate'])
    bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss()

    # set up tensorboard and model checkpoints
    writer = tensorboard.SummaryWriter()
    if not bucket.blob('checkpoints').exists():
        bucket.blob('checkpoints').upload_from_string('')
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    # train the model
    for epoch in range(hparams['num_epochs']):
        for i, (audio, pts) in enumerate(dataloader):
            audio = audio.cuda()
            pts = pts.cuda()

            optimizer.zero_grad()
            output = model(audio)
            # Output is shape (batch_size, song_length, 21)
            # 0-4 _lineIndex, 4-7 _lineLayer, 7-11 _type, 11-20 _time, 20 exists
            # BCE loss on 20, and then cross entropy loss on the rest.
            loss = bce(output[:, :, 20], pts[:, :, 20]) * hparams['exists_weight']
            loss += ce(output[:, :, :4].permute(0, 2, 1), pts[:, :, :4])
            loss += ce(output[:, :, 4:7].permute(0, 2, 1), pts[:, :, 4:7])
            loss += ce(output[:, :, 7:11].permute(0, 2, 1), pts[:, :, 7:11])
            loss += ce(output[:, :, 11:20].permute(0, 2, 1), pts[:, :, 11:20])

            loss.backward()
            optimizer.step()

            print('Epoch: {}, Batch: {}, Loss: {}'.format(epoch, i, loss.item()))
            writer.add_scalar('Loss', loss.item(), epoch * len(dataloader) + i)

        if epoch % hparams['save_interval'] == 0:
            torch.save(model.state_dict(), 'checkpoints/epoch_{}.pt'.format(epoch))
            bucket.blob('checkpoints/epoch_{}.pt'.format(epoch)).upload_from_filename('checkpoints/epoch_{}.pt'.format(epoch))


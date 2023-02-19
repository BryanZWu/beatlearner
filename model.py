import torch
import torchaudio 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import set_seed, JukeboxVQVAE

import json
import os

class ConditionedSparseAttention(nn.Module):
    '''
    ConditionedSparseAttention is a sparse attention mechanism that takes in a conditioning sequence
    (of audio features) with shape (batch, cond_time, cond_size). It takes in an input sequence 
    of shape (batch, input_time, input_size) and returns an output sequence of shape (batch, input_time, output_size).

    It uses torch.nn.MultiheadAttention to compute the attention weights, but only attends to the last
    attention_window frames of the input sequence and the last attention_window frames of the conditioning sequence.
    '''
    def __init__(self, embed_dim, num_heads, dropout, attention_window):
        super(ConditionedSparseAttention, self).__init__()
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

    def __init__(self, ndims, num_heads, dropout, num_layers, attention_window):
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

        
def run_model(data):
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

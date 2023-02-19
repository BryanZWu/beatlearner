
import json
import os
import modal
from google.cloud import storage
from google.oauth2 import service_account
import time


stub = modal.Stub("jukebox", image=modal.Image.debian_slim().pip_install(\
    ["transformers","torchaudio","torch","google-cloud-storage"]))

# ## Defining a function
#
# Here we define a Modal function using the `modal.function` decorator.
# The body of the function will automatically be run remotely.
# This particular function is pretty silly: it just prints "hello"
# and "world" alternatingly to standard out and standard error.
volume = modal.SharedVolume().persist("jukebox-checkpoint")
volume2 = modal.SharedVolume().persist("vae-checkpoint")
CACHE_DIR = "/cache"
VAE_CACHE = "/vae_cache"

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
    for _ in range(2):
        model.encoders.pop(-1)
    model.decoders = torch.nn.ModuleList()
    model.to(device)
    print('model moved to device')
    start = time.time()
    input_audio = data.permute(0, 2, 1).to(device)
    results = model.encode(input_audio) # data.swapaxes(1,2), start_level=1
    end = time.time()
    print("Time to encode: ", end - start)
    print("IN_SHAPE", data.shape) # should be batched
    print("OUT_SHAPE", results[1].shape)
    # start = time.time()
    # results = model.encode(data, bs_chunks = 100) # data.swapaxes(1,2)
    # end = time.time()
    # print("Time to encode: ", end - start)
    return results[1]
    # model.prune_heads({
    # -1: [i for i in range(20)]

@stub.function(secret=modal.Secret.from_name("my-googlecloud-secret")) # this is run in the cloud
def load_data():
    import torchaudio
    import torch
    from google.cloud import storage
    from google.oauth2 import service_account
    import io
    service_account_info = json.loads(os.environ["CLOUD_INFO"])
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    client = storage.Client(credentials=credentials)
    bucket = client.get_bucket("sabermaps")
    i = 0 
    datas = []
    for blob in bucket.list_blobs(): # prefix='zipped'
        # TODO: IGNORE ALL ZIPS
        # if blob.name.endswith(".zip"):
        #     continue
        if blob.name.endswith(".egg"):
            file_as_string = blob.download_as_string()
            # convert the string to bytes and then finally to audio samples as floats 
            # and the audio sample rate
            data, sample_rate = torchaudio.load(io.BytesIO(file_as_string))
            # data = data.mean(0, keepdim=True).T
            #print(data.unsqueeze(0).shape)
            # out = list(run_model.call(data.unsqueeze(0)))
            print(data.shape)
            datas.append(data.mean(0, keepdim=True).T)
            #print(blob.name)
            i+=1
        if i==2: break

    datas = torch.nn.utils.rnn.pad_sequence(datas, batch_first=True).swapaxes(1,2) #torch.stack(datas)
    # print(datas.shape)
    res = list(run_model.call(datas))

@stub.local_entrypoint
def main():
    # data, _ = torchaudio.load('Seishun Complex.egg')
    load_data.call()

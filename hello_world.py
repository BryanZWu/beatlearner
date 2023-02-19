
import json
import os
import modal
from google.cloud import storage
from google.oauth2 import service_account

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
               shared_volumes={VAE_CACHE: volume2},
    # Set the transformers cache directory to the volume we created above.
    # For details, see https://huggingface.co/transformers/v4.0.1/installation.html#caching-models
    secret=modal.Secret.from_name("huggingface-read-token")) # this is run in the cloud
def run_model(data):
    from transformers import set_seed, JukeboxVQVAE
    # test_sample, sample_rate = torchaudio.load('Seishun Complex.egg')
    # print(test_sample.shape, sample_rate)
    # How does this work with jukebox sample??
    # if os.environ["TRANSFORMERS_CACHE"] is None:
    #     os.environ["TRANSFORMERS_CACHE"] = JukeboxModel.from_pretrained("openai/jukebox-1b-lyrics", min_duration=0).eval()
    #model = os.environ["TRANSFORMERS_CACHE"]

    # actually it was running fine without the tokens hmmm
    model = JukeboxVQVAE.from_pretrained("openai/jukebox-1b-lyrics",\
                                        cache_dir=VAE_CACHE).eval()
    
    set_seed(0)
    print(model(data.swapaxes(1,2)).shape)

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
            #print(data.shape)
            datas.append(data.mean(0,keepdim=True).T)
            print(blob.name)
            i+=1
        if i==2: break
    datas = torch.nn.utils.rnn.pad_sequence(datas, batch_first=True).swapaxes(1,2) #torch.stack(datas)
    print(datas.shape)
    run_model.map(datas)

@stub.local_entrypoint
def main():
    # data, _ = torchaudio.load('Seishun Complex.egg')
    load_data.call()

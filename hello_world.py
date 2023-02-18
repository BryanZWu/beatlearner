
import sys
import json
import os
import modal
from google.cloud import storage
from google.oauth2 import service_account
# from google.cloud import storage

stub = modal.Stub("jukebox", image=modal.Image.debian_slim().pip_install(\
    ["transformers","torchaudio","google-cloud-storage"]))

# ## Defining a function
#
# Here we define a Modal function using the `modal.function` decorator.
# The body of the function will automatically be run remotely.
# This particular function is pretty silly: it just prints "hello"
# and "world" alternatingly to standard out and standard error.


@stub.function(gpu="A100") # this is run in the cloud
def run_model():
    from transformers import JukeboxModel, JukeboxConfig
    import torchaudio
    from transformers import AutoTokenizer, JukeboxModel, set_seed

    test_sample, sample_rate = torchaudio.load('Seishun Complex.egg')
    print(test_sample.shape, sample_rate)
    # How does this work with jukebox sample??
    model = JukeboxModel.from_pretrained("openai/jukebox-1b-lyrics", min_duration=0).eval()

    # model.prune_heads({
    # -1: [i for i in range(20)]

    
@stub.function(gpu="A100", secret=modal.Secret.from_name("my-googlecloud-secret")) # this is run in the cloud
def load_data():
    from google.cloud import storage
    from google.oauth2 import service_account
    service_account_info = json.loads(os.environ["CLOUD_INFO"])

    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    client = storage.Client(credentials=credentials)
    bucket = client.get_bucket("sabermaps")
    
    for blob in bucket.list_blobs(): # prefix='zipped'
        # TODO: IGNORE ALL ZIPS
        print(blob.name)

@stub.local_entrypoint
def main():
    # print(torchaudio.info('Seishun Complex.egg'))
    # test_sample, sample_rate = torchaudio.load('Seishun Complex.egg')
    # test_sample.shape, sample_rate
    print(load_data.call())
    #return bucket.list_blobs()


# ## What happens?
#
# When you call this, Modal will execute the function `f` **in the cloud,**
# not locally on your computer. It will take the code, put it inside a
# container, run it, and stream all the output back to your local
# computer.
#
# Try doing one of these things next.
#
# ### Change the code and run again
#
# For instance, change the `print` statement in the function `f`.
# You can see that the latest code is always run.
#
# Modal's goal is to make running code in the cloud feel like you're
# running code locally. You don't need to run any commands to rebuild,
# push containers, or go to a web UI to download logs.
#
# ### Map over a larger dataset
#
# Change the map range from 20 to some large number. You can see that
# Modal will create and run more containers in parallel.
#
# The function `f` is obviously silly and doesn't do much, but you could
# imagine something more significant, like:
#
# * Training a machine learning model
# * Transcoding media
# * Backtesting a trading algorithm.
#
# Modal lets you parallelize that operation trivially by running hundreds or
# thousands of containers in the cloud.

import sys
import modal
# from google.cloud import storage
from transformers import JukeboxModel, JukeboxConfig

import torchaudio
from transformers import AutoTokenizer, JukeboxModel, set_seed
import torch

stub = modal.Stub("example-hello-world")

@stub.function # this is run in the cloud
def run_model():
    model = JukeboxModel.from_pretrained("openai/jukebox-1b-lyrics", min_duration=0).eval()
    model.prune_heads({
    -1: [i for i in range(20)]
})
    
@stub.local_entrypoint
def main():
    print(torchaudio.info('Seishun Complex.egg'))
    test_sample, sample_rate = torchaudio.load('Seishun Complex.egg')
    test_sample.shape, sample_rate
    # Call the function directly.
    print(run_model.call())
    # Parallel map.
    # this loop is parallelized
    # for ret in f.map(range(1000)):
    #     total += ret
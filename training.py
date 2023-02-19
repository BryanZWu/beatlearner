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
from model import MapNet, get_vq_vae, MapNetDecoder

hparams = {
    'batch_size': 1,
    'learning_rate': 1e-4,
    'num_epochs': 100,
    'num_workers': 1,
    'save_interval': 20, # In batches
    'exists_weight': 10, # Weight for the exists loss compared to each one of the other losses
}


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up the dataset
    # credentials = service_account.Credentials.from_service_account_file('credentials.json')
    client = storage.Client(project='beat-saber-ml')
    bucket = client.get_bucket('sabermaps')
    dataset = MapDataset()
    dataloader = data.DataLoader(dataset, batch_size=hparams['batch_size'], shuffle=True, num_workers=hparams['num_workers'])
    print('Dataset loaded!')

    encoder = get_vq_vae()
    decoder = MapNetDecoder(128, 8, 0.1, 3, 128)

    # set up the model
    model = MapNet(encoder, decoder)
    model = model.to(device)
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
            audio = audio.to(device)
            pts = pts.to(device)

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



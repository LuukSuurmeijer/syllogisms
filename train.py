from DFSDataset import *
from ComprehensionModel import *
from dfs_utils import *

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange
from functools import partial
import argparse
import sys
import math
import wandb
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

def summarize(model, hyperparameters, bs):
    data = {name: [name, [*param.data.shape], param.numel()] for name, param in model.named_parameters() if param.requires_grad}
    print("{:<25} {:<25} {:<25}".format("Layer", "Dim", "Number of parameters"))
    print(("="*25)*3)
    for key, value in data.items():
        name, shape, num_param = value
        print("{:<25} {:<25} {:<25}".format(name, str(shape), num_param))
    print(("="*25)*3)
    total = sum([param[2] for param in data.values()])
    print(f"Total trainable parameters: {total}" )
    print(f"Estimated memory required: {((total * 4) * (10**-6)) * bs} MB")
    for key, value in hyperparameters.items():
        print("{:<15}: {:<15}".format(key, str(value)))

def genplot(counter, values, avg, xlabel, ylabel, dataset, name):
    plt.scatter(counter, values, color='green', zorder=1)
    plt.plot(list([i * len(dataset) for i in range(EPOCHS)]), avg, color='black', zorder=2)#, s=14)
    plt.xlabel(f'{xlabel}')
    plt.ylabel(f'{ylabel}')
    plt.savefig(f'{name}.pdf')
    plt.clf()

parser = argparse.ArgumentParser(description='Train the neural network.')
parser.add_argument('--num_layers', type=int, default=1, help='number of recurrent layers')
parser.add_argument('--epochs', type=int, default=15, help='number of epochs')
parser.add_argument('--hiddens', type=int, default=120, help='number of hidden units per layer')
parser.add_argument('--type', help="LSTM/RNN", default='RNN')
parser.add_argument('--batchsize', type=int, default=1, help="Batch size, must be 1 for CPU (i think)")
parser.add_argument('--lr', type=float, default=0.03, help="learning rate")
parser.add_argument('--momentum', type=float, default=0.8, help="learning rate")
parser.add_argument('--loginterval', type=int, default=15, help="log interval")
parser.add_argument('--meshdata', help="data dir")
parser.add_argument('--output', default=None, help="save model")
parser.add_argument('--gpu', type=int, default=2, help="gpu index")
args = parser.parse_args()

#some constants
wandb.init(project="syllogisms", entity="luuksuurmeijer")
wandb.config = args
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
EPOCHS = args.epochs
bs = args.batchsize

#initialize training data
train_dataset = DFSDataset(f"{args.meshdata}", '<EOS>')
vocab_size = len(train_dataset.word2id)
obs_size = train_dataset.shape
train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

#define model
model = ComprehensionModel(vocab_size, args.hiddens, obs_size, n_layers=1, type='RNN').to(device)
criterion = nn.MSELoss(reduction='mean').to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
summarize(model, vars(args), bs)

def train():
    wandb.watch(model)
    print("TRAINING")

    #keep track of some stuff
    train_counter = []

    train_losses = []
    train_cosine = []

    avg_epoch_cosine = []
    avg_epoch_loss = []
    loss = 0
    #initialize, TODO: somehow my initialization hurts a fuckton
    #for layer in [param for name,param in model.named_parameters() if param.requires_grad]:
    #    model.initHidden(layer)
    model.train()
    #setup progress bar
    progressbar = trange(EPOCHS, desc='Bar desc', leave=True)
    for epoch in progressbar:
        if epoch % 20 == 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.9
            print(optimizer.param_groups[0]['lr'])
        running_loss = 0.0
        running_cosine = 0.0
        for id, example in enumerate(train_dataloader):

            sent = example[0].to(device) # input (bs, seq_len, vocab_size)
            target = example[1].to(device) # target (bs, obs_size)

            optimizer.zero_grad()
            hidden_seq, pred = model(sent, target) #forward pass
            pred = pred.to(torch.float64)

            # calculate evaluation metrics
            loss = criterion(pred, target)
            with torch.no_grad():
                cos = model.cosine(pred[0][-1], target[0][-1])
            loss.backward() #backward pass
            optimizer.step() #update weights

            # print statistics
            running_loss += loss.item()
            running_cosine += cos.item()
            train_losses.append(loss.item())
            train_cosine.append(cos.item())
            train_counter.append(((epoch*len(train_dataloader)) + id)*bs)

            wandb.log({"train_loss" : loss, "train_acc" : acc.item()})#, "train_accuracy" : acc})
            print_loss = running_loss / (len(train_dataloader)/bs)

            if args.loginterval != -1:
                if id % args.loginterval == 0:
                    print("Epoch: {:<12} | loss: {:<12}".format(f"{epoch+1} ({id}/{len(train_dataloader)/bs})", loss.item()))

        progressbar.set_description(f"Loss after epoch {epoch+1}: {running_loss}", refresh=True)
        #wandb.log({"train_cumulative_epoch_loss" : running_loss, "train_average_epoch_loss" : running_loss/(len(train_dataloader)/bs), "epoch_PPL" : ppl_final, "epoch_acc" : running_acc/(len(train_dataloader)/bs)})
        avg_epoch_loss.append(running_loss/(len(train_dataloader)/bs))
        avg_epoch_cosine.append(running_cosine/(len(train_dataloader)/bs))
    print(f"Avg Loss: {avg_epoch_loss[-1]}")
    print(f"Avg cosine: {avg_epoch_cosine[-1]}")


        # plot error
    genplot(train_counter, train_cosine, avg_epoch_cosine, 'Number of examples', 'Cosine', train_dataset, "Cosine")
    genplot(train_counter, train_losses, avg_epoch_loss, 'Number of examples', 'MSELoss', train_dataset, "Loss")

train()

#for id, example in enumerate(train_dataloader):
#    with torch.no_grad():
#        sent = example[0].to(device) # input (bs, seq_len, vocab_size)
#        target = example[1].to(device) # target (bs, obs_size)
#        pred = model(sent, target) #forward pass
#
#        inf = inferenceScore(pred[0][-1], pred[0][-1])
#        print(train_dataset.translate_one_hot(sent), inf)

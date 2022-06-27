import argparse
import csv

# import wandb
import json

import ccobra.syllogistic

from ComprehensionModel import *
from DFSdataset import *
from trainingloop import *
import time
from torch.utils.data import Subset, RandomSampler, DataLoader
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import dfs
import sys


def plotcurve(xlabel, ylabel, stats, cum):
    plt.xlabel(f"{xlabel}")
    plt.ylabel(f"{ylabel}")
    plt.plot(
        [sum(stats[j:j + cum]) / cum for j in range(0, len(stats), cum)])
    plt.show()


def delete_multiple_lines(n):
    """Delete the last line in the STDOUT."""
    for _ in range(n):
        sys.stdout.write("\x1b[1A")  # cursor up one line
        sys.stdout.write("\x1b[2K")  # delete the last line


def summarize(model, hyperparameters, bs):
    data = {name: [name, [*param.data.shape], param.numel()] for name, param in model.named_parameters() if
            param.requires_grad}
    print("{:<25} {:<25} {:<25}".format("Layer", "Dim", "Number of parameters"))
    print(("=" * 25) * 3)
    for key, value in data.items():
        name, shape, num_param = value
        print("{:<25} {:<25} {:<25}".format(name, str(shape), num_param))
    print(("=" * 25) * 3)
    total = sum([param[2] for param in data.values()])
    print(f"Total trainable parameters: {total}")
    print(f"Estimated memory required: {((total * 4) * (10 ** -6)) * bs} MB")
    # for key, value in hyperparameters.items():
    #    print("{:<15}: {:<15}".format(key, str(value)))


def genplot(counter, values, avg, xlabel, ylabel, dataset, name):
    plt.scatter(counter, values, color='green', zorder=1, s=2)
    plt.plot(list([i * len(dataset) for i in range(EPOCHS)]), avg, color='black', zorder=2)
    plt.xlabel(f'{xlabel}')
    plt.ylabel(f'{ylabel}')
    plt.savefig(f'{name}.pdf')
    plt.clf()


parser = argparse.ArgumentParser(description='Train the neural network.')
parser.add_argument('--num_layers', type=int, default=1, help='number of recurrent layers')
parser.add_argument('--epochs', type=int, default=150, help='number of epochs')
parser.add_argument('--hiddens', type=int, default=150, help='number of hidden units per layer')
parser.add_argument('--type', help="LSTM/RNN", default='RNN')
parser.add_argument('--radius', type=float, help="Zero error radius", default=0.98)
parser.add_argument('--batchsize', type=int, default=1, help="Batch size, must be 1 for CPU (i think)")
parser.add_argument('--lr', type=float, default=0.03, help="learning rate")
parser.add_argument('--momentum', type=float, default=0.8, help="learning rate")
parser.add_argument('--loginterval', type=int, default=15, help="log interval")
parser.add_argument('--meshdata', help="data dir")
parser.add_argument('--output', default=None, help="save model")
parser.add_argument('--gpu', type=int, default=2, help="gpu index")
args = parser.parse_args()


# some constants
# wandb.init(project="syllogisms", entity="luuksuurmeijer")
# wandb.config = args

# draw [num_items] random examples from train_dataset
# random_idx = np.random.randint(12, size=num_items)
# sample_ds = Subset(train_dataset, random_idx)
# sampler = RandomSampler(sample_ds)

def train_model(final_dataset, vocab_size, hyperparams):
    train_dataloader = DataLoader(final_dataset, batch_size=bs, shuffle=True)
    # define model
    model = ComprehensionModel(vocab_size, hyperparams.hiddens, obs_size, n_layers=1, type='RNN').to(device)
    model.to(torch.double)
    criterion = nn.MSELoss(reduction='sum').to(device)

    stats_dict = defaultdict(list)
    start_t = time.time()
    for epoch in range(EPOCHS):
        if epoch % 50 == 0:
            hyperparams.lr = hyperparams.lr * 0.9
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.lr)#, momentum=hyperparams.momentum)
        model = train(model, train_dataloader, criterion, optimizer, stats_dict, radius=hyperparams.radius, log_interval=hyperparams.loginterval)

        print("=" * 70)
        print("Epoch: {:2} of {:2}, Running Time: {:7.2f} sec, Total Loss: {:7.4f}".format(
            epoch + 1, EPOCHS, time.time() - start_t, sum(stats_dict['loss']) / len(stats_dict['loss'])))
        print("=" * 70)
        delete_multiple_lines(3)

    return model, stats_dict


device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
EPOCHS = args.epochs
bs = args.batchsize

with open('syll_data/syll_categories.json') as f:
    syll_categories = json.load(f)
for key in syll_categories:
    syll_categories[key] = [task for task, proportion in syll_categories[key]]
# initialize training data
raw = read_mesh('gen/10k_to_150.mesh')
#sent_notaut = [tup for tup in raw if tup[0][-1] != tup[0][-2]]
#premis_dict = {' '.join(tup[0]) : tup[1] for tup in sent_notaut}
#train_dataset_premises = DFSdatasetPhrase(mesh_data=(sent_notaut, sent_dict), delim='')
# train_dataset.data = [pair for pair in train_dataset.data if (pair[0].task == 'AA1' or pair[0].task == 'AA2') and pair[0].is_valid()]
# train_dataset.data.extend(train_dataset_premises.data)
train_dataset = DFSdatasetPhrase(f"{args.meshdata}", '')
full_data = train_dataset.data
obs_size = train_dataset.obs_size


#TODO: Invalids are not learned at all, except if I remove the is_valid flag?
color_dict = {'known_valid' : 'red', 'unknown_valid' : 'blue', 'known_invalid' : 'purple', 'unknown_invalid' : 'green'}
category_stacks = {'known_valid' : [], 'unknown_valid' : [], 'known_invalid' : [], 'unknown_invalid' : []}
bad = []
for category, sylls in syll_categories.items():
    if category == category:
        for i, syll in enumerate(sylls):
            final_dataset = DFSdatasetPhrase(mesh_data=([pair for pair in full_data if
                                                         pair[0].task == syll and pair[0].is_valid()],
                                                        train_dataset.sen_sem_dict), delim='')

            vocab_size = len(final_dataset.vocab)
            print('\r' + syll, f'{i+1}/{len(sylls)}')
            m, s = train_model(final_dataset, vocab_size, args)
            sys.stdout.write("\x1b[1A")  # cursor up one line

            # plot the loss at each epoch for each model. Since dataset sizes differ we normalize by the length of the training data
            cum = len(final_dataset)
            inf_trail = [sum(s['cosine'][i:i + cum]) / cum for i in range(0, len(s['cosine']), cum)]

            if inf_trail[-1] < 0.8:
                bad.append((syll, len(final_dataset)))

            category_stacks[category].append(inf_trail[0:200])
            plt.plot([sum(s['cosine'][i:i + cum]) / cum for i in range(0, len(s['cosine']), cum)], alpha=0.25, lw=1, color=color_dict[category])

        #plot the mean of the syllogism category
        plt.plot(np.mean(np.array(category_stacks[category]), axis=0), alpha=1,
                   color=color_dict[category], label=category, lw=1.5)

plt.legend()#bbox_to_anchor=(0,1,1,0), loc='lower left', mode="expand", ncol=2)
plt.ylabel('Inf')
plt.xlabel('Epochs')
plt.show()
print(bad)


# fig, axs = plt.subplots(1, 3, figsize=(10, 4))
# axs[0].plot([sum(stats_dict['loss'][i:i+cum])/cum for i in range(0, len(stats_dict['loss']), cum)])
# axs[0].set_title('MSE Loss')
# axs[1].plot([sum(stats_dict['cosine'][i:i+cum])/cum for i in range(0, len(stats_dict['cosine']), cum)])
# axs[1].set_title('Cosine similarity')
# axs[2].plot([sum(stats_dict['inf'][i:i+cum])/cum for i in range(0, len(stats_dict['inf']), cum)])
# axs[2].set_title('Inference score')
# axs.set_xlabel('Iterations')

# TODO: Poor generalization? Some examples perfectly learned, others not at all
# TODO: gold inferences are <1 for some valid syllogisms

# sequences = []
# with torch.no_grad():
#     data = [train_dataset[idx] for idx in random_idx]
#     for sent, sem in data:
#         sent = torch.unsqueeze(sent, dim=0)
#         sem = torch.unsqueeze(sem, dim=0)
#         prev_state = model.init_state()
#         pred, hidden_seq = model(sent, prev_state)
#
#         syll = train_dataset.decode_training_item(train_dataset.vocab.translate_one_hot(sent[0]))
#         gold_inf = dfs.inference_score(
#         train_dataset.sen_sem_dict[' '.join(syll.conclusion)],
#         dfs.conjunction(train_dataset.sen_sem_dict[' '.join(syll.premises[0])],
#         train_dataset.sen_sem_dict[' '.join(syll.premises[0])])
#         )
#
#         sequences.append([' '.join(train_dataset.vocab.translate_one_hot(sent[0])),
#                           model.cosine(sem[0][-1], pred[0][-1]).item(),
#                           criterion(pred, sem).item(),
#                           dfs.inference_score(sem[0][-1], pred[0][-1]).item(),
#                           gold_inf.item(),
#                           (syll.task, syll.conclusion_type), syll.is_valid(),
#                           dfs.prior_prob(sem[0][-1]).item()])
#     with open('performance_training.csv', 'w') as f:
#         writer = csv.writer(f)
#         writer.writerow(['sent', 'cosine', 'loss', 'model_inf', 'gold_inf', 'task/conclusion', 'valid', 'prob'])
#         writer.writerows(sequences)

import argparse
import csv

# import wandb

from ComprehensionModel import *
from DFSdataset import *
from trainingloop import *
import time
from torch.utils.data import Subset, RandomSampler, DataLoader
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import dfs


def summarize(model, hyperparameters, bs):
    data = {name: [name, [*param.data.shape], param.numel()] for name, param in model.named_parameters() if param.requires_grad}
    print("{:<25} {:<25} {:<25}".format("Layer", "Dim", "Number of parameters"))
    print(("="*25)*3)
    for key, value in data.items():
        name, shape, num_param = value
        print("{:<25} {:<25} {:<25}".format(name, str(shape), num_param))
    print(("="*25)*3)
    total = sum([param[2] for param in data.values()])
    print(f"Total trainable parameters: {total}")
    print(f"Estimated memory required: {((total * 4) * (10**-6)) * bs} MB")
    for key, value in hyperparameters.items():
        print("{:<15}: {:<15}".format(key, str(value)))


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
parser.add_argument('--batchsize', type=int, default=1, help="Batch size, must be 1 for CPU (i think)")
parser.add_argument('--lr', type=float, default=0.03, help="learning rate")
parser.add_argument('--momentum', type=float, default=0.8, help="learning rate")
parser.add_argument('--loginterval', type=int, default=15, help="log interval")
parser.add_argument('--meshdata', help="data dir")
parser.add_argument('--output', default=None, help="save model")
parser.add_argument('--gpu', type=int, default=2, help="gpu index")
args = parser.parse_args()

#some constants
#wandb.init(project="syllogisms", entity="luuksuurmeijer")
#wandb.config = args
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
EPOCHS = args.epochs
bs = args.batchsize

#initialize training data
raw = read_mesh('gen/10k_to_150.mesh')
sent_notaut = [tup for tup in raw if tup[0][-1] != tup[0][-2]]
sent_dict = {' '.join(tup[0]) : tup[1] for tup in sent_notaut}

num_items = 24
#train_dataset = DFSdataset(mesh_data=(sent_notaut, sent_dict), delim='')
train_dataset = DFSdatasetPhrase(f"{args.meshdata}", '<EOS>')
vocab_size = len(train_dataset.vocab)
obs_size = train_dataset.obs_size

#draw [num_items] random examples from train_dataset
random_idx = np.random.randint(12, size=num_items)
sample_ds = Subset(train_dataset, random_idx)
sampler = RandomSampler(sample_ds)

train_dataloader = DataLoader(sample_ds, batch_size=bs, sampler=sampler)

#define model
model = ComprehensionModel(vocab_size, args.hiddens, obs_size, n_layers=1, type='RNN').to(device)
criterion = nn.MSELoss(reduction='mean').to(device)
#, momentum=args.momentum)
summarize(model, vars(args), bs)
time.sleep(3)

stats_dict = defaultdict(list)
start_t = time.time()
for epoch in range(EPOCHS):
    if epoch % 50 == 0:
        args.lr = args.lr * 0.9
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model = train(model, train_dataloader, criterion, optimizer, stats_dict, log_interval=args.loginterval)
    #time.sleep(3)

    os.system('clear')
    print("\n" + "=" * 70)
    print("Epoch: {:2} of {:2}, Running Time: {:7.2f} sec, Total Loss: {:7.4f}".format(
        epoch + 1, EPOCHS, time.time() - start_t, sum(stats_dict['loss']) / len(stats_dict['loss'])))
    print("=" * 70 + "\n")


def plotcurve(xlabel, ylabel, stats, cum):

    plt.xlabel(f"{xlabel}")
    plt.ylabel(f"{ylabel}")
    plt.plot(
        [sum(stats[i:i+cum])/cum for i in range(0, len(stats), cum)])
    plt.show()

#plotcurve('iters', 'Cosine', stats_dict['cosine'], len(train_dataset))
#plotcurve('iters', 'MSE Loss', stats_dict['loss'], len(train_dataset))
#plotcurve('iters', 'inference score', stats_dict['inf'], len(train_dataset))


cum = num_items
fig, axs = plt.subplots(1, 3, figsize=(10, 4))
axs[0].plot([sum(stats_dict['loss'][i:i+cum])/cum for i in range(0, len(stats_dict['loss']), cum)])
axs[0].set_title('MSE Loss')
axs[1].plot([sum(stats_dict['cosine'][i:i+cum])/cum for i in range(0, len(stats_dict['cosine']), cum)])
axs[1].set_title('Cosine similarity')
axs[2].plot([sum(stats_dict['inf'][i:i+cum])/cum for i in range(0, len(stats_dict['inf']), cum)])
axs[2].set_title('Inference score')
#axs.set_xlabel('Iterations')

plt.show()

#TODO: Poor generalization? Some examples perfectly learned, others not at all
#TODO: gold inferences are <1 for some valid syllogisms

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

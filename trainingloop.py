import torch
import torch.optim as optim
import math
from dfs import *

#TODO: Zero-error-radius ?
def train(model, dataloader, criterion, optimizer, stats_dict, log_interval=5):
    model.train()
    running_loss = 0.0

    for id, example in enumerate(dataloader):
        optimizer.zero_grad()  # set gradients to zero
        sent = example[0]
        semantics = example[1]

        prev_state = model.init_state(b_size=sent.shape[0])  # reset RNN state

        pred, hidden_seq = model(sent, prev_state)  # forward pass
        #pred = pred.to(torch.double)

        loss = criterion(pred, semantics)  # compute loss
        # Gather statistics
        #print(model.cosine(pred[0][-1], semantics[0][-1]).item())
        stats_dict['loss'].append(loss.item())
        with torch.no_grad():
            stats_dict['cosine'].append(model.cosine(pred[0][-1], semantics[0][-1]).item())
            stats_dict['inf'].append(inferenceScore(pred[0][-1], semantics[0][-1]))
        running_loss += loss.item()

        loss.backward()  # backward pass
        optimizer.step()  # update weights

        if log_interval and (id % log_interval) == 0:
            print("Iteration: {:{}}/{}, Loss: {:8.4f}".format(
                id+1, int(math.log(id+1, 10)) + 2, id+1,
                           running_loss / float(log_interval)))
            running_loss = 0
    return model
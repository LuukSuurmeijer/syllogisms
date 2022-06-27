from dfs import *
import torch
import sys
from ComprehensionModel import *

def delete_multiple_lines(n):
    """Delete the last line in the STDOUT."""
    for _ in range(n):
        sys.stdout.write("\x1b[1A")  # cursor up one line
        sys.stdout.write("\x1b[2K")  # delete the last line

#TODO: Zero-error-radius ?
def train(model, dataloader, criterion, optimizer, stats_dict, radius, log_interval):
    model.train()
    running_loss = 0.0

    for id, example in enumerate(dataloader):
        optimizer.zero_grad()  # set gradients to zero
        sent = example[0]
        semantics = example[1]

        prev_state = model.init_state(b_size=sent.shape[0])  # reset RNN state

        pred, hidden_seq = model(sent, prev_state)  # forward pass
        pred = zero_error_radius(pred, semantics, radius)
        #if inference_score(semantics[0][-1], pred[0][-1]) < 1.:
         #   print(pred[0][-1])
          #  print(semantics[0][-1])

        #assert torch.equal(pred, semantics)

        loss = criterion(pred, semantics)  # compute loss
        # Gather statistics
        #print(model.cosine(pred[0][-1], semantics[0][-1]).item())
        stats_dict['loss'].append(loss.item())
        with torch.no_grad():
            stats_dict['cosine'].append(model.cosine(pred[0][-1], semantics[0][-1]).item())
            stats_dict['inf'].append(inference_score(semantics[0][-1], pred[0][-1]))
        running_loss += loss.item()

        loss.backward()  # backward pass
        optimizer.step()  # update weights

        if log_interval and (id % log_interval) == 0:
            print('\r' + "Iteration: {}/{}, Loss: {:8.4f}".format(
                id+1, len(dataloader),
                           running_loss / float(log_interval)))
            #delete_multiple_lines(1)
            running_loss = 0
    return model
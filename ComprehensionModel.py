import torch
import torch.nn as nn


class ComprehensionModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, observation_size, n_layers=1, type='RNN'):
        super(ComprehensionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # Some activation functions
        self.cosine = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
        self.LReLU = torch.nn.LeakyReLU(negative_slope=0.3, inplace=False)
        self.cappedLRelu = lambda x: torch.clamp(self.LReLU(x), max=1)  # upper bound = 1

        self.activation = self.cappedLRelu
        # The RNN takes one hot encoded tensors as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if type == 'LSTM':
            self.input2hidden = nn.LSTM(vocab_size, hidden_dim, batch_first=True, num_layers=self.n_layers, nonlinearity='relu')  #dim: (vocab_size, hidden_dim)
        else:
            self.input2hidden = nn.RNN(vocab_size, hidden_dim, batch_first=True, num_layers=self.n_layers, nonlinearity='relu')  #dim: (vocab_size, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, observation_size) #dim: (hidden_dim, observation_size) linear
        self.double()

    def forward(self, sentence, prev_state):
        #sentence is a torch one hot tensor of (bs, seq_len, vocab_size)
        #target is a torch tensor of (bs, observation_size) containing truth values

        hidden, hidden_seq = self.input2hidden(sentence, prev_state) #[bs, seq_len, vocab_size] -> [bs, seq_len, hidden_dim]
        hidden_activ = self.cappedLRelu(hidden)
        return torch.sigmoid(self.hidden2tag(hidden_activ)), hidden_seq #[bs, seq_len, hidden_dim] -> [bs, seq_len, observation_size]

    def init_state(self, b_size=1):
        # we need to reset the hidden state (NOT the weights) after each sentence to zero.
        # The previous state is of dim [n_layers, bs, hidden_dim]
        return torch.zeros(self.n_layers, b_size, self.hidden_dim).double()

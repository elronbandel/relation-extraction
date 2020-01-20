from torch.nn import Parameter, Module, Embedding, LSTM, Linear, CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam
from torch import nn, Tensor
import torch

from models.utils import train
from dataset.dataset import RexDataset
from torch.nn import Module, Embedding, Linear


class LSTMAcceptor(Module):
    def __init__(self, input_size, hidden_size, num_layers=2, bidirectional=True, dropout=0):
        super().__init__()
        self.lstm = LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional, dropout=dropout)

    def forward(self, padded_seqs, lens):
        #packed = pack_padded_sequence(padded_seqs, lens, enforce_sorted=False)
        output, (ht, ct) = self.lstm(padded_seqs)
        #output, out_lens = pad_packed_sequence(output)
        last_seq = output[lens - 1, torch.arange(output.shape[1])]
        return last_seq


class Tagger(Module):
    def __init__(self, embedding_vecs, lstm_hidden, out_dim, padding_idx=1, projected_dim=200, bidirectional=True, dropout=0):
        super().__init__()
        self.embedding = Embedding.from_pretrained(embedding_vecs, padding_idx=padding_idx, freeze=True)
        self.projection = Linear(embedding_vecs.shape[1], projected_dim)
        self.lstm2out = Linear(lstm_hidden * 2 if bidirectional else lstm_hidden, out_dim)
        self.lstm = LSTMAcceptor(projected_dim, lstm_hidden, bidirectional=bidirectional, dropout=dropout)

    def forward(self, data):
        sentences, lengths = data
        embedded = self.embedding(sentences)
        projected = self.projection(embedded)
        output = self.lstm(projected, lengths)
        return self.lstm2out(output)



if __name__ == "__main__":
    torch.manual_seed(1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ds = RexDataset('../dataset')
    model = Tagger(ds.fields['text'].vocab.vectors, 200, len(ds.fields['label'].vocab)).to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    loss_func = CrossEntropyLoss()
    train(model, loss_func, 10, optimizer, ds.train(30, device), ds.dev(30, device))
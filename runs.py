from torch.nn import Parameter, Module, Embedding, LSTM, Linear, CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam
from torch import nn, Tensor
import torch

from models.utils import train
from dataset.dataset import RexDataset
from models.bilstm import Tagger

if __name__ == "__main__":
    torch.manual_seed(1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ds = RexDataset('dataset')
    model = Tagger(ds.fields['text'].vocab.vectors, 200, len(ds.fields['label'].vocab)).to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    loss_func = CrossEntropyLoss()
    train(model, loss_func, 10, optimizer, ds.train(30, device), ds.dev(30, device))
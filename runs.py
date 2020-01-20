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
    model = Tagger(ds.fields['text'].vocab.vectors, 200, len(ds.fields['label'].vocab) - 2, dropout=0.2).to(device)
    optimizer = Adam(model.parameters(), lr=0.0001)
    #loss_func = CrossEntropyLoss(torch.FloatTensor([0.04, 0.24, 0.24, 0.24, 0.24]).cuda())
    loss_func = CrossEntropyLoss()
    train(model, loss_func, 1000, optimizer, ds.train(10, device), ds.dev(100, device))
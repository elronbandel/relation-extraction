from torchtext.datasets import SequenceTaggingDataset
from torchtext.data import Field, BucketIterator
from torchtext import data, datasets, vocab


class RexDataset():
    def __init__(self, path='', glove_name='6B', glove_dim=300):
        fields = [
            ('text', Field(include_lengths=True, sequential=True)),
            ('cand1', Field( dtype=int)),
            ('cand2', Field(dtype=int)),
            ('label', Field(is_target=True))
        ]
        self.train_set, self.dev_set = SequenceTaggingDataset.splits(path=path, train='train.tsv', validation='dev.tsv',fields = fields)
        self.fields = dict(fields)
        self.fields['text'].build_vocab(self.train_set, self.dev_set, vectors=vocab.GloVe(name=glove_name, dim=glove_dim))
        self.fields['cand1'].build_vocab(self.train_set)
        self.fields['cand2'].build_vocab(self.train_set)
        self.fields['label'].build_vocab(self.train_set)


    def train(self, batch_size, device):
        return BucketIterator(self.train_set, batch_size=batch_size, device=device)

    def dev(self, batch_size, device):
        return BucketIterator(self.dev_set, batch_size=batch_size, device=device)





if __name__ == "__main__":
    ds = RexDataset()
    batch = next(iter(ds.train(2, 'cuda:0')))
    ((sent, len), cand1, cand2), target = batch
    print(sent)


from datatools.load import read_csv_to_dicts, data_annot_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

class DataSet:
    def __init__(self, dir):
        self.dir = dir
        self.train_data, self.train_labels, self.train_annot_data = self.load('train.csv')
        self.dev_data, self.dev_labels, self.dev_annot_data = self.load('dev.csv')
        self.data_vec, self.labels_vec = DictVectorizer(), LabelEncoder()
        self.train = self.data_vec.fit_transform(self.train_data), self.labels_vec.fit_transform(self.train_labels)
        self.dev = self.data_vec.transform(self.dev_data), self.labels_vec.transform(self.dev_labels)

    def load(self, file):
        return data_annot_split(read_csv_to_dicts(self.dir + '/' + file))

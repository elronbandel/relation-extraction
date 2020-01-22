import csv
def read_csv_to_dicts(file_name):
    with open(file_name, mode='r') as csv_file:
        csv_reader = [{k: v for k, v in row.items()} for row in csv.DictReader(csv_file)]
    return csv_reader

def data_label_split(dicts):
    data, labels = [], []
    for dic in dicts:
        dic_data = dic.copy()
        del dic_data['label']
        data.append(dic_data), labels.append(dic['label'])
    return data, labels


def dicts_to_vectors(list_dicts):
    from sklearn.feature_extraction import DictVectorizer
    vec = DictVectorizer()
    vec.fit_transform(list_dicts)
    return vec
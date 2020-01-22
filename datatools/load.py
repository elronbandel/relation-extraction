import csv

def read_csv_to_dicts(file_name):
    with open(file_name, mode='r') as csv_file:
        csv_reader = [{k: v for k, v in row.items()} for row in csv.DictReader(csv_file)]
    return csv_reader

def data_annot_split(dicts):
    data, label ,data_for_annot_file = [], [], []
    for dic in dicts:
        annot_sent = {}
        dic_data = dic.copy()
        del dic_data['label']
        del dic_data['id']
        del dic_data['ent1']
        del dic_data['ent2']
        del dic_data['sent']

        annot_sent['id'] = dic['id']
        annot_sent['ent1'] = dic['ent1']
        annot_sent['ent2'] = dic['ent2']
        annot_sent['sent'] = dic['sent']

        data.append(dic_data), label.append(dic['label']), data_for_annot_file.append(annot_sent)
    print(label)
    return data, label, data_for_annot_file

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
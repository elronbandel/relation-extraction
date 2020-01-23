from operator import itemgetter

import spacy
from collections import defaultdict, Counter
import extract as ey
import csv

# loading the corpus itself:
def load_corpus(path):
    sentences = dict()
    with open(path) as corpus:
        for line in corpus:
            tokens = line.split()
            sentences[int(tokens[0][4:])] = ' '.join(tokens[1:])
    return sentences

#loadig the annotation of the corpus
def load_annotations(path):
    annots = defaultdict(list)
    with open(path) as data:
        for line in data:
            tokens = line.split('\t')
            annots[int(tokens[0][4:])].append(tokens[1:])
    return annots

def ent(sentences, nlp):
    vocab = set()
    for sentence in sentences:
        for token in nlp(sentence):
            vocab.add(token.text)
    return vocab

def corpus_entetities(corpus, nlp):
    ents = set()
    for id, sent in corpus.items():
        for ent in nlp(sent).ents:
            ents.add(ent.text)
    return ents

def anotations_entitites(anotations):
    ents = set()
    for id, anots in anotations.items():
        for anot in anots:
            ents.add(anot[0]), ents.add(anot[2])
    return ents

relsyms = {'Work_For':'WRK', 'Live_In':'LIV'}

def normalize_entitiy(raw_ent, sent, nlp):
    ents = set(map(str, nlp(sent).ents))
    if raw_ent in ents:
        return raw_ent
    for ent in ents:
        ent = str(ent)
        if ent in raw_ent or raw_ent in ent:
            print(raw_ent + '->' + ent)
            return ent
    return raw_ent



def process_one_anotation(anot, nlp):
    arg1, rel, arg2, sent = anot
    sent = sent[1:-1].replace("-LRB-", "(").replace("-RRB-", ")")
    arg1 = normalize_entitiy(arg1, sent, nlp)
    arg2 = normalize_entitiy(arg2, sent, nlp)
    rel = relsyms[rel]
    return (arg1, rel, arg2, sent)


def process_annotations(annotations, nlp):
    res = defaultdict(list)
    for id, anots in annotations.items():
        for anot in anots:
            if anot[1] in relsyms:
                res[id].append(process_one_anotation(anot, nlp))
    return res

def extract_sentence(sentence, annotations, nlp):
    processed = nlp(sentence)
    tokenized = [token.text for token in processed]

from itertools import combinations

def annotation_record(sentence, candidate1,  candidate2, label):
    return "\n".join([token if i !=0 else '\t'.join([token, str(candidate1), str(candidate2), label]) for i, token in enumerate(sentence)])

good_annot = ["Live_In", "Work_For" ]


def add_sent_data_to_dict_features(id, sent, ent1, ent2, feature_dict):
    feature_dict['id'] = id
    feature_dict['ent1'] = ent1
    feature_dict['ent2'] = ent2
    feature_dict['sent'] = sent
    # print(feature_dict)
    return feature_dict


# generate sentences with annotations and direction and entities
def generate_data_extract_feature(corpus, annots, nlp):
    data = []
    possible_types = set()
    for id, sent in corpus.items():
        sent_split = nlp(sent)
        ents = list(filter(lambda tok: tok.ent_type_ != '', sent_split))
        for ent1, ent2 in combinations(ents, 2):
            feature_dict_12 = ey.extract_features(ent1, ent2, sent_split)
            feature_dict_21 = ey.extract_features(ent2, ent1, sent_split)
            feature_dict_12 = add_sent_data_to_dict_features(id, sent, ent1, ent2, feature_dict_12)
            feature_dict_21 = add_sent_data_to_dict_features(id, sent, ent2, ent1, feature_dict_21)
            # print(feature_dict_12)
            for annot in annots[id]:
                if annot[0] == ent1.text and annot[2] == ent2.text:
                    feature_dict_12['label'] = annot[1]
                    possible_types.add(feature_dict_12['concatenated-types'])
                    break
                if annot[0] == ent2.text and annot[2] == ent1.text:
                    feature_dict_21['label'] = annot[1]
                    possible_types.add(feature_dict_21['concatenated-types'])
                    break
            data.append(feature_dict_12)
            data.append(feature_dict_21)
    print('possible-ent-combinations:' + str(possible_types))
    return list(filter(lambda x: x['concatenated-types'] in possible_types or x['concatenated-types'] in possible_train or x['concatenated-types'] in possible_dev, data))

possible_train = {'GPEGPE', 'PERSONORG', 'PERSONGPE', 'PERSONFAC', 'ORGFAC', 'ORGORG', 'PERSONNORP', 'DATEORG', 'GPENORP', 'ORGGPE', 'ORGNORP', 'PERSONPERSON'}
possible_dev = {'PERSONFAC', 'PERSONORG', 'PERSONGPE', 'ORGFAC', 'ORGORG', 'PERSONNORP', 'PERSONLOC', 'WORK_OF_ARTORG', 'ORGGPE', 'ORGLOC', 'PERSONPERSON'}

def data_stats(data):
    print(Counter(map(itemgetter('label'), data)))


def make_csv(section, nlp):
    corpus = load_corpus(f'data/Corpus.{section}.txt')
    annotations = load_annotations(f'data/{section}.annotations')
    processed = process_annotations(annotations, nlp)
    data = generate_data_extract_feature(corpus, processed, nlp)
    data_stats(data)
    write_dictionary_to_csv_file(data, section)
    # open(f'{section.lower()}.tsv', 'w+').write("\n\n".join(data))


def write_dictionary_to_csv_file(dicts, section):
    fields = list(sum((Counter(dic.keys()) for dic in dicts), Counter()).keys())
    with open(section.lower() + '.csv', 'w+') as file:
        writer = csv.DictWriter(open(section.lower() + '.csv', 'w+'), fieldnames=fields)
        writer.writeheader()
        writer.writerows(dicts)






if __name__ == "__main__":
    nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe(nlp.create_pipe('merge_entities'))
    make_csv('TRAIN', nlp)
    make_csv('DEV', nlp)




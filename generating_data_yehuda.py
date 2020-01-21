import spacy
import difflib
from collections import defaultdict
import extract_yehuda as ey

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

# generate sentences with annotations and direction and entities
def generate_directed_data(corpus, annots, nlp):
    data = []
    for id, sent in corpus.items():
        text, ents = [], []
        for i, token in enumerate(nlp(sent)):
            text.append(token.text)
            if not token.ent_type_ == '':
                ents.append((token.text, i))
        for (arg1, i1), (arg2, i2) in combinations(ents, 2):
            for annot in annots[id]:
                if arg1 == annot[0] and arg2 == annot[2]:
                    rel = f'{annot[1]}->'
                elif arg2 == annot[0] and arg1 == annot[2]:
                    rel = f'{annot[1]}<-'
                else:
                    rel = 'NON'
                data.append(annotation_record(text, i1, i2, rel))
    return data

good_annot = ["Live_In", "Work_For" ]

# generate sentences with annotations and direction and entities
def generate_directed_data_with_extract_feature(corpus, annots, nlp):
    data = []
    for id, sent in corpus.items():
        text, ents = [], []
        sent_split = nlp(sent)
        for i, token in enumerate(sent_split):
            text.append(token.text)
            if not token.ent_type_ == '':
                ents.append((token, i))
        for (arg1, i1), (arg2, i2) in combinations(ents, 2):
            for annot in annots[id]:
                feature_dict = extract_feature(arg1,arg2, sent_split)
                # head_ent1, head_ent2, ent_ner_tag1, ent_ner_tag2
                print(feature_dict)
                # if annot[1] != "Live_In" or annot[1] != "Work_For":
                #     relation = 'NON'

                # if arg1 == annot[0] and arg2 == annot[2]:
                #     rel = f'{annot[1]}->'
                # elif arg2 == annot[0] and arg1 == annot[2]:
                #     rel = f'{annot[1]}<-'
                # else:
                #     rel = 'NON'
                # data.append(annotation_record(text, i1, i2, relation))
    return data


def extract_feature(annot, arg1, i1, arg2, i2):

    if annot[1] != "Live_In" or annot[1] != "Work_For":
        relation = 'NON'






def generate_data(corpus, annots, nlp):
    data = []
    for id, sent in corpus.items():
        labels = set()
        for annot in annots[id]:
            labels.add(annot[1])
        label = ''.join((sorted(list(labels)))) if len(labels) > 1 else 'NON'
        data.append("\n".join(word.text if i else '\t'.join([word.text, label])for i, word in enumerate(nlp(sent))))
    return data


def make_tsv(section, nlp):
    corpus = load_corpus(f'data/Corpus.{section}.txt')
    annotations = load_annotations(f'data/{section}.annotations')
    processed = process_annotations(annotations, nlp)
    data = generate_directed_data_with_extract_feature(corpus, processed, nlp)
    open(f'{section.lower()}.tsv', 'w+').write("\n\n".join(data))


def check_intersection(corpus, annots):
    corp_ents = corpus_entetities(corpus, nlp)
    anot_ents = anotations_entitites(annots)
    print(corp_ents)
    print(anot_ents)
    inter = anot_ents.intersection(corp_ents)
    print(len(inter) / len(anot_ents))
    failed = anot_ents.difference(corp_ents)
    print(sorted(list(failed)))
    print(len(failed))




if __name__ == "__main__":
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe(nlp.create_pipe('merge_entities'))
    make_tsv('TRAIN', nlp)
    make_tsv('DEV', nlp)




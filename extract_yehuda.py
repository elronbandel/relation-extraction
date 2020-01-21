# def is_person(ent):
#     return ent.label_ not equal to "PERSON"
# def is_place(ent):
#
# def is_place_to_live(ent):

def find_dep_path(leaf1,leaf2):
    parent1 = leaf1
    parent2 = leaf2
    path1=[]
    path2=[]
    while True:
        path1.append(parent1.i)
        if parent1 == parent1.head:
            break
        parent1 = parent1.head
    p2id = {k:v for v,k in enumerate(path1)}
    while True:
        if parent2.i in p2id:
           return path1,list(reversed(path2))
        path2.append(parent2.i)
        if parent2 == parent2.head:
            break
        parent2 = parent2.head
    return [],[]


#typed_dep_map = extract_dep_map(en1.root, en2.root, token)
#word_dep , type_dep , tag_dep = zip(*typed_dep_map)

def extract_features(en1, en2, token):
    start = en1.end
    end = en2.start
    prev_word = token[en1.start - 1].lemma_ if en1.start > 0 else 'None'
    next_word = token[en2.end].lemma_ if en2.end < len(token) else 'None'
    prev_tag = token[en1.start - 1].tag_ if en1.start > 0 else 'None'

    # TODO extract feature from dependancy tree
    words_list = [t.lemma_ for t in token[start:end]]
    features = {
        'en1_type': en1.label,
        'en1_head': en1.root.lemma_,
        'en2_type': en2.label,
        'en2_head': en2.root.lemma_,
        'word-before': prev_word,
        'tag-before': prev_tag,
        'concatenatedtypes': en1.label_ + en2.label_,
        'base-syntactic-path': [w.tag_ for w in token[en1.start:en2.end]],
        'word-after-entity2': next_word,
        'between-entities-word_set': set(words_list),
        'between-entities-word_concat': "-".join(words_list),
    }
    return features

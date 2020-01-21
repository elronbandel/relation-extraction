# def is_person(ent):
#     return ent.label_ not equal to "PERSON"
# def is_place(ent):
#
# def is_place_to_live(ent):

def find_dep_path_directional(right_token, left_token, tokens):
    right_path, left_path= [], []


    return right_path, left_path, head


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


def print_path(en1, en2):
    up_p, down_p = find_dep_path(en1.root, en2.root)
    print("up_p:", up_p)
    print("down_p", down_p)


def extract_features(en1, en2, tokens):
    # start = en1.end
    # end = en2.start
    # prev_word = tokens[en1.start - 1].lemma_ if en1.start > 0 else 'None'
    # next_word = tokens[en2.end].lemma_ if en2.end < len(tokens) else 'None'
    # prev_tag = tokens[en1.start - 1].tag_ if en1.start > 0 else 'None'

    # words_list = [t.lemma_ for t in tokens[start:end]]
    # TODO extract feature from dependancy tree


    features = {
        'en1_type': en1.ent_type_,

        # 'en1_head': en1.root.lemma_,
        'en2_type': en2.ent_type_,
        # 'en2_head': en2.root.lemma_,
        'concatenatedtypes': en1.ent_type_ + en2.ent_type_,


        # 'word-after-entity2': next_word,
        # 'between-entities-word_set': set(words_list),
        # 'between-entities-word_concat': "-".join(words_list),
    }

    return features
def extract_dependency_features():

     dependency_features = {



     }




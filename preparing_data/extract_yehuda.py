# def is_person(ent):
#     return ent.label_ not equal to "PERSON"
# def is_place(ent):
#
# def is_place_to_live(ent):
from collections import Counter
from operator import attrgetter



def extract_features(en1, en2, tokens):
    features = dict()
    dependency = find_dependecny(en1, en2)
    if dependency:
        features = extract_features_from_dependency(dependency)
    features.update({
        'label': 'NON',
        'ent1-type': en1.ent_type_,
        'ent2-type': en2.ent_type_,
        'concatenated-types': en1.ent_type_ + en2.ent_type_,
        'ent1-dep': en1.dep_,
        'ent2-dep': en2.dep_,
    })
    return features


def find_dependecny(right_token, left_token):
    right_path, left_path = [right_token] + list(right_token.ancestors), [left_token] + list(left_token.ancestors)
    intersect = set(right_path).intersection(set(left_path))
    if not intersect:
        return None
    head = right_path[min(map(right_path.index, intersect))]
    right_path = right_path[:right_path.index(head)]
    left_path = left_path[:left_path.index(head)]
    return right_path, left_path, head


def extract_features_from_dependency(dep):
    right_path, left_path, head = dep
    features = {
        'head-tag': head.pos_,
        'dep-dist': len(right_path) + len(left_path) + 1,
        'dep-left-dist': len(left_path),
        'dep-right-dist': len(right_path),
    }
    count_dep_pos_right = Counter(map(attrgetter('tag_'), right_path))
    count_dep_pos_left = Counter(map(attrgetter('tag_'), left_path))
    count_dep_pos = count_dep_pos_left + count_dep_pos_right
    features.update({f'dep-right-{key.lower()}-tags': val for key, val in count_dep_pos_right.items()})
    features.update({f'dep-left-{key.lower()}-tags': val for key, val in count_dep_pos_left.items()})
    count_dep_deps_right = Counter(map(attrgetter('dep_'), right_path))
    count_dep_deps_left = Counter(map(attrgetter('dep_'), left_path))
    features.update({f'dep-right-{key.lower()}-deps': val for key, val in count_dep_deps_right.items()})
    features.update({f'dep-left-{key.lower()}-deps': val for key, val in count_dep_deps_left.items()})

    return features


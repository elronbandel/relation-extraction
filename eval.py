import sys
from collections import defaultdict
from datatools.generating_data import load_annotations

matched_list = []
def is_matched_relation(gold_annot, pred_annot, is_true_positive):
    is_matched = gold_annot[1] == pred_annot[1] and \
              (gold_annot[0] in pred_annot[0] or pred_annot[0] in gold_annot[0]) and \
              (gold_annot[2] in pred_annot[2] or pred_annot[2] in gold_annot[2])
    if is_matched and is_true_positive:
        s = "relation:{}=={} ent1:{}=={} ent2:{}=={}".format(gold_annot[1], pred_annot[1], gold_annot[0], pred_annot[0] , gold_annot[2], pred_annot[2])
        matched_list.append(s)
        # print(s)

    return is_matched


def is_live_in_or_work_relation(relation):
    # print(relation, relation == 'Live_In' or relation == 'Work_For')
    return relation == 'Live_In' or relation == 'Work_For'


# def print_good_relation_to_file(correct_relations_sents):
#     with open("good_relation.txt", 'w') as f:
#         for annot in correct_relations_sents:


def eval_func():
    gold = load_annotations(sys.argv[1])
    pred = load_annotations(sys.argv[2])
    # relation_possible = {'Work_For': 'Work_For', 'Live_In': 'Live_In'}
    true_positive = 0
    true_positive_flag = false_negative_flag = False
    false_negative = false_positive = 0
    perfect = 0
    for sent_id in gold:
        for annot in gold[sent_id]:
            if is_live_in_or_work_relation(annot[1]):
                perfect += 1
                for found in pred[sent_id]:
                    if is_matched_relation(annot, found, True):
                        true_positive += 1
                        true_positive_flag = True
                        break
                if not true_positive_flag:
                    false_positive += 1
                true_positive_flag = False
            else:
                gold[sent_id].remove(annot)

        for found in pred[sent_id]:
            for correct in gold[sent_id]:
                if is_live_in_or_work_relation(correct[1]) and is_matched_relation(correct, found, False):
                    if correct == found:
                        false_negative_flag = True
                        break
            if not false_negative_flag:
                false_negative += 1
            false_negative_flag = False

    percision = true_positive / float(true_positive + false_positive)
    recall = true_positive / float(true_positive + false_negative)

    print('number of perfect recognition - work_for and live_in:', perfect)
    print('flase negative: ', false_negative)
    print('false positive:', false_positive)
    print('true positive: ', true_positive)
    print('recall', recall)
    print('precision', percision)

    f1 = 2 * ((recall * percision) / (recall + percision))
    print('F1 score: {}'.format(f1))
    with open("good_pred_compare.txt", 'w') as f:
        f.write('\n'.join(matched_list))

if __name__ == "__main__":
    eval_func()
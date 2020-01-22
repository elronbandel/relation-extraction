from dataset.dataset import DataSet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# Load numpy
import numpy as np

# Set random seed
#np.random.seed(3)
good_annot = ["Live_In", "NONE", "Work_For" ]

def print_annot_file(annot_sent_dic, label):
    line = 'sent' + str(annot_sent_dic['id']) + '\t' + str(annot_sent_dic['ent1']) + '\t' + good_annot[label] + '\t' \
           + str(annot_sent_dic['ent2']) + '\t' + '( ' + str(annot_sent_dic['sent']) + ' )'+'\n'
    return line



if __name__ == "__main__":
    ds = DataSet('dataset')
    # Create a random forest Classifier. By convention, clf means 'Classifier'
    clf = RandomForestClassifier(random_state=3, class_weight={0:4, 1:1, 2:4}, max_depth=12, )
    #clf = LogisticRegression(random_state=0, class_weight={0:4, 1:1, 2:4})


    # Train the Classifier to take the training features and learn how they relate
    # to the training y (the species)
    X, y = ds.train
    clf.fit(X, y)
    X_dev, y_dev = ds.dev
    annot_data_dev = ds.dev_annot_data
    y_pred = clf.predict(X_dev)
    from sklearn import metrics
    print('f1-WRK-LIV:', metrics.f1_score(y_dev, y_pred, labels=[0, 2], average='weighted'))
    print('f1-NON:', metrics.f1_score(y_dev, y_pred, labels=[1], average='weighted'))
    with open('annot_pred.txt', 'w') as f:
        for i, pred in enumerate(y_pred):
            # print(pred)
            if pred != 1: #NON
                line = print_annot_file(annot_data_dev[i], pred)
                f.write(line)


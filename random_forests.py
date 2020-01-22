from dataset.dataset import DataSet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# Load numpy
import numpy as np

# Set random seed
#np.random.seed(3)

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
    y_pred = clf.predict(X_dev)
    from sklearn import metrics
    print('f1-WRK-LIV:', metrics.f1_score(y_dev, y_pred, labels=[0, 2], average='weighted'))
    print('f1-NON:', metrics.f1_score(y_dev, y_pred, labels=[1], average='weighted'))


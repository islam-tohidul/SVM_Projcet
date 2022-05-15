# Import libraries
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

# load the cancer dataset
cancer = datasets.load_breast_cancer()

print(cancer.feature_names)

print(cancer.target_names)

# take features and target
X = cancer.data
y = cancer.target

# split into train and test
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

# making classifier
clf = svm.SVC(kernel='linear', C=1)

# train the classifier
clf.fit(X_train, y_train)

# let's predict values
predictions = clf.predict(X_test)

# testing classifier accuracy
acc = metrics.accuracy_score(y_test, predictions)
print(acc)



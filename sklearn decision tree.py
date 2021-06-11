import sklearn
from sklearn import datasets, tree

from sklearn.model_selection import cross_val_score
# Load iris dataset
iris = datasets.load_iris()

x=iris.data
y=iris.target

clf = tree.DecisionTreeClassifier()
scores = cross_val_score(clf, x, y, cv=5)

print(len(y),y)
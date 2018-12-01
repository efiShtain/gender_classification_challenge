from sklearn import tree
from sklearn.neural_network import MLPClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import numpy as np


# CHALLENGE - create 3 more classifiers...
classfiers = {
 'decisionTree':tree.DecisionTreeClassifier(),
 'mlp':MLPClassifier(),
 'randomForest':RandomForestClassifier(),
}


# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
for name, clf in classfiers.items():
     clf.fit(X,Y)


currentMaxAccuracy = 0
mostAccurateClassifier = ''
for name, clf in classfiers.items():
     result = clf.predict([[190, 70, 43]])
     resultOrigin = clf.predict(X)
     accuracy = accuracy_score(Y,resultOrigin)
     if(accuracy>currentMaxAccuracy):
          currentMaxAccuracy = accuracy
          mostAccurateClassifier = name
     
     print(name, result[0])

print('most accurate classifier:' ,mostAccurateClassifier,currentMaxAccuracy)


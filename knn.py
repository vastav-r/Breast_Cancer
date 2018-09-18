import pandas as pd
from sklearn import neighbors
import numpy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

masses_data = pd.read_csv('mammographic_masses.data.txt', na_values=['?'], names = ['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity'])
masses_data.dropna(inplace=True)
all_features = masses_data[['age', 'shape',
                             'margin', 'density']].values


all_classes = masses_data['severity'].values

feature_names = ['age', 'shape', 'margin', 'density']
numpy.random.seed(1234)

(x_train,x_test,y_train,y_test) = train_test_split(all_features, all_classes, test_size=0.25, train_size=0.75, random_state=1)


clf = neighbors.KNeighborsClassifier(n_neighbors=10)
cv_scores = cross_val_score(clf, all_features, all_classes, cv=10)

print(cv_scores.mean()*100)

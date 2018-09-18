from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import cross_val_score

masses_data = pd.read_csv('mammographic_masses.data.txt', na_values=['?'], names = ['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity'])

masses_data.dropna(inplace=True)
all_features = masses_data[['age', 'shape','margin', 'density']].values


all_classes = masses_data['severity'].values

feature_names = ['age', 'shape', 'margin', 'density']

numpy.random.seed(1234)

(training_inputs,
 testing_inputs,
 training_classes,
 testing_classes) = train_test_split(all_features, all_classes, test_size=0.25, train_size=0.75, random_state=1)

clf= DecisionTreeClassifier(random_state=1)


clf.fit(training_inputs, training_classes)

clf = RandomForestClassifier(n_estimators=10, random_state=1)
cv_scores = cross_val_score(clf, all_features, all_classes, cv=10)

print(cv_scores.mean()*100)


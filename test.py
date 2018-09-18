import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn import tree
from pydotplus import graph_from_dot_data
import pylab

masses_data = pd.read_csv('mammographic_masses.data.txt', na_values=['?'], names = ['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity'])

masses_data.dropna(inplace=True)

all_features = masses_data[['age', 'shape','margin', 'density']].values
all_classes = masses_data['severity'].values

feature_names = ['age', 'shape', 'margin', 'density']

#scaler = preprocessing.StandardScaler()
#all_features_scaled = scaler.fit_transform(all_features)

np.random.seed(1234)
#(x_train, x_test, y_train, y_test) = train_test_split(all_features, all_classes, train_size=0.75, test_size=0.25, random_state=1)

clf= DecisionTreeClassifier(random_state=1)

#clf.fit(all_features_scaled,all_classes)
clf.fit(all_features, all_classes)

print('Age:')
age=input()
print('Shape:')
shape=input()
print('Margin:')
margin=input()
print('Density:')
density=input()

x_test=[[age,shape,margin,density]]
#x_test.reshape(1, -1)
y_pred=clf.predict(x_test)
if(y_pred==1):
    print('\nMalignant')
else:
    print('\nBenign')

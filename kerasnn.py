import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import preprocessing


masses_data = pd.read_csv('mammographic_masses.data.txt', na_values=['?'], names = ['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity'])
masses_data.dropna(inplace=True)
all_features = masses_data[['age', 'shape','margin', 'density']].values


all_classes = masses_data['severity'].values

feature_names = ['age', 'shape', 'margin', 'density']

numpy.random.seed(1234)

scaler = preprocessing.StandardScaler()
all_features_scaled = scaler.fit_transform(all_features)

(x_train,x_test,y_train,y_test) = train_test_split(all_features_scaled, all_classes, test_size=0.25, train_size=0.75, random_state=1)

model = Sequential()
model.add(Dense(6, input_dim=4, kernel_initializer='normal', activation='relu'))    
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    

model.fit(x_train, y_train,batch_size=150,epochs=250,verbose=2,validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test,batch_size=10,verbose=2)
print('\n\n\nTest accuracy:', acc*100)

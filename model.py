import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv(r'D:\Datasets\winequality-red.csv')
#print(data.head())

threshold = 5
data['quality'] = np.where(data['quality']>threshold,1,0)
#print(data.quality.value_counts())

x = data.drop('quality',axis=1)
y = data['quality']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=75)

from lazypredict.Supervised import LazyClassifier
lpc = LazyClassifier()
models,predictions = lpc.fit(x_train,x_test,y_train,y_test)

print(models)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)

y_pred = rfc.predict(x_test)

from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred))

import pickle

pickle.dump(rfc,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(rfc.predict([[11.20,0.28,0.56,1.90,0.07,17.00,60.00,1.00,3.16,0.58,9.80]]))
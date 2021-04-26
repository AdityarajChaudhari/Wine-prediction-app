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

print("Without Hyperparamter Tuning :- ")

from sklearn import metrics
print("Accuracy Score :- ",metrics.accuracy_score(y_test,y_pred))
print("Confusion Matrix :- ",metrics.confusion_matrix(y_test,y_pred))
print("Classification Report :- ",metrics.classification_report(y_test,y_pred))

from sklearn.model_selection import RandomizedSearchCV
n_estimators = [x for x in range(10,3000,100)]
criterion = ['gini','entropy']
max_features = ['auto','log2',None]
min_samples_leaf = [int(x) for x in range(1,100,1)]
min_samples_split = [x for x in range(2,100,1)]
max_depth = [x for x in range(1,53,1)]
param_grid = {
    'n_estimators' : n_estimators,
    'criterion' : criterion,
    'max_features' : max_features,
    'min_samples_leaf' : min_samples_leaf,
    'min_samples_split' : min_samples_split,
    'max_depth' : max_depth
}

rf_randomcv = RandomizedSearchCV(estimator=rfc,param_distributions=param_grid,n_iter=300,cv=5,n_jobs=-1,random_state=100,verbose=True)
rf_randomcv.fit(x_train,y_train)

rf_best=rf_randomcv.best_estimator_
print(rf_best)
y_pre = rf_best.predict(x_test)
#from sklearn import metrics
print("After Hyperparameter Tuning (Randomized Search CV)")
print("Accuracy Score :- ",metrics.accuracy_score(y_test,y_pre))
print("Confusion Matrix :- ",metrics.confusion_matrix(y_test,y_pre))
print("Classification Report :- ",metrics.classification_report(y_test,y_pre))

from sklearn.model_selection import GridSearchCV

criterion = [rf_randomcv.best_params_['criterion']]
max_features = [rf_randomcv.best_params_['max_features']]
max_depth = [rf_randomcv.best_params_['max_depth']-3,rf_randomcv.best_params_['max_depth']-2,rf_randomcv.best_params_['max_depth']-1,
    rf_randomcv.best_params_['max_depth']+1,rf_randomcv.best_params_['max_depth']+2,rf_randomcv.best_params_['max_depth']+3]
min_samples_leaf = [rf_randomcv.best_params_['min_samples_leaf']-2,rf_randomcv.best_params_['min_samples_leaf']-1,
    rf_randomcv.best_params_['min_samples_leaf']+1,rf_randomcv.best_params_['min_samples_leaf']+2,rf_randomcv.best_params_['min_samples_leaf']+3]
n_estimators = [rf_randomcv.best_params_['n_estimators']-30,rf_randomcv.best_params_['n_estimators']-20,rf_randomcv.best_params_['n_estimators']-10,
                rf_randomcv.best_params_['n_estimators']+10,rf_randomcv.best_params_['n_estimators']+20,rf_randomcv.best_params_['n_estimators']+30]
min_samples_split = [rf_randomcv.best_params_['min_samples_split']-2,rf_randomcv.best_params_['min_samples_split']-1,rf_randomcv.best_params_['min_samples_split']+1,rf_randomcv.best_params_['min_samples_split']+2]

grid = {
    'n_estimators' : n_estimators,
    'criterion' : criterion,
    'max_features' : max_features,
    'min_samples_leaf' : min_samples_leaf,
    'min_samples_split' : min_samples_split,
    'max_depth' : max_depth
}

rf_gridsearch = GridSearchCV(estimator=rfc,param_grid=grid,cv=5,verbose=True,n_jobs=-1)
rf_gridsearch.fit(x_train,y_train)

rf_best_gridcv=rf_gridsearch.best_estimator_
print(rf_best_gridcv)
y_predict = rf_best_gridcv.predict(x_test)
#from sklearn import metrics
print("After Hyperparameter Tuning (Grid Search CV)")
print("Accuracy :- ",metrics.accuracy_score(y_test,y_predict))
print("Confusion Matrix :- ",metrics.confusion_matrix(y_test,y_predict))
print("Classification Report :- ",metrics.classification_report(y_test,y_predict))
import pickle

pickle.dump(rf_gridsearch,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(rfc.predict([[11.20,0.28,0.56,1.90,0.07,17.00,60.00,1.00,3.16,0.58,9.80]]))
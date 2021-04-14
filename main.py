import pandas as pd

import Utils
import HyperModel
from sklearn.metrics import confusion_matrix
path='../../DS/CICIDS/numeric/'
df=pd.read_csv(path+'Train_standard.csv')
label='Classification'
X,y =Utils.getXY(df,label)
df=pd.read_csv(path+'Test.csv')
print(X.shape)
X_test,y_test=Utils.getXY(df,label)
n_class=2
model, time =HyperModel.hypersearch(X,y, X_test,y_test, 'prova', path, n_class)
model.save(path+'NN.h5')
y_pred=model.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
import pandas as pd
import numpy as np
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
model, time =HyperModel.hypersearch(X,y, X_test,y_test, path, n_class)
model.save(path+'NN.h5')
Y_predicted=model.predict(X_test)
Y_predicted = np.argmax(Y_predicted, axis=1)
cm=confusion_matrix(y_test,Y_predicted)
print(cm)
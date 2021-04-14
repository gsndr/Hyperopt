from __future__ import print_function

import numpy as np

# random seeds must be set before importing keras & tensorflow
import Utils

my_seed = 123
np.random.seed(my_seed)
import random

random.seed(my_seed)
import tensorflow as tf


tf.random.set_seed(my_seed)
import csv
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin
import hyperopt

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.utils import np_utils

from keras import callbacks
from sklearn.metrics import confusion_matrix

from keras.optimizers import RMSprop, Adadelta, Adagrad, Nadam, Adam
import global_config
from sklearn.model_selection import train_test_split
import time
import keras.backend as K




SavedParameters = []

def NN(x_train, y_train, params):
    print(params)
    input_shape = (x_train.shape[1],)
    print(input_shape)
    input = Input(input_shape)

    l1 = Dense(params['neurons1'], activation='relu', kernel_initializer='glorot_uniform')(input)
    l1 = Dropout(params['dropout1'])(l1)

    # l1= BatchNormalization()(l1)

    l1 = Dense(params['neurons2'], activation='relu', kernel_initializer='glorot_uniform')(
        l1)
    l1 = Dropout(params['dropout2'])(l1)
    l1 = Dense(params['neurons3'], activation='relu', kernel_initializer='glorot_uniform')(
        l1)
    softmax = Dense(global_config.n_class, activation='softmax', kernel_initializer='glorot_uniform')(l1)

    adam = Adam(lr=params['learning_rate'])
    model = Model(inputs=input, outputs=softmax)
    #model.summary()
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=adam)

    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10,
                                restore_best_weights=True),
    ]

    XTraining, XValidation, YTraining, YValidation = train_test_split(x_train, y_train, stratify=y_train,
                                                                      test_size=0.2)  # before model building
    tic = time.time()
    h = model.fit(x_train, y_train,
                  batch_size=params["batch"],
                  epochs=5,
                  verbose=2,
                  callbacks=callbacks_list,
                  validation_data=(XValidation, YValidation))

    toc = time.time()
    time_tot=toc-tic
    scores = [h.history['val_loss'][epoch] for epoch in range(len(h.history['loss']))]
    score = min(scores)
    print(score)
    y_test = np.argmax(YValidation, axis=1)

    Y_predicted = model.predict(XValidation, verbose=0, use_multiprocessing=True, workers=12)

    Y_predicted = np.argmax(Y_predicted, axis=1)

    cf = confusion_matrix(y_test, Y_predicted)
    return model, h, {"val_loss": score , "time": time_tot,
                   "TP_val": cf[0][0],
                   "FN_val": cf[0][1], "FP_val": cf[1][0], "TN_val": cf[1][1]
                   }

def fit_and_score(params):

    global SavedParameters
    y_train = np_utils.to_categorical(global_config.train_Y, global_config.n_class)
    model, h, val = NN(global_config.train_X, y_train, params)
    print(val)

    print("start predict")

    Y_predicted = model.predict(global_config.test_X, verbose=0, use_multiprocessing=True, workers=12)
    Y_predicted = np.argmax(Y_predicted, axis=1)
    elapsed_time = val['time']
    cf = confusion_matrix(global_config.test_Y, Y_predicted)
    #print(cf)
    # print("test F1_score: " + str(f1_score(YTestGlobal, Y_predicted)))
    K.clear_session()
    SavedParameters.append(val)
    print(SavedParameters)
    global best_val_acc
    #global best_test_acc
    # print("val acc: " + str(val["F1_score_val"]))


    SavedParameters[-1].update(
        {
         "learning_rate": params["learning_rate"],
         "batch": params["batch"],
         "dropout1": params["dropout1"],
         "dropout2": params["dropout2"],
         "neurons_layer1": params["neurons1"],
         "neurons_layer2": params["neurons2"],
         "neurons_layer3": params["neurons3"],
         "time": time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
         })
    cm_val = [[SavedParameters[-1]["TP_val"], SavedParameters[-1]["FN_val"]],
          [SavedParameters[-1]["FP_val"], SavedParameters[-1]["TN_val"]]]

    r = Utils.getResult(cm_val, global_config.n_class)
    SavedParameters[-1].update({
        "OA_val": r[0],
        "P_val": r[2],
        "R_val": r[3],
        "F1_val": r[4],
        "FAR_val": r[5],
        "TPR_val": r[6]
    })
    SavedParameters[-1].update({
        "TP_test": cf[0][0],"FN_test": cf[0][1], "FP_test": cf[1][0], "TN_test": cf[1][1]
    })

    cm_test = [[SavedParameters[-1]["TP_test"], SavedParameters[-1]["FN_test"]],
               [SavedParameters[-1]["FP_test"], SavedParameters[-1]["TN_test"]]]
    r = Utils.getResult(cm_test, False)
    SavedParameters[-1].update({
        "OA_test": r[0],
        "P_test": r[2],
        "R_test": r[3],
        "F1_test": r[4],
        "FAR_test": r[5],
        "TPR_test": r[6]
    })
    # Save model
    if SavedParameters[-1]["F1_val"] > global_config.best_accuracy:
        print("new saved model:" + str(SavedParameters[-1]))
        global_config.best_model=model
        global_config.best_accuracy= SavedParameters[-1]["F1_val"]

    '''
    if SavedParameters[-1]["F1_test"] > best_test_acc:
        print("new saved model Test:" + str(SavedParameters[-1]))
        model.save(Name.replace(".csv", "_Test_model.h5"))
        best_test_acc = SavedParameters[-1]["F1_test"]
   '''
    SavedParameters = sorted(SavedParameters, key=lambda i: i['F1_val'], reverse=True)

    try:
        with open(global_config.test_path+'Results.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=SavedParameters[0].keys())
            writer.writeheader()
            writer.writerows(SavedParameters)
    except IOError:
        print("I/O error")
    return {'loss': -val["F1_val"], 'status': STATUS_OK}  # cambia


def reset_global_variables(train_X, train_Y, test_X, test_Y):
    global_config.train_X = train_X
    global_config.train_Y = train_Y
    global_config.test_X = test_X
    global_config.test_Y = test_Y

    global_config.best_score = 0
    global_config.best_scoreTest = 0
    global_config.best_accuracy=0
    global_config.best_model = None
    global_config.best_model_test = None
    global_config.best_time = 0





def hypersearch(train_X, train_Y, test_X, test_Y, modelName, testPath, n_class):
    reset_global_variables(train_X, train_Y, test_X, test_Y)
    global_config.n_class=n_class
    global_config.test_path = testPath
    bs = [32, 64, 128, 256, 512]
    space = {"batch": hp.choice("batch", [32, 64, 128, 256, 512]),
                                   'dropout1': hp.uniform("dropout1", 0, 1),
                                   'dropout2': hp.uniform("dropout2", 0, 1),
                                   "learning_rate": hp.uniform("learning_rate", 0.0001, 0.001),
                                   "neurons1" : hp.choice("neurons1",[32, 64, 128, 256, 512]),
                                   "neurons2": hp.choice("neurons2", [32, 64, 128, 256, 512]),
                                   "neurons3": hp.choice("neurons3", [32, 64, 128, 256, 512])}
    trials = Trials()
    best = fmin(fit_and_score, space, algo=tpe.suggest, max_evals=10, trials=trials,
                rstate=np.random.RandomState(my_seed))
    best_params = hyperopt.space_eval(space, best)


    return global_config.best_model, global_config.best_time

# -*- coding: utf-8 -*-
"""
Created on 2021

"""
 
from __future__ import print_function
from tensorflow import keras
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlwt

from sklearn.metrics import roc_auc_score

def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y
  
nb_epochs = 1000

#flist = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 
#'DiatomSizeReduction', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR', '50words', 'FISH', 'Gun_Point', 'Haptics', 
#'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MALLAT', 'MedicalImages', 'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 
#'NonInvasiveFatalECG_Thorax2', 'OliveOil', 'OSULeaf', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'SwedishLeaf', 'Symbols', 
#'synthetic_control', 'Trace', 'TwoLeadECG', 'Two_Patterns', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer', 'WordsSynonyms', 'yoga']

flist  = ['PD180']
for each in flist:
    fname = each
    x_train, y_train = readucr(fname+'/'+fname+'_TRAIN')
    x_test, y_test = readucr(fname+'/'+fname+'_TEST')
    nb_classes = len(np.unique(y_test))
    batch_size = 10 # min(x_train.shape[0]/10, 16)
    
    #y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
    #y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)
    
    Y_train = keras.utils.to_categorical(y_train, nb_classes)
    Y_test = keras.utils.to_categorical(y_test, nb_classes)
    
    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean)/(x_train_std)
     
    x_test = (x_test - x_train_mean)/(x_train_std)
    x_train = x_train.reshape(x_train.shape + (1,1,))
    x_test = x_test.reshape(x_test.shape + (1,1,))

    x = keras.layers.Input(x_train.shape[1:])
#    drop_out = Dropout(0.2)(x)
    conv1 = keras.layers.Conv2D(64, 8, 1, padding='same')(x)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation('relu')(conv1)
    
#    drop_out = Dropout(0.2)(conv1)
    conv2 = keras.layers.Conv2D(128, 5, 1, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)
    
#    drop_out = Dropout(0.2)(conv2)
    conv3 = keras.layers.Conv2D(64, 3, 1, padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)
    
    full = keras.layers.GlobalAveragePooling2D()(conv3)
    out = keras.layers.Dense(nb_classes, activation='softmax')(full)
    
    
    model = keras.models.Model(inputs=x, outputs=out)
     
    optimizer = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
     
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                      patience=50, min_lr=0.0001) 
    hist = model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
              verbose=1, 
              validation_data=(x_test, Y_test),
              callbacks = [reduce_lr])
    #Print the testing results which has the lowest training loss.
    log = pd.DataFrame(hist.history)
    print(model.summary())
    #flops = get_flops(model, batch_size=16)
    #print(f"FLOPS: {flops / 10 ** 9:.03} G")
    
    
    # 创建一个workbook 设置编码
    workbook = xlwt.Workbook(encoding = 'utf-8')
    # 创建一个worksheet
    worksheet = workbook.add_sheet('My Worksheet')
    print(Y_test.shape[0])
    num = Y_test.shape[0]
    y_pred = model.predict(x_test)
    
    auc_score = roc_auc_score(Y_test,y_pred)
    print(auc_score)
    
    for i in range(len(y_pred)):
      max_value=max(y_pred[i])
      for j in range(len(y_pred[i])):
        if max_value==y_pred[i][j]:
          y_pred[i][j]=1
        else:
          y_pred[i][j]=0
    print(classification_report(Y_test, y_pred))
    
    
    for i  in range(0,1):
        y_row=Y_test[i]
        p_row=y_pred[i]
        #print(y_row)
        #print(p_row) 
        print(np.argmax(y_row))
        print(np.argmax(p_row))
        worksheet.write(i,0,str(np.argmax(y_row)))
        worksheet.write(i,1,str(np.argmax(p_row)))
    # 保存
    workbook.save('test.xls')
    print(log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['val_acc'])
    
epochs=range(len(hist.history['acc']))
plt.figure()
plt.plot(epochs,hist.history['acc'],'b',label='Training acc')
plt.plot(epochs,hist.history['val_acc'],'r',label='Validation acc')
plt.title('Traing and Validation accuracy')
plt.legend()
plt.savefig('1_acc.jpg')

plt.figure()
plt.plot(epochs,hist.history['loss'],'b',label='Training loss')
plt.plot(epochs,hist.history['val_loss'],'r',label='Validation val_loss')
plt.title('Traing and Validation loss')
plt.legend()
plt.savefig('1_loss.jpg')
import numpy as np
import matplotlib.pyplot as plt
#import pandas
import math
import keras.callbacks
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM, Merge
from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.objectives import mean_absolute_percentage_error as MAPE
import scipy.io as scio
from keras.layers.normalization import BatchNormalization
# fix random seed for reproducibility
np.random.seed(7)

#alldata = scio.loadmat('dataset/trainAndtestData_norConf_czh3.mat')  
alldata = scio.loadmat('dataset/trainAndtestData.mat')   #creatData10_30_DLSTM
trainData = alldata['trainData']
testData = alldata['testData']
trainTarget = alldata['trainTarget']
testTarget = alldata['testTarget']

# calculate mean of root mean squared error 
def ME_get(truey,predy):
    temp = 0
    for i in range(len(truey[:,0])):
        temp = temp + math.sqrt(2*mean_squared_error(predy[i],truey[i]))
    mapescore = temp/len(truey[:,0])
    return mapescore

class EpochME(keras.callbacks.Callback):
    def __init__(self):
        self.mapes = []
        self.val_mapes = []
        self.predicted_y = []

    def on_epoch_end(self, epoch, logs={}):
        
                #lasses = model.predict_classes(X_test, batch_size=32)
        trainPredict = model.predict(trainData, batch_size=32)
        testPredict = model.predict(testData,batch_size=32)
            
        acc = ME_get(trainTarget,trainPredict)
        val_acc = ME_get(testTarget,testPredict)
                #print('acc:'+str(acc)+' - val_acc:'+str(val_acc))
        self.mapes.append(acc)
        self.val_mapes.append(val_acc)
        self.predicted_y.append(testPredict)


model = Sequential()
model.add(LSTM(30, input_dim= 9, activation='sigmoid', return_sequences=True))  #look_back
model.add(LSTM(40, activation='sigmoid', return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(60,activation='sigmoid'))
model.add(Dropout(0.5))

model.add(Dense(2))
model.compile(loss='mean_squared_error', optimizer='adam') #adam


epochmapes = EpochME()
history = model.fit(trainData, trainTarget, validation_data=(testData, testTarget), nb_epoch=60, batch_size=5, verbose=2, callbacks=[epochmapes]) 



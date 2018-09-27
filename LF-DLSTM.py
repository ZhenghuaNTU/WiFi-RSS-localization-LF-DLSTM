### at the begin
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#####

import numpy as np
import math
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, LSTM,Activation
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
import numpy as np
import scipy.io as scio
from sklearn import preprocessing
from keras.utils import np_utils
from sklearn.metrics import precision_score
import keras.callbacks
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

np.random.seed(7)

alldata = scio.loadmat('SimData1.mat')  
trainData = alldata['trainData']
testData = alldata['testData']
trainTarget = alldata['trainTarget']
testTarget = alldata['testTarget']

# calculate mean of root mean squared error 
def MAPE_get(truey,predy):
    temp = 0
    for i in range(len(truey[:,0])):
        temp = temp + math.sqrt(2*mean_squared_error(predy[i],truey[i]))
    mapescore = temp/len(truey[:,0])
    return mapescore

class EpochMAPE(keras.callbacks.Callback):
    def __init__(self):
        self.mapes = []
        self.val_mapes = []
        self.predicted_y = []

    def on_epoch_end(self, epoch, logs={}):
        
                #lasses = model.predict_classes(X_test, batch_size=32)
        trainPredict = model.predict(trainData, batch_size=32)
        testPredict = model.predict(testData,batch_size=32)
            
        acc = MAPE_get(trainTarget,trainPredict)
        val_acc = MAPE_get(testTarget,testPredict)
                #print('acc:'+str(acc)+' - val_acc:'+str(val_acc))
        self.mapes.append(acc)
        self.val_mapes.append(val_acc)
        self.predicted_y.append(testPredict)

hidDim = [40]
for i in range(len(hidDim)):
    model_input = Input(shape=(trainData.shape[1],trainData.shape[2]))
    lstm_1 = LSTM(30,activation='sigmoid',return_sequences=True)(model_input)
    lstm_2 = LSTM(hidDim[i],activation='sigmoid',return_sequences=False)(lstm_1)
    drop1 = Dropout(0.5)(lstm_2) 

    dense_1 = Dense(60,activation='sigmoid')(drop1)#activation='sigmoid')) # 100
    drop2 = Dropout(0.5)(dense_1) 
    model_outputs = Dense(2)(drop2)#activation='sigmoid'))
    model = Model(model_input, model_outputs)
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001,beta_1=0.9, beta_2=0.999, epsilon=1e-8))#'adam') #adam


    epochmapes = EpochMAPE()

    history = model.fit(trainData, trainTarget, validation_data=(testData, testTarget), nb_epoch=20, batch_size=5, verbose=2, callbacks=[epochmapes]) #0.33 validation_data=(testX, testY),

    #print 'rawQLDdata_W'
    print(epochmapes.val_mapes[-1])


# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:53:50 2019

@author: Sam
"""

#Additional Packages to Install
# !pip install pydrive
# !pip install oauth2client
# !pip install tensorboardcolab

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorboardcolab import *
from keras.layers import Dense,Conv1D,RepeatVector,TimeDistributed,Input
from keras.layers import Conv1D,MaxPooling1D,UpSampling1D
from keras.layers import Activation,BatchNormalization
from keras.models import Model,Sequential
from keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
from PreProcessingData import PreProcessingMusic
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

class AudioDenoising:
  
  def __init__(self,audio_filepath,length = 2**9):

  #Authenticate/Create PyDrive client 
    #auth.authenticate_user()
    #gauth = GoogleAuth()
    #gauth.credentials = GoogleCredentials.get_application_default()
    #drive = GoogleDrive(gauth)
  
    self.length = length
    proc = PreProcessingMusic(audio_filepath,length = self.length)
    self.clean_audio, self.noisy_audio = proc.run()
  
  
  def build_autoencoder(self):
    inputs = Input(shape = (self.length,1))
  
    encoded = Conv1D(32,3, padding = 'same')(inputs)
    encoded = BatchNormalization()(encoded)
    encoded = Activation('relu')(encoded)
    encoded = MaxPooling1D(2,padding = 'same')(encoded)
  
  
    encoded = Conv1D(16,3,padding = 'same')(encoded)
    encoded = Activation('relu')(encoded)
    encoded = MaxPooling1D(2,padding = 'same')(encoded)
  
    encoded = Conv1D(8,3, padding = 'same')(encoded)
    encoded = Activation('relu')(encoded)
    encoded = MaxPooling1D(2,padding = 'same')(encoded)
  
  
    decoded = Conv1D(8,3,padding = 'same')(encoded)
    decoded = Activation('relu')(decoded)
    decoded = UpSampling1D(2)(decoded)
  
    decoded = Conv1D(16,3, padding = 'same')(decoded)
    decoded = Activation('relu')(decoded)
    decoded = UpSampling1D(2)(decoded)
  
    decoded = Conv1D(32,3,padding = 'same')(decoded)
    decoded = Activation('relu')(decoded)
    decoded = UpSampling1D(2)(decoded)
  
    decoded = Conv1D(1,3,activation = 'sigmoid',padding = 'same')(decoded)
  
    self.autoencoder = Model(inputs,decoded)
    
    
  def trainNN(self,summary=False,tb=False):
    self.epochs = 1
    self.optimizer = RMSprop(lr=1e-3)
    self.batch_size = 32
    #tbc=TensorBoardColab()
    
    self.autoencoder.compile(optimizer = self.optimizer, loss = 'mse')
    
    if summary:
      self.autoencoder.summary()
  
    
      
    self.autoencoder.fit(self.noisy_audio,self.clean_audio,
                    epochs=self.epochs, 
                    verbose = 1)
                    #callbacks=[TensorBoardColabCallback(tbc)])
   
   
  def saveModel(self):
    
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)


    self.autoencoder.save('AudioDenoising.h5')    
    model_file = drive.CreateFile({'title' : 'AudioDenoising.h5'})
    model_file.SetContentFile('AudioDenoising.h5')
    model_file.Upload()

    self.autoencoder.save_weights('AudioDenoising_weights.h5')
    weights_file = drive.CreateFile({'title' : 'AudioDenoising_weights.h5'})
    weights_file.SetContentFile('AudioDenoising_weights.h5')
    weights_file.Upload()
    
    drive.CreateFile({'id': weights_file.get('id')})
    drive.CreateFile({'id': model_file.get('id')})
    
    print('Saved Model')

AE = AudioDenoising('trimed data.zip')
AE.build_autoencoder()
AE.trainNN()
AE.saveModel()




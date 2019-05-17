# -*- coding: utf-8 -*-
"""
Created on Thu May 16 23:22:09 2019

@author: Sam
"""

import numpy as np
from keras.models import load_model
import librosa
from sklearn.preprocessing import MinMaxScaler
from IPython.display import Audio
import matplotlib.pyplot as plt

# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from google.colab import auth
# from oauth2client.client import GoogleCredentials


class PostProcessing:
  
  def __init__(self,model_path,audio_path,save_path,length = 2**9):
    
      self.autoencoder = load_model(model_path)
      self.save_path = save_path
      self.audio, self.sr = librosa.load(audio_path)
      plt.plot(self.audio)
      self.orig_length = self.audio.shape
      self.length = length
  
  def padData(self):
    excess = self.audio.size%self.length
    self.padLength = self.length-excess
    zeros = np.zeros((1,self.padLength))
    self.audio = np.concatenate((self.audio,zeros),axis = None)
    
    
  def scale(self):
    self.audio = self.audio.reshape(1,-1)
    self.scaler = MinMaxScaler(feature_range = (0,1)).fit(self.audio)
    self.audio = self.scaler.transform(self.audio)
  
  def shapeAudio(self):
    self.audio = self.audio.reshape(-1,self.length,1)
    
  
  
  def predict(self):
    self.clean_audio = self.autoencoder.predict(self.audio,batch_size = self.audio.shape[0])
  
  def rescale(self):
    self.clean_audio = self.clean_audio.reshape(1,-1)
    self.clean_audio = self.scaler.inverse_transform(self.clean_audio)
  
  def reshape(self):
    self.clean_audio = self.clean_audio.reshape(-1)
  
  def saveAudio(self,mono = False):
    
    
    if mono:
      librosa.output.write_wav(self.save_path, self.clean_audio, self.sr)
    
    else:
      self.stereo_clean_audio = np.asarray([])
      self.stereo_clean_audio = np.vstack((self.clean_audio,self.clean_audio))
      
      librosa.output.write_wav(self.save_path, self.stereo_clean_audio,self.sr)
 
  def run(self):
    self.padData()
    self.scale()
    self.shapeAudio()
    self.predict()
    self.rescale()
    self.reshape()
    self.saveAudio()
    plt.plot(self.clean_audio)
    
    
    
      
proc = PostProcessing(model_path = 'AudioDenoising.h5',
                     audio_path = 'ww2.wav',
                     save_path = 'test.wav')
proc.run()
Audio(proc.clean_audio,rate= proc.sr)
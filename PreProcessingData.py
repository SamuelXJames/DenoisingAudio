# -*- coding: utf-8 -*-
"""
Created on Mon May 13 22:40:22 2019

@author: Sam
"""

#!pip install librosa
#!pip install mido==1.2.6
#!pip install madmom
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import librosa
import os
import zipfile
from datetime import datetime
import IPython.display as ipd
import librosa.display
from IPython.display import Audio
#import madmom
#from google.colab import drive


class PreProcessingMusic:

  def __init__(self,data_filepath,length = 22050):
    self.data_filepath = data_filepath #'Data.zip'
    self.sr_list = []
    self.audio_list = []
    self.dist_audio_list = []
    self.snr = 50 #dB
    self.length = length
    self.start = datetime.now()
  


  def gatherData(self):
  
    z = zipfile.ZipFile(self.data_filepath)
    names = z.namelist()
    names = names[0:len(names)]
    
    for name in names:
      
      x,sr = librosa.load(z.extract(name))
      #librosa.display.waveplot(x, sr=sr)
      self.audio_list.append(x)
      self.sr_list.append(sr)
      
    
    
    
  
  def addDistortion(self,plot=True):
    self.clean_audio = self.audio_list
    self.noisy_audio = []
    mu = 0
    sigma = 0.1
    
    for audio in self.audio_list:
      audio_power=np.sqrt(np.mean(audio**2))
      #audio_db = 10*np.log10(audio_power)
      #noise = sig_pow-self.snr
      #noise = 10 ** (noise/ 10)
      #print(noise)
      #noise = np.random.normal(0,sigma,len(audio))
      noise = np.random.normal(mu,sigma,len(audio))
      noise_power = np.sqrt(np.mean(noise**2))
      #noise_db = 10*np.log10(noise_power)
      sig_noise = audio_power/noise_power
      print('SNR: {0} \nSignal: {1}\nNoise: {2}'.format(sig_noise,audio_power,noise_power))
      dist = audio+noise
      self.noisy_audio.append(dist)
    
    if plot:
      count, bins, ignored = plt.hist(noise, 30, density=True)
      plt.title('Noise Profile')
      plt.plot(bins, 
               1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
               linewidth=2, color='r')
      plt.show()
      
    self.clean_audio = np.asarray(self.clean_audio)
    self.noisy_audio = np.asarray(self.noisy_audio)
    return dist
  
  def trimData(self):
    
    excess_length = self.clean_audio.size%self.length 
    self.clean_audio = self.clean_audio.reshape(1,-1)
    self.noisy_audio = self.noisy_audio.reshape(1,-1)
    
    if excess_length is not 0:
      self.clean_audio = self.clean_audio[0][0:-excess_length]
      self.noisy_audio = self.noisy_audio[0][0:-excess_length]
    
  def scaleData(self,method_1 = False,method_2 = False,method_3 = False):
    ''' There are three scaling methods. The first uses the same scale for both
    noisy signals and clean signals. This is to make it easier for the nn map 
    the noisy signal onto the clean signal. The problem is the sscaler may not
    be dynamic enough for noise. The second scales them seperatly. Not sure if 
    this works. It might give the same thing as method 1. and the third uses two 
    seperate scales.'''
    
    self.scaler = MinMaxScaler(feature_range = (0,1))
    self.scaler_2 = MinMaxScaler(feature_range = (0,1))
    
    if method_1:
      data  = np.concatenate((self.clean_audio,self.noisy_audio),axis=None)
      data = data.reshape(1,-1)
      data = self.scaler.fit_transform(data)
      data = data.reshape(-1,1)
      self.clean_audio = data[0:self.clean_audio.size]
      self.noisy_audio = data[self.clean_audio.size:data.size]
    
    if method_2:
      self.clean_audio = self.scaler.fit_transform(self.clean_audio.reshape(1,-1))
      self.noisy_audio = self.scaler.fit_transform(self.noisy_audio.reshape(1,-1))
      self.clean_audio = self.clean_audio.reshape(-1,1)
      self.noisy_audio = self.noisy_audio.reshape(-1,1)
    
    else:
      self.clean_audio = self.scaler.fit_transform(self.clean_audio.reshape(1,-1))
      self.noisy_audio = self.scaler_2.fit_transform(self.noisy_audio.reshape(1,-1))
      self.clean_audio = self.clean_audio.reshape(-1,1)
      self.noisy_audio = self.noisy_audio.reshape(-1,1)
  
  def shapeData(self):
    self.clean_audio = self.clean_audio.reshape(-1,self.length)
    self.noisy_audio = self.noisy_audio.reshape(-1,self.length)
  
  def run(self):
    self.gatherData()
    self.addDistortion()
    self.trimData()
    self.scaleData()
    self.shapeData()
    print('Clean Data Shape: {}',format(proc.clean_audio.shape))
    print('Noisy Data Shape: {}'.format(proc.noisy_audio.shape))
    end_time = datetime.now()
    print('Time to process data: {}'.format(end_time - self.start))
  
    return self.clean_audio,self.noisy_audio
      
      
    


proc = PreProcessingMusic('Test.zip')
proc.run()

#Audio(d,rate = proc.sr_list[0],autoplay=True)



  
  

  
  
  
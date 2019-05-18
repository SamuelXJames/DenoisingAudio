#!pip install librosa
#!pip install mido==1.2.6
#!pip install madmom
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import librosa
import zipfile
from datetime import datetime
import IPython.display as ipd
import librosa.display
from IPython.display import Audio
#import madmom
#from google.colab import drive


class PreProcessingMusic:

  def __init__(self,data_filepath,length = 22050,method_1=True,
              method_2 = False, method_3 = False,plotNoise = False ):
    self.data_filepath = data_filepath #'Data.zip'
    self.sr_list = []
    self.audio_list = []
    self.dist_audio_list = []
    self.snr = 50 #dB
    self.length = length
    self.start = datetime.now()
    self.method_1 = method_1
    self.method_2 = method_2
    self.method_3 = method_3
    self.plotNoise = plotNoise
  


  def gatherData(self):
  
    z = zipfile.ZipFile(self.data_filepath)
    names = z.namelist()
    names = names[0:len(names)]
    
    for name in names:
      
      x,sr = librosa.load(z.extract(name))
      #librosa.display.waveplot(x, sr=sr)
      self.audio_list.append(x)
      self.sr_list.append(sr)
    
    
   
    
  
  def addDistortion(self):
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
      #print('SNR: {0} \nSignal: {1}\nNoise: {2}'.format(sig_noise,audio_power,noise_power))
      dist = audio+noise
      self.noisy_audio.append(dist)
    
    if self.plotNoise:
      count, bins, ignored = plt.hist(noise, 30, density=True)
      plt.title('Noise Profile')
      plt.plot(bins, 
               1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
               linewidth=2, color='r')
      plt.show()
      
    self.clean_audio = np.asarray(self.clean_audio)
    self.noisy_audio = np.asarray(self.noisy_audio)
    
    #self.audio_shape = self.clean_audio.shape
    return dist
  
  def padData(self):
    for i in range(self.clean_audio.shape[0]):
      excess_length = self.clean_audio[i].size%self.length 
    
      if excess_length is not 0:
        padLength = self.length-excess_length
        zeros = np.zeros((1,padLength))
        #print(zeros.shape)
        #test = np.concatenate((self.clean_audio[i].reshape(-1),zeros.reshape(-1)))
        
        self.clean_audio[i] = np.concatenate((self.clean_audio[i].reshape(-1),zeros.reshape(-1)),axis = None)
        self.noisy_audio[i] = np.concatenate((self.noisy_audio[i],zeros),axis = None)
        self.audio_shape = self.clean_audio.shape 
        #print(test.shape)
    
  def scaleData(self):
    ''' There are three scaling methods. The first uses the same scale for both
    noisy signals and clean signals. This is to make it easier for the nn map 
    the noisy signal onto the clean signal. The problem is the sscaler may not
    be dynamic enough for noise. The second scales them seperatly. Not sure if 
    this works. It might give the same thing as method 1. and the third uses two 
    seperate scales.'''
    
    self.scaler = MinMaxScaler(feature_range = (0,1))
    self.scaler_2 = MinMaxScaler(feature_range = (0,1))
    
    for i in range(self.clean_audio.shape[0]):
      if self.method_1:
        data  = np.concatenate((self.clean_audio[i],self.noisy_audio[i]),axis=None)
        data = data.reshape(-1,1)
        data = self.scaler.fit_transform(data)
        self.clean_audio[i] = data[0:self.clean_audio[i].size]
        self.noisy_audio[i] = data[self.clean_audio[i].size:data.size]
        print('Scaling Method: 1')

      elif self.method_2:
        self.clean_audio[i] = self.scaler.fit_transform(self.clean_audio[i].reshape(-1,1))
        self.noisy_audio[i] = self.scaler.fit_transform(self.noisy_audio[i].reshape(-1,1))
        print('Scaling Method: 2')


      elif self.method_3:
        self.clean_audio[i] = self.scaler.fit_transform(self.clean_audio[i].reshape(-1,1))
        self.noisy_audio[i] = self.scaler_2.fit_transform(self.noisy_audio[i].reshape(-1,1))
        print('Scaling Method: 3')

  
  def shapeData(self):
    self.clean_data = np.asarray([])
    self.noisy_data = np.asarray([])
    for i in range(self.clean_audio.shape[0]):
      self.clean_data = np.concatenate((self.clean_data,self.clean_audio[i]), axis = None)
      self.noisy_data = np.concatenate((self.noisy_data,self.noisy_audio[i]), axis = None)
      
    self.clean_audio = self.clean_data.reshape(-1,self.length,1)
    self.noisy_audio = self.noisy_data.reshape(-1,self.length,1)
  
  def run(self):
    self.gatherData()
    self.addDistortion()
    self.padData()
    self.scaleData()
    self.shapeData()
    print('Clean Data Shape: {}'.format(self.clean_audio.shape))
    print('Noisy Data Shape: {}'.format(self.noisy_audio.shape))
    end_time = datetime.now()
    print('Time to process data: {}'.format(end_time - self.start))
  
    return self.clean_audio,self.noisy_audio
      
      
    


# proc = PreProcessingMusic('Alabama.zip')
# x,y = proc.run()
# x.shape
#Audio(y[0],rate = 22050)



  
  

  
  
  
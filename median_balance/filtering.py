from __future__ import division  #force floating point division
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import rsf.api
import math

'''
Created on Feb 24, 2015

@author: mlai
'''
            #Plot frequency and phase response
def plotFrequencyAndPhaseResponse(b,a=1):
    frequency_response_support,frequency_response = signal.freqz(b,a)
    frequency_response_db = 20 * np.log10 (np.abs(frequency_response))
    
    fig, axes_array = plt.subplots(2,1)
    axes = axes_array[0]
    axes.plot(frequency_response_support/np.max(frequency_response_support),
              frequency_response_db)
    axes.set_ylim(-150, 5)
    axes.set_ylabel('Magnitude (db)')
    axes.set_xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    axes.set_title(r'Frequency response')
    
    h_Phase = np.unwrap(np.arctan2(np.imag(frequency_response),np.real(frequency_response)))
    axes = axes_array[1]
    axes.plot(frequency_response_support/max(frequency_response_support),h_Phase)
    axes.set_ylabel('Phase (radians)')
    axes.set_xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    axes.set_title(r'Phase response')
    fig.subplots_adjust(hspace=0.5)

            #Plot step and impulse response
def plotStepAndImpulseResponse(b,a=1):
    l = len(b)
    impulse = np.repeat(0.,l); 
    impulse[0] =1.
    x = np.arange(0,l)
    response = signal.lfilter(b,a,impulse)
    fig, axes_array = plt.subplots(2,1)
    axes = axes_array[0]
    axes.stem(x, response)
    axes.set_ylabel('Amplitude')
    axes.set_xlabel(r'n (samples)')
    axes.set_title(r'Impulse response')
    
    axes = axes_array[1]
    step = np.cumsum(response)
    axes.stem(x, step)
    axes.set_ylabel('Amplitude')
    axes.set_xlabel(r'n (samples)')
    axes.set_title(r'Step response')
    fig.subplots_adjust(hspace=0.5)
    
if __name__ == '__main__':
    n = 61
    
    a = signal.firwin(n, cutoff = 0.3, window = "hamming")
    plotFrequencyAndPhaseResponse(a)
    plt.show()
    
    plotStepAndImpulseResponse (a)
    plt.show()
# speaker.py 
# Group 7 ECE 4900
# Edward Reehorst w/ help from
# http://minhdo.ece.illinois.edu/teaching/speaker_recognition/
# Text independent speaker recognition system based on mel frequency coeffiecient
# features and vector quantization
import array
import numpy as np
import scipy.fftpack as fft
import scipy.io.wavfile as wav
import wave
import pyaudio
from scipy.io.wavfile import read

import math

import os,sys


# Records audio signal to 
def record_audio(record_seconds):
	no_of_recordings=1
	CHUNK = 512
	INPUT_CHANNELS = 2
	FORMAT = pyaudio.paInt16
	CHANNELS = 1
	RATE = 48000
	filename= 's.wav'

	p = pyaudio.PyAudio()
        
	stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,input=INPUT_CHANNELS,frames_per_buffer=CHUNK)  
	print("* recording")
        data=array.array('h')
	for i in range(0, int(RATE / CHUNK * record_seconds)):
		data.fromstring(stream.read(CHUNK))   
        
	print("* done recording")
	print('*******************************************')
    
	stream.stop_stream()
	stream.close()
	p.terminate()

	d = np.array(data,dtype='i')
	print d.size    
	return (d,RATE)

# DISTEU Pairwise Euclidean distances between columns of two matrices
#
# Input:
#       x, y:   Two matrices whose each column is an a vector data.
#
# Output:
#       d:      Element d(i,j) will be the Euclidean distance between two
#               column vectors X(:,i) and Y(:,j)
#
# Note:
#       The Euclidean distance D between two vectors X and Y is:
#       D = sum((x-y).^2).^0.5
def disteu(x, y):

	[M, N] = x.shape
	[M2, P] = y.shape
	
	if M != M2:
		print 'Matrix dimensions do not match.'
		return -1

	d = np.zeros((N, P))

	for n in range(0,N):
		for p in range(0,P):
			d[n,p] = np.sum(np.power(x[:,n]-y[:,p],2),axis=0)

	d = np.power(d,0.5)
	return d


# VQLBG Vector quantization using the Linde-Buzo-Gray algorithm
#
# Inputs:
#       d contains training data vectors (one per column)
#       k is number of centroids required
#
# Outputs:
#       c contains the result VQ codebook (k columns, one for each centroids)
def vqlbg(d, k):
	# Constants
	split = 0.01;
	sigma = 0.001;

	#Initial Codebook of one entry contains single centroid
	c = np.mean(d, axis=1);
	c = c[:,np.newaxis]

	m = 1;

	# Continue 
	while m < k :
		# (Randomly) Split into two codebooks
		c = np.concatenate((np.multiply(c,(1 + split)), np.multiply(c,(1 - split))),axis=1)
	

		m = 2*m
		Dpast = float("inf")
		D = 10000
		while (Dpast - D)/D > sigma:
			Dpast = D;
			# Nearest Neighbor Search
			z = disteu(d, c);

			dist = np.amin(z, axis=1);
			ind = np.argmin(z, axis=1);

			D = np.mean(dist);

			# Update Centroids
			for j in range(0,m):
				c[:, j] = np.mean(d[:, ind==j], axis=1);

	return c

# FROM https://github.com/jameslyons/python_speech_features/blob/master/features/base.py
def get_filterbanks(nfilt=20,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"
    
    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbank = np.zeros([nfilt,nfft/2+1])
    for j in xrange(0,nfilt):
        for i in xrange(int(bin[j]),int(bin[j+1])):
            fbank[j,i] = (i - bin[j])/(bin[j+1]-bin[j])
        for i in xrange(int(bin[j+1]),int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i)/(bin[j+2]-bin[j+1])
    return fbank   

# FROM https://github.com/jameslyons/python_speech_features/blob/master/features/base.py
def hz2mel(hz):
    """Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1+hz/700.0)

# FROM https://github.com/jameslyons/python_speech_features/blob/master/features/base.py
def mel2hz(mel):
    """Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)


def toMono(sig):
	if sig.ndim > 1:
		return sig[:,0]
	return sig

# MFCC Calculate the mel frequencey cepstrum coefficients (MFCC) of a signal
#
# Inputs:
#       s       : speech signal
#       fs      : sample rate in Hz
#
# Outputs:
#       c       : MFCC output, each column contains the MFCC's for one speech frame			
def mfcc(s, fs):
	#Constants
	N = 256
	M = 100
	P = 30
	l = int(math.ceil((s.size-N+1)/M))

	#Allocate c array
	c = np.zeros((P,l));

	for x in range(0,l-1):
		#Frame
		start = x * M;
		frame = s[start:start+N];

		#Window
		w = np.hamming(N)
		windFrame = frame * w

		#FFT
		frameFFT = np.fft.fft(windFrame)

		#Mel-Frequency Wrapping
		m = get_filterbanks(P,N,fs)
		n2 = math.floor(N/2)
		ms = np.dot(m , abs(np.power(frameFFT[0:n2+1],2)))
		#Last step, compute mel-frequency cepstrum coefficients
		c[:,x] = fft.dct(np.log(ms.clip(min=0.00001)));
	np.delete(c,0,0)    # exclude 0'th order cepstral coefficient
	
	return c
	
def train():
# Speaker Recognition: Training Stage
#
# Input:
#       trainfile : string name of .wav file to use for training
#
# Output:
#       code     : trained VQ codebooks, code input speaker
#

	k = 16;                         # number of centroids required
	[s,fs]= record_audio(2)
	
	
	s = toMono(s)
	v = mfcc(s, fs)          # Compute MFCC's
	return vqlbg(v, k)      # Train VQ codebook

def test(code):
# Speaker Recognition: Testing Stage
#
# Input:
#       testfile : string name of test sound file
#       code    : codebooks of one trained speaker
#
# Output:
#		match: returns true if testfile matchs code

	Dmax = 43;

	[s,fs] = record_audio(2)
	  
	s = toMono(s)
	v = mfcc(s, fs)           # Compute MFCC's

	#for l in range(0,len(code)):    # each trained codebook, compute distortion
	d = disteu(v, code); 
	dist = sum(np.amin(d,axis=1)) / d.shape[0]

	print dist

	if dist<Dmax:
		return True
	
	return False

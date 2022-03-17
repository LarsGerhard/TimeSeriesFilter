import scipy.io.wavfile as wav
import sounddevice as sd
from matplotlib.pyplot import plot, legend, show, figure, subplot, title, tight_layout
from numpy import arange, sqrt, real, imag
from scipy.fftpack import fft

# time dimension

file_name = 'noisefile.wav'  # Path to your downloaded sound file
Fs, f = wav.read(file_name)

nt = len(f)
T = nt / Fs  # Time period of record
dT = 1 / Fs  # sec   time between samples

t = arange(0, T, dT)  # time array in seconds using arange(start,stop,step)
#   note that arange actually stops *before* stop time which
#   is what we want (in a periodic function t=0 and t=T are the same)

# frequency dimension

freqf = 1 / T  # Hz   fundamental frequency (lowest frequency)
nfmax = int(nt / 2)  # number of frequencies resolved by FFT

freqmax = freqf * nfmax  # Max frequency (Nyquist)

freq = arange(0, freqmax, 1 / T)  # frequency array using arange(start,stop,step)
# Note:
#     include freq=0 (constant term), so freq[0]=0
#     end one term before the  Nyquist (max) frequency, so freq[-1]=freqmax-freqf

print('Fundamental period and Nyquist Freq', T, freqmax)

# take FFT of this function
F = fft(f)

# get the coeffs
a = 2 * real(F[:nfmax]) / nt  # form the a coefficients
a[0] = a[0] / 2

b = -2 * imag(F[:nfmax]) / nt  # form the b coefficients

p = sqrt(a ** 2 + b ** 2)  # form power spectrum

# make some plots

f = f[:len(t)]

figure(1)

subplot(2, 1, 1)
plot(t, f)
title('Signal')

subplot(2, 1, 2)
plot(freq, a, 'o', label='Cosine')
plot(freq, b, '*', label='Sine')
plot(freq, p, '-', label='Power')
legend()

title('FFT Fourier Coefficients')

tight_layout()  # prevent squished plot (matplotlib kludge)

sd.play(f, Fs)

show()
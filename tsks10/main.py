import numpy as np
import scipy
import scipy.io.wavfile
import matplotlib
import matplotlib.pyplot as plot
from scipy.signal import butter, lfilter, freqz

from scipy.fftpack import fft, ifft

SAMPLE_RATE = 400 * 10**3
SAMPLE_SPACING = 1 / SAMPLE_RATE

CARRIER_FREQ = 150 * 10**3

CUTOFF_FREQ = 5000


#----------------------------------------------------------------------------
# The following 2 functions are taken from stack overflow
#http://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
#----------------------------------------------------------------------------


def plot_fft(transformed, sample_rate, sample_amount):
    sample_spacing = 1 / sample_rate
    xf = np.linspace(0.0, 1.0 / (2.0 * sample_spacing), sample_amount // 2)

    plot.plot(xf, 2.0/sample_amount * np.abs(transformed[0:sample_amount // 2]))


def iq_demodulate(sample_rate, data, carrier_freq):
    sample_spacing = 1 / sample_rate
    sample_amount = len(data)

    #generate a cosine wave at the correct sample rate
    samples = np.linspace(0.0, sample_amount*sample_spacing, sample_amount)
    cosine = 2 * np.cos(np.pi * 2 * carrier_freq * samples)

    #Multiply the data with the cos wave
    modulated = data * cosine;

    plot_fft(fft(modulated), sample_rate, sample_amount)

    #Filter the modulated signal
    filtered = butter_lowpass_filter(modulated, CUTOFF_FREQ, sample_rate)
    plot_fft(fft(filtered), sample_rate, sample_amount)

    plot.plot(filtered)

    scipy.io.wavfile.write("filtered.wav", sample_rate, filtered / 4000.)

def main():
    (fs, data) = scipy.io.wavfile.read("signal.wav")

    transformed = fft(data)

    plot_fft(transformed, fs, len(data))

    iq_demodulate(fs, data, CARRIER_FREQ)

    plot.show()

main()

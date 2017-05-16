#!/bin/python3

import numpy as np
import scipy
import scipy.io.wavfile
import matplotlib
import matplotlib.pyplot as plot
import math
from scipy.signal import butter, lfilter, freqz

from scipy.fftpack import fft, ifft

import random

SAMPLE_RATE = 400 * 10**3
SAMPLE_SPACING = 1 / SAMPLE_RATE

CARRIER_FREQ = 150 * 10**3
#CARRIER_FREQ = 113 * 10**3

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


def remove_carrier_frequency(sample_rate, data, carrier_freq):
    sample_spacing = 1 / sample_rate
    sample_amount = len(data)

    #generate a cosine wave at the correct sample rate
    samples = np.linspace(0.0, sample_amount*sample_spacing, sample_amount)
    cosine = 2 * np.cos(np.pi * 2 * carrier_freq * samples)

    #Multiply the data with the cos wave
    modulated = data * cosine;

    #plot_fft(fft(modulated), sample_rate, sample_amount)

    #Filter the modulated signal
    filtered = butter_lowpass_filter(modulated, CUTOFF_FREQ, sample_rate)
    #plot_fft(fft(filtered), sample_rate, sample_amount)

    #plot.plot(filtered)

    scipy.io.wavfile.write("filtered.wav", sample_rate, filtered / 4000.)

    return filtered

#Returns the sample at which the echo starts affecting the signal
def find_echo_delay(data):
    # To avoid waiting for the long calculation to finish
    #return 164003
    return 163999

    samples_until_echo_start = 100000

    #only check the first second of transmission because of the restriction that delay < 500ms
    samples_until_end_check = 441000

    start_samples = data[:samples_until_echo_start]
    check_samples = data[samples_until_echo_start:samples_until_end_check]

    correlated = np.correlate(check_samples, start_samples, mode='valid')

    correlated /= np.max(np.abs(correlated))

    #plot.plot(check_samples)
    plot.plot(correlated)
    #plot.plot(start_samples)

    return correlated.argmax() + samples_until_echo_start


def remove_echo(data, echo_start):
    echo_amplitude = 0.9;

    result = np.array(data)

    for i in range(echo_start, len(result)):
        result[i] = result[i] - echo_amplitude * data[i-echo_start]

    return result
    #if len(data) > echo_start*2:
    #    print("running echo removal step, len: {}", len(data))
    #    clean = data[:echo_start]
    #    cleaned = data[echo_start:echo_start*2] - echo_amplitude*clean

    #    new_data = np.append(cleaned, data[echo_start*2:])
    #    return np.append(clean, remove_echo(new_data, echo_start))
    #else:
    #    return data


def main():
    (fs, data) = scipy.io.wavfile.read("signal.wav")

    transformed = fft(data)

    interesting_signal = remove_carrier_frequency(fs, data, CARRIER_FREQ)

    normalized_signal = interesting_signal / np.max(np.abs(interesting_signal))

    echo_delay = find_echo_delay(normalized_signal)

    print("Echo delay in seconds: {}, in samples: {}".format(echo_delay / SAMPLE_RATE, echo_delay))

    without_echo = remove_echo(interesting_signal, echo_delay)

    scipy.io.wavfile.write("no_echo.wav", SAMPLE_RATE, without_echo / 4000.)

    #plot.plot(data)
    #plot.plot(without_echo)


    plot.show()

main()

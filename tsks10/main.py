#!/bin/python3

import numpy as np
import scipy
import scipy.io.wavfile
import matplotlib
import matplotlib.pyplot as plot
import math
from scipy.signal import butter, lfilter, freqz
import sounddevice

from scipy.fftpack import fft, ifft


SAMPLE_RATE = 400 * 10**3
SAMPLE_SPACING = 1 / SAMPLE_RATE

#CARRIER_FREQ = 75 * 10**3
CARRIER_FREQ = 151 * 10**3
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

def remove_carrier_frequency(sample_rate, data, carrier_freq, delta=0):
    print(delta)
    sample_spacing = 1 / sample_rate
    sample_amount = len(data)

    samples = np.linspace(0, sample_amount*sample_spacing, sample_amount)

    #cosine and sine waves at the correct sample rate
    cosine = 2 * np.cos((np.pi * 2 * carrier_freq * samples))
    sine = 2 * np.sin((np.pi * 2 * carrier_freq * samples))

    #Multiply the data with the cos waves
    modulated_i = data * cosine;
    modulated_q = data * sine;

    #Filter the demodulated signals
    filtered_i = butter_lowpass_filter(modulated_i, CUTOFF_FREQ, sample_rate)
    filtered_q = butter_lowpass_filter(modulated_q, CUTOFF_FREQ, sample_rate)
    #plot_fft(fft(filtered), sample_rate, sample_amount)


    #scipy.io.wavfile.write("filtered.wav", sample_rate, filtered / 4000.)

    final_i = filtered_i*np.cos(delta) - filtered_q*np.sin(delta)
    final_q = filtered_i*np.sin(delta) + filtered_q*np.cos(delta)

    return (final_i, final_q)



#Returns the sample at which the echo starts affecting the signal
def find_echo_delay(data):
    # To avoid waiting for the long calculation to finish
    return 164000

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
        result[i] = result[i] - echo_amplitude * result[i-echo_start]

    return result


def main():
    (fs, data) = scipy.io.wavfile.read("signal.wav")

    transformed = fft(data)
    plot_fft(transformed, fs, len(data))
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 15}

    matplotlib.rc('font', **font)
    plot.xlabel("Frekvens (Hz)", fontsize=13)
    plot.ylabel("Amplitud", fontsize=13)

    step_amount=0
    for i in range(0, step_amount):
        print(i)
        #find_delta(fs, data, CARRIER_FREQ)
        #interesting_signals = remove_carrier_frequency(fs, data, CARRIER_FREQ, np.pi / 2 * i/step_amount)
        #interesting_signals = remove_carrier_frequency(fs, data, CARRIER_FREQ, math.pi / 2 * 43/100)
        interesting_signals = remove_carrier_frequency(fs, data, CARRIER_FREQ, math.pi / 2 * 75/100)

        normalized_signal = interesting_signals[0] / np.max(np.abs(interesting_signals[0]))

        echo_delay = find_echo_delay(normalized_signal)

        print("Echo delay in seconds: {}, in samples: {}".format(echo_delay / SAMPLE_RATE, echo_delay))

        without_echo_i = remove_echo(interesting_signals[0], echo_delay)
        without_echo_q = remove_echo(interesting_signals[1], echo_delay)

        #scipy.io.wavfile.write("no_echo.wav", SAMPLE_RATE, without_echo / 4000.)
        downsampled_i = without_echo_i[0:len(without_echo_i):10]
        downsampled_q = without_echo_q[0:len(without_echo_q):10]
        sounddevice.play(downsampled_i/8000, 44100, blocking=True)
        sounddevice.play(downsampled_q/8000, 44100, blocking=True)
        #scipy.io.wavfile.write("no_echo_i.wav", SAMPLE_RATE, without_echo_q / 4000.)
        #scipy.io.wavfile.write("no_echo_q.wav", SAMPLE_RATE, without_echo_i / 4000.)


    plot.show()

main()

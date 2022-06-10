#!/usr/bin/python

from scipy import signal
"""
fir bandpass filter for LFP data

"""


def fir_bandpass(data, fs, cut_off_low, cut_off_high, width=2.0, ripple_db=10.0):
    """
    The FIR bandpass filter
    Args:
        data: 1-d array
        fs: frequency (sampling rate)
        cut_off_low: the low threshold
        cut_off_high: the high threshold
        width: the time windows for filtering
    Return:
        filtered data, N, delay (for plotting)
        when plot, align the axis by `plt.plot(time[N-1:]-delay, filtered_data[N-1:])`
    """
    nyq_rate = fs / 2.0
    wid = width/nyq_rate
    N, beta = signal.kaiserord(ripple_db, wid)
    taps = signal.firwin(N, cutoff = [cut_off_low, cut_off_high],
                  window = 'hamming', pass_zero = False, fs=fs)
    filtered_signal = signal.lfilter(taps, 1.0, data)
    delay = 0.5 * (N-1) / fs
    return filtered_signal, N, delay
# a collection of functions for processing my recording signals

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def schr(f0=100, lohar=1, hihar=400, c=0, fs=100, dur=30, func='sin'):
    """ create schroeder waveform
    inputs:
        f0::double    unit: Hz
        lohar::int    unit: harmonic number
        hihar::int
        c::double     unit: curvature
        fs::double    unit: kHz
        dur::double   unit: ms
	func::string  'sin' (default) or 'cos'
    returns:
        wv::1-D np.ndarray
    """
    f0_khz = f0 / 1000
    t = np.arange(0, dur, 1/fs)
    n = hihar - lohar + 1
    schr_wv = np.zeros_like(t)
    ith_wv = np.zeros_like(t)
    for i in range(lohar, hihar+1):
        ph = np.pi * i * (i - 1) / n * c
        if func == 'sin':
            ith_wv = np.sin(2 * np.pi * i * f0_khz * t + ph)
        else:
            ith_wv = np.cos(2 * np.pi * i * f0_khz * t + ph)
        schr_wv = schr_wv + ith_wv
    return schr_wv

def plt_schr_vs_instfreq(f0=100, lohar=1, hihar=100, curvs=np.linspace(-1, 1, 9), fs=100, cycles=2):
    """ plot schr waveform vs instantaneous frequency estimated from Hilbert transform
    inputs:
        f0::double      unit: Hz
        lohar::int      unit: harmonic number
        hihar::int
        curvs::1-D np.array unit: curvature
        fs::double      unit: kHz
        cycles::int     how many cycles to plot
    returns: 
        fig, ax
    """
    # create waveforms
    schr_wv = dict()
    highest_freq_khz = f0 / 1000 * hihar 
    # period: ms
    T = 1000 / f0; 
    dur = T * cycles
    for c in curvs:
        schr_wv[str(c)] = schr(f0=f0, lohar=lohar, hihar=hihar, c=c, fs=fs, dur=dur)
    t = np.arange(0, dur, 1 / fs)

    # plot
    fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(10, 7), sharex=True, sharey='row')
    for idx, c in enumerate(curvs):
        irow = idx if c <= 0 else 3 - np.mod(idx, 5) 
        icol = idx // 5
        wv = schr_wv[str(c)]
        # wv amp. normalize to highest frequency
        wv = wv / max(wv) * highest_freq_khz
        ax[irow, icol].plot(t, wv)
        ax[irow, icol].text(0, -10, 'c = %0.2f' % c)
        # only plot inst.freq when c != 0
        if c != 0:
            inst_freq = instantFreq(wv, fs)
            # low-pass filter inst_freq
            inst_freq_lp = lpfiltfilt(inst_freq, fs, cutoff_hz=5)    
            # plt_window: effective time window for schroeder
            npnts_to_plt = abs(c) * T * fs - 1
            if c < 0:
                indices = np.arange(0, npnts_to_plt, dtype=int)
            else:
                indices = np.arange(T * fs - npnts_to_plt, T * fs - 1, dtype=int)
            if cycles > 1:
                len_one_indices = len(indices)
                indices = np.tile(indices, cycles)
                idx_offset = np.arange(0, fs * T * cycles, fs * T)
                indices = indices + np.repeat(idx_offset, len_one_indices) 

            indices = np.array(indices, dtype = 'int')
            # set instfrequencies not in the indices to nan
            # so that there will be gaps between cycles
            inst_freq_lp[~np.isin(np.arange(0, len(inst_freq_lp)), indices)] = np.nan
            ax[irow, icol].plot(t[:-1], inst_freq_lp)
        if c == 0:
            # plot at ax[4, 1]
            ax[irow, icol + 1].plot(t, wv)
            ax[irow, icol + 1].text(0, -10, 'c = %0.2f' % c)
    ax[0, 0].set_ylim(-11, 11)
    ax[2, 0].set_ylabel('Sound amplitude \n Instantaneous frequency (kHz)')
    ax[-1, 0].set_xlabel('Time (ms)')
    ax[-1, 1].set_xlabel('Time (ms)')
    example_lines = ax[0, 0].lines
    fig.legend(handles=example_lines, labels=['sound', 'inst. freq'], loc='upper right')
    fig.show()
    return fig, ax
         


def instantFreq(wv, fsam):
    """ Returns the instant frequency of the wave
    inputs:
        wv::1-D np.ndarray
        fsam::double
    returns:
        inst_freq::1_D np.ndarray
    """
    z = signal.hilbert(wv)
    # instant phase
    inst_ph = np.unwrap(np.angle(z))
    # instant frequency
    inst_freq = (np.diff(inst_ph) / (2.0 * np.pi) * fsam)
    return inst_freq

def envelope(wv, fsam):
    """ Returns the envelope of the wave
    inputs:
        wv::1-D np.ndarray
        fsam::double
    returns:
        env::1-D np.ndarray
    """
    z = signal.hilbert(wv)
    # envelope amplitudes
    env = np.abs(z)
    return env

def lowpass_kaiser(fsam, cutoff_hz=10):
    """ returns the coefficients of a kaiser lowpass filter
    Copied from https://scipy-cookbook.readthedocs.io/items/FIRFilter.html
    inputs:
        fsam::double
        cutoff_hz::int
    returns:
        taps::1-D np.ndarray
    """
    # The Nyquist rate of the signal
    nyq_rate = fsam / 2.0

    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 5 Hz transition width.
    width = 5.0/nyq_rate

    # The desired attenuation in the stop band, in dB
    ripple_db = 60.0

    # compute the order and Kaiser parameter for the FIR filter
    N, beta = signal.kaiserord(ripple_db, width)

    # coefficients of a lowpass FIR filter using Kaiser window
    taps = signal.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))

    return taps

def lowpass_butter(fsam, cutoff_hz=10):
    """ returns the coefficients of a butterworth lowpass filter
    copied from https://medium.com/analytics-vidhya/how-to-filter-noise-with-a-low-pass-filter-python-885223e5e9b7
    inputs:
        fsam::double
        cutoff_hz::int
    returns:
        (b, a)::a tuple of two 1-D np.ndarrays
    """
    # The Nyquist rate of the signal
    nyq_rate = fsam / 2.0
    # normalized cutoff frequency
    cutoff_norm = cutoff_hz / nyq_rate
    # order
    order = 2
    # use signal.butter to obtain the coefficients
    b, a = signal.butter(order, cutoff_norm, btype='low', analog=False)
    return (b, a)

def highpass_butter(fsam, cutoff_hz=10):
    """ returns the coefficients of a butterworth highpass filter
    copied from https://medium.com/analytics-vidhya/how-to-filter-noise-with-a-low-pass-filter-python-885223e5e9b7
    inputs:
        fsam::double
        cutoff_hz::int
    returns:
        (b, a)::a tuple of two 1-D np.ndarrays
    """
    # The Nyquist rate of the signal
    nyq_rate = fsam / 2.0
    # normalized cutoff frequency
    cutoff_norm = cutoff_hz / nyq_rate
    # order
    order = 2
    # use signal.butter to obtain the coefficients
    b, a = signal.butter(order, cutoff_norm, btype='highpass', analog=False)
    return (b, a)

def hpfiltfilt(wv, fsam, cutoff_hz=10, method='butter'):
    """ use filtfilt to return a high-pass filtered signal
    Currently only supports Butterworth
    inputs:
        wv::1-D np.ndarray
        fsam::double        unit: Hz
        cutoff_hz::int
        method::string      'butter'
    returns:
        wv_filtered::1-D np.ndarray
    """
    if method=='butter':
        b, a = highpass_butter(fsam, cutoff_hz)
        wv_filtered = signal.filtfilt(b, a, wv)
    return wv_filtered

def lpfiltfilt(wv, fsam, cutoff_hz=10, method='kaiser'):
    """ use filtfilt to return a low-pass filtered signal
    inputs:
        wv::1-D np.ndarray
        fsam::double        unit: Hz
        cutoff_hz::int
    returns:
        wv_filtered::1-D np.ndarray
    """
    if method=='kaiser':
        taps = lowpass_kaiser(fsam, cutoff_hz)
        wv_filtered = signal.filtfilt(taps, 1.0, wv)
    if method=='butter':
        b, a = lowpass_butter(fsam, cutoff_hz)
        wv_filtered = signal.filtfilt(b, a, wv)
    return wv_filtered


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    copied from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal with the *SAME* length as the input
        i.e. return y[(window_len // 2) : -(window_len // 2)] instead of y

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len // 2) : -(window_len // 2)]


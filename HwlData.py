from scipy.io import loadmat
from scipy.signal import find_peaks

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import sys
import glob
import h5py as h5
import ipywidgets as widgets
import Hsignals as hs
import pywt
from skimage.restoration import denoise_wavelet

class HwlDS(object):
    """A data format equivalent to hwlClass in matlab
    """
    homepath = './early_processeddata'
    metapath = homepath + '/earlydata_analysis'
    stimpath = homepath + '/earlydata_stim'
    spktpath = homepath + '/earlydata_spkt'
    calibpath = homepath + '/earlydata_calib'
    cellpath = homepath + '/earlydata_cell'
    h5path = homepath + '/earlydata_h5'

    def __init__(self, expname, iDataset):
        self.expname = expname
        self.allmeta = None
        self.meta = None
        self.cmeta = None
        if type(iDataset) == str:
            self.iDataset = self.find_iDataset(iDataset)
        else:
            self.iDataset = iDataset
        self._set_paths()

        # default parameters to None 
        # to save object creation time
        self.spkt = None
        self.stimparam = None
        self.ncond = None
        self.nreprec = None
        # a dictionary containing excluded reps for each cond#
        # key: cond# - 1; items: rep# - 1
        self.excluded_dict = {}

    def IDstring(self):
        """ returns the IDstring of the ds
        """
        meta = self.get_meta()
        iCell = meta.iCell.item()
        iRecOfCell = meta.iRecOfCell.item()
        StimType = meta.StimType.item()
        return '%d-%d-%s' % (iCell, iRecOfCell, StimType)

    def find_iDataset(self, IDstring):
        """ Use IDstring to find iDataset
        inputs:
            IDstring::str   e.g. '1-2'
        returns:
            iDataset::int   e.g. 2
        """
        iCell = int(IDstring.split('-')[0])
        iRecOfCell = int(IDstring.split('-')[1])
        self.fname_meta = '%s/%s.mat' % (self.metapath, self.expname)
        allmeta = self._load_allmeta()
        meta = allmeta[(allmeta['iCell'] == iCell) & (allmeta['iRecOfCell'] == iRecOfCell)]
        return meta['iDataset'].item()

    def _set_paths(self):
        self.fname_spkt = '%s/%s/%s_%05d.mat' % (
               self.spktpath, self.expname, self.expname, self.iDataset)
        self.fname_meta = '%s/%s.mat' % (self.metapath, self.expname)
        self.fname_stim = '%s/%s.mat' % (self.stimpath, self.expname)
        self.fname_h5 = '%s/%s/%s_%05d.h5' % (
                self.h5path, self.expname, self.expname, self.iDataset)
 
    def _load_spkt(self):
        """ Load spkt from self.fname_spkt
        Returns:
            spkt::dictionary in which
                spkt['on'] is spkt.on in Matlab
                spkt['off'] is spkt.off in Matlab
                spkt['exludedReps'] is spkt.excludedReps in Matlab
        """
        spkt = loadmat(self.fname_spkt, squeeze_me=True)
        return spkt

    def get_spkt(self, on_off='off', toload=False):
        """ Return a spkt['off'] if it exists
        inputs:
            on_off::string      'on' or 'off
            toload::boolean     True or False to load from .mat
        returns:
            spkt::ndarray       a ncond-by-nrep array; each item.shape=(1,) 
        """
        if self.spkt is None or toload:
            self.spkt = self._load_spkt()

        if self.spkt['off'].size == 0:
            spkt = self.spkt['on']
        else:
            spkt = self.spkt[on_off]

        spkt = self._expand_spkt_dim(spkt)

        return spkt

    def _expand_spkt_dim(self, spkt):
        """ Make every float item in spkt a 1-D np.ndarray
        input:
            spkt::nc-by-nr ndarray  e.g. 9-by-200 array for nc=9, ir=200
        returns:
            spkt::nc-by-nr ndarray   e.g. 9-by-200 array for nc=9, ir=200
        """
        for spkt_ic in spkt:
            for ir, spkt_ir in enumerate(spkt_ic):
                if type(spkt_ir) == float:
                    spkt_ic[ir] = np.expand_dims(spkt_ir, axis=0)
        return spkt

    def _get_excludedreps(self, toload=False):
        """ Return excludedReps::np.array
               e.g. ["Cond#1, Rep#200", "Cond#2, Rep#1"] 
        """
        if self.spkt is None or toload:
            self.spkt = self._load_spkt()
        # return [] if there's no 'excludedReps' key
        excluded =  self.spkt.get('excludedReps', [])
        if type(excluded) == str:
            excluded = [excluded]
        return excluded

    def _get_excl_con_rep(self, exclstr):
        """ Returns a tuple of (excluded cond#, excluded rep#)
        inputs:
            excstr::str
                a string; e.g. "Cond#1, Rep#200"
        returns:
            a tuple: (excluded cond#-1, excluded rep#-1)
            NOTE: This follows the zero-indexing tradition in Python
        """
        [cstr, rstr] = exclstr.split(', ')
        [cstr, cnum] = cstr.split('#')
        [rstr, rnum] = rstr.split('#')
        return (int(cnum) - 1, int(rnum) - 1)

    def get_excluded_dict(self, toload=False):
        """ return a dictionary of exlcuded reps
        input:
            toload::bool    True to reload self.spkt; False not
        returns:
            exc_dict::dictionary
                a dictionary whose keys are cond#-1 and items are rep#-1
                NOTE: I follow the zero-indexing tradition in python
        """
        if toload == False and len(self.excluded_dict) > 0:
            return self.excluded_dict

        exc_arr = self._get_excludedreps(toload)
        for e in exc_arr:
            exc_tuple = self._get_excl_con_rep(e)
            if exc_tuple[0] in self.excluded_dict.keys():
                self.excluded_dict[exc_tuple[0]].append(exc_tuple[1])
            else:
                self.excluded_dict[exc_tuple[0]] = [exc_tuple[1]]
        return self.excluded_dict

    def get_meta(self, toload=False):
        """ Return a pd.DataFrame equivalent to hwlClass.meta
        """
        if self.meta is None or toload:
            allmeta = self.get_allmeta()
            self.meta = allmeta[allmeta['iDataset'] == self.iDataset]
        return self.meta

    def get_meta_thiscell(self):
        """ Returns the meta from this hiCell
        """
        allmeta = self.get_allmeta()
        hiCell = self.get_meta().hiCell.item()
        meta_thiscell = allmeta[allmeta.hiCell == hiCell]
        return meta_thiscell

    def _load_allmeta(self):
        """ Load allmeta from self.fname_meta
        Return a pd.DataFrame equivalent to hwlClass.allmeta
        """
        allmeta = loadmat(self.fname_meta, squeeze_me=True)
        allmeta_df = pd.DataFrame(allmeta['S'])
        return allmeta_df

    def get_allmeta(self, toload=False):
        """ return a pd.DataFrame equivalent to hwlClass.allmeta
        """
        if self.allmeta is None or toload:
            self.allmeta = self._load_allmeta()
        return self.allmeta

    def _load_stim(self):
        """ Load self.fname_stim
        Returns a tuple: (stimparam, nreprec) 
            stimparam: ndarray
            nreprec: an n-element ndarray where n is ncond
        """
        stim = loadmat(self.fname_stim, squeeze_me=True)
        stim = stim['EXP']
        stim = stim[stim['iDataset'] == self.iDataset]
        stimparam = stim['stimparam'][0]
        nreprec = stim['NrepRec'][0]
        return (stimparam, nreprec)

    def get_stimparam(self, toload=False):
        """ Return self.stimparam
        """
        if self.stimparam is None or toload:
            (self.stimparam, self.nreprec) = self._load_stim()
        return self.stimparam


    def get_reps(self):
        """ Return the indices of reps that are good for analysis
        output:
            reps::dict 
                reps[i] is the good reps of (i-1)th cond
                NOTE: rep# follows zero-indexing tradition in Python
        """
        nreprec = self.get_nreprec()
        excluded_dict = self.get_excluded_dict()
        reps = {}
        for ic in range(len(nreprec)):
            excluded_reps = excluded_dict.get(ic, [])
            reps[ic]  = list(set(range(nreprec[ic])) - set(excluded_reps))

        return reps


    def get_nreprec(self, toload=False):
        """ Return the equivalent of self.stim.NrepRec
        output:
            an n-element ndarray where n is the number of conditions
        """
        if self.nreprec is None or toload:
            (self.stimparam, self.nreprec) = self._load_stim()
        return self.nreprec

    def get_ncond(self, toload=False):
        """ Return the equivalent of self.nCond
        output:
            an integer
        """
        if self.ncond is None or toload:
            (self.stimparam, self.nreprec) = self._load_stim()
            try:
                self.ncond = len(self.nreprec)
            except TypeError:
                print('TypeError; ncond = 1')
                self.ncond = 1
        return self.ncond

    def get_spkt_trimmed(self, ic):
        """ Return a copy of spkt whose unwanted reps are excluded
        only includes nreprec[ic] reps and exclude reps in exlcudedreps
        input:
            ic::int                 cond#
        output:
            spkt_excl::np.ndarray   an n-element array, n = number of good reps
        """
        spkt = self.get_spkt()
        ex_dict = self.get_excluded_dict()
        nreprec = self.get_nreprec()
        # remove unwanted reps
        spkt_tr = np.delete(spkt[ic][:nreprec[ic]], ex_dict.get(ic, []), axis=0)
        return spkt_tr

    def concat_spkt(self, spkt):
        """ concatenate all spkt into an array
        input:
            spkt::array     an nrep-element array, each element is an array
        returns:
            x::array     an n-element array; n is total number of spikes 
        """
        # avoid error when concatenating an empty array
        if spkt.size > 0:
            x = np.concatenate(spkt)
        else:
            x = spkt
        return x

    def get_cyc_avg_vm(self, icond, 
            method='quantile', 
            t_start=None, 
            t_end = None,
            T = None,
            denoise=False
            ):
        """ Return the cycle average of Vm of the specified condition
        NOTE: only calculate ONE cycle
        input:
            self::HwlDS object
            icond::int
                ZERO-indexed condition number
            method::string
                'quantile' (default), 'mean', 'all'
            t_start::double  unit: ms
                default: None --> will use the smallest k*T where k*T >= 10, k = 1, 2, 3...
            t_end::None or double (ms)
                end time
                default: None --> will use bustdur - T
            T::None or double (ms)
                cycle period
                default: None --> will automatically look for T using self.get_f0
            denoise::bool (default: False)
                denoise the wave using denoise_wavelet with sym4
        Returns:
            y_cavg::np.array
                if method == 'quantile':
                    a 2-D np.array; shape=(3, nsamples)
                if method == 'mean':
                    a 1-D np.array; shape=(nsample,)
                if method == 'all':
                    a 2-D np.array; shape=(ncycles, nsamples)
        """
        stim = self.get_stimparam()
        # automatically look for T
        if T is None:
            f0 = self.get_f0()
            if type(f0) == np.ndarray:
                T = 1000 / f0[icond]
            else:
                T = 1000 / f0
        burstdur = stim['BurstDur'][()]
        if t_end is None:
            t_end = burstdur - T 
        # t_start > 10 and mod(t_start, T) = 0
        if t_start is None:
            t_start = np.ceil(10 / T) * T
        (y, x, t0, dt, scale) = self.get_vm_fromh5(icond, denoise=denoise)
        exdict = self.get_excluded_dict()
        # remove unwanted reps
        y = np.delete(y, exdict.get(icond, []), axis=0)    
        # calculate cycle average
        y_cavg = self.cyc_average(y, t0, dt, T, t_start, t_end, T, method=method)
        return y_cavg

    def cyc_average(self, y, t0, dt, T, t_start, t_end, extractT, method='quantile'):
        """ a general func to calculate cycle average of the input waveform
        input:
            y::np.array
                a 2-D array; shape=(nrep, nsamples) 
            t0::double
                the start time of the trace
            dt::double
                the interval of the trace
            T::int
                the period of the cycle
            t_start::double
                the start time of the trace to extract cycle-average
            t_end::double
                the end time of the trace to extract cycle-average
            extractT::double 
                the duration to be extraced from each cycle
                i.e. extract the first extractT time of each cyle 
                Exmplae: T=10, extractT=20
                ==> extract the first 20ms of each cycle whose T=10ms
            method::string
                'quantile' (Default): return the 1st, 2nd, 3rd quantile 
                'mean': return the mean of the trace
        Returns:
            y_cyc::np.array
                if method == 'quantile':
                    a 2-D np.array; shape=(3, nsamples)
                if method == 'mean':
                    a 1-D np.array; shape=(nsample,)
                if method == 'all':
                    a 2-D np.array; shape=(ncycles, nsamples)
        """
        # check if y is nan
        if (y.size == 1) and (np.isnan(y)):
            y_cyc = y
        else:
            numreps = y.shape[0]
            x = np.arange(t0, y.shape[1]*dt + t0, dt)
            numcycles = int((t_end - t_start) / T)
            x_cyc = np.arange(0, T, dt)
            numpnts = x_cyc.size
            y_cyc = np.zeros((numcycles * numreps, numpnts))
            for irep, yrep in enumerate(y):
                for cyc in range(numcycles):
                    t_start_thiscyc = cyc * T + t_start
                    y_thiscyc = yrep[(x >= t_start_thiscyc)][:numpnts]
                    y_cyc[irep * numcycles + cyc] = y_thiscyc

        if method == 'quantile':
            return np.quantile(y_cyc, [0.25, 0.5, 0.75], axis=0)
        elif method == 'mean':
            return np.mean(y_cyc, axis=0)
        elif method == 'all':
            return y_cyc

    def get_eachcycle(self, y, t0, dt, T, t_start, t_end, extractT):
        """ a general func to get each cycle of the input waveform
        input:
            y::np.array
                a 2-D array; shape=(nrep, nsamples) 
            t0::double
                the start time of the trace
            dt::double
                the interval of the trace
            T::double
                the period of the cycle
            t_start::double
                the start time of the trace to extract cycle-average
            t_end::double
                the end time of the trace to extract cycle-average
            extractT::double 
                the duration to be extraced from each cycle
                i.e. extract the first extractT time of each cyle 
                Exmplae: T=10, extractT=20
                ==> extract the first 20ms of each cycle whose T=10ms
        Returns:
            y_cyc::np.array
                a 2-D np.array; shape=(ncycles, nsamples)
        """
        numreps = y.shape[0]
        x = np.arange(t0, y.shape[1]*dt + t0, dt)
        numcycles = int((t_end - t_start) / T)
        x_cyc = np.arange(0, extractT, dt)
        numpnts = x_cyc.size
        y_cyc = np.zeros((numcycles * numreps, numpnts))
        for irep, yrep in enumerate(y):
            for cyc in range(numcycles):
                t_start_thiscyc = cyc * T + t_start
                y_cyc[irep * numcycles + cyc] = yrep[(x >= t_start_thiscyc)][:numpnts]
        return y_cyc

    def get_vrest(self, method='mean'):
        """ return vrest by averaging Vm[t > burstdur + 5] over all conditions
        inputs:
            self::hwlDS object
            method::string
                'mean': return average of all conditions
                'each': return an array of vrest for each condition
        returns:
            vrest::np.array or double
                average vrest over all conditions if method='mean'
                vrest for each condition if method='each'
        """
        ncond = self.get_ncond()
        reps = self.get_reps()
        burstdur = self.get_stimparam()['BurstDur'].item()
        vrest_percond = np.empty(self.get_ncond())
        for ic in range(ncond):
            vm, t, t0, dt, scale = self.get_vm_fromh5(ic)
            vrest_percond[ic] = np.mean(vm[reps[ic]][:, t > burstdur + 5])
            
        if method == 'mean':
            return np.mean(vrest_percond)
        if method == 'each':
            return vrest_percond

    def get_f0(self):
        """ Return f0 of the ith cond
        Returns:
            f0::np.array
                an 1-D array; shape=(ncond,)
        """
        stim = self.get_stimparam()
        stimtype = stim['StimType'][()]
        if stimtype in ('RC', 'SCHR', 'CFS', 'CSPL'):
            f0 = stim['Fcar'][()]
        elif stimtype in ('MTF', 'RCM'):
            f0 = stim['Fmod'][()]
        else:
            print('%s not supported', stimtype)
            f0 = None
        return f0

    def plt_dotraster(self, icond, ax=None):
        """ Plot dotraster of ith condition in the input axis
        Use spkt['off'] if exists
        input:
            ax::ax          e.g. from plt.subplots(); default:None; 
            icond::int      ith cond number
        """
        if ax is None:
            fig, ax = plt.subplots()
        x = self.get_spkt_trimmed(icond)
        numreps = len(x)
        numspkt_per_rep = [xval.size for xval in x]
        # yvals for each spike 
        y = np.repeat(np.arange(numreps), numspkt_per_rep)
        allspkt = self.concat_spkt(x)
        ax.plot(allspkt, y, 'k.', markersize=2)
        plt.show()

    def get_cycle_hist(self, ic, ncycles=1):
        """ returns REGULAR (not polar) cycle histogram's height and edges of one condition
        inputs:
            ic::int
                ith condition
            ncycles::int (default: 1)
                the number of cycles to be returned
                ONLY accepts 1 or 2
        returns:
            hist::np.array
                the y values of the histogram
            edges::np.array (unit: cycle)
                the x values of the histogram;
                NOTE: it has the SAME length as the hist!
                DIFFERENT from what you get from np.histogram
        """

        # unit: ms
        T = self.get_T()[ic]
        binsize = 0.1
        nbins = int(T / binsize)

        spkc = self.get_spkt_in_cycle(ic, method='cycle')
        hist, edges = np.histogram(spkc, bins=nbins, range=(0, 1))
        edges = edges[1:]
        if ncycles == 2:
            hist = np.concatenate([hist, hist])
            edges = np.concatenate([edges, edges + 1])
        return (hist, edges)
        

    def plt_cycle_hist(self, ic, ax=None, method='regular'):
        """ plot cycle histogram as regular or polar form in the specified axis
        input:
            ic::int         condition number
            ax::axis        axis object
            method::str     'regular' or 'polar'
        """
        if ax is None:
            if method == 'regular':
                fig, ax = plt.subplots()
            elif method == 'polar':
                fig, ax = plt.subplots(subplot_kw=dict(polar=True))
        binsize = 0.1
        T = self.get_T()
        if method == 'regular':
            hist, edges = self.get_cycle_hist(ic, ncycles=2)
            ax.step(edges, hist)
        elif method == 'polar':
            spkt_cycle = self.get_spkt_in_cycle(ic, method='radian')
            ax.hist(spkt_cycle, bins=int(T[ic] / binsize), histtype='step')

    def get_cycle_raster(self, ic, subtract_wbdelay=True, two_cycles=True):
        """ returns the cycle number (from 1st cycle in 1st rep to last cycle in last rep)
        and its corresponding spike time in cycles

        inputs:
            ic::int
                condiction number
            subtract_wbdelay::boolean
                True to subtract wbdelay in the acoustic system
                False not
            two_cycles::boolean
                True to return two cycles
                False not
        returns:
            x::spike times (unit: cycle)
                will return 2 cycles
            y::cycle number
                if there are 100 cycles in each rep, and thare are 10 reps
                1st cycle in 1st rep will be 0
                last cycle in last rep will be 100 * 10 - 1 = 999
        """
        T = self.get_T()[ic]
        burstdur = self.get_stimparam()['BurstDur']
        if subtract_wbdelay:
            wbdelay = self.get_wbdelay()
        else:
            wbdelay = 0
        t_start = 10 - wbdelay
        t_end = burstdur - 10 - wbdelay
        spkt = self.get_spkt()
        spkt_ic = np.concatenate(spkt[ic, :])
        # analysis time window
        spkt_ic = spkt_ic[(spkt_ic > t_start) & (spkt_ic < t_end)]
        # spkt in cycles (unwrapped)
        spkt_ic_unwrap = spkt_ic / T
        # cycle number for each spkt
        cycle_number = np.floor(spkt_ic_unwrap)
        # wrap spkt in cycles
        spkt_ic_wrap = np.mod(spkt_ic_unwrap, 1)
        # 2 cycles
        if two_cycles:
            x = np.concatenate([spkt_ic_wrap, spkt_ic_wrap + 1])
            y = np.concatenate([cycle_number, cycle_number])
        else:
            x = spkt_ic_unwrap
            y = cycle_number

        return (x, y)
    
    def plt_fs_allrate(self, figsize=(1.2, 1.2), **kwargs):
        """ plot the rate level of all schr responses in the cell
        inputs:
            figsize::tuple
                figsize
            **kwargs
                kwargs for plotting
        returns:
            fig::plt.Figure
            ax::plt.axes
        """
        df = self.fs_param_thiscell()
        fig, ax = plt.subplots(figsize=figsize)
        for ds in df.iloc:
            hds = HwlDS(self.expname, ds.iDataset)
            x = hds.cond_val()[0]
            try:
                y = hds.rate_curve()
            except:
                print('Problem getting rate at %s-%d' % (self.expname, ds.iDataset))
                continue
            ax.plot(x, y, label='%d dB' % (ds.spl), **kwargs)
        ylabel = 'Spikes / sec'
        ax.set_ylabel(ylabel)
        (ymin, ymax) = ax.get_ylim()
        ymin = -0.05
        ax.set_ylim([ymin, ymax])
        return fig, ax


    def plt_schr_allrate(self, normalized=True, figsize=(1.2, 1.2), **kwargs):
        """ plot the rate level of all schr responses in the cell
        inputs:
            normalized::bool
                True: plot normalized rate (spikes / cycle)
                False: plot absolute rate (spikes / sec)
            figsize::tuple
                figsize
            **kwargs
                kwargs for plotting
        returns:
            fig::plt.Figure
            ax::plt.axes
        """
        df = self.schr_param_thiscell()
        fig, ax = plt.subplots(figsize=figsize)
        for ds in df.iloc:
            hds = HwlDS(self.expname, ds.iDataset)
            x = hds.cond_val()[0]
            y = hds.rate_curve()
            if normalized:
                y /= ds.f0
                ylabel = 'Spikes / cycle'
            else:
                ylabel = 'Spikes / sec'
            marker, color = self.set_schr_marker_and_color(ds.spl, ds.f0)
            ax.plot(x, y, label='%d Hz, %d dB' % (ds.f0, ds.spl), color=color, marker=marker, **kwargs)
        ax.set_ylabel(ylabel)
        (ymin, ymax) = ax.get_ylim()
        ymin = -0.05
        ax.set_ylim([ymin, ymax])
        return fig, ax

    def plt_schr_allcontrast(self, what='rate', figsize=(1.2, 1.2), **kwargs):
        """ plot the rate contrast of all schr responses in the cell
        inputs:
            what::string
                'rate': rate contrast
                'vs': vector strength contrast
            figsize::tuple
                figsize
            **kwargs
                kwargs to plotting
        returns:
            fig::plt.Figure
            ax::plt.axes
        """
        df = self.schr_param_thiscell()
        fig, ax = plt.subplots(figsize=figsize)
        for ds in df.iloc:
            hds = HwlDS(self.expname, ds.iDataset)
            if what == 'rate':
                x, y = hds.schr_contrast()
                ylabel = 'rate contrast'
            elif what == 'vs':
                x, y = hds.schr_vscontrast()
                ylabel = 'vs contrast'
            if y is None:
                continue
            marker, color = self.set_schr_marker_and_color(ds.spl, ds.f0)
            ax.plot(x, y, label='%d Hz, %d dB' % (ds.f0, ds.spl), color=color, marker=marker, **kwargs)
        ax.set_ylim([-1.05, 1.05])
        xmin = 1e-2
        xmax = 7e1
        ax.set_xlim([xmin, xmax])
        ax.set_xscale('log')
        ax.hlines(-0.33, xmin, xmax, linestyle=':')
        ax.hlines(0.33, xmin, xmax, linestyle=':')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('kHz / ms')
        return fig, ax

    def set_schr_marker_and_color(self, spl, f0):
        """ set schr marker and color
        useful for summarizing graphs
        inputs:
            spl::double
                dB SPL
            f0::double
                fundamental freq
        returns:
            color::string
                dependent on spl 
            marker::string
                dependent on f0
        """
        if spl < 10:
            color = 'lightblue'
        elif spl < 20:
            color = 'blue'
        elif spl < 30:
            color = 'magenta'
        elif spl < 40:
            color = 'orange'
        elif spl < 50:
            color = 'red'
        else:
            color = 'k'
        
        if f0 == 50:
            marker = '^'
        elif f0 == 100:
            marker = 'o'
        elif f0 == 200:
            marker = 's'
        elif f0 == 300:
            marker = 'x'
        elif f0 == 400:
            marker = '.'
        
        return (marker, color)


    def plt_cycle_raster(self, ic, ax=None, method='regular', markersize=3, **kwargs):
        """ plot cycle raster as regular or polar form in the specified axis
        inputs:
            **kwargs: kwargs for plt.plot
        """
        if ax is None:
            if method == 'regular':
                fig, ax = plt.subplots()
            elif method == 'polar':
                fig, ax = plt.subplots(subplot_kw=dict(polar=True))
        if method == 'regular':
            T = self.get_T()[ic]
            burstdur = self.get_stimparam()['BurstDur']
            t_start = 10
            t_end = burstdur - 10
            spkt = self.get_spkt()
            spkt_ic = np.concatenate(spkt[ic, :])
            # analysis time window
            spkt_ic = spkt_ic[(spkt_ic > t_start) & (spkt_ic < t_end)]
            # spkt in cycles (unwrapped)
            spkt_ic_unwrap = spkt_ic / T
            # cycle number for each spkt
            cycle_number = np.floor(spkt_ic_unwrap)
            # wrap spkt in cycles
            spkt_ic_wrap = np.mod(spkt_ic_unwrap, 1)
            # plot 2 cycles
            x = np.concatenate([spkt_ic_wrap, spkt_ic_wrap + 1])
            y = np.concatenate([cycle_number, cycle_number])
            ax.plot(x, y, '.', markersize=markersize, **kwargs)
        elif method == 'polar':
            spkt_cycle = self.get_spkt_in_cycle(ic, method='radian')
            yval = np.arange(len(spkt_cycle))
            ax.plot(spkt_cycle, yval, '.', markersize=markersize)

    def plt_schr_cycle_vm_quantile(self, quantiles=[.25, .5, .75], cycles=2):
        """ plot cycle quantiles of vm to SCHR stims
        ASSUMES there are 9 conditions
        inputs:
            quantiles::list or np.array
                quantiles to plot; default:[.25, .5. 75]
            cycles::int
                cycles to show in the graph; default: 2
        returns:
            fig, ax: the figure and ax objects
        """
        t0, dt = self.get_t0_dt_fromh5()
        ncond = self.get_ncond()
        # ith cond where c = 0; 
        # it should be ncond // 2
        icond_c0 = ncond // 2
        numrows = icond_c0 + 1
        curvatures = self.get_stimparam()['C'][()]

        fig, ax = plt.subplots(nrows=numrows, 
                               figsize=(5, 10), 
                               sharey=True, 
                               gridspec_kw=dict(wspace=0, hspace=0))

        for ic in range(ncond):
            irow = abs(icond_c0 - ic)
            if ic > icond_c0:
                colorstr = 'r'
            else:
                colorstr = 'k'
            cycv = self.get_cyc_avg_vm(ic, method='all')
            cycvq = np.quantile(cycv, quantiles, axis=0)
            if cycles == 2:
                cycvq = np.concatenate([cycvq, cycvq], axis=1)
            cyct = np.arange(0, dt*cycvq.shape[1], dt)
            # plot median first
            ax[irow].plot(cyct, cycvq[1],
                          label='c = %0.2f' % (curvatures[ic]),
                          color=colorstr, 
                          alpha=0.8)
            # fill in between quantiles
            ax[irow].fill_between(cyct, cycvq[0], y2=cycvq[2],
                        color=colorstr,
                        alpha=0.3)
        # add legend when all lines are plotted
        for irow in range(numrows):
            if irow > 0:
                lines = ax[irow].get_lines()
                ax[irow].legend(handles=[lines[0], lines[1]])
            elif irow == 0:
                lines = ax[irow].get_lines()
                ax[irow].legend(handles=[lines[0]])
        fig.suptitle(self.titles_for_plt())

        return fig, ax
            

    def plt_dotraster_all(self, ax=None, marker='.', linewidth=0, markersize=3, **kwargs):
        """ Plot dotraster of all conditions in a single figure
        Use spkt['off'] if it exists
        input:
            ax::ax      input axis object; default: None
            marker::str
                marker for dotraster
                default: '.'
            linewidth::int
                linewidth for dotraster
                default: 0
            markersize::int
                markersize for dotraster
                default: 3
            **kwargs::
                kwargs to plt.plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        ncond = self.get_ncond()
        colors = ['k', 'r']
        exdict = self.get_excluded_dict()
        for ic in range(ncond):
            # only includes nreprec and remove unwanted reps
            spkt_thiscond = self.get_spkt_trimmed(ic)
            numreps = len(spkt_thiscond)
            numspkt_per_rep = [s.size for s in spkt_thiscond]
            y = np.repeat(ic + np.linspace(-0.4, 0.4, numreps),
                        numspkt_per_rep)
            x = self.concat_spkt(spkt_thiscond)
            ax.plot(x, y, 
                    marker=marker,
                    linewidth=linewidth,
                    markersize=markersize,
                    color=colors[np.mod(ic, 2)], 
                    **kwargs)
        (ytick_labels, ylabel_str) = self.cond_val()
        ax.set_yticks(np.arange(ncond)) 
        ax.set_yticklabels(ytick_labels)
        ax.set_ylabel(ylabel_str)
        plt.show()

    def plt_thr(self):
        """ plot all threshold curves of the cell on the current axis
        """
        hiCell = self.get_meta().hiCell.item()
        allmeta = self.get_allmeta()

        if allmeta[(allmeta.hiCell == hiCell) & (allmeta.StimType == 'THR')].size > 0:
            thrmeta = allmeta[(allmeta.StimType == 'THR') & (allmeta.hiCell == hiCell)]
            thrcreated = thrmeta.created.unique()
            for cr in thrcreated:
                thr_seri = thrmeta[thrmeta.created == cr].iloc[0]
                thr_ds = HwlDS(thr_seri.expname, thr_seri.iDataset)
                freq, thr_smth = thr_ds.get_thr()
                plt.plot(freq, thr_smth, label=thr_ds.IDstring())
                plt.xscale('log')
                plt.legend()
        else:
            print('No THR recorded in this cell')
            return
        plt.show()

    def get_thr(self):
        """ returns the freq, smoothed thr for the dataset
        self.StimType must be 'THR'
        returns:
            freq::1-D np.array
                frequencies
            thr_smth::1-D np.array
                smoothed thr values

        """
        if self.get_stimparam()['StimType'] != 'THR':
            print('StimType not THR, quit')
            return
        freq = self._load_spkt()['freq']
        thr = self._load_spkt()['thr']
        thr_smth = hs.smooth(thr, window_len=5)
        # exceptions
        # L19066-19 has good thr when freq > 1000
        if self.expname == 'L19066' and self.iDataset == 19:
            oldf = freq.copy()
            freq = freq[oldf > 1000]
            thr_smth = thr_smth[oldf > 1000]
        return (freq, thr_smth)
        


    def titles_for_plt(self):
        """ Returns a string for the title of the plot
        Returns:
            title_str::str  a string for the title
        """
        meta = self.get_meta()
        # default string: expname-iCell-iRecOfCell
        title_str = '%s-%d-%d, hiCell: %d' % (self.expname,
                                meta['iCell'].item(),
                                meta['iRecOfCell'].item(),
                                meta['hiCell'].item())

        stim = self.get_stimparam()
        stimtype = stim['StimType'][()]
        Fcar = np.unique(stim['Fcar'][()])
        SPL = np.unique(stim['SPL'][()])

        if stimtype == 'SCHR':
            title_str = title_str +' F0: %d Hz, SPL: %d dB' % (Fcar, SPL)
        elif stimtype == 'CFS':
            title_str = title_str +' SPL: %d dB' % (SPL)
        elif stimtype == 'FS':
            Fmod = stim['ModFreq'][()]
            ModDepth = stim['ModDepth'][()]
            title_str = title_str +\
                ' SPL: %d dB; Fmod: %d Hz, mod depth: %d %%' % (SPL, Fmod, ModDepth)
        elif stimtype in ('RC', 'RCM', 'MTF', 'CSPL'):
            title_str = title_str + ' Fcar: %d Hz' % (Fcar)
 
        return title_str

    def get_wbdelay(self):
        """ return the wideband delay in the acoustic system
        returns:
            wbdelay::double (unit: ms)
                wideband delay in the acoustic system
        """
        E = EarlyExp(self.expname)
        (DL, Dphi, wbdelay) = E.calibrate(1000)
        return wbdelay


    def get_spkt_in_cycle(self, ic, method='cycle', t_start=10, minus_wbdelay=True):
        """ Returns spkt in cycle for ith condition
        ONLY spkt in self.reps is included
        input:
            ic::int
                condition number
            method::str
                'cycle' or 'radian'
            t_start::double     unit: ms
                default: 10
            minus_wbdelay::boolean (default: True)
                True: subtract wideband delay in the acoustic system
                False: not
        Returns:
            spkt_cycle::np.array
                spkt in unit of radians or cycles of that condition
        """
        stim = self.get_stimparam()
        t_end = stim['BurstDur']
        # calibrate for wbdelay
        if minus_wbdelay:
            wbdelay = self.get_wbdelay()
            t_start -= wbdelay
            t_end -= wbdelay

        spkt_ic = self.get_spkt_trimmed(ic)
        spkt_concat = self.concat_spkt(spkt_ic)
        spkt_concat = spkt_concat[(spkt_concat > t_start) & 
                                    (spkt_concat < t_end)]
        T = self.get_T()
        spkt_cycle = self.spkt_in_cycle(spkt_concat, T[ic], method=method)
        return spkt_cycle

    def spkt_in_cycle(self, spkt, T, method='radian'):
        """ a general function to return spkt in cycle or radians
        input:
            spkt::np.array
                an array of spike times
            T::int
                period of 
            method::str
                'radian' or 'cycle'
        Returns:
            spkt_cycle::np.array
                an array of spike times in the units of cycle or radians
        """
        if method == 'radian':
            spkt_cycle = np.mod(spkt, T) / T * 2 * np.pi
        elif method == 'cycle':
            spkt_cycle = np.mod(spkt, T) / T
        return spkt_cycle

    def plt_psth(self, icond, binwidth=0.1, xlim=None):
        """ Plot PSTH of icond
        inputs:
            icond::int
                ith condition (0 indexed)
            binwidth::double (unit: ms)
                bin width
                default: 0.1
            xlim::np.array (default: None)
                the xlim
                default: [0, burstdur]
        """
        spkt = self.get_spkt_trimmed(icond)
        allspkt = self.concat_spkt(spkt)
        # make sure it's an scalar
        stimparam = self.get_stimparam()
        burstdur = stimparam['BurstDur'].item()
        numbins = int(burstdur/binwidth)

        if xlim == None:
            xlim = [0, burstdur]
        plt.hist(allspkt,
                                bins=numbins,
                                range=xlim,
                                histtype='step')
        plt.show()

    def get_psth(self, icond, binwidth=0.1, xlim=None):
        """ return the PSTH of icond
        inputs:
            icond::int
                ith condition (0 indexed)
            binwidth::double (unit: ms)
                bin width
                default: 0.1
            xlim::np.array (default: None)
                the xlim
                default: [0, burstdur]
        returns: what np.histogram returns
            hist::np.array; size = n
                the counts in each bin
            bin_edges::np.array; size = n + 1
                the bin_edges 
        """
        spkt = self.get_spkt_trimmed(icond)
        allspkt = self.concat_spkt(spkt)
        # make sure it's an scalar
        stimparam = self.get_stimparam()
        burstdur = stimparam['BurstDur'].item()
        numbins = int(burstdur/binwidth)

        if xlim == None:
            xlim = [0, burstdur]
        hist, bin_edges = np.histogram(allspkt,
                                    bins=numbins,
                                    range=xlim)
        return hist, bin_edges



    def psth_peak_to_steady(self):
        """ return the peak-to-steady state ratio from a PSTH
        of all conditions
        peak_rate: the average rate over 1-ms window centered at the PSTH peak
        steady_rate: the average rate over [10, burstdur] window in the PSTH
        returns:
            ratios::np.array, size = ncond
                the ratio of each condition
                NaN if no available spikes in the condition
        """
        ncond = self.get_ncond()
        ratios = np.zeros(ncond)
        # set default to nan
        ratios.fill(np.nan)
        reps = self.get_reps()
        for ic in range(ncond):
            if len(reps[ic]) == 0:
                continue
            n, bins = self.get_psth(ic)
            edges = bins[:-1]
            idx = np.argmax(n)
            rate_peak = np.mean(n[idx -5: idx + 5])
            rate_steady = np.mean(n[edges > 10])
            if rate_steady == 0:
                continue
            ratios[ic] = rate_peak / rate_steady
        return ratios

    def rate_curve(self, sustained=False):
        """ Return the spike rate of each condition
        input:
            sustained::bool     True to omit the first 10-ms spikes 
                False: start from 0 ms
                True: analysis window between 10 and burstdur ms
        returns:
            rate::array (unit: spk / sec)
                an ncond-element array
        """
        if sustained:
            t_start = 10
        else:
            t_start = 0
        stim = self.get_stimparam()
        burstdur = stim['BurstDur']
        t_end = burstdur
        twindow = t_end - t_start
        ncond = self.get_ncond()
        rate = np.empty([ncond])
        for ic in range(ncond):
            spkt = self.get_spkt_trimmed(ic)
            allspkt = self.concat_spkt(spkt)
            numspikes = np.sum((allspkt < t_end) & (allspkt > t_start))
            if spkt.size == 0:
                rate[ic] = np.nan
            else:
                rate[ic] = (numspikes / spkt.size) / (twindow/1000)
        return rate

    def plt_ratecurve(self, ax=None, sustained=False):
        """ plot the rate curve
        input:
            ax::axis; default: None
                axis to be plotted
                if not assigned, the method will generate a new figure
            sustained::bool     True to omit the first 10-ms spikes 
                False: start from 0 ms
                True: analysis window between 10 and burstdur ms
        """
        if ax is None:
            fig, ax = plt.subplots()
        rate = self.rate_curve(sustained=sustained)
        cval, cunit = self.cond_val()
        ax.plot(cval, rate, '-o')
        ax.set_xlabel(cunit)
        ymax = ax.get_ylim()[1]
        ax.set_ylim([-0.5, ymax])
            
    def cond_val(self):
        """ Return a numerical value for each condition
        Useful for plotting spike rates, vector strengths, etc
        Returns:
            cval::array     an ncond-element array
            unit::string    'Hz', 'dB SPL', etc
        """
        ncond = self.get_ncond()
        cval = np.zeros(ncond)
        unit = ''
        stim = self.get_stimparam()
        if stim['StimType'] in ('FS', 'CFS', 'STEP'):
            cval = np.round(stim['Fcar'][()])
            unit = 'Hz'
        elif stim['StimType'] in ('RC', 'RCN', 'RCM', 'CSPL'):
            cval = np.round(stim['SPL'][()])
            unit = 'dB SPL'
        elif stim['StimType'] in ('MTF', ):
            cval = np.round(stim['Fmod'][()])
            unit = 'Hz'
        elif stim['StimType'] in ('SCHR', ):
            cval = stim['C'][()]
            unit = 'Curvature'
        return (cval, unit)

    def vector_strength(self, spkt, T):
        """ Return vector strength, average phase, SD of t
        input:
            spkt::array             n-element array, n = spike times
            T::double               the period of a cycle
        returns:
            (vs, ph, sdt)::tuple    
            vs::double              vector strength of spike times
            ph::double              average phase in cycle
            sdt::double             SD of spike times in ms
            pval::double            p-values of rayleigh test
        """
        spkt_radian = 2 * np.pi * spkt / T
        z = spkt_radian * 1j
        # make z.dtype to be complex; otherwise causes error
        spkt_z = np.exp(z.astype(complex))
        # average phase
        ph = np.angle(np.mean(spkt_z)) / (2 * np.pi)
        # vector strength
        vs = np.abs(np.mean(spkt_z))
        # rayleigh statistic
        n = len(spkt)
        R = vs * n
        pval = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - R**2)) - (1 + 2 * n))
        # circular standard deviation
        spkt_radianSD = np.sqrt(-2 * np.log(vs))
        # SD in cycle
        sdc = spkt_radianSD / (2 * np.pi)
        # SD in time
        sdt = sdc * T
        return (vs, ph, sdt, pval)

    def vector_strength_allcond(self):
        """ Return vector strengths, average phase, sdt, p-values of all conditions
        Returns:
            vs::array       n-element array; n = ncond
            ph::array       n-element array; n = ncond
            sdt::array      n-element array; n = ncond
            pval::array     n-element array; n = ncond
        """
        ncond = self.get_ncond()
        stim = self.get_stimparam()
        burstdur = stim['BurstDur']
        t_start = 10
        t_end = burstdur
        vs = np.repeat(np.nan, ncond)
        ph = np.repeat(np.nan, ncond)
        sdt = np.repeat(np.nan, ncond)
        pval = np.repeat(np.nan, ncond)
        T = self.get_T()
        for ic in range(ncond):
            spkt = self.get_spkt_trimmed(ic)
            alls = self.concat_spkt(spkt)
            alls = alls[(alls > t_start) & (alls < t_end)]
            if alls.size < 10:
                print("Cond# %d has < 10 spikes, skip" % (ic))
                continue
            (vs[ic], ph[ic], sdt[ic], pval[ic]) = self.vector_strength(alls, T[ic])

        return (vs, ph, sdt, pval)
            
    def get_T(self):
        """ Return period T for the particular stim
        Useful for calculating spike vector strength, average phase, etc
        Returns:
            T::array    an n-element array; n=ncond
        """
        ncond = self.get_ncond()
        stim = self.get_stimparam()
        stimtype = stim['StimType'][()]
        if stimtype in ('RC', 'FS', 'CSPL', 'CFS', 'SCHR'):
            T = 1000 / stim['Fcar'][()]
        if stimtype in ('MTF', 'RCM'):
            T = 1000 / stim['Fmod'][()]
        # make sure T.shape = (ncond, )
        if T.size == 1:
            T = np.repeat(T[()], ncond)
        return T

    def schr_rate_diff(self):
        """ returns the rate difference (spk / cycle) between up / down sweeps
        Returns:
            y::1-D np.array     up - down rate difference (spike / cycle); 
            x::1-D np.array     sweep speed ( kHz / ms)
        """
        if any(self.get_nreprec() < 5):
            print('At least one nreprec < 5, skip')
            return (None, None)
        hschr = Hstim_schr(self.get_stimparam())
        # sweep speed: kHz / ms
        sweepspeed = hschr.sweepspeed / 1e6
        # rate: spike / cycle
        rate = self.rate_curve() / hschr.f0
        # check if c values are symmetric
        c = hschr.c
        if any(c[c < 0] - c[c > 0][::-1] * (-1) > 1e-2):
            print('Curvature not symmetric, quit')
            return (None, None)

        # calculate rate difference
        x = sweepspeed[c < 0]
        y = rate[c < 0] - rate[c > 0][::-1]
        return (x, y)

    def schr_contrast(self):
        """ Return contrast ratio between spike rates to up vs downward sweeps
        * NOTE: if rate_up + rate_down < 2 spikes / sec
        set the contrast at that speed to np.nan
        * use sustained rate; i.e. rate 10 ms after onset
        Returns:
            x::1-D array        sweep speeds; unit: kHz / ms
            y::1-D array        contrast ratio; (up - down) / (up + down)
        """
        if any(self.get_nreprec() < 5):
            print('At least one nreprec < 5, skip')
            return (None, None)
        hschr = Hstim_schr(self.get_stimparam())
        # sweep speed: kHz / ms
        sweepspeed = hschr.sweepspeed / 1e6
        # rate: spike / sec 
        # use sustained rate
        rate = self.rate_curve(sustained=True) 
        # check if c values are symmetric
        c = hschr.c
        if any(c[c < 0] - c[c > 0][::-1] * (-1) > 1e-2):
            print('Curvature not symmetric, quit')
            return (None, None)
        # calculate contrast ratio
        x = sweepspeed[c < 0] 
        rate_up = rate[c < 0]
        rate_down = rate[c > 0][::-1]
        rate_sum = rate_up + rate_down
        y = (rate_up - rate_down) / rate_sum
        # set y to np.nan if rate_sum < 2 spike / sec
        for idx, r in enumerate(rate_sum):
            if r < 2:
                y[idx] = np.nan

        return (x, y)

    def schr_contrast_vs(self):
        """ Return contrast ratio between VECTOR STRENGTHS in up vs downward sweeps
        Returns:
            x::1-D array        sweep speeds; unit: kHz / ms
            y::1-D array        contrast ratio; (up - down) / (up + down)
        """
        if any(self.get_nreprec() < 5):
            print('At least one nreprec < 5, skip')
            return (None, None)
        hschr = Hstim_schr(self.get_stimparam())
        # sweep speed: kHz / ms
        sweepspeed = hschr.sweepspeed / 1e6
        # rate: spike / sec 
        # use sustained rate
        vs, ph, sdt, pval = self.vector_strength_allcond() 
        # check if c values are symmetric
        c = hschr.c
        if any(c[c < 0] - c[c > 0][::-1] * (-1) > 1e-2):
            print('Curvature not symmetric, quit')
            return (None, None)
        # calculate contrast ratio
        x = sweepspeed[c < 0] 
        vs_up = vs[c < 0]
        vs_down = vs[c > 0][::-1]
        vs_sum = vs_up + vs_down
        y = (vs_up - vs_down) / vs_sum

        return (x, y)
 
       
    def schr_vs_diff(self, method='raw'):
        """ Returns a difference in vector strength of same sweep speeds
        Input:
            method::string
                'raw' for raw difference; 'contrast' for contrast ratio
        Returns:
            vs_diff::np.array
                an n-element array of raw or contrast difference, n=(ncond-1)/2
        """
        stim = self.get_stimparam()
        c = stim['C'][()]
        if any(abs(c[c < 0][::-1]) != c[c > 0]):
            print('curvature not symmetric, quit')
            return None
        (vs, ph, sdt, pval) = self.vector_strength_allcond()
        vs_up = vs[c < 0][::-1]
        vs_down = vs[c > 0]
        if method == 'raw':
            vs_diff = vs_up - vs_down
        elif method == 'contrast':
            vs_diff = (vs_up - vs_down) / (vs_up + vs_down)
        return vs_diff


    def get_vm_fromh5(self, icond, denoise=False):
        """ returns the vm of the ith cond (icond) from .h5
        input:
            icond::int
                ZERO-indexed condition number
            denoise::bool (default: False)
                denoise the wave using denoise_wavelet with sym4
        returns: 
            (y, x, t0, dt, scale)::tuple
            where
                y::ndarray
                x::ndarray
                t0::double
                dt::double
                scale::2-element ndarray
        """
        if glob.glob(self.fname_h5) == []:
            print('No file found')
            return 0

        # junction potential
        v_junction = 10

        # NOTE: in .h5 file, cond# is ONE-indexed, i.e. /vm/c001
        with h5.File(self.fname_h5, 'r') as f:
            dsetname = '/vm/c%03d' % (icond + 1)
            dset = f[dsetname]
            dt = f['/vm/dt'][...].item()
            t0 = f['/vm/t0'][...].item()
            scale = f['/vm/scale'][...]
            x = np.arange(t0, dt*dset.shape[1] + t0, dt)
            y = dset[:]*scale[0] + scale[1]
        # if patch recording, subtract 10 mV junction potential
        if self.expname.startswith('B'):
            y -= v_junction
        # denoise the waveform using sym4 
        if denoise:
            y = denoise_wavelet(y, wavelet='sym4', channel_axis=0)
        return (y, x, t0, dt, scale)

    def get_t0_dt_fromh5(self):
        """ Reuturn the t0, dt of the vm from the .h5 file
        Returns:
            t0::double
                t0 for vm trace; unit: ms
            dt::double
                delta t for vm trace; unit: ms
        """
        if glob.glob(self.fname_h5) == []:
            print('No file found')
            return 0
        # NOTE: in .h5 file, cond# is ONE-indexed, i.e. /vm/c001
        with h5.File(self.fname_h5, 'r') as f:
            dt = f['/vm/dt'][...].item()
            t0 = f['/vm/t0'][...].item()
        return (t0, dt)

    def get_cyc_vm_supra_sub(self, ic, t_start=None, t_end=None, ncycles=2, denoise=False):
        """ get vm from .h5 file in each cycle, separated by supra or subthreshold
        inputs:
            ic::int
                ith condition, zero-indexed
            t_start::double unit:ms
                start time for extraction
                default: None --> use the k*T >= 10, k: the smallest positive integer
            t_end::double unit:ms
                end time for extraction;
                default: None; which will be set to (burstdur - 10)
            ncycles::1 or 2 (default: 2)
                1: return one cycle
                2: return 2 cycles
            denoise::bool (default: False)
                denoise the wave using denoise_wavelet with sym4
        returns: a tuple
            (y_supra, y_sub)
            y_supra::2-D array
                ncycles-by-nsamples
            y_sub::2-D array
                ncycles-by-nsamples
            dt::double
                
        """
        if self.get_nreprec()[ic] == 0:
            print("Cond#%d has no recording" % (ic))
            return

        T = 1000 / (self.get_f0()[ic])
        if t_start is None:
            t_start = np.ceil(10 / T) * T
        if t_end is None:
            t_end = self.get_stimparam()['BurstDur'] - T
        spkt = self.get_spkt()
        (y, x, t0, dt, scale) = self.get_vm_fromh5(ic, denoise=denoise)
        npnts_percyc = int(T // dt)
        nreps = len(self.get_reps()[ic])
        ncycs_per_rep = int((t_end - t_start) // T)
        # pre-allocate
        y_supra = np.zeros([ncycs_per_rep * nreps, npnts_percyc])
        y_supra.fill(np.nan)
        y_sub = np.zeros([ncycs_per_rep * nreps, npnts_percyc])
        y_sub.fill(np.nan)
        supra_count = 0
        sub_count = 0
        for irep in self.get_reps()[ic]:
            for icyc in range(ncycs_per_rep):
                x_start = T * icyc + t_start
                i_start = np.nonzero(x >= x_start)[0][0]
                i_end = i_start + npnts_percyc
                y_icyc = y[irep][i_start:i_end]
                # if there is a spike in this cycle
                if any((spkt[ic][irep] >= x[i_start]) & (spkt[ic][irep] < x[i_end])):
                    y_supra[supra_count] = y_icyc
                    supra_count += 1
                else:
                    y_sub[sub_count] = y_icyc
                    sub_count += 1

        # trim the arrays
        if supra_count > 0:
            y_supra = y_supra[:supra_count]
        else:
            # make sure it's an 1-by-npnt np.nan array
            y_supra = y_supra[:supra_count + 1]
        if sub_count > 0:
            y_sub = y_sub[:sub_count]
        else:
            y_sub = y_sub[:sub_count + 1]

        if ncycles == 2:
            y_supra = np.concatenate([y_supra, y_supra], axis=1)
            y_sub = np.concatenate([y_sub, y_sub], axis=1)

        return (y_supra, y_sub, dt)

    def plt_schr_cycvm_multi_spls_horiz(self, 
            t_start=10, 
            f0=100, 
            harlow=1, harhigh=400, 
            recquality=('S', 'A'), subtract_vrest=True,
            denoise=False
            ):
        """ overlay cycle-median vm to multiple sound levels from the same cell
        ONLY applies to c = -1:0.25:1
        Plot HORZONTALLY, for vertical plots, see:
        self.plt_schr_cycvm_multi_spls_vert()
        inputs:
            t_start::double
                start time for extracting cycle
            f0::double
                f0
            harlow::int
                lowest #har
            harhigh::int
                highest #har
            recquality::tuple
                ('S', 'A')
            subtract_vrest::bool
                True: subtract vrest of C=0 in each dataset, and then overlay
                        show the average vrest of all datasets as baseline
                False: show original vrest in each dataset
            denoise::bool (default: False)
                use denoise_wavelet to denoise
        outputs:
            fig
            axs
        """
        curvs = np.arange(-1, 1.25, 0.25)
        sparam = self.schr_param_thiscell()
        spf0 = sparam[(sparam.f0 == f0) &
                      (sparam.harlow == harlow) &
                      (sparam.harhigh == harhigh) &
                      (sparam.recquality.isin(recquality)) 
                     ]
        # drop curvs not in -1:0.25:1
        spf0.index = range(spf0.shape[0])
        idtodrop = []
        for i in spf0.index:
            if not np.all(spf0.loc[i].c == curvs):
                print('%s-%d: not all Curvs are -1:0.25:1' % (self.expname, spf0.iloc[i].iDataset))
                idtodrop.append(i)
        spf0 = spf0.drop(index=idtodrop)

        spf0 = spf0.sort_values(by='spl', ascending=False)
        colors = ['black', 'red', 'orange', 'magenta', 'cyan', 'blue', 'darkblue']
        # use color map to plot multiple SPLs
        #colors = plt.cm.gnuplot2(np.linspace(0, 0.8, spf0.shape[0]))

        fig, axs = plt.subplots(figsize=(6.5, 3), nrows=5, ncols=5, gridspec_kw=dict(wspace=0.1, hspace=0, height_ratios=[0.25, 1, 0.05, 0.25, 1]))

        schr_plotted = np.zeros(9)
        vrest_list = []
        for ids, ds in enumerate(spf0.iloc):
            # only plot c = -1:0.25:1
            if not np.all(ds.c == curvs):
                print('%s-%d: not all Curvs are -1:0.25:1' % (self.expname, ds.iDataset))
                continue
            hdsf0 = HwlDS(self.expname, ds.iDataset)
            if hdsf0.get_vm_fromh5(0) == 0:
                print('%s-%d: no vm.h5 ' % (self.expname, ds.iDataset))
                continue

            # plot c=0 first, get the resting membrane potential
            ic = 4
            vm = hdsf0.get_cyc_avg_vm(ic, t_start=t_start, denoise=denoise)[1]
            vm = np.concatenate([vm, vm])
            #vrest = np.median(vm)
            # 2022.04.17 use hdsf0.get_vrest()
            vrest = hdsf0.get_vrest()
            vrest_list.append(vrest)
            print('%s-%d Vrest: %0.1f mV' % (self.expname, ds.iDataset, vrest))
            t0, dt = hdsf0.get_t0_dt_fromh5()

            for ic in range(9):
                if curvs[ic] <= 0:
                    irow_schr = 0
                    irow_vm = 1
                    icol = ic
                else:
                    irow_schr = 3
                    irow_vm = 4
                    icol = 8 - ic
                vm = hdsf0.get_cyc_avg_vm(ic, t_start=t_start, denoise=denoise)[1]
                # make it two cycles
                vm = np.concatenate([vm, vm])
                # subtract vrest
                if subtract_vrest:
                    vm -= vrest
                t = np.linspace(0, vm.size * dt, vm.size)
                # plot schr wave
                if not schr_plotted[ic]:
                    T = 1000 / ds.f0
                    dur = T * 2
                    # wv: schr waveform
                    wv = hs.schr(ds.f0, int(ds.harlow), int(ds.harhigh), c=curvs[ic], dur=dur)
                    # normalize waveform
                    wv = wv / max(wv)
                    # t = np.arange(0, dur, 1/100)
                    t_schr = np.linspace(0, dur, len(wv))
                    axs[irow_schr, icol].plot(t_schr, wv)
                    axs[irow_schr, icol].axis('off')
                    # set plotted = True
                    schr_plotted[ic] = 1
                axs[irow_vm, icol].plot(t, vm, color=colors[ids], label=ds.spl)

        # remove axes of row#2:
        for a in axs[2, :].flat:
            a.axis('off')

        # remove last column in row#3, 4
        axs[3, -1].axis('off')
        axs[4, -1].axis('off')


        # set ylim for row#1 and row#4
        ymax_row1 = [a.get_ylim()[1] for a in axs[1, :]]
        ymax_row4 = [a.get_ylim()[1] for a in axs[4, :4]]
        ymax = max(max(ymax_row1), max(ymax_row4))

        ymin_row1 = [a.get_ylim()[0] for a in axs[1, :]]
        ymin_row4 = [a.get_ylim()[0] for a in axs[4, :4]]
        ymin = min(min(ymin_row1), min(ymin_row4))

        for a in axs[[1, 4], :].flat:
            a.set_ylim(ymin, ymax)

        for a in axs[1, :-1]:
            a.xaxis.set_visible(False)
            a.spines['bottom'].set_visible(False)

        for a in axs[[1, 4], 1:].flat:
            a.yaxis.set_visible(False)
            a.spines['left'].set_visible(False)

        # set yticks to reveal average vrest
        if subtract_vrest:
            vrest_avg = np.round(np.mean(vrest_list))
            new_yticks = axs[1, 0].get_yticks() + vrest_avg
            axs[1, 0].set_yticklabels(new_yticks)
            axs[-1, 0].set_yticklabels(new_yticks)

        axs[-1, 3].legend()
        
        return fig, axs

    def plt_schr_cycvm_multi_spls_vert(self, 
            t_start=10, 
            f0=100, 
            harlow=1, 
            harhigh=400, 
            recquality=('S', 'A'), 
            subtract_vrest=True, 
            show_curv=True,
            denoise=False
            ):
            """ overlay cycle-median vm to multiple sound levels from the same cell
            ONLY applies to c = -1:0.25:1
            plot VERTICALLY, for horizontal plots, see:
            self.plt_schr_cycvm_multi_spls_horiz()
            inputs:
                t_start::double
                    start time for extracting cycle
                f0::double
                    f0
                harlow::int
                    lowest #har
                harhigh::int
                    highest #har
                recquality::tuple
                    ('S', 'A')
                subtract_vrest::bool 
                    True: subtract vrest of C=0 in each dataset, and then overlay
                            show the average vrest of all datasets as baseline
                    False: show original vrest in each dataset
                show_curv::bool (default: True)
                    True: show curvature on the left of each subplot
                    False: do not show
                denoise::bool (default: False)
                    denoise the waveform using denoise_wavelet
            outputs:
                fig
                axs
            """
            curvs = np.arange(-1, 1.25, 0.25)
            sparam = self.schr_param_thiscell()
            spf0 = sparam[(sparam.f0 == f0) &
                          (sparam.harlow == harlow) &
                          (sparam.harhigh == harhigh) &
                          (sparam.recquality.isin(recquality)) 
                         ]

            # drop curvs not in -1:0.25:1
            spf0.index = range(spf0.shape[0])
            idtodrop = []
            for i in spf0.index:
                if not np.all(spf0.loc[i].c == curvs):
                    print('%s-%d: not all Curvs are -1:0.25:1' % (self.expname, spf0.iloc[i].iDataset))
                    idtodrop.append(i)
            spf0 = spf0.drop(index=idtodrop)

            spf0 = spf0.sort_values(by='spl', ascending=False)
            colors = ['black', 'red', 'orange', 'magenta', 'cyan', 'blue', 'darkblue']
            
            fig, axs = plt.subplots(figsize=(1.65, 3.65), nrows=9, ncols=1, sharex=True, sharey=True, gridspec_kw=dict(hspace=0))

            curv_shown = np.zeros(9)
            vrest_list = []
            for ids, ds in enumerate(spf0.iloc):
                # only plot c = -1:0.25:1
                if not np.all(ds.c == curvs):
                    continue
                hdsf0 = HwlDS(self.expname, ds.iDataset)
                if hdsf0.get_vm_fromh5(0) == 0:
                    print('%s-%d: no vm.h5 ' % (self.expname, ds.iDataset))
                    continue

                # plot c=0 first, get the resting membrane potential
                ic = 4
                vm = hdsf0.get_cyc_avg_vm(ic, t_start=t_start, denoise=denoise)[1]
                vm = np.concatenate([vm, vm])
                vrest = np.median(vm)
                vrest_list.append(vrest)
                print('%s-%d Vrest: %0.1f mV' % (self.expname, ds.iDataset, vrest))
                t0, dt = hdsf0.get_t0_dt_fromh5()

                for ic in range(9):
                    irow = 8 - ic
                    vm = hdsf0.get_cyc_avg_vm(ic, t_start=t_start, denoise=denoise)[1]
                    # make it two cycles
                    vm = np.concatenate([vm, vm])
                    # subtract vrest
                    if subtract_vrest:
                        vm -= vrest
                    t = np.linspace(0, vm.size * dt, vm.size)

                    axs[irow].plot(t, vm, color=colors[ids], label=ds.spl)
                    if show_curv and not curv_shown[ic]:
                        axs[irow].text(-0.5, 3, 'C = %0.2f' % (curvs[ic]), ha='right')
                        curv_shown[ic] = True

            # remove axes and spines
            for a in axs[:-1].flat:
                a.axis('off')
            axs[-1].yaxis.set_visible(False)
            axs[-1].spines['left'].set_visible(False)
            # vm scale
            t_end = 1000 / f0 * 2
            axs[-1].plot([t_end + 1, t_end + 1], [0, 10], color='k')
            axs[-1].text(t_end + 1.5, 3, '10 mV')
            
            return fig, axs


    def plt_schr_vm_trigfreq(self, 
            hschr, 
            ic, 
            ax, 
            delay=0, 
            pkheight=20, 
            prominence=20, 
            distance_ms=1,
            denoise=False
            ):
        """ plot schr wave, vm, and mark the triggering frequency of the vm 
        in a cycle-basis
        inputs:
            hschr::Hstim_schr
                Hstim_schr object
            ic::int
                condition number; zero-indexed
            ax::2-element array; each is an axes
                ax[0] for schr wave;
                ax[1] for vm
            delay::double (unit: ms)
                the latency 
            pkheight::double
                height argument for find_peak in cyclehistogram
            prominence::double
                prominence argument for find_peak in cyclehistogram
            distance_ms::double (unit: ms)
                minimal distance to for two neighboring peaks
            denoise::bool (default: False)
                use denoise_wavelet to denoise the waveform with sym4
        """
        T = 1000 / hschr.f0
        dur = T * 2
        cval, cunit = self.cond_val()
        # wv: schr waveform
        wv = hs.schr(hschr.f0, int(hschr.harlow), int(hschr.harhigh), c=cval[ic], dur=dur)
        # normalize waveform
        wv = wv / max(wv)
        # t = np.arange(0, dur, 1/100)
        t = np.linspace(0, dur, len(wv))
        ax[0].plot(t, wv, linewidth=0.5, zorder=1)

        # 2 cycles of vm
        # plot vm
        # t_start has to be >=10 10 and mod(t_start, T) has to be 0
        vm = self.get_cyc_avg_vm(ic, denoise=denoise)
        y = np.concatenate([vm[0], vm[0]])
        t0, dt = self.get_t0_dt_fromh5()
        x = np.linspace(0, dur, y.size)
        ax[1].plot(x, y)
        # find peaks in the cycle histogram, distance=1ms(10 pnts)
        distance = round(distance_ms / dt)
        pkidx, props = find_peaks(y, height=pkheight, prominence=prominence, distance=distance)
        if len(pkidx) > 0:
            # time of triggering frequency, remove duplicates
            x_peak = np.unique(np.round(np.mod(x[pkidx] - delay, T), 1))

            # remove peaks within 0.1ms
            distant_peaks = np.diff(x_peak) > 0.11
            x_peak = np.concatenate([x_peak[:1], x_peak[1:][distant_peaks]])
            
            print(pkidx, x_peak)
            # find triggering frequency
            triggerfreq = hschr.cycle2freq(x_peak / T, cval[ic])
            for i_xpk, x_pk in enumerate(x_peak):
                # label only non-zero c
                if abs(cval[ic]) > 0:
                    ax[0].text(x_pk, 1 + i_xpk * 0.75, '%0.1fk' % (triggerfreq[i_xpk] * 1e-3), size=6)
                # use vline to mark the triggering frequency in the waveform
                ax[0].vlines(x_pk, -1, 1, linestyle='-', color='magenta', zorder=1.1, linewidth=1)
                # use hline to mark the delay
                ax[0].hlines(-1, x_pk, x_pk + delay, color='magenta', zorder=1.1, linewidth=1)
        # remove axis lines in the 1st row
        ax[0].axis('off')


    def plt_schr_vm_trigfreq_allcond(self, 
            figsize=(6.5, 3), 
            pkheight=-55, 
            prominence=1, 
            distance_ms=1,
            denoise=False
            ):
        """ plot the schr waveform, vm from all conditions
        also mark the triggering frequency by a vline and
        mark the delay by a hline
        inputs:
            figsize::2-element tuple
                figure size in inches
            pkheight::double
                height argument for find_peak in cyclehistogram
            prominence::double
                prominence argument for find_peak in cyclehistogram
            distance_ms::double (unit: ms)
                minimal distance to for two neighboring peaks
            denoise::bool (default:False)
                use denoise_wavelet to denoise the waveform with sym4
        """
        ncols = 5
        fig, axs = plt.subplots(nrows=5, sharex=True, ncols=ncols, figsize=figsize, gridspec_kw=dict(wspace=0.1, hspace=0, height_ratios=[0.25, 1, 0.05, 0.25, 1]))

        # hschr
        hschr = Hstim_schr(self.get_stimparam())
        T = 1000 / hschr.f0
        # get the delay to C = 0
        vs, ph, sdt, pval = self.vector_strength_allcond()
        delay = np.mod(ph[4], 1) * T

        # C = -1:0.25:0
        for icol, ic in enumerate(np.arange(0, 5)):
            self.plt_schr_vm_trigfreq(hschr, ic, 
                    axs[:2, icol], delay=delay, 
                    pkheight=pkheight, prominence=prominence, 
                    distance_ms=distance_ms, denoise=denoise)
            if icol > 0:
                axs[1, icol].spines['left'].set_visible(False)
                axs[1, icol].set_yticks([])
        # C = 1:-0.25:0.25
        for icol, ic in enumerate(np.arange(8, 4, -1)):
            self.plt_schr_vm_trigfreq(hschr, ic, 
                    axs[3:, icol], delay=delay, 
                    pkheight=pkheight, prominence=prominence, 
                    distance_ms=distance_ms, denoise=denoise)
            if icol > 0:
                axs[4, icol].spines['left'].set_visible(False)
                axs[4, icol].set_yticks([])

        # remove axes of row#2:
        for a in axs[2, :].flat:
            a.axis('off')

        # remove last column in row#3, 4
        axs[3, -1].axis('off')
        axs[4, -1].axis('off')


        # set ylim for row#1 and row#4
        ymax_row1 = [a.get_ylim()[1] for a in axs[1, :]]
        ymax_row4 = [a.get_ylim()[1] for a in axs[4, :4]]
        ymax = max(max(ymax_row1), max(ymax_row4))

        ymin_row1 = [a.get_ylim()[0] for a in axs[1, :]]
        ymin_row4 = [a.get_ylim()[0] for a in axs[4, :4]]
        ymin = min(min(ymin_row1), min(ymin_row4))

        for a in axs[1, :]:
            a.set_ylim(ymin, ymax)
            a.xaxis.set_visible(False)
            a.spines['bottom'].set_visible(False)

        for a in axs[4, :]:
            a.set_ylim(ymin, ymax)

        return fig, axs



    def _iplt_vm(self, 
                icond, 
                irep, 
                xcenter=-5, 
                xwinsize=600,
                ycenter=-30, 
                ywinsize=80):
        """ plot function for interactive plot of vm
        input:
            icond::int      ith condition
            irep::int       ith rep
        """
        if self.get_nreprec()[icond] == 0:
            print('No recordings in cond#%d' % (icond))
            return
        (y, x, t0, dt, scale) = self.get_vm_fromh5(icond)
        fig, axs = plt.subplots(figsize=(12, 5), 
                                nrows=2, 
                                sharex=True,
                                gridspec_kw=dict(height_ratios=[1.5, 0.5]))
        # all traces in the condition
        for i in range(y.shape[0]):
            axs[0].plot(x, y[i], color='gray', alpha=0.3)
        # highlight specific trace
        axs[0].plot(x, y[irep], color='black')
        # spike times
        spkt = self.get_spkt()[icond][irep]
        spky = np.ones(spkt.shape)
        axs[1].scatter(spkt, spky, marker='|', s=300)
        # set xlim ylim
        axs[1].set_xlim(xcenter - xwinsize / 2,
                        xcenter + xwinsize / 2)
        axs[0].set_ylim(ycenter - ywinsize / 2,
                        ycenter + ywinsize / 2)
        #fig.show()

    def iplt_vm(self):
        """ interactive plot of vm
        NOTE: ONLY WORKS IN JUPYTER NOTEBOOK
        user can switch icond, irep, 
        scale or pan along x, y axes
        """
        stimparam = self.get_stimparam()
        burstdur = stimparam['BurstDur'].item()
        isi = stimparam['ISI'].item()
        nreprec = self.get_nreprec()
        ncond = self.get_ncond()
        icond_w = widgets.IntSlider(
                value=0,
                min=0, 
                max=ncond - 1,
                continuous_update=False)
        irep_w = widgets.IntSlider(
                value=0, 
                min=0, 
                max=(nreprec[icond_w.value] != 0) * (nreprec[icond_w.value] - 1),
                continuous_update=False)
        # irep_w.max is dependent on icond_w.value
        def update_irep_w(*args):
            irep_w.max = (nreprec[icond_w.value] != 0) * (nreprec[icond_w.value] - 1)
        icond_w.observe(update_irep_w, 'value')

        xcenter_w = widgets.FloatSlider(
                value=0,
                min=-10, 
                max=isi, 
                step=0.1,
                continuous_update=False)
        xwinsize_w = widgets.FloatSlider(
                value=burstdur,
                min=0.1, 
                max=isi * 1.2, 
                step=0.1,
                continuous_update=False)
        ycenter_w = widgets.FloatSlider(
                value=-30, 
                min=-100, 
                max=40, 
                step=0.1,
                continuous_update=False)
        ywinsize_w = widgets.FloatSlider(
                value=50,
                min=0.1, 
                max=100, 
                step=0.1,
                continuous_update=False)
        # interactive plot
        interactive_plot = widgets.interactive(
                self._iplt_vm,
                icond=icond_w, 
                irep=irep_w,
                xcenter=xcenter_w,
                xwinsize=xwinsize_w,
                ycenter=ycenter_w,
                ywinsize=ywinsize_w)
        display(interactive_plot)

    def schr_param(self):
        """ return a dataframe showing schr param
        
        """
        if self.get_stimparam()['StimType'] != 'SCHR':
            print('Not SCHR, quit')
            return

        hschr = Hstim_schr(self.get_stimparam())
        df = pd.DataFrame(dict(f0=hschr.f0,
                                spl=hschr.spl,
                                harlow=hschr.harlow,
                                harhigh=hschr.harhigh,
                                freqlow=hschr.freqlow,
                                freqhigh=hschr.freqhigh,
                                c=[hschr.c],
                                sweepspeed=[hschr.sweepspeed]))
        return df

    def schr_param_thiscell(self):
        """ return a dataframe showing all schr params in this cell
        """
        df = self._stimparam_thiscell_wrapper('SCHR')
        return df

    def fs_param(self):
        """ return a dataframe showing fs param
        
        """
        if self.get_stimparam()['StimType'] != 'FS':
            print('Not FS, quit')
            return

        stim = self.get_stimparam()
        df = pd.DataFrame(dict(spl=stim['SPL'].item(),
                                fmod=stim['ModFreq'].item(),
                                fmoddepth=stim['ModDepth'].item(),
                                fstart=stim['StartFreq'].item(),
                                fend=stim['EndFreq'].item(),
                                fstep=stim['StepFreq'].item(),
                                fstepunit=stim['StepFreqUnit'].item(),
                                fcar=[stim['Fcar'].item()]))
        return df

    def fs_param_thiscell(self):
        """ return a dataframe showing all fs params in this cell
        """
        df = self._stimparam_thiscell_wrapper('FS')
        return df

    def _stimparam_thiscell_wrapper(self, stimtype):
        """ a wrapper function for stimparam of this cell
        """
        # use function handle to return stimparam
        if stimtype == 'FS':
            func_handle = HwlDS.fs_param
        elif stimtype == 'CFS':
            func_handle = HwlDS.cfs_param
        elif stimtype == 'SCHR':
            func_handle = HwlDS.schr_param
        elif stimtype == 'RC':
            func_handle = HwlDS.rc_param
        elif stimtype == 'FM':
            func_handle = HwlDS.fm_param

        # find ds for the stimtype in this cell
        meta_thiscell = self.get_meta_thiscell()
        meta = meta_thiscell[meta_thiscell.StimType == stimtype]
        if meta.size == 0:
            print('No %s in this cell' % (stimtype))
            return

        # get schr_param df from every ds
        for idx, m in enumerate(meta.iloc):
            hds = HwlDS(m.expname, m.iDataset)
            if idx == 0:
                df = func_handle(hds)
                df.insert(0, 'iDataset', m.iDataset)
                df.insert(0, 'iCell', m.iCell)
                df.insert(0, 'iRecOfCell', m.iRecOfCell)
                df.insert(0, 'recquality', m.recquality)
            else:
                newdf = func_handle(hds)
                newdf.insert(0, 'iDataset', m.iDataset)
                newdf.insert(0, 'iCell', m.iCell)
                newdf.insert(0, 'iRecOfCell', m.iRecOfCell)
                newdf.insert(0, 'recquality', m.recquality)
                df = pd.concat([df, newdf])
        return df

    def cfs_param(self):
        """ return a dataframe showing cfs param
        
        """
        if self.get_stimparam()['StimType'] != 'CFS':
            print('Not CFS, quit')
            return

        stim = self.get_stimparam()
        df = pd.DataFrame(dict(spl=stim['SPL'].item(),
                                fstart=stim['StartFreq'].item(),
                                fend=stim['EndFreq'].item(),
                                fstep=stim['StepFreq'].item(),
                                fcar=[stim['Fcar'].item()]))
        return df
        
    def cfs_param_thiscell(self):
        """ return a dataframe showing all cfs params in this cell
        """
        df = self._stimparam_thiscell_wrapper('CFS')
        return df

    def fm_param(self):
        """ return a dataframe showing FM param
        
        """
        if self.get_stimparam()['StimType'] != 'FM':
            print('Not FM, quit')
            return

        stim = self.get_stimparam()
        df = pd.DataFrame(dict(spl=[stim['SPL'].item()],
                                fstart=stim['StartFreq'].item(),
                                fend=stim['EndFreq'].item(),
                                fmode=stim['SweepMode'].item(),
                                updur=stim['upDur'].item(),
                                holddur=stim['holdDur'].item(),
                                downdur=stim['downDur'].item(),
                                burstdur=stim['BurstDur'].item(),
                                risedur=stim['RiseDur'].item(),
                                falldur=stim['FallDur'].item()))
        return df
 
    def fm_param_thiscell(self):
        """ return a dataframe showing all FM params in this cell
        """
        df = self._stimparam_thiscell_wrapper('FM')
        return df

    def rc_param(self):
        """ return a dataframe showing rc param
        
        """
        if self.get_stimparam()['StimType'] != 'RC':
            print('Not RC, quit')
            return

        stim = self.get_stimparam()
        df = pd.DataFrame(dict(fcar=np.unique(stim['Fcar'].item()).item(),
                                risedur=stim['RiseDur'].item(),
                                phasestart=stim['StartPhase'].item(),
                                spl=[stim['SPL'].item()]))
        return df
 
    def rc_param_thiscell(self):
        """ return a dataframe showing all rc params in this cell
        """
        df = self._stimparam_thiscell_wrapper('RC')
        return df

    def get_cmeta(self, reload=False):
        """ returns a cmeta dataframe for the cell of this dataset
        inputs:
            reload::bool
                True: to reload .mat file
                False: not to reload .mat file, use default self.cmeta if it exists
        returns:
            cmeta_df::pd.DataFrame
                contains following columns:
                    hiCell
                    celltype
                    cf
                    thrval
                    sr
                    q10
                    cfmethod
                    IDstring
                    labeled
                    PenDepth
                    iPen
        """
        fname_cmeta = self.cellpath + '/' + self.expname + '.mat'
        if reload:
            cmeta = loadmat(fname_cmeta, squeeze_me=True, struct_as_record=False)['Sc']
        elif self.cmeta is None:
            self.cmeta = self.get_cmeta(reload=True)
            return self.cmeta
        else:
            return self.cmeta

        # if there's only one cell, 'cf' is in dir(cmeta)
        # make cmeta a list
        if 'cf' in dir(cmeta):
            cmeta = [cmeta]
        cmeta_dict = {'expname':[self.expname for c in cmeta],
                      'hiCell': [c.hiCell for c in cmeta],
                      'celltype': [c.celltype for c in cmeta],
                      'cf': [c.cf.cf if 'cf' in dir(c.cf) else None for c in cmeta],
                      'thrval': [c.cf.thrval if 'cf' in dir(c.cf) else None for c in cmeta],
                      'sr': [c.cf.sr if 'cf' in dir(c.cf) else None for c in cmeta],
                      'q10': [c.cf.q10 if 'cf' in dir(c.cf) else None for c in cmeta], 
                      'method': [c.cf.method if 'cf' in dir(c.cf) else None for c in cmeta],
                      'IDstring': [c.cf.IDstring if 'cf' in dir(c.cf) else None for c in cmeta],
                      'labeled':  [c.labeled for c in cmeta],
                      'PenDepth': [c.PenDepth for c in cmeta],
                      'iPen': [c.iPen for c in cmeta]
                     }
        cmeta_exp = pd.DataFrame(data=cmeta_dict)
        cmeta_df = cmeta_exp[cmeta_exp.hiCell == self.get_meta().hiCell.item()]
        return cmeta_df


 



# class Hwlgroup(object):
#     """ A class that's similar to my matlab's hwlgroup
#     """
#     homepath = os.getenv("HOME") + '/early_processeddata'
#     metapath = homepath + '/earlydata_analysis'
#     stimpath = homepath + '/earlydata_stim'
#     spktpath = homepath + '/earlydata_spkt'
#     calibpath = homepath + '/earlydata_calib'
#     cellpath = homepath + '/earlydata_cell'
#     h5path = homepath + '/earlydata_h5'

#     def __init__(self):
#         self.allmeta = self._get_allmeta_from_all_exp()
#         self.allcmeta = self.get_cmeta_from_list(np.unique(self.allmeta.expname))

#     def _get_allmeta_from_all_exp(self):
#         """ Load all meta files from metapath
        
#             and store it in a pd.DataFrame
#             Returns:
#                 a pd.Dataframe
#         """
#         file_list = glob.glob(self.metapath + "/*.mat")
        
#         for i, f in enumerate(file_list):
#             this_meta = self._get_allmeta_from_one_exp(f)
#             if i == 0:
#                 allmeta_df = this_meta
#             else:
#                 allmeta_df = pd.concat([allmeta_df, this_meta])
#         # add cellname
#         cellname_list = []
#         for exp, hi in zip(allmeta_df['expname'], allmeta_df['hiCell']):
#             cellname_list.append((exp + '-' + str(hi)))
#         allmeta_df['cellname'] = cellname_list
#         # remove duplicates (mostly THR)
#         allmeta_df = allmeta_df.drop_duplicates(subset=['created'])
#         # set recquality as category
#         allmeta_df.recquality = allmeta_df.recquality.astype('category')

#         return allmeta_df


#     def _get_allmeta_from_one_exp(self, filename):
#         """ Return a pd.DataFrame equivalent to hwlClass.allmeta
#         """
#         allmeta = loadmat(filename, squeeze_me=True)
#         if allmeta['S'].size == 0:
#             return None
#         elif allmeta['S'].size == 1:
#             # NOTE: works with pandas 1.2.4; doesn't work with 1.3.0
#             allmeta_df = pd.DataFrame(allmeta['S'], range(1))
#         else:
#             allmeta_df = pd.DataFrame(allmeta['S'])
#         # remove duplicates (mostly THR)
#         allmeta_df = allmeta_df.drop_duplicates(subset=['created'])
#         # set recquality as category
#         allmeta_df.recquality = allmeta_df.recquality.astype('category')
#         return allmeta_df
    
#     def get_cmeta_from_list(self, expname_list):
#         """ returns a concatnated cmeta_df from all experiments in the expname_list
#         inputs:
#             expname_list::np.array
#                 a np.array containing expnames
#                 e.g. np.array(['B19091', 'B19092'])
#         returns:
#             allcmeta::pd.DataFrame
#                 a concatenated data frame from all cmeta_df 
#                 contains following columns:
#                     expname
#                     hiCell
#                     celltype
#                     cf
#                     thrval
#                     sr
#                     cfmethod
#                     IDstring
#                     q10
#                     labeled
#                     PenDepth
#                     iPen            
#         """
#         allcmeta = self.get_cmeta_df(expname_list[0])
#         for expname in expname_list[1:]:
#             cmeta_df = self.get_cmeta_df(expname)
#             allcmeta = pd.concat([allcmeta, cmeta_df])
#         return allcmeta
    
#     def get_cmeta_df(self, expname):
#         """ returns a cmeta dataframe for expname
#         inputs:
#             expname::String
#         returns:
#             cmeta_df::pd.DataFrame
#                 contains following columns:
#                     hiCell
#                     celltype
#                     cf
#                     thrval
#                     sr
#                     q10
#                     cfmethod
#                     IDstring
#                     labeled
#                     PenDepth
#                     iPen
#         """
#         fname_cmeta = self.cellpath + '/' + expname + '.mat'
#         cmeta = loadmat(fname_cmeta, squeeze_me=True, struct_as_record=False)['Sc']
#         # if there's only one cell, 'cf' is in dir(cmeta)
#         # make cmeta a list
#         if 'cf' in dir(cmeta):
#             cmeta = [cmeta]
#         cmeta_dict = {'expname':[expname for c in cmeta],
#                       'hiCell': [c.hiCell for c in cmeta],
#                       'celltype': [c.celltype for c in cmeta],
#                       'cf': [c.cf.cf if 'cf' in dir(c.cf) else None for c in cmeta],
#                       'thrval': [c.cf.thrval if 'cf' in dir(c.cf) else None for c in cmeta],
#                       'sr': [c.cf.sr if 'cf' in dir(c.cf) else None for c in cmeta],
#                       'q10': [c.cf.q10 if 'cf' in dir(c.cf) else None for c in cmeta], 
#                       'method': [c.cf.method if 'cf' in dir(c.cf) else None for c in cmeta],
#                       'IDstring': [c.cf.IDstring if 'cf' in dir(c.cf) else None for c in cmeta],
#                       'labeled':  [c.labeled for c in cmeta],
#                       'PenDepth': [c.PenDepth for c in cmeta],
#                       'iPen': [c.iPen for c in cmeta]
#                      }
#         return pd.DataFrame(data=cmeta_dict)

        



# class Hstim_schr(object):
#     """ create schr stim
#     """
#     def __init__(self, stimparam):
#         """ initialize
#         input:
#             hds.stimparam::np.ndarray       the stimparam of SCHR
#         """
#         if len(np.unique(stimparam['Fcar'][()])) > 1:
#             print('Has more than 1 F0, not support yet')
#             return
#         self.EARLYparam = stimparam
#         self.f0 = np.unique(stimparam['Fcar'][()])
#         self.spl = stimparam['SPL'][()]
#         self.c = stimparam['C'][()]
#         (self.harlow, self.harhigh) = self._set_harlimit()
#         (self.freqlow, self.freqhigh) = self._set_freqlimit()
#         self.sweepspeed = self._set_sweepspeed()

    
#     def _set_harlimit(self):
#         """ set real self.harlow, self.harhigh
#         Returns:
#             (low, high)::tuple
#                 low::np.(int)        lowest harmonic number
#                 high::np.(int)       highest harmonic number
#         """
#         stim = self.EARLYparam
#         # EARLY param
#         eHarHigh = np.unique(stim['HarHigh'][()])
#         eFreqHigh = np.unique(stim['FreqHigh'][()])
#         if self.f0 * eHarHigh > eFreqHigh:
#             high = np.floor(eFreqHigh / self.f0)
#         else:
#             high = eHarHigh

#         eHarLow = np.unique(stim['HarLow'][()])
#         eFreqLow = np.unique(stim['FreqLow'][()])
#         if self.f0 * eHarLow > eFreqLow:
#             low = eHarLow
#         else:
#             low = np.ceil(eFreqLow / self.f0)

#         return (low, high)

#     def _set_freqlimit(self):
#         """ set self.freqlow, self.freqhigh
#         Returns:
#             (low, high)::tuple
#                 low::np.(double)     lowest frequency component
#                 high::np.(double)    highest frequency component
#         """
#         low = self.f0.astype('double') * self.harlow
#         high = self.f0.astype('double') * self.harhigh
#         return (low, high)

#     def _set_sweepspeed(self):
#         """ set self.sweepspeed (unit: "Hz / sec")
#         Returns:
#             sweepspeed::np.array        n-element array where n=ncond
#                                         upward is positive; downward is negative
#         """
#         t0 = 1 / self.f0
#         sweepspeed = (self.freqlow - self.freqhigh) / (self.c * t0)
#         return sweepspeed

#     def cycle2freq(self, cycle, c):
#         """ convert instant cycle to instant freq in the schroeder waveform
#         inputs:
#             cycle::double or 1-D array
#                 cycle number; range=[0, 1)
#             c::double
#                 curvature; range=[-1, 1]
#         returns:
#             freq::double or 1-D array
#                 interpolated frequency
#         """
#         if c == 0:
#             return np.nan

#         # make sure cycle is within [0, 1)
#         cycle = np.mod(cycle, 1)

#         if c < 0:
#             t_start = 0
#             t_end = np.abs(c)
#             f_start = self.freqlow.item()
#             f_end = self.freqhigh.item()
#         else:
#             t_start = 1 - c
#             t_end = 1
#             f_start = self.freqhigh.item()
#             f_end = self.freqlow.item()
#         freq = np.interp(cycle, [t_start, t_end], [f_start, f_end])
#         return freq


class EarlyExp(object):
    """ Early Exp objects
    """

    datadir = './earlydata'

    def __init__(self, expname):
        self.expname = expname
        self.matpath = '%s/%s/%s.ExpDef' % (self.datadir, expname, expname)
        self.set_E()

    def set_E(self):
        """ set self.E as E obj in expname.ExpDef
        set squeeze_me=True and struct_as_record=False
        so that user can use . to access the fields
        """
        self.E = loadmat(self.matpath, squeeze_me=True, struct_as_record=False)['E']

    def get_ndataset(self):
        """ return number of datasets recorded
        """
#        return self.E['Status'][0, 0]['Ndataset'][0, 0].item()
        return self.E.Status.Ndataset

    def get_type(self):
        """ return E.ID.Type
                e.g. "gerbil AN", "gerbil OCA"
        """
        return self.E.ID.Type

    def get_allsaved_df(self):
        """ return AllSaved as in E.Status.AllSaved
        returns:
            df::pd.DataFrame
                df['iDataset']
                df['iCell']
                df['iRecOfCell']
                df['StimType']
        """
        df = pd.DataFrame(columns=['iDataset', 'iCell', 'iRecOfCell', 'StimType'])
        df.iDataset = np.array([a.iDataset for a in self.E.Status.AllSaved])
        df.iCell = np.array([a.iCell for a in self.E.Status.AllSaved])
        df.iRecOfCell = np.array([a.iRecOfCell for a in self.E.Status.AllSaved])
        df.StimType = np.array([a.StimType for a in self.E.Status.AllSaved])
        return df

    def get_calib(self, auto=True):
        """ returns the calibration profile loaded from .EarCalib['EC']
        use squeeze_me=True and struct_as_record=False
        if there are more than one file, as the user to choose
        input:
            auto::bool
                True (default): automatically choose the last file
                False : the function will ask the user to choose a file
        returns:
            calib::matlab struct
        """
        fname_wildcard = '/'.join([self.datadir, self.expname, '*.EarCalib'])
        fname_list = glob.glob(fname_wildcard)
        if len(fname_list) > 1:
            if auto:
                fname = fname_list[-1]
            else:
                print(fname_list)
                choice = input('Enter a number to choose one file: ')
                fname = fname_list[int(choice) - 1]
                print('use file: %s' % (fname))
        else:
            fname = fname_list[0]
            
        return loadmat(fname, squeeze_me=True, struct_as_record=False)['EC']

    def calibrate(self, freqs, auto=True):
        """ returns (DL, Dphi, wbdelay) according to input freqs
        inputs:
            freqs::1-D np.array
            auto::bool
                True (default): automatically chooses the ipsi side
                False: the function will ask the user to choose side
        returns: a tuple (DL, Dphi, wbdelay)
            DL::np.array: unit: dB
                calibrated attenuation for each freq in freqs
            Dphi::np.array; unit: cycle
                calibrated phase for each freq in freqs
            wbdelay::double; unit: ms
                wb delay
        """
        calib = self.get_calib()
        transf = calib.Transfer
        recordingside = self.E.Recording.General.Side
        if len(transf) > 1:
            if auto:
                if recordingside == 'Right':
                    ichan = 1
                else:
                    ichan = 0
            else:
                print('Recording side is: ', recordingside)
                ichan = input('Choose calibration side: 1 for left, 2 for right: ')
                ichan = int(ichan) - 1
            transf = calib.Transfer[ichan]
        return self.general_calibrate(transf, freqs)

    def general_calibrate(self, transf, freqs):
        """ returns (DL, Dphi, wbdelay)
        This is a general calibrate function for any transfer struct
        inputs:
            transf::matlab struct
                loaded from calib.Transfer[ichan]
            freqs::np.array
                frequencies to be calibrated
        output:
            DL::np.array; unit: dB
                calibrated magnitude
            Dphi::np.array; unit: cycle
                calibrated phase
            wbdelay::double; unit: ms
                wideband delay
        """
        M = np.interp(freqs, transf.Freq, transf.Ztrf)
        DL = -20 * np.log10(np.abs(M) + 1e-20)
        Dphi = -np.angle(M) / (2 * np.pi)
        wbdelay = transf.WB_delay_ms
        return (DL, Dphi, wbdelay)

    def plt_calib(self, method='amp'):
        """ plot the transfer function
        inputs:
            method::string (default: 'amp')
                method='amp' -> plot magnitude spectrum
                method='phsae' -> plot phase spectrum
        """
        #TODO


        
##### General functions




# HWL's NEURON implementation to feed recorded ANF spikes to an octopus cell model

import HwlData as hw
import importlib
import cnmodel.cells
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from neuron import h
from neuron.units import ms, mV
# use high-level simulation
h.load_file('stdrun.hoc')
from scipy.io import loadmat
import pickle
# import Hsignals as hs
plt.style.use('./paper.mplstyle')

def synStim_many(recordingCell, spkt, numSynapses=10, gsyn=2*1e-3):
    ''' A function only in jupyter notebook for testing

    create and stimulate a synapse on cellList[0]
    use h.VecStim, a class defined by vecevent.mod
    *NOTE* must use nrnivmodl to compile vecevent.mod first
    then, in the folder that contains the original vecevent.mod, execute the .py code
    x
    Input:
        recordingCell::cnmodel class
            a cell class from cnmodel
        spkt::np.ndarray
            the spkt recorded from EARLY
        numSynapses::int
            number of synapses onto the soma
        gsyn::double
            synaptic conductance of each synapse
            default: 2 * 1e-3 (2 nS), from Cao2011
    Returns:
        (soma_v, soma_gklt, soma_syn_i, soma_syn_g, t)
        soma_v::ndarray
            soma Vm
        soma_gklt::ndarray
            soma gklt; condutance of KLT
        soma_syn_i::ndarray
            soma synaptic current
        soma_syn_g::ndarray
            soma synaptic conductance
        t::ndarray
            time points (ms)
        syn_i::ndarray
            synaptic current
    '''

    # create h instances using dictionary
    synapses = {}
    # spkt_: the spike times delivered to ith synapse
    spkt_ = {}
    vecstims = {}
    connections = {}
    syn_i = {}
    syn_g = {}
    for i in range(numSynapses):
        synapses[i] = h.ExpSyn(recordingCell.soma(i/numSynapses))
        synapses[i].tau = 0.35 * ms
        spkt_[i] = h.Vector(spkt[i])
        vecstims[i] = h.VecStim()
        vecstims[i].play(spkt_[i])
        connections[i] = h.NetCon(vecstims[i], synapses[i])
        connections[i].delay = 100 * ms
        # synaptic weight, unit: microS
        # use 2 nS, which is the larger end from Cao2011
        connections[i].weight[0] = gsyn

        syn_i[i] = h.Vector().record(synapses[i]._ref_i)
        syn_g[i] = h.Vector().record(synapses[i]._ref_g)


    # determine what to record
    # vm
    soma_v = h.Vector().record(recordingCell.soma(0.5)._ref_v)
    # conductance of KLT
    soma_gklt = h.Vector().record(recordingCell.soma(0.5)._ref_gklt_klt)
    # time
    t = h.Vector().record(h._ref_t)

    # run simulation
    h.finitialize(-65 * mV)
    h.continuerun(150 * ms)

    # convert recorded data to numpy arrays
    soma_v = np.array(soma_v)
    soma_gklt = np.array(soma_gklt)
    soma_syn_i = np.zeros_like(soma_v)
    soma_syn_g = np.zeros_like(soma_v)
    for i in range(numSynapses):
        soma_syn_i += np.array(syn_i[i])
        soma_syn_g += np.array(syn_g[i])
    t = np.array(t)

    return (soma_v, soma_gklt, soma_syn_i, soma_syn_g, t, syn_i)

def get_simu_new(datasets, ncond, modelcell, ntrials=5, gsyn=2*1e-3):
    """ get simulation results (vm, syn_i, syn_g, t) and store it in arrays
    inputs:

        datasets::list of tuples
            the inputs to be fed in to the cell
        ncond::int
            number of conditions to try

        modelcell::an NUERON cell instance

        ntrials::int
            number of trials
        gsyn::double
            conductance of each synapse
            default: 2*1e-3 (2 nS), Cao2011
    returns:
        t::1-D array
            time axis
        vmarray::3-D array, shape=[ncond][ntrials][vm.size]
            vm simulated
        syn_iarray::3-D array, shape=[ncond][ntrials][syn_i.size]
            synaptic current simulated
        syn_garray::3-D array, shape=[ncond][ntrials][syn_g.size]
            synaptic conductance simulated
        spkt_array::3-D array, shape=[ncond][ntrials][syn_g.size]
            spike histogram fed into that synapse, binsize = 0.1 ms

    """
    for itry in range(ntrials):
        for ic in range(ncond):
            spkt_input = choose_dataset_new(datasets, ic)
            spkt_input = np.concatenate(spkt_input)
            spkt_input.sort()
            spkt_input = np.array([spkt_input])
            hist, edges = np.histogram(spkt_input, bins=200, range=(0, 20))
            (vm, soma_gklt, syn_i, syn_g, t, each_syn_i) = synStim_many(modelcell, spkt_input, numSynapses=1, gsyn=gsyn)
            if (itry == 0) and (ic == 0):
                vmarray = np.empty((ncond, ntrials, vm.size))
                syn_iarray = np.empty((ncond, ntrials, vm.size))
                syn_garray = np.empty((ncond, ntrials, vm.size))
                spkt_array = np.empty((ncond, ntrials, hist.size))

            vmarray[ic][itry] = vm
            syn_iarray[ic][itry] = syn_i
            syn_garray[ic][itry] = syn_g
            spkt_array[ic][itry] = hist

    return (t, vmarray, syn_iarray, syn_garray, spkt_array)

def plt_simu_new(ax, t, data, ic, itry, plt_type='line', threshold=-40):
    """ plot simulation results in one axis
    inputs:
        ax::plt.axes
            the axis to be plotted
        t::1-D array
            time axis for the data
        data::3-D array with shape = [ncond][ntrials][nsamples]
            can be vmarray, syn_iarray, syn_garray, or spkt_array
            spkt_array has size
        ic::int
            condition number
        itry::array
            trial numbers in an 1-D array
        plt_type::str
            'vm': for plotting vm (separating supra vs subthreshold)
            'line': for plotting vm, syn_i, syn_g
            'hist': for plotting spkt_array, which is essentially the y values of cyclehistogram
        threshold::double
            threshold for detecting spikes
            default: -40
    returns:
        pspike::double
            spike probability; only if plt_type=='vm'
    """
    if plt_type == 'line':
        ax.plot(t, data[ic][itry].T, alpha=0.2, linewidth=0.5, color='k')
    elif plt_type == 'vm':
        itry_supra = np.any(data[ic] > threshold, axis=1)
        itry_sub = np.all(data[ic] <= threshold, axis=1)
        nsupra = np.sum(itry_supra)
        nsub = np.sum(itry_sub)
        if nsupra > 0:
            ax.plot(t, data[ic][itry_supra].T, alpha=0.2, color='k')
        if nsub > 0:
            ax.plot(t, data[ic][itry_sub].T, alpha=0.2, color='r')
        # texts showing how many supra vs sub cycles are there
#         ax.text(0.05, 0.9, '%d / %d' % (nsupra, nsupra + nsub), transform=ax.transAxes, color='k', ha='left')
#         ax.text(0.05, 0.75, '%d supra' % (nsupra), transform=ax.transAxes, color='k', ha='left')
#         ax.text(0.05, 0.9, 'p = %0.2f' % (nsupra / (nsupra + nsub)), transform=ax.transAxes, color='k', ha='left')
        return nsupra / (nsupra + nsub)
    elif plt_type == 'hist':
        xsize = data[ic][0].size
        binsize = 0.1
        x = np.arange(0.1, xsize * binsize + binsize, binsize)
        ax.plot(x, data[ic][itry].T, alpha=0.2, color='k')
    ax.grid('on', linestyle=':')

def choose_dataset_new(datasets, ic):
    """ return the spike inputs to the octopus cell
    input:
        datasets::list of tuples
            datasets to be chosen to feed
            a list of tuples, each tuple has the form('expname', 'IDstring', ninputs)
        ic::condition number
            the condition number to be chosen
    returns:
        spkt_input::list
            a list of spike times to be fed into the NEURON model
    """
    spkt_input = []
    for ds in datasets:
        hds = hw.HwlDS(ds[0], ds[1])
        reps = hds.get_reps()[ic]
        spkt = hds.get_spkt()[ic]
        wbdelay = hds.get_wbdelay()

        T = hds.get_T()[ic]
        burstdur = hds.get_stimparam()['BurstDur'].item()
        ncycles = int(burstdur / T)

        # original method; not good because it assumes the ANF always spike in this cycle
#         spkc = hds.get_spkt_in_cycle(ic)
#         spkt_thiscycle = spkc * T

        nsynapses = ds[2]
        # choose n spikes
#         nspikes = 3
        # assign spike train to each synapse
        for i in range(nsynapses):
#             spkt_chosen = np.random.choice(spkt_thiscycle, nspikes)
#             spkt_chosen.sort()
            # select a rep; then select a cycle;
            irep = np.random.choice(reps)
            # choose spkt in this cycle; subtract wbdelay
            spkt_irep = spkt[irep]
            # select a cycle
            # skip the first cycle and the last cycle
            icyc = np.random.choice(ncycles - 1) + 1
            t_start = icyc * T - wbdelay
            t_end = t_start + T
            # select spike times in this cycle
            spkt_chosen = spkt_irep[(spkt_irep >= t_start) & (spkt_irep < t_end)]
            # skip this synapse if there's no spikes
            if spkt_chosen.size == 0:
                continue
            # convert it so that it starts with 0
            spkt_chosen = np.mod(spkt_chosen, T)
            # make it two cycles
            spkt_chosen = np.concatenate([spkt_chosen, spkt_chosen + T])
            spkt_input.append(spkt_chosen)
    return spkt_input

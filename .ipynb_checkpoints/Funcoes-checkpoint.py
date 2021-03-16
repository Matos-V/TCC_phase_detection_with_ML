from qampy import signals, impairments, equalisation, phaserec, helpers
from qampy.theory import ber_vs_es_over_n0_qam as ber_theory
from qampy.helpers import normalise_and_center as normcenter
from qampy.core.filter import rrcos_pulseshaping as lowpassFilter
import numpy as np
from collections.abc import Sequence


def signal_phase_min(M:int,Fb:int,SpS:int, SNR:float,rolloff = 0.01):
    Fs = SpS*Fb
    s = signals.ResampledQAM(M, 2**16, fb=Fb, fs=Fs, nmodes=1,
                         resamplekwargs={"beta": rolloff, "renormalise": True})
    s = impairments.simulate_transmission(s, snr=SNR)
    sfilt = normcenter(lowpassFilter(s, Fs, 1/Fb, 0.001, taps=4001))
    sfm = sfilt.copy()
    t = np.arange(0, s[0].size)*1/s.fs
    A = (np.max(np.abs(sfilt)))*np.exp(1j*np.deg2rad(45))
    Δf = 2*np.pi*(sfilt.fb/2)*t
    sfm = A + sfilt*np.exp(1j*Δf)

    return sfm

def abs_and_phases(sfm):
    amplitudes = np.abs(sfm[0])
    phases = np.angle(sfm[0])
    return {'amplitudes':amplitudes, 'phases':phases}

def dataset_01(sfm, n_features):
    size = 3000
    data = abs_and_phases(sfm)
    amplitudes = data['amplitudes'].copy()
    phases = data['phases'].copy()
    X = np.zeros((size,n_features))
    for n in range(size):
        aux = amplitudes[n:n_features+n]
        X[n] = aux
    y = phases[n_features-1:size+n_features]
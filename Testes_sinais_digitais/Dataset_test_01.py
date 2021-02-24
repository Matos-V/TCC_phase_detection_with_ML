# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.2
# ---

# %%
from qampy import signals, impairments, equalisation, phaserec, helpers
from qampy.theory import ber_vs_es_over_n0_qam as ber_theory
from qampy.helpers import normalise_and_center as normcenter
from qampy.core.filter import rrcos_pulseshaping as lowpassFilter
import matplotlib.pyplot as plt
import numpy as np
# %%
plt.rcParams['font.size'] = 18
plt.rcParams['figure.figsize'] = [16, 8]
plt.rcParams['lines.linewidth'] = 2
# %%
# Geração do sinal QAM

M = 64        # ordem da modulação
Fb = 40e9      # taxa de símbolos
SpS = 4         # amostras por símbolo
Fs = SpS*Fb    # taxa de amostragem
SNR = 40        # relação sinal ruído (dB)
rolloff = 0.01  # Rolloff do filtro formatador de pulso

# Gera sequência de símbolos QAM e a filtra com um filtro formatador de pulso rrc (root-raised cosine)
s = signals.ResampledQAM(M, 2**16, fb=Fb, fs=Fs, nmodes=2,
                         resamplekwargs={"beta": rolloff, "renormalise": True})

# Adiciona ruído gaussiano
s = impairments.simulate_transmission(s, snr=SNR)
# %%
# Plota espectro do sinal QAM em banda base
fig = plt.figure(figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
plt.magnitude_spectrum(s[0], Fs=s.fs, scale='dB', color='C1')
plt.magnitude_spectrum(s[1], Fs=s.fs, scale='dB', color='C0')
plt.title('Base band QAM signal spectrum before LPF')
plt.grid(True)

# Filtra ruído fora da banda do sinal (out-of-band noise)
sfilt = normcenter(lowpassFilter(s, Fs, 1/Fb, 0.001, taps=4001))
fig = plt.figure(figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
plt.magnitude_spectrum(sfilt[0], Fs=s.fs, scale='dB', color='C1')
plt.magnitude_spectrum(sfilt[1], Fs=s.fs, scale='dB', color='C0')
plt.title('Base band QAM signal spectrum after LPF')
plt.grid(True)

# %%
# Gera sinal de fase mínima (sfm(t) = A + s(t)*exp(j*2π*Δf*t))

sfm = sfilt.copy()

t = np.arange(0, s[0].size)*1/s.fs

A = (np.max(np.abs(sfilt)))*np.exp(1j*np.deg2rad(45))
Δf = 2*np.pi*(sfilt.fb/2)*t
sfm = A + sfilt*np.exp(1j*Δf)

plt.figure(figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
plt.magnitude_spectrum(sfm[0], Fs=s.fs, scale='dB', color='C1')
plt.magnitude_spectrum(sfm[1], Fs=s.fs, scale='dB', color='C0')
plt.title('QAM signal spectrum after PM operation')
plt.grid(True)

# %%
plt.figure(figsize=(16, 8))
plt.plot(sfm[0, :10000].real, sfm[0, :10000].imag, linestyle='-', marker='o',
         markerfacecolor='tab:red',
         markeredgecolor='tab:red')
plt.legend(['recieved signal'], loc='lower right')
plt.xlabel('real')
plt.ylabel('imag')
plt.title('Constelation after PM operation')
plt.grid(True)

# %%
n_features = SpS*2
# valor absoluto do sinal -> entrada da rede
amplitudes_train = np.abs(sfm[0,])
phases_train = np.angle(sfm[0, n_features::n_features])  # fase do sinal     -> saída desejada

# valor absoluto do sinal  -> entrada da rede
amplitudes_test = np.abs(sfm[1,])
phases_test = np.angle(sfm[1, n_features::n_features])  # fase do sinal      -> saída desejada

L = 10
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 12))

axes[0].set_title("ABS of the QAM Sinal")
axes[0].plot(t[0:int(n_features*L)], amplitudes_train[0:int(n_features*L)], '-o', color='C0')
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)

axes[1].plot(t[0:L], phases_train[0:L], '-o', color='C1')
axes[1].set_title("Phase of the QAM Sinal")
axes[1].set_xlabel("Tempo (s)")
axes[1].set_ylabel("Phase (rad)")
axes[1].grid(True)

# %%
size = 5000
X_train = amplitudes_train.reshape(-1, n_features)[:size]
X_test = amplitudes_test.reshape(-1, n_features)[:size]

# %%
y_train = phases_train.reshape(-1, 1)[:size]

y_test = phases_test.reshape(-1, 1)[:size]

#%%
dataset_train = np.concatenate((X_train,y_train),axis=1)
dataset_test = np.concatenate((X_test,y_test),axis=1)
# %%

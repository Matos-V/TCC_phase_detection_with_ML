# -*- coding: utf-8 -*-
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
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from qampy import signals, impairments, equalisation, phaserec, helpers
from qampy.theory import ber_vs_es_over_n0_qam as ber_theory
from qampy.helpers import normalise_and_center as normcenter
from qampy.core.filter import rrcos_pulseshaping as lowpassFilter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor

from Funcoes import *

# %%
# %matplotlib inline

# %%
plt.rcParams['font.size'] = 18
plt.rcParams['figure.figsize'] = [16, 8]
plt.rcParams['lines.linewidth'] = 2

# %%
M = 64        # ordem da modulação
Fb = 40e9      # taxa de símbolos
SpS = 4         # amostras por símbolo
Fs = SpS*Fb    # taxa de amostragem
SNR = 40        # relação sinal ruído (dB)
rolloff = 0.01  # Rolloff do filtro formatador de pulso
sfm = qam_signal_phase_min(M,Fb,SpS,SNR)
ordem = 4
dataset , X , y = dataset_01(sfm,ordem)

# %%
X_train = X[:50000]
X_test = X[50000:]

y_train = y[:50000]
y_test = y[50000:]
# %%
scaler = MinMaxScaler()

# %%
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# %%
y_train.shape

# %%
forest = RandomForestRegressor(200)
forest.fit(X_train, y_train)
# %%
y_preds = forest.predict(X_test)
# %%
print('rmse = ', np.sqrt(mean_squared_error(y_test, y_preds)))
print('r2 = ', r2_score(y_test, y_preds))
# %%
plt.figure(figsize=(16, 8))
plt.plot(y_test[:50], '-o')
plt.plot(y_preds[:50], '-o')
plt.xlabel('Symbol')
plt.ylabel('phase (rad)')
plt.legend(['True phases', 'predicted phases'])
plt.title('True and predicted phases comparison')
plt.grid(True)
plt.show()

# %%
sig_abs = scaler.inverse_transform(X_test)[:].reshape((-1))
size = sig_abs.shape[0]
# %%
dataset['amplitudes'].shape

# %%
y_preds.shape

# %%
sinal = dataset['amplitudes'][50000:]*np.exp(1j*y_preds)
# %%
sinal.shape

# %%
plt.magnitude_spectrum(sinal, Fs=Fs, scale='dB')

# %%

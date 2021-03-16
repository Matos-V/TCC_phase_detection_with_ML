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

# %%
plt.rcParams['font.size'] = 18
plt.rcParams['figure.figsize'] = [16, 8]
plt.rcParams['lines.linewidth'] = 2

# %%
dataset_train = pd.read_pickle('Testes_sinais_digitais/dataset_train_03.pkl')
dataset_test = pd.read_pickle('Testes_sinais_digitais/dataset_test_03.pkl')
data_shape = dataset_train.shape[-1]
num_features = data_shape-1

# %%
X_train = dataset_train.drop(
    data_shape-1, axis=1).values.reshape(-1, num_features)
X_test = dataset_test.drop(
    data_shape-1, axis=1).values.reshape(-1, num_features)

y_train = dataset_train[data_shape-1].values.reshape(-1, 1)
y_test = dataset_test[data_shape-1].values.reshape(-1, 1)
# %%
scaler = MinMaxScaler()

# %%
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
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
M = 64        # ordem da modulação
Fb = 40e9      # taxa de símbolos
SpS = 4         # amostras por símbolo
Fs = SpS*Fb    # taxa de amostragem
SNR = 40        # relação sinal ruído (dB)
rolloff = 0.01  # Rolloff do filtro formatador de pulso
# %%
sig_abs = scaler.inverse_transform(X_test)[:].reshape((-1))
size = sig_abs.shape[0]
sinal = sig_abs*np.exp(1j*y_test.reshape(-1))
# %%
t = np.arange(0, s[0].size)*1/Fs

A = (np.max(np.abs(sfilt)))*np.exp(1j*np.deg2rad(45))
Δf = 2*np.pi*(sfilt.fb/2)*t
sfm = A + sfilt*np.exp(1j*Δf)

# %%
signals.from_

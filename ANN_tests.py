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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
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
ordem = 128
dataset , X , y = dataset_02(sfm,ordem)

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
stop = EarlyStopping(monitor='val_loss', patience=5)
# %%
# define model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(ordem,)))
model.add(Dense(128, activation='relu'))
Dropout(0.5)

model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# %%
model.summary()
# %%
model.fit(X_train, y_train, epochs=300, callbacks=[stop],
          validation_data=(X_test, y_test), batch_size=64)
# %%
plt.plot(np.sqrt(model.history.history['loss']))
plt.plot(np.sqrt(model.history.history['val_loss']))
plt.xlabel('Epochs')
plt.ylabel('Root Mean Square Error - RMSE')
plt.grid(True)

# %%
preds = model.predict(X_test)
# %%
print('rmse = ', np.sqrt(mean_squared_error(y_test, preds)))
print('r2 = ', r2_score(y_test, preds))

# %%
plt.figure(figsize=(16, 8))
plt.plot(y_test[:50], '-o')
plt.plot(preds[:50], '-o')
plt.xlabel('Symbol')
plt.ylabel('phase (rad)')
plt.legend(['True phases', 'predicted phases'])
plt.title('True and predicted phases comparison')
plt.grid(True)
plt.show()

# %%
sinal = dataset['amplitudes'][50000:]*np.exp(1j*preds.reshape(-1,))
# %%
dataset['amplitudes'][50000:].shape
# %%
preds.shape
# %%
sinal.shape

# %%
plt.figure(dpi=100, facecolor='w', edgecolor='k')
plt.magnitude_spectrum(sinal, Fs=Fs, scale='dB', color='C1')
plt.title('QAM signal spectrum after phase detection')
plt.grid(True)
#%%
plt.figure(figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
plt.magnitude_spectrum(sfm[0], Fs=Fs, scale='dB', color='C1')
plt.title('QAM signal spectrum after PM operation')
plt.grid(True)
# %%
plt.figure(figsize=(16, 8))
plt.plot(sinal.real[::SpS], sinal.imag[::SpS], linestyle='-', marker='o',
         markerfacecolor='tab:red',
         markeredgecolor='tab:red')
plt.legend(['recieved signal'], loc='lower right')
plt.xlabel('real')
plt.ylabel('imag')
plt.title('Constelation after PM operation')
plt.grid(True)
# %%

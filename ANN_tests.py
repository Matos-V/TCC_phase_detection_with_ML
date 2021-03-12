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
# %%
# %%
plt.rcParams['font.size'] = 18
plt.rcParams['figure.figsize'] = [16, 8]
plt.rcParams['lines.linewidth'] = 2
# %%
dataset_train = pd.read_pickle('Testes_sinais_digitais/dataset_train_02.pkl')
dataset_test = pd.read_pickle('Testes_sinais_digitais/dataset_test_02.pkl')
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
stop = EarlyStopping(monitor='val_loss', patience=5)
# %%
# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(num_features,)))
model.add(Dense(100, activation='relu'))
Dropout(0.5)


model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# %%
model.summary()
# %%
model.fit(X_train, y_train, epochs=300, callbacks=[stop],
          validation_data=(X_test, y_test), batch_size=200)
# %%
plt.plot(np.sqrt(model.history.history['loss']))
plt.plot(np.sqrt(model.history.history['val_loss']))
plt.xlabel('Epochs')
plt.ylabel('Root Mean Square Error - RMSE')
plt.grid(True)

# %%
preds = model.predict(X_test)
# %%
test = np.concatenate((preds, y_test), axis=1)
print(test[:5])

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
#%%
M = 64        # ordem da modulação
Fb = 40e9      # taxa de símbolos
SpS = 4         # amostras por símbolo
Fs = SpS*Fb    # taxa de amostragem
SNR = 40        # relação sinal ruído (dB)
rolloff = 0.01  # Rolloff do filtro formatador de pulso
# %%
sig_abs = scaler.inverse_transform(X_test)[:,0].ravel()
size = sig_abs.shape[0]
preds_phase = preds.ravel()
sinal = sig_abs*np.exp(1j*preds_phase)
#%%
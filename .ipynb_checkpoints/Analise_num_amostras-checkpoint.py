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
import pickle
# %%
# %matplotlib inline
# %%
plt.rcParams['font.size'] = 18
plt.rcParams['figure.figsize'] = [16, 8]
plt.rcParams['lines.linewidth'] = 2
# %%
resultados = dict()
for amostras in [2**x for x in range(5,11)]:
    print(f'{"#"*20} {amostras} Amostras {"#"*20}')
    M = 16        # ordem da modulação
    Fb = 40e9      # taxa de símbolos
    SpS = 4         # amostras por símbolo
    Fs = SpS*Fb    # taxa de amostragem
    SNR = 40        # relação sinal ruído (dB)
    rolloff = 0.01  # Rolloff do filtro formatador de pulso
    sfm, A = sinal_qam_fase_min(M,Fb,SpS,SNR)
    #amostras = 128
    dataset , X , y = dataset_02(sfm,amostras)

    X_train = X[:50000]
    X_test = X[50000:]

    y_train = y[:50000]
    y_test = y[50000:]
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    stop = EarlyStopping(monitor='val_loss', patience=5)
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(amostras,)))
    model.add(Dense(128, activation='relu'))
    Dropout(0.5)

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=300, callbacks=[stop],
            validation_data=(X_test, y_test), batch_size=64,verbose=2)
    predicoes = model.predict(X_test)
    sinal_predito = (dataset['amplitudes'][50000:]*np.exp(1j*predicoes.reshape(-1,))).reshape(1,-1)
    sinal_predito_revertido = reverter_sinal_fase_min(sinal_predito ,A).reshape(1,-1)
    sinal_predito_filtrado = normcenter(lowpassFilter(sinal_predito_revertido, Fs, 1/Fb, 0.001, taps=4001))
    sinal_base_revertido = reverter_sinal_fase_min(sfm[:,50000:60001],A)
    teste = str(amostras) + ' Amostras'
    resultados[teste] = {'Sinais fase minima':(sfm[:,50000:60001], sinal_predito),
                         'Sinais revertidos':(sinal_base_revertido,sinal_predito_revertido),
                         'Sinal predito filtrado':(sinal_predito_filtrado)}
    print('\n\n')
print(' FIM ')

#%%
# nome_arquivo = 'Result_dif_num_amostras.pkl'
# arquivo = open(nome_arquivo,'wb')
# pickle.dump(resultados,arquivo)
# arquivo.close()
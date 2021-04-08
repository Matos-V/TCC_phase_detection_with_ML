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
for M in [4 , 8 , 16 , 32 , 64 , 128]:
    print(f'{"#"*20} {M} QAM {"#"*20}')
    #M = 64        # ordem da modulação
    Fb = 40e9      # taxa de símbolos
    SpS = 4         # amostras por símbolo
    Fs = SpS*Fb    # taxa de amostragem
    SNR = 40        # relação sinal ruído (dB)
    rolloff = 0.01  # Rolloff do filtro formatador de pulso
    sfm, A = sinal_qam_fase_min(M,Fb,SpS,SNR)
    ordem = 128
    dataset , X , y = dataset_02(sfm,ordem)

    X_train = X[:50000]
    X_test = X[50000:]

    y_train = y[:50000]
    y_test = y[50000:]
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    stop = EarlyStopping(monitor='val_loss', patience=5)
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(ordem,)))
    model.add(Dense(128, activation='relu'))
    Dropout(0.5)

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=300, callbacks=[stop],
            validation_data=(X_test, y_test), batch_size=64,verbose=2)
    preds = model.predict(X_test)
    sinal = (dataset['amplitudes'][50000:]*np.exp(1j*preds.reshape(-1,))).reshape(1,-1)
    sinal_revertido = reverter_sinal_fase_min(sinal ,A)
    teste = str(M) + ' QAM'
    resultados[teste] = (sfm, sinal_revertido.reshape(1,-1))
    print('\n\n')
print(' FIM ')
# %%
resultados['128 QAM'][-1][:,::SpS].cal_ber()
# %%
10*np.log10(resultados['4 QAM'][-1][:,::SpS].est_snr())
#%%
QAM64 = normcenter(lowpassFilter(resultados['64 QAM'][-1], Fs, 1/Fb, 0.001, taps=4001))
# %%
plot_constelação(QAM64[:,::SpS])
# %%
plot_constelação(resultados['64 QAM'][-1][:,::SpS])
# %%
QAM64[:,::SpS].cal_ber()
#%%
# nome_arquivo = 'Result_dif_ordens_QAM.pkl'
# arquivo = open(nome_arquivo,'wb')
# pickle.dump(resultados,arquivo)
# arquivo.close()
# %%
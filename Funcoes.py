from qampy import signals, impairments, equalisation, phaserec, helpers
from qampy.theory import ber_vs_es_over_n0_qam as ber_theory
from qampy.helpers import normalise_and_center as normcenter
from qampy.core.filter import rrcos_pulseshaping as lowpassFilter
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def generate_signal(M: int, Fb: int, SpS: int, SNR: float, rolloff=0.01):
    """Criação do sinal QAM em fase mínima.

    Args:
        M (int): Ordem da modulação.
        Fb (int): Taxa de símbolos.
        SpS (int): Amostras por símbolo.
        SNR (float): Relação sinal-ruído.
        rolloff (float, optional): Fator de roll off do filtro cosseno levantado. Defaults to 0.01.

    Returns:
        QAM_signal (ResampledQAM: np.ndarray): Símbolos complexos do sinal QAM gerado.
    """
    Fs = SpS*Fb
    QAM_signal = signals.ResampledQAM(M, 2**16, fb=Fb, fs=Fs, nmodes=1,
                             resamplekwargs={"beta": rolloff, "renormalise": True})
    QAM_signal = impairments.simulate_transmission(QAM_signal, snr=SNR)
    QAM_signal = normcenter(lowpassFilter(QAM_signal, QAM_signal.fs, 1/QAM_signal.fb, 0.001, taps=4001))
    return QAM_signal

def qam_signal_phase_min(signal,A=None):
    """Criação do sinal QAM em fase mínima.

    Args:
        M (int): QAM order.
        Fb (int): Symbol rate.
        SpS (int): samples per symbol.
        SNR (float): Signal-to-noise relation.
        rolloff (float, optional): Roll-off factor for a raised cossine filter. Defaults to 0.01.

    Returns:
        sfm (ResampledQAM: np.ndarray): Complex QAM symbols.
        A (float): 
    """
    sfm = signal.copy()
    t = np.arange(0, sfm[0].size)*1/sfm.fs
    if A is None:
      A = (np.max(np.abs(sfm)))*np.exp(1j*np.deg2rad(45))
    Δf = 2*np.pi*(sfm.fb/2)*t
    sfm = A + sfm*np.exp(1j*Δf)

    return sfm , A

def abs_and_phases(sfm):
    """ Divisão do sinal em fase mínima em componentes de amplitudes e fases.

    Args:
        sfm (ResampledQAM: np.ndarray): Sinal QAM reamostrado com frequência de amostragem maior 
        que a taxa de símbolos.

    Returns:
        data (dict): Dicionário com os arrays contendo as informações de
        amplitudes e fases do sinal.
    """
    amplitudes = np.abs(sfm.copy()[0])
    phases = np.angle(sfm.copy()[0])
    data = {'amplitudes': amplitudes, 'phases': phases}
    return data

def dataset_01(sfm, ordem: int):
    """O dataset criado pela função é o resultado de uma convolução simples ao longo
    do vetor das amplitudes, para a criação das features, e a fase correspondente à
    ultima amostra de amplitude em cada passo da convolução.

    Args:
        sfm (ResampledQAM: np.ndarray): Símbolos do sinal QAM a ser observado.
        ordem (int): Número de amostras de amplitudes a serem observadas para a análise
            de uma amostra de fase.

    Returns:
        data (dict[np.ndarray, np.ndarray]): Dicionário com os arrays contendo as informações de
            amplitudes e fases do sinal.
        X (np.ndarray): Matriz contendo as informações de amplitudes do sinal dispostas de tal 
            forma que qualquer algorítmo de regressão de ML pode utilizar como features.
        y (np.ndarray): Vetor coluna contendo as informações de fases do sinal dispostas de tal 
            forma que qualquer algorítmo de regressão de ML pode utilizar como features.
    """
    size = 60000
    data = abs_and_phases(sfm)
    amplitudes = data['amplitudes'].copy()
    phases = data['phases'].copy()
    X = np.zeros((size, ordem))
    for n in range(size):
        aux = amplitudes[n:ordem+n]
        X[n] = aux
    y = phases[ordem-1:size+ordem-1]
    data['amplitudes'] = data['amplitudes'][ordem-1:size+ordem-1]
    data['phases'] = data['phases'][ordem-1:size+ordem-1]

    return data, X, y.reshape(-1,)

def dataset_02(sfm, ordem: int,size):
    """O dataset criado pela função é o resultado de uma convolução simples ao longo
    do vetor das amplitudes, para a criação das features, e a fase correspondente à
    amostra de amplitude central da janela em cada passo da convolução.

    Args:
        sfm (ResampledQAM: np.ndarray): Símbolos do sinal QAM a ser observado.
        ordem (int): Número de amostras de amplitudes a serem observadas para a análise
            de uma amostra de fase.

    Returns:
        data (dict[np.ndarray, np.ndarray]): Dicionário com os arrays contendo as informações de
            amplitudes e fases do sinal.
        X (np.ndarray): Matriz contendo as informações de amplitudes do sinal dispostas de tal 
            forma que qualquer algorítmo de regressão de ML pode utilizar como features.
        y (np.ndarray): Vetor coluna contendo as informações de fases do sinal dispostas de tal 
            forma que qualquer algorítmo de regressão de ML pode utilizar como features.
    """
    data = abs_and_phases(sfm)
    amplitudes = data['amplitudes'].copy()
    phases = data['phases'].copy()
    X = np.zeros((size, ordem))
    for n in range(size):
        aux = amplitudes[n:ordem+n]
        X[n] = aux
    y = phases[int(ordem/2):size+int(ordem/2)]
    data['amplitudes'] = data['amplitudes'][int(ordem/2):size+int(ordem/2)]
    data['phases'] = data['phases'][int(ordem/2):size+int(ordem/2)]

    return data, X, y

def dataset_03(sfm, ordem: int):
    """O dataset criado pela função é o resultado de uma convolução simples ao longo
    do vetor das amplitudes, para a criação das features, e a fase correspondente à
    primeira amostra de amplitude em cada passo da convolução.

    Args:
        sfm (ResampledQAM: np.ndarray): Símbolos do sinal QAM a ser observado.
        ordem (int): Número de amostras de amplitudes a serem observadas para a análise
            de uma amostra de fase.

    Returns:
        data (dict[np.ndarray, np.ndarray]): Dicionário com os arrays contendo as informações de
            amplitudes e fases do sinal.
        X (np.ndarray): Matriz contendo as informações de amplitudes do sinal dispostas de tal 
            forma que qualquer algorítmo de regressão de ML pode utilizar como features.
        y (np.ndarray): Vetor coluna contendo as informações de fases do sinal dispostas de tal 
            forma que qualquer algorítmo de regressão de ML pode utilizar como features.
    """
    size = 60000
    data = abs_and_phases(sfm)
    amplitudes = data['amplitudes'].copy()
    phases = data['phases'].copy()
    X = np.zeros((size, ordem))
    for n in range(size):
        aux = amplitudes[n:ordem+n]
        X[n] = aux
    y = phases[:size]
    data['amplitudes'] = data['amplitudes'][:size]
    data['phases'] = data['phases'][:size]

    return data, X, y.reshape(-1,)

def dataset_02_CNN(sfm, ordem: int):
    """O dataset criado pela função é o resultado de uma convolução simples ao longo
    do vetor das amplitudes, para a criação das features, e a fase correspondente à
    primeira amostra de amplitude em cada passo da convolução.

    Args:
        sfm (ResampledQAM: np.ndarray): Símbolos do sinal QAM a ser observado.
        ordem (int): Número de amostras de amplitudes a serem observadas para a análise
            de uma amostra de fase.

    Returns:
        data (dict[np.ndarray, np.ndarray]): Dicionário com os arrays contendo as informações de
            amplitudes e fases do sinal.
        X (np.ndarray): Matriz contendo as informações de amplitudes do sinal dispostas de tal 
            forma que qualquer algorítmo de regressão de ML pode utilizar como features.
        y (np.ndarray): Vetor coluna contendo as informações de fases do sinal dispostas de tal 
            forma que qualquer algorítmo de regressão de ML pode utilizar como features.
    """
    size = 60000
    data = abs_and_phases(sfm)
    amplitudes = data['amplitudes'].copy()
    phases = data['phases'].copy()
    X = np.zeros((size, ordem, ordem))
    for n in range(size):
        aux = amplitudes[n:int(ordem*ordem)+n].reshape(ordem,ordem)
        X[n] = aux
    X = X.reshape((size,ordem,ordem,1))
    y = phases[:size]
    data['amplitudes'] = data['amplitudes'][:size]
    data['phases'] = data['phases'][:size]

    return data, X, y.reshape(-1,)

def train_test_datasets(X,y,size):
    X_train = X[:int(0.8*size)]
    X_test = X[int(0.8*size):]
    y_train = y[:int(0.8*size)]
    y_test = y[int(0.8*size):]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train,X_test, y_test

def ANN_model(X_train, y_train,X_test, y_test,patience = 5):
    stop = EarlyStopping(monitor='val_loss', patience=patience)
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(X_test.shape[1],)))
    model.add(Dense(4,activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=300, callbacks=[stop],
          validation_data=(X_test, y_test), batch_size=32)
    return model

def predict_signal(model,dataset,X_test,size):
    preds = model.predict(X_test)
    predicted = dataset['amplitudes'][int(0.8*size):]*np.exp(1j*preds.reshape(-1,))
    return predicted.reshape((1,-1))

def revert_sfm(sfm, A):
    sinal_fase_min = sfm.copy()
    t = np.arange(0, sinal_fase_min[0].size)*1/sinal_fase_min.fs
    Δf = 2*np.pi*(sinal_fase_min.fb/2)*t
    sinal_revertido = (sinal_fase_min[0] - A)/np.exp(1j*Δf)
    sinal_revertido = sinal_revertido.reshape((1,-1))
    sinal_revertido = normcenter(lowpassFilter(sinal_revertido, sinal_revertido.fs, 1/sinal_revertido.fb, 0.001, taps=4001))
    return sinal_revertido.reshape((1,-1))

def plot_constelation(sinal,SpS):
    ber = sinal[0,::SpS].cal_ber()
    plt.plot(sinal[0,:5000:SpS].real, sinal[0,:5000:SpS].imag, 'o')
    plt.xlabel('real')
    plt.ylabel('imaginário')
    plt.title(f'Signal constelation - BER = {ber}')
    plt.grid(True)
    plt.show()

def plot_spectrum(sinal):
    plt.figure(figsize=(16, 8), dpi=100, facecolor='w', edgecolor='k')
    plt.magnitude_spectrum(sinal[0,:5000], Fs=sinal.fs, scale='dB', color='C1')
    plt.grid(True)
    plt.show()
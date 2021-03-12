#%%
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
from sklearn.neighbors import KNeighborsRegressor

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
sfilt = normcenter(lowpassFilter(s, Fs, 1/Fb, 0.001, taps=4001))

sfm = sfilt.copy()

t = np.arange(0, s[0].size)*1/s.fs

A = (np.max(np.abs(sfilt)))*np.exp(1j*np.deg2rad(45))
Δf = 2*np.pi*(sfilt.fb/2)*t
sfm = A + sfilt*np.exp(1j*Δf)

# %%
# valor absoluto do sinal -> entrada da rede
amplitudes_train = np.abs(sfm[0])
phases_train = np.angle(sfm[0, ::SpS])  # fase do sinal     -> saída desejada

# valor absoluto do sinal  -> entrada da rede
amplitudes_test = np.abs(sfm[1])
phases_test = np.angle(sfm[1, ::SpS])  # fase do sinal      -> saída desejada

# %%
X_train = amplitudes_train.reshape(-1, SpS)[:5000]
X_test = amplitudes_test.reshape(-1, SpS)[:5000]

# %%
y_train = phases_train.reshape(-1, 1)[:5000]

y_test = phases_test.reshape(-1, 1)[:5000]

# %%
scaler = MinMaxScaler()
#scaler = StandardScaler()

# %%
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#%%
n_neighbors = 5

for i, weights in enumerate(['uniform', 'distance']):
    knn = KNeighborsRegressor(n_neighbors, weights=weights)
    y_preds = knn.fit(X_train, y_train).predict(X_test)
    print('rmse = ', np.sqrt(mean_squared_error(y_test, y_preds)))
    print('r2 = ', r2_score(y_test, y_preds))

    plt.subplot(2, 1, i + 1)
    plt.plot(y_test[:50], color='darkorange', label='data',)
    plt.plot(y_preds[:50], color='navy', label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                weights))
# %%
r2 = []
for n_neighbors in range(1,30):
    knn = KNeighborsRegressor(n_neighbors, weights='distance',)
    y_preds = knn.fit(X_train, y_train).predict(X_test)
    score = r2_score(y_test, y_preds)
    r2.append(score)
r2 = np.array(r2)
plt.plot(r2)
plt.grid(True)
max_r2 = np.max(r2)
best_neighbors = np.argmax(r2)+1
plt.legend([f'r2 max = {max_r2}'])
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

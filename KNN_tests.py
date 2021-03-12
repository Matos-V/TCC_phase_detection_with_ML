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
from sklearn.neighbors import KNeighborsRegressor

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
n_neighbors = 9

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
for n_neighbors in range(1, 65):
    knn = KNeighborsRegressor(n_neighbors, weights='distance',)
    y_preds = knn.fit(X_train, y_train).predict(X_test)
    score = r2_score(y_test, y_preds)
    r2.append(score)
r2 = np.array(r2)
plt.plot(r2)
plt.grid(True)
max_r2 = np.max(r2)
best_neighbors = np.argmax(r2)+1
plt.legend([f'best neighbors = {best_neighbors}'])
plt.title(f'r2 max = {np.round(max_r2,2)}')
# %%

knn = KNeighborsRegressor(best_neighbors, weights='distance',)
y_preds = knn.fit(X_train, y_train).predict(X_test)
#%%
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

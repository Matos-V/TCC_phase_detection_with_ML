#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %%
dados = '..\\Testes_sinais_digitais\\'

# %%
features = pd.read_pickle(dados+'features.pkl')
labels = pd.read_pickle(dados+'labels.pkl')
# %%
features.head()
# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression

# %%
X_train, X_test, y_train, y_test = train_test_split(features.values, labels.values, test_size=0.25, random_state=42)
# %%
scaler_ft = StandardScaler()
scaler_ft = MinMaxScaler()
X_train = scaler_ft.fit_transform(X_train)
X_test = scaler_ft.transform(X_test)

# %%
linear = LinearRegression()

# %%
linear.fit(X_train,y_train)
# %%
preds = linear.predict(X_test)
# %%
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
# %%
r2_score(y_test, preds)
# %%
np.sqrt(mean_squared_error(y_test, preds))
# %%
mean_absolute_error(y_test,preds)
# %%

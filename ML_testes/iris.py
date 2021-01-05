#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_datasets as tfds
# %%
dataset,info= tfds.load('wine_quality',split='train',with_info=True)
# %%
dataset = tfds.as_dataframe(dataset.take(4500),info)
dataset.head()
# %%

import pickle
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns

with open('mod_05_topic_10_various_data.pkl', 'rb') as fl:
    datasets = pickle.load(fl)
    
autos = datasets['autos']
autos['stroke_ratio'] = autos['stroke'] / autos['bore']
autos[['stroke', 'bore', 'stroke_ratio']].head()


accidents = datasets['accidents']
accidents['LogWindSpeed'] = accidents['WindSpeed'].apply(np.log1p)

sns.set_theme()

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
sns.kdeplot(accidents.WindSpeed, fill=True, ax=axs[0])
sns.kdeplot(accidents.LogWindSpeed, fill=True, ax=axs[1])

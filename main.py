import pickle
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns

with open('../mod_05_topic_10_various_data.pkl', 'rb') as fl:
    datasets = pickle.load(fl)

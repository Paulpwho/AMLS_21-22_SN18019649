import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

mri_labels = pandas.read_csv('./datasets/label.csv')
mri_labels["label"] = mri_labels["label"] == "no_tumor"
mri_labels["label"] = mri_labels["label"] * 1
# no_tumour = 1, tumour = 0
print(mri_labels)
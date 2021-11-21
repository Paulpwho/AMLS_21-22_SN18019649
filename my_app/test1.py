import pandas as pd
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image

image_h = 512
image_w = 512

labels_path = './datasets/label.csv'
image_path = './datasets/image/'
#filename = 'IMAGE_0000.jpg'

mri_labels = pd.read_csv(labels_path)
mri_labels["label"] = mri_labels["label"] == "no_tumor"
mri_labels["label"] = mri_labels["label"] * 1
# no_tumour = 1, tumour = 0
print(mri_labels)


print("Opening files")
image_list = [] # [1]
for filename in mri_labels["file_name"]:
    image_list.append(io.imread(image_path + filename))

print("Converting to np array")
im = np.array(image_list)
im = im[:, :, :, 1]

num_samples = len(mri_labels["file_name"])
im = np.reshape(im, (num_samples, -1))

X = pd.DataFrame(im)
Y = mri_labels["label"]

print("Splitting data")
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, random_state=0)

print("Done")


# [1] https://scikit-image.org/docs/dev/user_guide/getting_started.html
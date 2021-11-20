import pandas
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
filename = 'IMAGE_0000.jpg'

mri_labels = pandas.read_csv(labels_path)
mri_labels["label"] = mri_labels["label"] == "no_tumor"
mri_labels["label"] = mri_labels["label"] * 1
# no_tumour = 1, tumour = 0
print(mri_labels)

images = np.empty((0, image_h * image_w), np.uint8)

for filename in mri_labels["file_name"]:
    im = np.array(Image.open(image_path + filename))
    im = im[:, :, 1]
    im = np.reshape(im, (1, -1))
    images = np.append(images, im, axis=0)
    print(filename)


print(images.shape)




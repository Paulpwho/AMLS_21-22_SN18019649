import pandas as pd
from skimage import io
from skimage.exposure import histogram
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, classification_report, \
    recall_score
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

image_h = 512
image_w = 512

labels_path = './datasets/label.csv'
image_path = './datasets/image/'
# filename = 'IMAGE_0006.jpg'
hist_size = 256

mri_labels = pd.read_csv(labels_path)
# mri_labels["label"][mri_labels["label"] == "no_tumor"] = 0
# mri_labels["label"][mri_labels["label"] == "meningioma_tumor"] = 1
# mri_labels["label"][mri_labels["label"] == "glioma_tumor"] = 2
# mri_labels["label"][mri_labels["label"] == "pituitary_tumor"] = 3
# no_tumour = 1, tumour = 0
print(mri_labels)

print("Opening files")
image_list = []  # [1]
for filename in mri_labels["file_name"]:
    image = io.imread(image_path + filename, as_gray=True)
    # io.imshow(image_list)
    # create and plot histogram according to [2]
    hist, hist_centers = histogram(image)
    image_list.append(hist)

print("Convert to np array")
im = np.array(image_list)
print("Scaling")
# [3] - takes about 6 mins to run
scaler = MinMaxScaler()
im = scaler.fit_transform(im)

# PCA from [5], [6]
def PCAPredict(X, k):
    '''
    Inputs
        X: dataset;
        k: number of Components.

    Return
        SValue: The singular values corresponding to each of the selected components.
        Variance: The amount of variance explained by each of the selected components.
                It will provide you with the amount of information or variance each principal component holds after projecting the data to a lower dimensional subspace.
        Vcomp: The estimated number of components.
    '''

    # the bulit-in function for PCA,
    # where n_clusters is the number of clusters.
    pca = PCA(n_components=k)

    # fit the algorithm with dataset
    principalComponents = pca.fit_transform(X)

    return principalComponents


# project
k1 = 5
recall_arr = []
spec_arr = []
PCA_on = True

if PCA_on:
    Vcomponent = PCAPredict(im, k1)

    X = pd.DataFrame(Vcomponent)
else:
    X = pd.DataFrame(im)

Y = mri_labels["label"]

# As SVM tries to identify numpy dtype=object as multiclass, you need to convert it into this, or alternatively, a list
# as per [5]'s recommendation
# Y = Y.astype('int')

print("Splitting data")
xTrain, xTest, yTrain, yTest = train_test_split(X, Y)

# [4]
def SVM(x_train,y_train, x_test):
    model = SVC()
    model.fit(x_train, y_train)
    #model.  #fit using x_train and y_train
    y_pred = model.predict(x_test)
    return y_pred
# Scikit learn library results
y_pred=SVM(xTrain, yTrain, xTest)

print(confusion_matrix(yTest, y_pred))
print(classification_report(yTest, y_pred))

recall_base = recall_score(yTest, y_pred, average = None)
print("Base model recall: " + str(recall_base))
spec_base = recall_score(yTest, y_pred, pos_label=0, average = None)
print("Base model specificity: " + str(spec_base))

print("done")
'''
#Plotting the Image and the Histogram of gray values
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].imshow(image_list, cmap=plt.cm.gray)
axes[0].axis('off')
axes[0].set_title(filename)
axes[1].plot(hist_centers, hist, lw=2)
axes[1].set_title('histogram of gray values')

fig.show()
'''

# [1] https://scikit-image.org/docs/dev/user_guide/getting_started.html
# [2] https://analyticsindiamag.com/image-feature-extraction-using-scikit-image-a-hands-on-guide/
# [3] From Task 3.9 lab exercises
# [4] From Task 6.3 lab exercises
# [5] https://stackoverflow.com/questions/45346550/valueerror-unknown-label-type-unknown

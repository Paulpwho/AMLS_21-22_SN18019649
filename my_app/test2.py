import pandas as pd
from skimage import io
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, classification_report
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

image_h = 512
image_w = 512

labels_path = './datasets/label.csv'
image_path = './datasets/image/'
# filename = 'IMAGE_0000.jpg'

mri_labels = pd.read_csv(labels_path)
mri_labels["label"] = mri_labels["label"] == "no_tumor"
mri_labels["label"] = mri_labels["label"] * 1
# no_tumour = 1, tumour = 0
print(mri_labels)

print("Opening files")
image_list = []  # [1]
for filename in mri_labels["file_name"]:
    image_list.append(io.imread(image_path + filename))

print("Converting to np array")
im = np.array(image_list)
im = im[:, :, :, 1]

num_samples = len(mri_labels["file_name"])
im = np.reshape(im, (num_samples, -1))

print("Applying scaler")

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
k1 = 2
Vcomponent = PCAPredict(im, k1)
print(Vcomponent)

X = pd.DataFrame(Vcomponent, columns=['principal component 1', 'principal component 2'])
Y = mri_labels["label"]

finalDf = pd.concat([X, Y], axis=1)
print(finalDf)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 component PCA', fontsize=20)
targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['label'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c=color
               , s=50)
ax.legend(["tumour", "no tumour"])
ax.grid()
fig.show()

print("done")

'''

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, random_state=0)





def logRegrPredict(xTrain, yTrain, xTest):
    logreg = LogisticRegression(solver='lbfgs')
    logreg.fit(xTrain, yTrain)
    yPred = logreg.predict(xTest)
    return yPred


yPred = logRegrPredict(xTrain, yTrain, xTest)
print(yPred)
print(confusion_matrix(yTest, yPred))
print('Accuracy on test set: ' + str(accuracy_score(yTest, yPred)))
print(classification_report(yTest, yPred))  # text report showing the main classification metrics
'''

'''
X = pd.DataFrame(im)
Y = mri_labels["label"]

print("Splitting data")
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, random_state=0)

print("Done")


# [1] https://scikit-image.org/docs/dev/user_guide/getting_started.html

'''
'''
# Plotting a histogram of the data [4]
df = pd.concat([Y, pd.DataFrame(X)], axis=1)
no_tumour = df.loc[df.label==1, 0]
tumour = df.loc[df.label==0, 0]
kwargs = dict(alpha=0.5, bins=100)

plt.hist(no_tumour, **kwargs, color='g', label='no_tumour')
plt.hist(tumour, **kwargs, color='b', label='tumour')
plt.legend();

# Historgram of data with MinMaxScaler
yTrain = yTrain.to_numpy()
pd.concat([pd.DataFrame(yTrain), pd.DataFrame(xTrain)], axis=1)
no_tumour = df.loc[df.label==1, 0]
tumour = df.loc[df.label==0, 0]
kwargs = dict(alpha=0.5, bins=100)

plt.hist(no_tumour, **kwargs, color='g', label='no_tumour')
plt.hist(tumour, **kwargs, color='b', label='tumour')
plt.legend();
'''

# [2] https://thispointer.com/count-values-greater-than-a-value-in-2d-numpy-array-matrix
# [3] From Task 3.9 lab exercises
# [4] https://www.machinelearningplus.com/plots/matplotlib-histogram-python-examples/
# [5] From Task 5.14 lab exercises
# [6] https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

import time

import pandas as pd
from skimage import io
from skimage.exposure import histogram
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, tree
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, classification_report, \
    recall_score
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

image_h = 512
image_w = 512

labels_path = './datasets/label.csv'
image_path = './datasets/image/'
# filename = 'IMAGE_0006.jpg'
hist_size = 256

mri_labels = pd.read_csv(labels_path)
mri_labels["label"] = mri_labels["label"] == "no_tumor"
mri_labels["label"] = mri_labels["label"] * 1
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

comp_arr = range(5, 50, 1)
recall_arr = []
spec_arr = []
PCA_on = True

for k1 in comp_arr:
    # tic = time.perf_counter() [4]
    if PCA_on:
        Vcomponent = PCAPredict(im, k1)

        X = pd.DataFrame(Vcomponent)
    else:
        X = pd.DataFrame(im)

    Y = mri_labels["label"]

    print("Splitting data")
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y)

    # From lecture notes task 4.6
    clf = RandomForestClassifier(n_estimators=100)

    clf.fit(xTrain, yTrain)

    y_pred = clf.predict(xTest)

    # toc = time.perf_counter()

    # print(f'Test feature {np.array(xTest.iloc[0])}\n True class {yTest.iloc[0]}\n predict class {y_pred[0]}')

    # print(confusion_matrix(yTest, y_pred))
    recall = recall_score(yTest, y_pred)
    recall_arr.append(recall)

    spec = recall_score(yTest, y_pred, pos_label=0)
    spec_arr.append(spec)
    print("Components: " + str(k1))
    # print(toc-tic)
    # print('Accuracy on test set: ' + str(accuracy_score(yTest, yPred)))
    # print(classification_report(yTest, y_pred))  # text report showing the main classification metrics
plt.figure()
plt.plot(comp_arr, recall_arr)
plt.xlabel("Number of principle components")
plt.ylabel("Recall")
plt.ylim(0, 1)
plt.grid()
plt.show()
print("done")

'''
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

'''
def visualise_tree(tree_to_print):
    plt.figure()
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), dpi=800)
    tree.plot_tree(tree_to_print,
                   feature_names=X.columns.tolist(),
                   class_names=["t", "no t"],
                   filled=True,
                   rounded=True);
    plt.show()


for index in range(0, 5):
    visualise_tree(clf.estimators_[index])
'''
# [1] https://scikit-image.org/docs/dev/user_guide/getting_started.html
# [2] https://analyticsindiamag.com/image-feature-extraction-using-scikit-image-a-hands-on-guide/
# [3] From Task 3.9 lab exercises
# [4] https://realpython.com/python-timer/

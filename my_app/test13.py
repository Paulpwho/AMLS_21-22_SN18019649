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

lab = True # Replace string labels with int labels?
plot_graph = True # plot an svm graph? (Takes a long time)

if lab:
    mri_labels["label"][mri_labels["label"] == "no_tumor"] = 0
    mri_labels["label"][mri_labels["label"] == "meningioma_tumor"] = 1
    mri_labels["label"][mri_labels["label"] == "glioma_tumor"] = 2
    mri_labels["label"][mri_labels["label"] == "pituitary_tumor"] = 3

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
if lab:
    Y = Y.astype('int')

print("Splitting data")
xTrain, xTest, yTrain, yTest = train_test_split(X, Y)

# [4]
model = SVC()
model.fit(xTrain, yTrain)
#model.  #fit using x_train and y_train
y_pred = model.predict(xTest)

print(confusion_matrix(yTest, y_pred))
print(classification_report(yTest, y_pred))

### Visualise the SVM ###
def make_meshgrid(x, y, a, b, c, h=.5):

    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    a_min, a_max = a.min() - 1, a.max() + 1
    b_min, b_max = b.min() - 1, b.max() + 1
    c_min, c_max = c.min() - 1, c.max() + 1
    xx, yy, aa, bb, cc = np.meshgrid(np.arange(x_min, x_max, h),
                                     np.arange(y_min, y_max, h),
                                     np.arange(a_min, a_max, h),
                                     np.arange(b_min, b_max, h),
                                     np.arange(c_min, c_max, h))
    return xx, yy, aa, bb, cc

def plot_contours(ax, clf, xx, yy, aa, bb, cc, **params):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel(), aa.ravel(), bb.ravel(), cc.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx[:, :, 9, 9, 9], yy[:, :, 9, 9, 9], Z[:, :, 9, 9, 9], **params)
    return out

if plot_graph:
    fig, sub = plt.subplots(1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    X0, X1, X2, X3, X4 = X[0], X[1], X[2], X[3], X[4]

    print("Making meshgrid")
    xx, yy, aa, bb, cc = make_meshgrid(X0, X1, X2, X3, X4)

    print("Plotting graph")
    plot_contours(sub, model, xx, yy, aa, bb, cc, cmap=plt.cm.coolwarm, alpha=0.8)
    sub.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    sub.set_xlim(xx.min(), xx.max())
    sub.set_ylim(yy.min(), yy.max())
    sub.set_xlabel('Principal Component X0')
    sub.set_ylabel('Principal Component X1')
    sub.set_xticks(())
    sub.set_yticks(())
    sub.set_title("SVC default settings")
    plt.savefig("test13_SVC.png")
    plt.clf()
# recall_base = recall_score(yTest, y_pred, average = None)
# print("Base model recall: " + str(recall_base))
# spec_base = recall_score(yTest, y_pred, pos_label=0, average = None)
# print("Base model specificity: " + str(spec_base))

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

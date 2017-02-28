from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#import cv2
import numpy as np
# sklearn==machine learning in python, other option is opencv
from sklearn import svm


mat_dict = loadmat('ex6data1.mat')
X = np.array(mat_dict['X'])
y = np.array(mat_dict['y'])

for x1,x2,yy in zip(X[:,0], X[:,1], y):
    if yy == 0:
        marker = 'o'
        color = 'red'
    elif yy==1:
        marker = 'x'
        color='blue'
    else:
        print("problem with y:",y)
        raise ValueError
    plt.scatter(x1, x2, marker=marker, color=color)

red_patch = mpatches.Patch(color='red', label='Not admitted')
blue_patch = mpatches.Patch(color='blue', label='Admitted')
plt.legend(handles=[red_patch, blue_patch])
#plt.show()

#==================== Part 2: Training Linear SVM ====================
#  The following code will train a linear SVM on the dataset and plot the
#  decision boundary learned.
#
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_svm/py_svm_opencv/py_svm_opencv.html#svm-opencv
# Load from ex6data1:

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)
print (X.shape, y.shape)

C = 1.0  # SVM regularization parameter
#svc = svm.SVC(kernel='linear', C=C).fit(X, y.flatten())
svc = svm.SVC(kernel='rbf', C=C).fit(X, y.flatten())

# create a mesh to plot in
h = .05  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
#plt.xlabel('Sepal length')
#plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
#plt.title(titles[i])

plt.show()

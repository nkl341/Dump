# program		vandenburg_demo_SVMvsNaiveBayes_irisData.py
# purpose	    Demonstrate SVM vs Basian Classifiers
# usage         script
# notes         (1) 
# date			02/20/2024
# programmer    Colton Vandenburg

import datetime          # used for getting the date
import os                # used for getting the basic file name (returns lower case)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing    # Import label encoder
from sklearn.model_selection import train_test_split
from sklearn import svm
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler  # Import StandardScaler for data normalization
from sklearn.datasets import load_iris
# ============== COMMON INITIALIZATION =====================
date_o = datetime.datetime.today()
date_c = date_o.strftime('%m/%d/%Y')
programName_c = os.path.basename(__file__)                     # case insensitive file name


script_dir = os.path.dirname(os.path.abspath(__file__))     # Get the directory of the current script
csv_file_path= os.path.join(script_dir, 'irisData.csv')   # Construct the full path to the CSV file, now Mac friendly :)


irisData_df = pd.read_csv(csv_file_path, usecols=[0, 1, 4])

ix = str.find(programName_c,'.')

fileName_c = 'irisData.csv'
programMsg_c = programName_c + ' (' + date_c + ')'


authorName_c = 'Colton Vandenburg'
figName_c = programName_c[:ix]+'_fig.png'

#=================Data Preprocessing=======================

tts_rs = 3

iris = load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=tts_rs)

#The reason for using load_iris is for the simplicity of the format of the data when imported this way. There is still a random state.

#=====SVM Model=====
#I had to google this part for visualization of the SVC algorithm.
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


model = svm.SVC(kernel='linear')
clf = model.fit(X, y)


#=================Naive Bayes Model=======================
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

plt.figure(num=1, figsize=(11.2, 5.2))
plt.subplot(1, 2, 1)
plt.subplots_adjust(wspace=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('SVM Decision Boundaries')
plot_contours(plt.gca(), clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Naive Bayes Decision Boundaries')
plot_contours(plt.gca(), gnb, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
plt.tight_layout()       #
plt.rcParams.update({'font.size': 8}) 
plt.subplot(position=[0.0500,    0.94,    0.02500,    0.02500]) # U-left
plt.axis('off')
plt.text(0,.5, programMsg_c, fontsize=8)

plt.subplot(position=[0.550,    0.94,    0.02500,    0.02500]) # U-right
plt.axis('off')
plt.text(0,.5, authorName_c, fontsize=8)

plt.subplot(position=[0.0500,    0.02,    0.02500,    0.02500]) # L-left
plt.axis('off')
plt.text(0,.5, fileName_c, fontsize=8)
plt.show()

plt.show()


# program		vandenburg_demo_KNNvsKMeans_irisData_v1.py
# purpose	    Demonstrate SVM vs Basian Classifiers
# usage         script
# notes         (1) 
# date			02/29/2024
# programmer    Colton Vandenburg

import datetime          # used for getting the date
import os                # used for getting the basic file name (returns lower case)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing    # Import label encoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler  # Import StandardScaler for data normalization
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans 
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
tts_rs = 4
irisDataT_df, irisDataV_df = train_test_split(irisData_df, test_size=0.5, random_state=tts_rs) 
irisDataT_num_df = irisDataT_df.drop("type", axis = 1) 

# ============== get labels, preprocess train data except for scaling  ======================
irisDataT_labels = irisDataT_df["type"]
label_encoder = preprocessing.LabelEncoder() 

irisDataT_labels_v = label_encoder.fit_transform(irisDataT_labels)
# ================= get labels, preprocess verify data ==================
irisDataV_num_df = irisDataV_df.drop("type", axis=1)    # numerical data only
irisDataV_labels = irisDataV_df["type"]
irisDataV_labels_v = label_encoder.fit_transform(irisDataV_labels)

# ================= K Nearest Neighbor ==================
# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=10)

# Train the model using the training data
knn.fit(irisDataT_num_df, irisDataT_labels_v)

# Define the meshgrid range
x_min, x_max = irisDataV_df["septalLength_cm"].min() - 1, irisDataV_df["septalLength_cm"].max() + 1
y_min, y_max = irisDataV_df["septalWidth_cm"].min() - 1, irisDataV_df["septalWidth_cm"].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict the labels for the meshgrid points
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Create a color map for the plot
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

# ============= K Means Classifier ================
# Create KMeans classifier
kmeans_classifier = KMeans(n_clusters=3, random_state=0)

# Train the model using the training data
kmeans_classifier.fit(irisDataT_num_df)

# Predict the labels for the verify data
kmeans_labels = kmeans_classifier.predict(irisDataV_num_df)

# Define the meshgrid range
x_min_kmeans, x_max_kmeans = irisDataV_num_df["septalLength_cm"].min() - 1, irisDataV_num_df["septalLength_cm"].max() + 1
y_min_kmeans, y_max_kmeans = irisDataV_num_df["septalWidth_cm"].min() - 1, irisDataV_num_df["septalWidth_cm"].max() + 1
xx_kmeans, yy_kmeans = np.meshgrid(np.arange(x_min_kmeans, x_max_kmeans, 0.02),
                                   np.arange(y_min_kmeans, y_max_kmeans, 0.02))

# Predict the labels for the meshgrid points
Z_kmeans = kmeans_classifier.predict(np.c_[xx_kmeans.ravel(), yy_kmeans.ravel()])
Z_kmeans = Z_kmeans.reshape(xx_kmeans.shape)

# Create a color map for the plot
cmap_light_kmeans = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

# ============= Plotting the decision boundaries ================

# Plotting the decision boundaries
plt.figure(figsize=(12, 6))

# Plot the decision boundaries for K Nearest Neighbor
plt.subplot(1, 2, 1)
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(irisDataV_num_df["septalLength_cm"], irisDataV_num_df["septalWidth_cm"], c=irisDataV_labels_v, cmap=cmap_light, edgecolor='k')
plt.xlabel('septalLength_cm')
plt.ylabel('septalWidth_cm')
plt.title('K Nearest Neighbor')

# Plot the decision boundaries for K Means Classifier
plt.subplot(1, 2, 2)
plt.pcolormesh(xx_kmeans, yy_kmeans, Z_kmeans, cmap=cmap_light_kmeans)
plt.scatter(irisDataV_num_df["septalLength_cm"], irisDataV_num_df["septalWidth_cm"], c=irisDataV_labels_v, cmap=cmap_light_kmeans, edgecolor='k')
plt.xlabel('septalLength_cm')
plt.ylabel('septalWidth_cm')
plt.title('K Means Classifier')


# ================== Labeling the Plot ===============================
plt.figure(num=1, figsize=(11.2, 5.2))        #
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





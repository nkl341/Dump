# program		vandenburg_demo_logRegression_irisData_v1.py
# purpose	    Demonstrate logistic regression on iris data
# usage         script
# notes         (1) 
# date			2/15/2024
# programmer   Colton Vandenburg

import datetime          # used for getting the date
import os                # used for getting the basic file name (returns lower case)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing    # Import label encoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# ============== COMMON INITIALIZATION =====================
date_o = datetime.datetime.today()
date_c = date_o.strftime('%m/%d/%Y')
programName_c = os.path.basename(__file__)                     # case insensitive file name


script_dir = os.path.dirname(os.path.abspath(__file__))     # Get the directory of the current script
csv_file_path= os.path.join(script_dir, 'irisData.csv')   # Construct the full path to the CSV file, now Mac friendly :)


irisData_df = pd.read_csv(csv_file_path)    # Load the CSV file

ix = str.find(programName_c,'.')

fileName_c = 'irisData.csv'
programMsg_c = programName_c + ' (' + date_c + ')'


authorName_c = 'Colton Vandenburg'
figName_c = programName_c[:ix]+'_fig.png'


# ========== get data frame, split, get numerical data ==============

tts_rs = 10
irisDataT_df, irisDataV_df = train_test_split(irisData_df, test_size=0.5, random_state=tts_rs) 
irisDataT_num_df = irisDataT_df.drop("type", axis = 1)    # numerical data only

# ============== get labels, preprocess train data except for scaling  ======================
irisDataT_labels = irisDataT_df["type"]
label_encoder = preprocessing.LabelEncoder() 

irisDataT_labels_v = label_encoder.fit_transform(irisDataT_labels)



# ================= get labels, preprocess verify data ==================
irisDataV_num_df = irisDataV_df.drop("type", axis=1)    # numerical data only
irisDataV_labels = irisDataV_df["type"]
irisDataV_labels_v = label_encoder.fit_transform(irisDataV_labels)


# ================== scaling ============================================
scaler = StandardScaler()
irisDataT_num_std = scaler.fit_transform(irisDataT_num_df)
irisDataV_num_std = scaler.transform(irisDataV_num_df)

scaler = MinMaxScaler()
irisDataT_num_norm = scaler.fit_transform(irisDataT_num_df)
irisDataV_num_norm = scaler.transform(irisDataV_num_df)

# ================== logistic regression ===============================
# Fit logistic regression model on standardized data
logreg_std = LogisticRegression()
logreg_std.fit(irisDataT_num_std, irisDataT_labels_v)

# Fit logistic regression model on normalized data
logreg_norm = LogisticRegression()
logreg_norm.fit(irisDataT_num_norm, irisDataT_labels_v)

# Fit logistic regression model on original data
logreg_orig = LogisticRegression()
logreg_orig.fit(irisDataT_num_df, irisDataT_labels_v)

# Compute accuracy scores for training data
train_scores = []
train_scores.append(accuracy_score(irisDataT_labels_v, logreg_std.predict(irisDataT_num_std)))
train_scores.append(accuracy_score(irisDataT_labels_v, logreg_norm.predict(irisDataT_num_norm)))
train_scores.append(accuracy_score(irisDataT_labels_v, logreg_orig.predict(irisDataT_num_df)))

# Compute accuracy scores for testing data
test_scores = []
test_scores.append(accuracy_score(irisDataV_labels_v, logreg_std.predict(irisDataV_num_std)))
test_scores.append(accuracy_score(irisDataV_labels_v, logreg_norm.predict(irisDataV_num_norm)))
test_scores.append(accuracy_score(irisDataV_labels_v, logreg_orig.predict(irisDataV_num_df)))

# Print accuracy scores for training data
print("Training Data Accuracy Scores:")
print("Standardized Data: ", train_scores[0])
print("Normalized Data: ", train_scores[1])
print("Original Data: ", train_scores[2])

# Print accuracy scores for testing data
print("Testing Data Accuracy Scores:")
print("Standardized Data: ", test_scores[0])
print("Normalized Data: ", test_scores[1])
print("Original Data: ", test_scores[2])

# ================== Confusion Matrix ===============================
confusion_mat = confusion_matrix(irisDataV_labels_v, logreg_norm.predict(irisDataV_num_norm))

plt.subplot(121)
plt.imshow(confusion_mat, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
# ================== Normalized Matrix ===============================
normalized_confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
np.fill_diagonal(normalized_confusion_mat, 0)

plt.subplot(122)
plt.imshow(normalized_confusion_mat, cmap='Blues')
plt.title('Normalized Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')

# ================== Labeling the Plot ===============================
plt.tight_layout()
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

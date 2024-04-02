# program		vandenburg_demo_tsne_irisData_v1.py
# purpose	    Demonstrate a tsne algorithm with iris data
# usage         script
# notes         (1) THIS WILL NOT WORK ON MAC. DONT ASK ME HOW I KNOW.
#               (2) It constantly throws this AttributeError:'NoneType' object has no attribute 'split'
#               (3) this runs fine on my home pc.
# date			2/5/2024
# programmer   Colton Vandenburg

import datetime          # used for getting the date
import os                # used for getting the basic file name (returns lower case)
from sklearn.manifold import TSNE as tsne   # t-distributed stochastic neighbor embedding
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing    # Import label encoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import seaborn as sns
import os
import numpy as np

# ============== COMMON INITIALIZATION =====================
date_o = datetime.datetime.today()
date_c = date_o.strftime('%m/%d/%Y')
programName_c = os.path.basename(__file__)                     # case insensitive file name


script_dir = os.path.dirname(os.path.abspath(__file__))     # Get the directory of the current script
csv_file_path= os.path.join(script_dir, 'irisData.csv')   # Construct the full path to the CSV file, now Mac friendly :)
#

irisData_df = pd.read_csv(csv_file_path)    # Load the CSV file

ix = str.find(programName_c,'.')

fileName_c = 'irisData.csv'
programMsg_c = programName_c + ' (' + date_c + ')'


authorName_c = 'Colton Vandenburg'
figName_c = programName_c[:ix]+'_fig.png'


# ========== get data frame, split, get numerical data ==============


tts_rs = 6
irisDataT_df, irisDataV_df = train_test_split(irisData_df, test_size=0.5, random_state=tts_rs) 
irisDataT_num_df = irisDataT_df.drop("type", axis = 1)    # numerical data only

# ============== get labels, preprocess train data except for scaling  ======================
irisDataT_labels = irisDataT_df["type"]
label_encoder = preprocessing.LabelEncoder() 

irisDataT_labels_v = label_encoder.fit_transform(irisDataT_labels)

imputer = SimpleImputer(strategy="median")


imputer.fit(irisDataT_num_df)
irisDataT_num_df = imputer.transform(irisDataT_num_df)

# ================= get labels, preprocess verify data ==================
irisDataV_num_df = irisDataV_df.drop("type", axis=1)    # numerical data only
irisDataV_labels = irisDataV_df["type"]
irisDataV_labels_v = label_encoder.fit_transform(irisDataV_labels)

irisDataV_num_df = imputer.transform(irisDataV_num_df)


# ======================== TSNE 1 and 2 ======================
tsne_model_1 = tsne(n_components=2, perplexity=30)
tsne_case1 = tsne_model_1.fit_transform(irisDataT_num_df)
tsne_case2 = tsne_model_1.fit_transform(irisDataV_num_df)


plt.subplot(2, 2, 1)
plt.scatter(tsne_case1[:, 0], tsne_case1[:, 1], c=irisDataT_labels_v)
plt.title('TSNE on 50/50 Data', fontsize=7) 

# Label true classes in approximate center or cluster
for label in np.unique(irisDataT_labels):
    indices = np.where(irisDataT_labels == label)
    center = np.mean(tsne_case1[indices], axis=0)
    plt.annotate(label, center, fontsize=6, alpha=0.7)


plt.subplot(2, 2, 2)
plt.scatter(tsne_case2[:, 0], tsne_case2[:, 1], c=irisDataV_labels_v)
plt.title('TSNE on 50/50 Data p=50', fontsize=7) 

# Label true classes in approximate center or cluster
for label in np.unique(irisDataV_labels):
    indices = np.where(irisDataV_labels == label)
    center = np.mean(tsne_case2[indices], axis=0)
    plt.annotate(label, center, fontsize=6, alpha=0.7)



#Intrestingly, the look symettrical of eachother.

# ======================== TSNE 3 ======================
tsne_model_2 = tsne(n_components=2, perplexity=30)

irisData_df_no_ = irisData_df.drop("type", axis=1)
tsne_case3 = tsne_model_2.fit_transform(irisData_df_no_)

plt.subplot(2, 2, 3)
label_encoder = preprocessing.LabelEncoder()
irisData_df['type_v'] = label_encoder.fit_transform(irisData_df['type'])
plt.scatter(tsne_case3[:, 0], tsne_case3[:, 1], c=irisData_df['type_v'])
plt.title('TSNE on Full Data p=30', fontsize=7)  

# Label true classes in approximate center of cluster
for label in np.unique(irisData_df['type']):
    indices = np.where(irisData_df['type'] == label)
    center = np.mean(tsne_case3[indices], axis=0)
    plt.annotate(label, center, fontsize=6, alpha=0.7)


# ======================== TSNE 4 ======================
tsne_model_3 = tsne(n_components=2, perplexity=48)

tsne_case4 = tsne_model_3.fit_transform(irisData_df_no_)

plt.subplot(2, 2, 4)
plt.scatter(tsne_case4[:, 0], tsne_case4[:, 1], c=irisData_df['type_v'])
plt.title('TSNE with p=48', fontsize=7)  

# Label true classes in approximate center of cluster
for label in np.unique(irisData_df['type']):
    indices = np.where(irisData_df['type'] == label)
    center = np.mean(tsne_case4[indices], axis=0)
    plt.annotate(label, center, fontsize=6, alpha=0.7)


#======================Plotting Stuff ====================
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
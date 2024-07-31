import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def Data_Cleaner(x,y,x_cols):
    sm = SMOTE()                                            #Using SMOTE from imblearn to resample Imbalanced Data
    x_res, y_res = sm.fit_resample(x, y)

    x_train,x_test,y_train,y_test=train_test_split(x_res,y_res,test_size=0.20) #Splitting into train and test after resampling
    scaler=StandardScaler()
    Encoder=LabelEncoder()

    x_train=scaler.fit_transform(x_train)                   #Scaling after splitting to avoid data leakage
    x_test=scaler.transform(x_test)

    y_train=Encoder.fit_transform(y_train)                  #Label Encoding categorical values
    y_test=Encoder.transform(y_test)


    pca=PCA()
    pca.fit(x_train)

    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_) #finding number of components required,
    explained_variance = pca.explained_variance_ratio_                          #  while getting maximum variance
    n_components_required = np.argmax(cumulative_explained_variance >= 0.95) + 1 
    #print(n_components_required)

    pca=PCA(n_components=n_components_required)             #recalibrating pca after knowing number of components required
    x_train_pca=pca.fit_transform(x_train)
    x_test_pca=pca.transform(x_test)

    return x_train_pca,x_test_pca,y_train,y_test,scaler,Encoder,pca                 #returning required objects
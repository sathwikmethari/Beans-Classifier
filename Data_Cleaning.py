import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def outlier_replacer(data):                     #Replaces Outliers with median
    x=data.iloc[:,:-1].values                   #Independent varibles Array
    y=data.iloc[:,-1].values                    #Deoendent varible Array

    num_columns=len(x[0])                       #finding no of columns of Independent variables
    num_rows=len(x)                             #finding no of rows

    data_num= data.select_dtypes(exclude='object')  #Numerical DataFrame
    cols=[x for x in data_num.columns]              #List containg all Numerical column names

    '''print('Columns list \n',cols)
    '''
    IQR=[0]*num_columns                         #Creating empty 0 lists for interquartile ranges, lower bounds and upper bounds of Independent Variables
    l_bound=[0]*num_columns
    u_bound=[0]*num_columns
    
    for a in range(num_columns):
        IQR[a]=float(data[cols[a]].quantile(0.75))-float(data[cols[a]].quantile(0.25))
    for b in range(num_columns):
        l_bound[b]=float(data[cols[b]].quantile(0.25))-1.5*IQR[b]
    for c in range(num_columns):
        u_bound[c]=float(data[cols[c]].quantile(0.75))+1.5*IQR[c]
    '''
    print('\nInter_Quartile_Range list \n',IQR)
    print('\n Lower_Bound list \n',l_bound)
    print('\nUpper_Bound list \n',u_bound)
    '''
    
    median_list=[float(x) for x in data[cols].median()]      #list has medians of all independent varibles
    
    '''print('\nMedian list \n',median_list)
    '''                    
    
    for i in range(num_rows):                               #Replacing ouliers with median
        for j in range(num_columns):
            if l_bound[j]>x[i][j] or u_bound[j]<x[i][j]:
                x[i][j]=median_list[j]
            else:
                pass
    return x,y,cols                                              #Returns Independent and Dependent variables
    


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

    df_xtrain=pd.DataFrame(x_train,columns=x_cols)          #turning into dataframes to fit to pca
    df_xtest=pd.DataFrame(x_test,columns=x_cols)
    pca=PCA()
    pca.fit(df_xtrain)

    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_) #finding number of components required,
    explained_variance = pca.explained_variance_ratio_                          #  while getting maximum variance
    n_components_required = np.argmax(cumulative_explained_variance >= 0.95) + 1 
    print(n_components_required)

    pca=PCA(n_components=n_components_required)             #recalibrating pca after knowing number of components required
    x_train_pca=pca.fit_transform(df_xtrain)
    x_test_pca=pca.transform(df_xtest)

    return x_train_pca,x_test_pca,y_train,y_test,scaler,Encoder,pca #returning required objects
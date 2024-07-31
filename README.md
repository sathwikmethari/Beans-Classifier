# Hobby Project-1 (Beans Classifier)
A Web application designed with Flask, that uses python Machine Learning algorithm to predict the Bean Class when given input.<br>
# DataSet:
Got the dataset from <a href="https://archive.ics.uci.edu/dataset/602/dry+bean+dataset">UC Irvine Machine Learning Repository.</a><br>
There are total of 17 features in the dataset, 16 independent features and 1 dependent feature.
###
![fact](https://github.com/user-attachments/assets/20524d42-e552-4dc8-bb3c-142cdcb295ce)<br>
![fact2](https://github.com/user-attachments/assets/b9650306-26f4-43b4-844b-d09fd9d82fe9)
# Preprocessing:
Read the data into a dataframe using Pandas *pd.read_excel()*.<br>
Imported functions *outlier_replacer*, *Data_Cleaner*  from the two .py files, prevoiusly read data was given as input into the function *outlier_replacer*.
This function replaces outliers with median and returns x,y and column names of x(the column names of independent varibles).<br>
### Bean class plot
![presampling](https://github.com/user-attachments/assets/9c6118a4-1767-4fec-ac03-2c3e0ae858c5)
###
As we can see the dataset is skewed, we can fix this using Sampling. Used SMOTE from <a href="https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html">imblearn</a> to do the sampling.(tried with class-weight parameter, using sampling gives better performance)<br>
### After Over Sampling
![postsampling](https://github.com/user-attachments/assets/dfcee6ac-0da3-4503-992c-97cf3c051fd2)
###
Using sampling has its disadvantages. As the data size is relatively low, we can use this.

## Correlation Map
![corr](https://github.com/user-attachments/assets/c46dc77c-1d22-4256-9433-07e22c3232bb)
###
The features are highly correlated, used Dimensionality reduction(PCA) on data to reduce number of columns.<br>
Found *n_components* parameter for PCA such that maximum variance is captured while reduction.
# Training :
###
Used most of the classfiers from sklearn library and xgboost and trained them on the dataset and tested for performannce.<br>
Selected XGBClassifier as final model as it produced the highest scores, further tuned the parameters using RandomizedSearchCV.
###
# Running
###
Download the Zip file and extract the classifier.<br>
1. *Windows* While in the Beans Classifier-main folder, type cmd in the search bar and press enter.<br>
2.  Via the prompt(pip), we need to install required libraries. For that write<br>
    *pip install -r requirements.txt* and press enter.<br>
3.  After installation of libraries, run the application.py by typing<br>
    *python application.py*
4. Click on the link(*http://127.0.0.1:5000*) or copy paste it in web browser.

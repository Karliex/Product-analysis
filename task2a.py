import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.tree import *
from sklearn.impute import SimpleImputer

#merge two datasets (record is not taken if it has no cooresponding life expectancy)
world = pd.read_csv("world.csv", header=0,low_memory=False)
life = pd.read_csv("life.csv", header=0,low_memory=False)
raw_data = pd.merge(world, life, on = 'Country Code')
raw_data.drop(["Country Name", "Time", "Country Code", "Country", "Year"], axis = 1, inplace = True)

raw_data = raw_data.replace('..', np.nan)
for column in raw_data.columns:
    raw_data[column] = pd.to_numeric(raw_data[column], downcast="float",errors='ignore')

#split into train and test datasets
#randomly select 66% of the instances to be training and the rest to be testing
classlabel = raw_data['Life expectancy at birth (years)']
X_train, X_test, y_train, y_test = train_test_split(raw_data.iloc[:,0:-1],classlabel,test_size=0.33333, random_state=100)


# Median Imputation
median_lst = []
imputed_train = X_train.copy()
imputed_test = X_test.copy()

for column in X_train.columns:
    median = round(X_train[column].median(),3)
    median_lst.append(median)
    imputed_train[column].fillna(median, inplace=True)
    
#impute X_test with median of X_train
for column in X_test.columns:
    for median in median_lst:
        imputed_test[column].fillna(median,inplace = True)

# Scale each feature by removing the mean and scaling to unit variance
mean = [round(i,3) for i in imputed_train.mean()]
std = [round(i,3) for i in imputed_train.std()]
scaler = preprocessing.StandardScaler(with_mean = mean, with_std = std).fit(imputed_train)

standardised_train = scaler.transform(imputed_train)
standardised_test = scaler.transform(imputed_test)

#static variables used for imputation and scaling
column = imputed_train.columns
df2a = pd.DataFrame({"feature":column, "median":median_lst, "mean":mean, "variance":std})
#df2a
df2a.to_csv('task2a.csv', index = False)


# Comparing Classification Algorithms
#decison_tree
decision_tree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
decision_tree.fit(standardised_train, y_train)
y_pred = decision_tree.predict(standardised_test)
dt_accuracy = round(accuracy_score(y_test, y_pred) *100, 3)
print(f'Accuracy of decision tree: {dt_accuracy}%')

#knn5
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(standardised_train, y_train)
y_pred = knn5.predict(standardised_test)
knn5_accuracy = round(accuracy_score(y_test, y_pred)*100, 3)
print(f'Accuracy of k-nn (k=5): {knn5_accuracy}%')

#knn10
knn10 = KNeighborsClassifier(n_neighbors=10)
knn10.fit(standardised_train, y_train)
y_pred = knn10.predict(standardised_test)
knn10_accuracy = round(accuracy_score(y_test, y_pred)*100, 3)
print(f'Accuracy of k-nn (k=10): {knn10_accuracy}%')






#!/usr/bin/env python
# coding: utf-8

# In[159]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing


# In[160]:


#Task2b

world = pd.read_csv("world.csv", na_values = "..")
life = pd.read_csv("life.csv")

#merge two datasets
combination = life.merge(world.iloc[:,2:], how = "left")

# data preprocessing
x = combination.iloc[:,4:]
y = combination.iloc[:,3]

median = [i for i in x.median(skipna = True)]
mean = [i for i in x.mean(skipna = True)]
std = [i for i in x.std(skipna = True)]
scaler = preprocessing.StandardScaler(with_mean = mean, with_std = std).fit(x)
x_scaled = scaler.transform(x)
x = pd.DataFrame(data=x_scaled, index=None, columns=x.columns)
# median imputation to impute missing values
x = x.groupby(x.columns, axis = 1).transform(lambda a: a.fillna(a.median()))
norm_xy = x.copy()
norm_xy['Life expectancy at birth (years)'] = y


# In[161]:


X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = 2/3, test_size = 1/3,random_state=100)

# train KNN for baseline, use all processors to speed up
knn5 = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

# Compute baseline accuracy for comparsion
knn5.fit(X_train, y_train)
baseline = knn5.score(X=X_test, y=y_test)
print("Accuracy baseline:", baseline)


# ## Kmeans

# In[172]:


kmean_accuracy = []
for k in range(2, 102, 2):
    # Split training and testing data. Random_state 100 is specificed as seed.
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = 2/3, test_size = 1/3,random_state=100)
    # Convert X_train, X_test to dataframes
    kmeans = KMeans(n_clusters=k).fit(X_train)
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    # Apply k-means clustering to the data in foodscaled and then use the resulting cluster labels as the values for a new feature fclusterlabel. 
    X_train['clusterlabel'] = kmeans.labels_
    # At test time, a label for a testing instance can be created by assigning it to its nearest cluster.
    X_test['clusterlabel'] = kmeans.predict(X_test)
    # Train knn classifier again
    knn5.fit(X_train, y_train)
    y_test_prediction = knn5.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_prediction)
    kmean_accuracy.append(accuracy)


# In[173]:


# Draw a plot of accuracy against number of clusters
kmean_df = pd.DataFrame(kmean_accuracy, [baseline for i in range(50)]).reset_index()
kmean_df.index = list(range(2, 102, 2))
kmean_df.columns = ['Baseline', 'Accuracy']

plot = kmean_df.plot(title='Number of clusters (k) vs. Accuracy')
plot.set_xlabel("Number of clusters (k) ")
plot.set_ylabel("Accuracy")
fig = plot.get_figure()
fig.savefig("task2bgraph1.png")


# In[174]:


k = kmean_df.idxmax(axis=0)['Accuracy']
print(f'The number of clusters which contributes to max accuracy: {k}')


# In[175]:


# Make a feature_engineering of scaled data
feature_eng_data = x.copy()
interaction = list()

# add interactiom term pairs
original_feature_num = len(feature_eng_data.columns)
for i in range(original_feature_num):
    for j in range(i,original_feature_num):
        if i < j:
            f1 = feature_eng_data.iloc[:,i].astype("float")
            f2 = feature_eng_data.iloc[:,j].astype("float")
            feature_eng_data[f"f{i}*f{j}"] = f1*f2

# add clustering labels (low, medium and high)
kmeans = KMeans(n_clusters = k).fit(x)
x["f_clusterlabel"] = kmeans.labels_            


# In[176]:


# Using mutual information, select only the feature with better performance.
def entropy(prob):  
    new_value = np.where(prob==0, 1, prob)  # equal zero return 1 else return prob
    return -prob.dot(np.log2(new_value))  # replace dot product equal zero

def mutual_info(df):
    
    Hx = entropy(df.iloc[:,0].value_counts(normalize=True, sort=False))
    Hy = entropy(df.iloc[:,1].value_counts(normalize=True, sort=False))
    
    counts = df.groupby(list(df.columns.values)).size()
    probs = counts/ counts.values.sum()
    H_xy = entropy(probs)

    # Mutual Information
    MI = Hx + Hy - H_xy
    NMI = MI/min(Hx,Hy)
    
    return {'MI':MI,'NMI':NMI} 


# In[177]:


feature_eng_train, feature_eng_test, y_train, y_test = train_test_split(feature_eng_data, y, train_size = 2/3, test_size=1/3, random_state = 100) 

# store the binning result by copying the feature_eng_train
binning_result = feature_eng_train.copy()

MI_dict = {}
bin_num = 211

for col in feature_eng_train.columns:
    # Cut each of the feature
    binning_result[col] = pd.cut(binning_result[col], bin_num)
    data = {'feature_bin':binning_result[col],'label_bin':y_train}
    df = pd.DataFrame(data)
    # Compute the Mutual Information
    MI_dict[col] = mutual_info(df)['MI']

sorted_MI = pd.DataFrame(list(MI_dict.items()), columns=['Feature', 'MI']).sort_values(by=['MI'],ascending=False)        
col_list = sorted_MI.iloc[0:4]['Feature'].values.tolist()

selected_train = feature_eng_train[col_list]
selected_test = feature_eng_test[col_list]


# In[178]:


knn= KNeighborsClassifier(n_neighbors = 5)
knn.fit(selected_train, y_train)

y_test_prediction = knn.predict(selected_test)
feature_eng_accuracy = round(accuracy_score(y_test, y_test_prediction)*100, 3)


# In[179]:


# PCA
pca = PCA(n_components = 4)

pca_df = pd.DataFrame(pca.fit_transform(x))
pca_train, pca_test, y_train, y_test = train_test_split(pca_df, y, train_size = 2/3, test_size = 1/3, random_state = 100)

knn_pca= KNeighborsClassifier(n_neighbors = 5)
knn_pca.fit(pca_train, y_train)

y_test_prediction_pca = knn_pca.predict(pca_test)
pca_accuracy = round(accuracy_score(y_test, y_test_prediction_pca)*100, 3)


# In[180]:


# first four features
first4_x_train, first4_x_test, y_train, y_test = train_test_split(x.iloc[:,0:4], y, train_size = 2/3, test_size = 1/3, random_state = 100)
knn_first_four= KNeighborsClassifier(n_neighbors = 5)
knn_first_four.fit(first4_x_train, y_train)

y_test_prediction_first4 = knn_first_four.predict(first4_x_test)
test_5_first4 = round(accuracy_score(y_test, y_test_prediction_first4)*100, 3)


# In[181]:


print(f'Accuracy of feature engineering: {feature_eng_accuracy}%')
print(f'Accuracy of PCA: {pca_accuracy}%')
print(f'Accuracy of first four features: {str(test_5_first4)}%')


#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing, metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#1. Principal Component Analysis
#a. Apply PCA on CC dataset.
#b. Apply k-means algorithm on the PCA result and report your observation if the silhouette score has improved or not?
#c. Perform Scaling+PCA+K-Means and report performance.

#Reading the csv file and printing the info about file
dataset_pd = pd.read_csv("CC GENERAL.csv")
dataset_pd.info()


# In[3]:


#To print first five rows of the dataset to inspect data format
dataset_pd.head()


# In[4]:


#checking missing values in dataset
dataset_pd.isnull().any()


# In[5]:


# Select numeric columns of the dataset
numeric_columns = dataset_pd.select_dtypes(include=[np.number]).columns.tolist()

# Replace missing values with mean of the respective columns
dataset_pd[numeric_columns] = dataset_pd[numeric_columns].fillna(dataset_pd[numeric_columns].mean())

dataset_pd.isnull().any()


# In[6]:


# Extracting input features and output labels from the pandas dataframe and printing their shapes
x = dataset_pd.iloc[:,1:-1]
y = dataset_pd.iloc[:,-1]
print(x.shape,y.shape)


# In[7]:


#1.a Apply PCA on CC Dataset 
pca = PCA(3)
x_pca = pca.fit_transform(x)
principalDf = pd.DataFrame(data = x_pca, columns = ['principal component 1', 'principal component 2', 'principal component 3'])
finalDf = pd.concat([principalDf, dataset_pd.iloc[:,-1]], axis = 1)
finalDf.head()


# In[8]:


#1.b Apply K Means on PCA Result
X = finalDf.iloc[:,0:-1]
y = finalDf.iloc[:,-1]


# In[10]:


# This is the k in kmeans
nclusters = 3 
km = KMeans(n_clusters=nclusters)
km.fit(X)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(X)


# Summary of the predictions made by the classifier
print(classification_report(y, y_cluster_kmeans, zero_division=1))
print(confusion_matrix(y, y_cluster_kmeans))


train_accuracy = accuracy_score(y, y_cluster_kmeans)
print("\nAccuracy for our Training dataset with PCA:", train_accuracy)


#Calculate sihouette Score
score = metrics.silhouette_score(X, y_cluster_kmeans)
print("Sihouette Score: ",score) 
"""
Sihouette Score- ranges from −1 to +1 , a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.
"""


# In[11]:


#1.c Scaling +PCA + KMeans
x = dataset_pd.iloc[:,1:-1]
y = dataset_pd.iloc[:,-1]
print(x.shape,y.shape)


# In[12]:


#Scaling
scaler = StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
#PCA
pca = PCA(3)
x_pca = pca.fit_transform(X_scaled_array)
principalDf = pd.DataFrame(data = x_pca, columns = ['principal component 1', 'principal component 2','principal component 3'])
finalDf = pd.concat([principalDf, dataset_pd.iloc[:,-1]], axis = 1)
finalDf.head()


# In[13]:


#Extraction of the features and target variable from the finalDf dataframe. 
#X contains all the columns of the dataframe except the last one
X = finalDf.iloc[:,0:-1]
#y contains the values from the last column.
y = finalDf["TENURE"]
print(X.shape,y.shape)


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.34,random_state=0)
nclusters = 3 
# this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(X_train,y_train)


# predict the cluster for each training data point
y_clus_train = km.predict(X_train)

# Summary of the predictions made by the classifier
print(classification_report(y_train, y_clus_train, zero_division=1))
print(confusion_matrix(y_train, y_clus_train))

train_accuracy = accuracy_score(y_train, y_clus_train)
print("Accuracy for our Training dataset with PCA:", train_accuracy)

#Calculate sihouette Score
score = metrics.silhouette_score(X_train, y_clus_train)
print("Sihouette Score: ",score) 

"""
Sihouette Score- ranges from −1 to +1 , a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.
"""


# In[15]:


# predict the cluster for each testing data point
y_clus_test = km.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_clus_test, zero_division=1))
print(confusion_matrix(y_test, y_clus_test))

train_accuracy = accuracy_score(y_test, y_clus_test)
print("\nAccuracy for our Training dataset with PCA:", train_accuracy)

#Calculate sihouette Score
score = metrics.silhouette_score(X_test, y_clus_test)
print("Sihouette Score: ",score) 

"""
Sihouette Score- ranges from −1 to +1 , a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.
"""


# In[16]:


# 2.Use pd_speech_features.csv
# a. Perform Scaling
# b. Apply PCA (k=3)
# c. Use SVM to report performance

dataset_pd = pd.read_csv('pd_speech_features.csv')
dataset_pd.info()


# In[17]:


dataset_pd.head()


# In[18]:


dataset_pd.isnull().any()


# In[19]:


# Create X and y arrays for machine learning modeling
# X is independent variable
X = dataset_pd.drop('class',axis=1).values
# Y is dependent variable
y = dataset_pd['class'].values


# In[20]:


#2.a Scaling Data
scaler = StandardScaler()
X_Scale = scaler.fit_transform(X)


# In[21]:


#2.b Apply PCA with k =3
pca3 = PCA(n_components=3)
principalComponents = pca3.fit_transform(X_Scale)

principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2','Principal Component 3'])

finalDf = pd.concat([principalDf, dataset_pd[['class']]], axis = 1)
finalDf.head()


# In[22]:


# Drop class column and extract remaining data as input features
X = finalDf.drop('class',axis=1).values
# Extracts the target variable 'class' from the dataset and assigns it to y.
y = finalDf['class'].values
# Randomly split the dataset into two sets: a training set and a testing set.
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.34,random_state=0)


# In[23]:


#2.c Using Support Vector Machine's (SVM)

from sklearn.svm import SVC

svmClassifier = SVC()
svmClassifier.fit(X_train, y_train)

y_pred = svmClassifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred, zero_division=1))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
glass_acc_svc = accuracy_score(y_pred,y_test)
print('accuracy is',glass_acc_svc )

#Calculate sihouette Score
score = metrics.silhouette_score(X_test, y_pred)
print("Sihouette Score: ",score) 


# In[24]:


#3.Apply Linear Discriminant Analysis (LDA) on Iris.csv dataset to reduce dimensionality of data to k=2. 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
dataset_iris = pd.read_csv('Iris.csv')
dataset_iris.info()


# In[25]:


dataset_iris.isnull().any()


# In[26]:


x = dataset_iris.iloc[:,1:-1]
y = dataset_iris.iloc[:,-1]
print(x.shape,y.shape)


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# In[28]:


#performs data preprocessing by standardizing the input features and encoding the target variable
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
le = LabelEncoder()
y = le.fit_transform(y)


# In[29]:


# Perform Linear Discriminant Analysis on the training data to reduce dimensionality to 2 components 
# and transform the training and test data to the reduced space
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
# Print the shape of the transformed training and test data
print(X_train.shape,X_test.shape)


# In[ ]:





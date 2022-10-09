# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# import libraries for scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MaxAbsScaler

# import libraries for encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# import libraries for clustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
# from pyclustering.cluster.clarans import clarans
# from pyclustering.utils import timedcall
# Didn't know how to use clarans well

# import libraries for scoring for certain clusters
from sklearn.metrics import silhouette_score

# Read in data
data = pd.read_csv("C://Users/User/Desktop/housing.csv")

# Data Analysis and Feature Engineering
# 1. Change all NULL values into 0 and drop the columns with 0 in it.
# 2. Drop medianhousevalue since we are not going to use them
data.dropna(axis=0, how='any', inplace=True)

target = data[['median_house_value']]
comp1 = data[['longitude']]
comp2 = data[['latitude']]
comp3 = data[['total_rooms']]
comp4 = data[['total_bedrooms']]
comp5 = data[['population']]
comp6 = data[['households']]
comp7 = data[['median_income']]
comp8 = data[['ocean_proximity']].replace(['INLAND', '<1H OCEAN', 'NEAR BAY', 'NEAR OCEAN'], [0, 1, 2, 3])

comp = [comp1, comp2, comp3, comp4, comp5, comp6, comp7, comp8]

# Encoding : Use 2 types of encoding method (One-Hot, Label)
OH = OneHotEncoder()
LE = LabelEncoder()

# Put the scalers in list for Big Function AutoML
encoder_list = ['OH', 'LE']

# Use 5 types of scaling method (Minmax, Robust, Maxabs, Normalizer, Standard)
MinMax = MinMaxScaler()
Robust = RobustScaler()
Maxabs = MaxAbsScaler()
Standard = StandardScaler()
Norm = Normalizer()

# Put the scalers in list for Big Function AutoML
scale_list = ['MinMax', 'Robust', 'Maxabs', 'Standard', 'Norm']

# Use 5 types of clustering algorithm. Each Clusters can have different parameters
# I will run K means twice with different hyperparameters
km_2 = KMeans(n_clusters = 2)
km_5 = KMeans(n_clusters = 5)

# I will run GMM twice with different hyperparameters
GMM_4 = GaussianMixture(n_components=4)
GMM_5 = GaussianMixture(n_components=5)

# I will run DBSCAN three times with different hyperparameters
DB_3 = DBSCAN(eps=0.375, min_samples=3)
DB_4 = DBSCAN(eps= 0.4, min_samples=4)
DB_5 = DBSCAN(eps=0.425, min_samples=5)

# I will run Mean Shift clustering 5 times with different hyperparameters
MS_1 = MeanShift(bandwidth=1)
MS_2 = MeanShift(bandwidth=2)
MS_3 = MeanShift(bandwidth=3)
MS_4 = MeanShift(bandwidth=4)
MS_5 = MeanShift(bandwidth=5)

# I couldn't find a way to use CLARANS
# Put the clusters in list for AutoML
clustering_list = ['km_2', 'km_5', 'GMM_4', 'GMM_5', 'DB_3', 'DB_4', 'DB_5', 'MS_1', 'MS_2', 'MS_3', 'MS_4', 'MS_5', 'CL']

# Write a huge function about running the algorithm. It takes data, scale it and encode it.
# Then, runs the clustering algorithms by order.
# First, make a function for scaling

score = 0

def scaling(new_data, scaler):
    if scaler == 'MinMax':
        MinMax.fit(new_data)
    if scaler == 'Robust':
        Robust.fit(new_data)
    elif scaler == 'Maxabs':
        Maxabs.fit(new_data)
    elif scaler == 'Standard':
        Standard.fit(new_data)
    elif scaler == "Norm":
        Norm.fit(new_data)

# Second, make a function for encoding
def encoding(new_data, encoder):
    if encoder == 'OH':
        OH.fit(new_data)
    elif encoder == 'LE':
        LE.fit(new_data)

# Lastly, make a function for clustering algorithm. The result will be plotted right away.
def clustering(new_data, scaler, encoder, cluster):
    if cluster == 'km_2':
        km_2.fit(new_data)
        labels = km_2.predict(new_data)
        score = silhouette_score(new_data, data['median_house_value'])
        print('------------------------------')
        print('Scaler Used : ', scaler)
        print('Encoder Used : ', encoder)
        print('Silhouette Score km_2', new_data.columns,' : ' ,score)
        print('------------------------------')
        plt.scatter(new_data, target, c=labels, s=40, cmap='viridis')
        plt.xlabel('K Means (K = 2)')
        plt.ylabel('median house value')
        plt.show(block=False)
        plt.pause(1)
        plt.close()

    elif cluster == 'km_5':
        km_5.fit(new_data)
        labels = km_5.predict(new_data)
        score = silhouette_score(new_data, data['median_house_value'])
        print('------------------------------')
        print('Scaler Used : ', scaler)
        print('Encoder Used : ', encoder)
        print('Silhouette Score km_5', new_data.columns,' : ' ,score)
        print('------------------------------')
        plt.scatter(new_data, target, c=labels, s=40, cmap='viridis')
        plt.xlabel('K Means (K = 5)')
        plt.ylabel('median house value')
        plt.show(block=False)
        plt.pause(1)
        plt.close()

    elif cluster == 'GMM_4':
        GMM_4.fit(new_data)
        labels = GMM_4.predict(new_data)
        score = silhouette_score(new_data, data['median_house_value'])
        print('------------------------------')
        print('Scaler Used : ', scaler)
        print('Encoder Used : ', encoder)
        print('Silhouette Score GMM_4', new_data.columns,' : ' ,score)
        print('------------------------------')
        plt.scatter(new_data, target, c=labels, s=40, cmap='viridis')
        plt.xlabel('Gaussian Mixture Model : 4')
        plt.ylabel('median house value')
        plt.show(block=False)
        plt.pause(1)
        plt.close()

    elif cluster == 'GMM_5':
        GMM_5.fit(new_data)
        labels = GMM_5.predict(new_data)
        score = silhouette_score(new_data, data['median_house_value'])
        print('------------------------------')
        print('Scaler Used : ', scaler)
        print('Encoder Used : ', encoder)
        print('Silhouette Score GMM_5', new_data.columns,' : ' ,score)
        print('------------------------------')
        plt.scatter(new_data, target, c=labels, s=40, cmap='viridis')
        plt.xlabel('Gaussian Mixture Model : 5')
        plt.ylabel('median house value')
        plt.show(block=False)
        plt.pause(1)
        plt.close()

    elif cluster == 'DB_3':
        db = DB_3.fit(new_data)
        labels = db.labels_
        score = silhouette_score(new_data, data['median_house_value'])
        print('------------------------------')
        print('Scaler Used : ', scaler)
        print('Encoder Used : ', encoder)
        print('Silhouette Score DB_3', new_data.columns,' : ' ,score)
        print('------------------------------')
        plt.scatter(new_data, target, c=labels, s=40, cmap='viridis')
        plt.xlabel('DBSCAN : 3')
        plt.ylabel('median house value')
        plt.show(block=False)
        plt.pause(1)
        plt.close()

    elif cluster == 'DB_4':
        db = DB_4.fit(new_data)
        labels = db.labels_
        score = silhouette_score(new_data, data['median_house_value'])
        print('------------------------------')
        print('Scaler Used : ', scaler)
        print('Encoder Used : ', encoder)
        print('Silhouette Score DB_4', new_data.columns,' : ', score)
        print('------------------------------')
        plt.scatter(new_data, target, c=labels, s=40, cmap='viridis')
        plt.xlabel('DBSCAN : 4')
        plt.ylabel('median house value')
        plt.show(block=False)
        plt.pause(1)
        plt.close()

    elif cluster == 'DB_5':
        db = DB_5.fit(new_data)
        labels = db.labels_
        score = silhouette_score(new_data, data['median_house_value'])
        print('------------------------------')
        print('Scaler Used : ', scaler)
        print('Encoder Used : ', encoder)
        print('Silhouette Score DB_5', new_data.columns,' : ', score)
        print('------------------------------')
        plt.scatter(new_data, target, c=labels, s=40, cmap='viridis')
        plt.xlabel('DBSCAN : 5')
        plt.ylabel('median house value')
        plt.show(block=False)
        plt.pause(1)
        plt.close()

    elif cluster == 'MS_1':
        MS_1.fit(new_data)
        labels = MS_1.predict(new_data)
        score = silhouette_score(new_data, data['median_house_value'])
        print('------------------------------')
        print('Scaler Used : ', scaler)
        print('Encoder Used : ', encoder)
        print('Silhouette Score MS_1', new_data.columns,' : ', score)
        print('------------------------------')
        plt.scatter(new_data, target, c=labels, s=40, cmap='viridis')
        plt.xlabel('Mean Shift : 1')
        plt.ylabel('median house value')
        plt.show(block=False)
        plt.pause(1)
        plt.close()

    elif cluster == 'MS_2':
        MS_2.fit(new_data)
        labels = MS_2.predict(new_data)
        score = silhouette_score(new_data, data['median_house_value'])
        print('------------------------------')
        print('Scaler Used : ', scaler)
        print('Encoder Used : ', encoder)
        print('Silhouette Score MS_2', new_data.columns,' : ', score)
        print('------------------------------')
        plt.scatter(new_data, target, c=labels, s=40, cmap='viridis')
        plt.xlabel('Mean Shift : 2')
        plt.ylabel('median house value')
        plt.show(block=False)
        plt.pause(1)
        plt.close()

    elif cluster == 'MS_3':
        MS_3.fit(new_data)
        labels = MS_3.predict(new_data)
        score = silhouette_score(new_data, data['median_house_value'])
        print('------------------------------')
        print('Scaler Used : ', scaler)
        print('Encoder Used : ', encoder)
        print('Silhouette Score MS_3', new_data.columns,' : ', score)
        print('------------------------------')
        plt.scatter(new_data, target, c=labels, s=40, cmap='viridis')
        plt.xlabel('Mean Shift : 3')
        plt.ylabel('median house value')
        plt.show(block=False)
        plt.pause(1)
        plt.close()

    elif cluster == 'MS_4':
        MS_4.fit(new_data)
        labels = MS_4.predict(new_data)
        score = silhouette_score(new_data, data['median_house_value'])
        print('------------------------------')
        print('Scaler Used : ', scaler)
        print('Encoder Used : ', encoder)
        print('Silhouette Score MS_4', new_data.columns,' : ', score)
        print('------------------------------')
        plt.scatter(new_data, target, c=labels, s=40, cmap='viridis')
        plt.xlabel('Mean Shift : 4')
        plt.ylabel('median house value')
        plt.show(block=False)
        plt.pause(1)
        plt.close()

    elif cluster == 'MS_5':
        MS_5.fit(new_data)
        labels = MS_5.predict(new_data)
        score = silhouette_score(new_data, data['median_house_value'])
        print('------------------------------')
        print('Scaler Used : ', scaler)
        print('Encoder Used : ', encoder)
        print('Silhouette Score MS_5', new_data.columns,' : ', score)
        print('------------------------------')
        plt.scatter(new_data, target, c=labels, s=40, cmap='viridis')
        plt.xlabel('Mean Shift : 5')
        plt.ylabel('median house value')
        plt.show(block=False)
        plt.pause(1)
        plt.close()

# Run Algorithm
def AutoML(new_data):
    for i in encoder_list:
        encoding(new_data, i)
        for j in scale_list:
            scaling(new_data, j)
            for k in clustering_list:
                clustering(new_data, j, i, k)

for i in comp:
    AutoML(i)


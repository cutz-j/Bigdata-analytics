import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import normalized_mutual_info_score as NMI

########### Data load ###########
np.random.seed(0)

all_data = pd.read_csv("data/KMeans_PCA_data.csv", index_col=0)
y_data = pd.read_csv("data/KMeans_PCA_label.csv", index_col=0)

########## PreProcessing ###########
# 4) permutation
n_sample = len(all_data)
order = np.random.permutation(n_sample)
all_data = all_data.iloc[order]
y_data = y_data.iloc[order]

# 1) from second column to 499th column
all_data = all_data.iloc[:, 1:500]

# 2) Label Encoding
le = LabelEncoder()
y_label = le.fit_transform(y_data)

# 3) K
K = len(np.unique(y_label))

# 5) Scaling
mms = MinMaxScaler()
X_data = mms.fit_transform(all_data)

########### K-means #############
# 1) K - clustering
km = KMeans(n_clusters=K)
km.fit(X_data)
y_hat = km.labels_

# 2) NMI
kmeans_NMI = NMI(y_label, y_hat, average_method='arithmetic')

########### PCA --> K-means ###########
# 1 + 2) PCA
pca = PCA(n_components=10, whiten=True, random_state=0)
X_data_pca = pca.fit_transform(X_data)
km = KMeans(n_clusters=K)
y_hat = km.fit_predict(X_data_pca)
PCA_NMI = NMI(y_label, y_hat, average_method='arithmetic')

print("K-Means NMI score before PCA: %f" %(kmeans_NMI))
print("K-Means NMI score after PCA: %f" %(PCA_NMI))
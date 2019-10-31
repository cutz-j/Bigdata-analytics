#%% md
# 3. K-Means, PCA
#%%
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import normalized_mutual_info_score as NMI
#%%
# data preprocessing
data = pd.read_csv("data/KMeans_PCA_data.csv")
label = pd.read_csv("data/KMeans_PCA_label.csv")

# 1)
x = data.drop(["Unnamed: 0"], axis=1)
x = x.iloc[:,0:500].values

# 2)
y = label["Class"].values
le = LabelEncoder()
y = le.fit_transform(y)

# 3)
K = len(np.unique(y))

# 4)
n_sample = x.shape[0]
np.random.seed(0)
order = np.random.permutation(n_sample)
x = x[order]
y = y[order]

# 5)
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
#%%
km = KMeans(n_clusters=K, random_state=0).fit(x)
pred_y = km.labels_

result = NMI(pred_y, y, average_method='arithmetic')
#%%
pca = PCA(n_components=10, whiten=True, random_state=0).fit(x)
x_pca = pca.transform(x)

km_pca = KMeans(n_clusters=K, random_state=0).fit(x_pca)
pred_y_pca = km_pca.labels_

pca_result = NMI(pred_y_pca, y, average_method='arithmetic')
#%%
print("K-Means NMI score before PCA : {}".format(result))
print("K-Means NMI score after PCA : {}".format(pca_result))
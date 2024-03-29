import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

######### Preprocess ##########
# Data Load
data = np.loadtxt('./data/SvmData.txt')
data.shape

# 2) Permutation
# 2) Seed 0
np.random.seed(0)
n_sample = len(data)
order = np.random.permutation(n_sample)
data = data[order]

# 1) Column split
x = data[:,:7]
y = data[:, [-1]]

# 3) Data split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify=y, random_state = 42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.25, stratify=y_train, random_state = 42)

# 4) Scaling
ss = StandardScaler()
X_train = ss.fit_transform(x_train)
X_val = ss.transform(x_val)
X_test = ss.transform(x_test)

########## Validation ##########
# 1) Kernel parameters
param_list = np.linspace(start=0.1, stop=1, num=10)
kernel_list = ['linear', 'rbf', 'poly', 'sigmoid']

for kernel in kernel_list: # kernel function
    f1_list = []
    for p in param_list: # parameter choice
        # 2) C & Gamma parameter validation
        svc = SVC(kernel=kernel, C=p if kernel=='linear' else 1.0, gamma='auto_deprecated' if kernel=='linear' else p)
        svc.fit(X_train, y_train)
        y_hat = svc.predict(X_val)
        f1_list.append(f1_score(y_val, y_hat, average='micro'))
    
########## Test ###########
    # 1) SVM fit
    best_f1 = param_list[np.argmax(f1_list)]
    svc_model = SVC(kernel=kernel, C=best_f1 if kernel=='linear' else 1.0, gamma='auto_deprecated' if kernel=='linear' else best_f1)
    svc_model.fit(X_train, y_train)
    y_hat = svc_model.predict(X_test)
    
    # 2) Print
    print("%s kernel SVM Best parameter C: %f" %(kernel, best_f1))
    print("%s kernel SVM F1 score : %f" %(kernel, f1_score(y_test, y_hat, average='micro')))
    print("%s kernel Confusion Matrix : \n" %(kernel), confusion_matrix(y_test, y_hat))

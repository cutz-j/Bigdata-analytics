import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

############# Preprocess #############
# Data load
data = np.loadtxt('data/RegressionData.txt', delimiter=',')
# 2) Permutation & Seed 0
np.random.seed(0)
n_sample = len(data)
order = np.random.permutation(n_sample)
data = data[order]

# 1) Column split
x_data = data[:,:-2]
y_data = data[:,-2:]

# 3) Data split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, random_state = 42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.25, random_state = 42)

# 4) Scaling
mms = MinMaxScaler()
X_train = mms.fit_transform(x_train)
X_val = mms.transform(x_val)
X_test = mms.transform(x_test)

################ Validation #################
# 2) alpha list
alpha_list = [0.0001, 0.001, 0.01, 0.1, 1]
label = ['latitute', 'longitude']

# 1) Except for LS, validation
for i in range(2): # 0: latitute, 1: longitude
    for model in ['LeastSquare', 'Ridge', 'Lasso']:
        mse_list = []
        for alpha in alpha_list:
            if model == 'LeastSquare':
                continue
            if model == 'Ridge':
                rm = linear_model.Ridge(alpha=alpha)
            if model == 'Lasso':
                rm = linear_model.Lasso(alpha=alpha)
            rm.fit(X_train, y_train[:,i])
            y_hat = rm.predict(X_val)
            mse_list.append(mean_squared_error(y_val[:,i], y_hat))
        
################## Test ######################
        if model == 'LeastSquare':
            rm = linear_model.LinearRegression()
        elif model == 'Ridge':
            rm = linear_model.Ridge(alpha=alpha_list[np.argmin(mse_list)])
        else:
            rm = linear_model.Lasso(alpha=alpha_list[np.argmin(mse_list)])
        
        rm.fit(X_train, y_train[:,i])
        y_hat = rm.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test[:,i], y_hat))
        var = r2_score(y_test[:,i], y_hat)
        
        # 2) Print
        if model == 'Ridge' or model == 'Lasso':
            print("%s best alpha of %s: %f" %(model, label[i], alpha_list[np.argmin(mse_list)]))
        print("%s RMSE of %s: %f" %(model, label[i], rmse))
        print("%s Variance of %s: %f" %(model, label[i], var))
        print("="*40)
        

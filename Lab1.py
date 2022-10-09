# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Read in data
data = pd.read_csv("C://Users/User/Desktop/breast-cancer-wisconsin.data.csv")

# Clean Dirty Data
data.replace({"?" : np.nan}, inplace=True)
data.dropna(axis=0, how='any', inplace=True)

# Separate data with data and target
X = data.drop(data.columns[-1], axis=1)
y = data[["class"]]

# No need for scaling, but I'll use 2 (MinMax, Robust)
MinMax = MinMaxScaler()
Robust = RobustScaler()
scale_list = ['No', 'MinMax', 'Robust']

# Models
entropy = DecisionTreeClassifier(criterion="entropy")
gini = DecisionTreeClassifier(criterion="gini")
logistic = LogisticRegression()
SVM = SVC(kernel='linear')
model_list = ['entropy', 'gini', 'logistic', 'SVM']

# K values for K fold cross validation
K = [3, 5, 10]

i = 0
score = []

# Function for scaling. When this Function runs, it scales the data by none / RobustScaler / MinMaxScaler
def scaling(X_train, X_test, scaler):
    if scaler == "No":
        X_train_scale = X_train
        X_test_scale = X_test
        return X_train_scale, X_test_scale, 'No'

    elif scaler == "Robust":
        robustScaler = RobustScaler()
        X_train_scale = robustScaler.fit_transform(X_train)
        X_test_scale = robustScaler.transform(X_test)
        return X_train_scale, X_test_scale,'Robust'

    elif scaler == "MinMax":
        mmScaler = MinMaxScaler()
        X_train_scale = mmScaler.fit_transform(X_train)
        X_test_scale = mmScaler.transform(X_test)
        return X_train_scale, X_test_scale,'Minmax'

# Function for model fitting. When this Function runs, it fits the data into model each: DecisionTree(Entropy) / DecisionTree(Gini Index) / Logistic Regression / Support Vector Machine
def fitting(X_train_scale, y_train, X_test, y_test, model, scaler):
    if model == "entropy":
        entropy.fit(X_train_scale, y_train)
        pred = entropy.predict(X_test)
        score.append(f1_score(y_test, pred, average='macro'))

    elif model == "gini":
        gini.fit(X_train_scale, y_train)
        pred = gini.predict(X_test)
        score.append(f1_score(y_test, pred, average='macro'))

    elif model == "logistic":
        logistic.fit(X_train_scale, y_train)
        pred = logistic.predict(X_test)
        score.append(f1_score(y_test, pred, average='macro'))

    elif model == "SVM":
        SVM.fit(X_train_scale, y_train)
        pred = SVM.predict(X_test)
        score.append(f1_score(y_test, pred, average='macro'))

# Training Start : Runs 3 times total (k = 3, 5, 10) , scales data and fit them into each models accordingly / Show results directly
for k in K:
    kf = KFold(n_splits=k, shuffle=True, random_state=50)
    print("**************KFold : ", k, "**************")
    for train, test in kf.split(X):
        X_train, X_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]

        for s in scale_list:
            X_train_scaled, X_test_scaled, scaler_used = scaling(X_train, X_test, s)

            for m in model_list:
                fitting(X_train_scaled, y_train, X_test_scaled, y_test, m, scaler_used)
                print("Model Used : ", m)
                print("Scaler Used : ", scaler_used)
                print("Score : ", score[i])
                i += 1
                print("-------------------------\n")


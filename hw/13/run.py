from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 導入資料集
data = datasets.load_breast_cancer()
X, y = data.data, data.target

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# 設定一個 4 層的 MLP 模型
mlp = MLPClassifier(hidden_layer_sizes=(10, 10),
                    max_iter=1000,
                    activation='relu',
                    solver='adam',
                    random_state=1234)

# 訓練
mlp.fit(X_train, y_train)

# 預測
y_pred = mlp.predict(X_test)

# 準確率
print(accuracy_score(y_test, y_pred))
# 0.9385964912280702
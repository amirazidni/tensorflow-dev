import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# membaca dataset dan mengubahnya menjadi dataframe
data = pd.read_csv('dataset/Salary_Data.csv')

# memisahkan atribut dan label
X = data['YearsExperience']
y = data['Salary']

# mengubah bentuk atribut
X = X[:, np.newaxis]

# membangun model dengan parameter C, gamma, dan kernel
model = SVR(C=1000, gamma=0.05, kernel='rbf')

# melatih model dengan fungsi fit
model.fit(X, y)


# memvisualisasikan model
plt.scatter(X, y)
plt.plot(X, model.predict(X))

plt.show()
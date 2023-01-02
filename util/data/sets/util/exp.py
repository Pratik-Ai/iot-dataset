import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
data = pd.read_csv('Smart Meters Dataset.csv')
data.head()
data = data.iloc[:, 2:5]
data.count()
data['energy_median'].mean()
data.plot(kind="box", subplots=True, layout=(7, 2), figsize=(15, 20))
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)
forest = IsolationForest(
    n_estimators=100, max_samples='auto', contamination=0.006, random_state=1)
predict = forest.fit_predict(data_pca)
# The isolation forest will return value 1 for inliers and value -1 for outliers. In this case we have assumed the outliers to be 0.6% of the whole dataset.
outlier_values = data.iloc[np.where(predict == -1)]
outlier_values_pca = pca.transform(outlier_values)

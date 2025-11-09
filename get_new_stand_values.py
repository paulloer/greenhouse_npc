import pandas as pd
import numpy as np

data = pd.read_csv('./online_optimization_and_learning/GH_Data.csv', delimiter=',')

features = list(data.columns)
features.pop(0)

mean_values = {feature: np.mean(data[feature]) for feature in features}
std_values = {feature: np.sqrt(sum((data[feature] - mean_values[feature])**2) / len(data[feature])) for feature in features}

print(mean_values)
print(std_values)


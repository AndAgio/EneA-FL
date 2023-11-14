import csv
import json
import os
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

data_dir = os.path.join(parent_path, 'data', 'raw_data', 'nbaiot.csv')

data_df = pd.read_csv(data_dir)

data = []
with open(data_dir, 'rt', encoding='ISO-8859-1') as f:
    reader = csv.reader(f)
    data = list(reader)

num_users = 1
users = ['0']
num_samples = len(data) - 1

data_df = data_df.sample(frac=1).reset_index(drop=True)

# get the list of columns except the type column
features = list(data_df.columns)
features.remove("type")

# encode labels in the same way for train and test
label_encoder = LabelEncoder()
data_df["type"] = label_encoder.fit_transform(data_df["type"])
rares_indexes = data_df["type"].value_counts().nsmallest(2).index.values.astype(int)


# scale data in the same way for train and test
scaler = MinMaxScaler()
data_df[features] = scaler.fit_transform(data_df[features])

# get all the values from all the columns except the type column
Xs = data_df[features].values
ys = data_df["type"].values

clf = ExtraTreesClassifier(n_estimators=50, n_jobs=-1)
clf = clf.fit(Xs, ys)
model = SelectFromModel(clf, prefit=True)
Xs = model.transform(Xs)
Xs = Xs.reshape((-1, Xs.shape[-1]))
print(Xs.shape)

user_data = {}
row_i = 0

for u in users:
    user_data[u] = {'x': [], 'y': []}

    while row_i < num_samples:
        user_data[u]['x'].append(Xs[row_i,:].tolist())
        user_data[u]['y'].append(int(ys[row_i]))

        row_i += 1

all_data = {}
all_data['users'] = users
all_data['num_samples'] = [num_samples]
all_data['user_data'] = user_data

file_path = os.path.join(parent_path, 'data', 'all_data', 'all_data.json')

with open(file_path, 'w') as outfile:
    json.dump(all_data, outfile)

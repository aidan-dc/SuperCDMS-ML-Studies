import numpy as np
from numpy import loadtxt, savetxt, asarray
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

filename = 'reduced_data.csv'

dataset = loadtxt(filename,delimiter=',',skiprows=1)

org_train_df = pd.read_csv(filename)
org_feature_df = org_train_df.drop(['Row', 'y','PBstart','PCstart','PDstart','PFstart'], 1)  # remove ID and postions/targets 
#org_feature_df = org_train_df.drop(['Row', 'y','PAamp','PBamp','PCamp','PDamp','PFamp'], 1)
all_features = org_feature_df.columns



x = np.array(org_feature_df)

y = dataset[:,len(dataset[0])-1]

#---normalize features to be the range of [0,1]
feature_df_np = org_feature_df.to_numpy()
scaler = StandardScaler()
scaler.fit(feature_df_np)
scaled_feature_np = scaler.transform(feature_df_np)
feature_df = pd.DataFrame(data=scaled_feature_np, columns=all_features)

x = np.array(feature_df)

print(np.shape(x))

y = dataset[:,len(dataset[0])-1]

data = []
test = []

y_norm = -41.9

for i in range(len(x)):
    temp = []
    for j in range(len(x[i])):
        temp.append(x[i][j])
    temp.append(float(y[i]))
    temp_y = int(np.round(abs(y[i]),0))
    if temp_y==13 or temp_y==30 or temp_y==42:
        test.append(temp)
    else:
        data.append(temp)

data = np.array(data)

training, validation = train_test_split(data, test_size = 0.2, random_state = 42,stratify=data[:,-1])

testing = np.array(test)

training[:, [5, 1]] = training[:, [1, 5]]
training[:, [10, 2]] = training[:, [2, 10]]
training[:, [5, 3]] = training[:, [3, 5]]
training[:, [6, 4]] = training[:, [4, 6]]
training[:, [11, 5]] = training[:, [5, 11]]
training[:, [10, 6]] = training[:, [6, 10]]
training[:, [12, 8]] = training[:, [8, 12]]
training[:, [11, 9]] = training[:, [9, 11]]
training[:, [12, 10]] = training[:, [10, 12]]
training[:, [13, 11]] = training[:, [11, 13]]

validation[:, [5, 1]] = validation[:, [1, 5]]
validation[:, [10, 2]] = validation[:, [2, 10]]
validation[:, [5, 3]] = validation[:, [3, 5]]
validation[:, [6, 4]] = validation[:, [4, 6]]
validation[:, [11, 5]] = validation[:, [5, 11]]
validation[:, [10, 6]] = validation[:, [6, 10]]
validation[:, [12, 8]] = validation[:, [8, 12]]
validation[:, [11, 9]] = validation[:, [9, 11]]
validation[:, [12, 10]] = validation[:, [10, 12]]
validation[:, [13, 11]] = validation[:, [11, 13]]

testing[:, [5, 1]] = testing[:, [1, 5]]
testing[:, [10, 2]] = testing[:, [2, 10]]
testing[:, [5, 3]] = testing[:, [3, 5]]
testing[:, [6, 4]] = testing[:, [4, 6]]
testing[:, [11, 5]] = testing[:, [5, 11]]
testing[:, [10, 6]] = testing[:, [6, 10]]
testing[:, [12, 8]] = testing[:, [8, 12]]
testing[:, [11, 9]] = testing[:, [9, 11]]
testing[:, [12, 10]] = testing[:, [10, 12]]
testing[:, [13, 11]] = testing[:, [11, 13]]


np.random.shuffle(testing)
np.random.shuffle(training)
np.random.shuffle(validation)

print(np.shape(training),np.shape(validation),np.shape(testing))

savetxt('conv_training_reduced.csv', asarray(training),fmt='%s', delimiter=',')
savetxt('conv_validation_reduced.csv', asarray(validation),fmt='%s', delimiter=',')
savetxt('conv_testing_reduced.csv', asarray(testing),fmt='%s', delimiter=',')

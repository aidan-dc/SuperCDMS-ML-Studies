import numpy as np
from numpy import loadtxt, savetxt, asarray
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

filename = 'reduced_data.csv'

dataset = loadtxt(filename,skiprows=1,delimiter=',')

org_train_df = pd.read_csv(filename)
org_feature_df = org_train_df.drop(['Row', 'y'], 1)  # remove ID and postions/targets ,'PBstart','PCstart','PDstart','PFstart'
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
np.random.shuffle(testing)
np.random.shuffle(training)
np.random.shuffle(validation)

print(np.shape(training),np.shape(validation),np.shape(testing))
quit()
savetxt('training_reduced.csv', asarray(training),fmt='%s', delimiter=',')
savetxt('validation_reduced.csv', asarray(validation),fmt='%s', delimiter=',')
savetxt('testing_reduced.csv', asarray(testing),fmt='%s', delimiter=',')

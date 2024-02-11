#Import necessary libraries

import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l1

#Choose data set for this list, comment out the others

train_dataset = np.loadtxt('reduced_datasets/training_reduced.csv',delimiter=',')
val_dataset = np.loadtxt('reduced_datasets/validation_reduced.csv',delimiter=',')
test_dataset = np.loadtxt('reduced_datasets/testing_reduced.csv',delimiter=',')
data_type = 'Reduced'

#Padding of reduced set for CNN
# train_dataset = np.c_[np.zeros(len(train_dataset)),train_dataset]
# val_dataset = np.c_[np.zeros(len(val_dataset)),val_dataset]
# test_dataset = np.c_[np.zeros(len(test_dataset)),test_dataset]

# train_dataset = np.loadtxt('full_datasets/training_full_fixed.csv',delimiter=',')
# val_dataset = np.loadtxt('full_datasets/validation_full_fixed.csv',delimiter=',')
# test_dataset = np.loadtxt('full_datasets/testing_full_fixed.csv',delimiter=',')
# data_type = 'FullFix'

# train_dataset = np.loadtxt('full_datasets/training_full_fixed_reordered.csv',delimiter=',')
# val_dataset = np.loadtxt('full_datasets/validation_full_fixed_reordered.csv',delimiter=',')
# test_dataset = np.loadtxt('full_datasets/testing_full_fixed_reordered.csv',delimiter=',')
# data_type = 'FullFixRO'

# train_dataset = np.loadtxt('full_datasets/training_full_fix_new_order.csv',delimiter=',')
# val_dataset = np.loadtxt('full_datasets/validation_full_fix_new_order.csv',delimiter=',')
# test_dataset = np.loadtxt('full_datasets/testing_full_fix_new_order.csv',delimiter=',')
# data_type = 'FullFixNO'

# train_dataset = np.delete(train_dataset,[0,1,2,3,4],axis=1)
# val_dataset = np.delete(val_dataset,[0,1,2,3,4],axis=1)
# test_dataset = np.delete(test_dataset,[0,1,2,3,4],axis=1)
# data_type = 'FullNPAFixNO'

# train_dataset = np.loadtxt('full_datasets_npa/training_full_npa_fixed.csv',delimiter=',')
# val_dataset = np.loadtxt('full_datasets_npa/validation_full_npa_fixed.csv',delimiter=',')
# test_dataset = np.loadtxt('full_datasets_npa/testing_full_npa_fixed.csv',delimiter=',')
# data_type = 'FullNPAFix'

# train_dataset = np.loadtxt('subset_data/training_subset1.csv',delimiter=',')
# val_dataset = np.loadtxt('subset_data/validation_subset1.csv',delimiter=',')
# test_dataset = np.loadtxt('subset_data/testing_subset1.csv',delimiter=',')
# data_type = 'Subset1'

# train_dataset = np.loadtxt('subset_data/training_subset2.csv',delimiter=',')
# val_dataset = np.loadtxt('subset_data/validation_subset2.csv',delimiter=',')
# test_dataset = np.loadtxt('subset_data/testing_subset2.csv',delimiter=',')
# data_type = 'Subset2'

#Split data between inputs and output labels
X = train_dataset[:,0:len(train_dataset[0])-1]
y = train_dataset[:,len(train_dataset[0])-1:]

X_val = val_dataset[:,0:len(val_dataset[0])-1]
y_val = val_dataset[:,len(val_dataset[0])-1:]

X_test = test_dataset[:,0:len(test_dataset[0])-1]
y_test = test_dataset[:,len(test_dataset[0])-1:]


def DNN2_Model(L1_Reg=0.0,Batch_Norm=False,Dropout_Rate=0.0):
    model = tf.keras.Sequential(name='DNN-2')
    model.add(tf.keras.layers.Dense(units=32,kernel_regularizer=l1(L1_Reg)))
    if Dropout_Rate>0.0:
        model.add(tf.keras.layers.Dropout(Dropout_Rate))
    if Batch_Norm==True:
        model.add(tf.keras.layers.BatchNormalization(momentum=0.1,epsilon=1e-05))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(units=1,input_shape=(32,), activation='linear'))
    model.compile(loss='mse', 
                  optimizer=tf.keras.optimizers.Adam(), 
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])   
    return model

def DNN10_Model(L1_Reg=0.0,Batch_Norm=False,Dropout_Rate=0.0):
    model = tf.keras.Sequential(name='DNN-10')
    for i in range(9):
        model.add(tf.keras.layers.Dense(units=32,kernel_regularizer=l1(L1_Reg)))
        if Dropout_Rate>0.0:
            model.add(tf.keras.layers.Dropout(Dropout_Rate))
        if Batch_Norm==True:
            model.add(tf.keras.layers.BatchNormalization(momentum=0.1,epsilon=1e-05))
        model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(units=1,input_shape=(32,), activation='linear'))
    model.compile(loss='mse', 
                  optimizer=tf.keras.optimizers.Adam(), 
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])    
    return model

def LargeDNN_Model(L1_Reg=0.0,Batch_Norm=False,Dropout_Rate=0.0):
    model = tf.keras.Sequential(name='LargeDNN')
    units_list = [100,100,50,25,10]
    for i in range(len(units_list)):
        model.add(tf.keras.layers.Dense(units=units_list[i], kernel_regularizer=l1(L1_Reg)))
        if Dropout_Rate>0.0:
            model.add(tf.keras.layers.Dropout(Dropout_Rate))
        if Batch_Norm==True:
            model.add(tf.keras.layers.BatchNormalization(momentum=0.1,epsilon=1e-05))
        model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(loss='mse', 
                  optimizer=tf.keras.optimizers.Adam(), 
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def CNN_Model(L1_Reg=0.0,Batch_Norm=False,Dropout_Rate=0.0,Input_Len=None):
    if type(Input_Len)!=int:
        if Input_Len<=0:
            print('Input length must be specified as a positive integer for CNN')
            quit()
    units_list = [50,25,10]
    Kernel_Size = int(Input_Len/5)
    Stride = Kernel_Size
    print('Kernel Size: {}, Stride: {}'.format(Kernel_Size,Stride))
    model = tf.keras.Sequential(name='CNN')
    model.add(tf.keras.layers.Conv1D(filters=100, 
                                     kernel_size=Kernel_Size, 
                                     strides=Stride,
                                     input_shape=(Input_Len,1),
                                     kernel_regularizer=l1(L1_Reg)))
    if Dropout_Rate>0.0:
        model.add(tf.keras.layers.Dropout(Dropout_Rate))
    if Batch_Norm==True:
        model.add(tf.keras.layers.BatchNormalization(momentum=0.1,epsilon=1e-05))    
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.GRU(100, return_sequences='false', kernel_regularizer=l1(L1_Reg)))
    if Dropout_Rate>0.0:
        model.add(tf.keras.layers.Dropout(Dropout_Rate))
    if Batch_Norm==True:
        model.add(tf.keras.layers.BatchNormalization(momentum=0.1,epsilon=1e-05))    
    for i in range(len(units_list)):
        model.add(tf.keras.layers.Dense(units=units_list[i], kernel_regularizer=l1(L1_Reg)))
        if Dropout_Rate>0.0:
            model.add(tf.keras.layers.Dropout(Dropout_Rate))
        if Batch_Norm==True:
            model.add(tf.keras.layers.BatchNormalization(momentum=0.1,epsilon=1e-05))
        model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(loss='mse', 
                  optimizer=tf.keras.optimizers.Adam(), 
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

results = [] #Final Results Array
min_rmse = 1000 #Placeholder, if you somehow break this something is very wrong

num_interations = 50 #How many different training iterations
Lambda_1 = 1.0 #Associated loss penalty for large weights in L1 Regularization, Set as 0.0 for no L1 Regularization, recommended ~1.0
Batch_Normalization = False #Boolean to include Batch Normalization in Network architecture, recommended False if Lambda_1 is non_zero
Dropout_Rate = 0.0 #Rate of dropout in network, set as 0.0 for no Dropout, limited [0.0,1.0]
batch_num = 128 
total_epochs = 500 
Early_Stopping = True 
training_verbosity = 0 
model_name = 'DNN-2' 

#For training-logging purposes
print('Model Name: ',model_name)
print('Dataset: ', data_type)
print('L1 Reg. Value: ', Lambda_1)
print('Dropout Rate: ', Dropout_Rate)
print('Batch Normalization: ', Batch_Normalization)
print('Early Stopping: ', Early_Stopping)
print('')

for iteration in range(num_interations):
    #Select desired model and comment out the others
    
    model = DNN2_Model(Lambda_1,Batch_Normalization,Dropout_Rate)
    # model = DNN10_Model(Lambda_1,Batch_Normalization,Dropout_Rate)
    # model = LargeDNN_Model(Lambda_1,Batch_Normalization,Dropout_Rate)
    # model = CNN_Model(Lambda_1,Batch_Normalization,Dropout_Rate,len(X[0]))
 
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',verbose=2,patience=1)
    if Early_Stopping == True:
        model.fit(X,y,epochs=total_epochs,
                  batch_size=batch_num,
                  verbose=training_verbosity,
                  validation_data=(X_val,y_val),
                  callbacks=[callback])
    if Early_Stopping == False:
        model.fit(X,y,epochs=total_epochs,
                  batch_size=batch_num,
                  verbose=training_verbosity,
                  validation_data=(X_val,y_val))

    if iteration==0:
        print(model.summary())

    train_mse, train_rmse = model.evaluate(X, y, batch_size=batch_num,verbose=0)
    val_mse, val_rmse = model.evaluate(X_val, y_val, batch_size=batch_num,verbose=0)
    test_mse, test_rmse = model.evaluate(X_test, y_test, batch_size=batch_num,verbose=0)
    #Print results from this iteration and save the results
    print(train_rmse,val_rmse,test_rmse)
    results.append((train_rmse,val_rmse,test_rmse))
    #Save best model
    if test_rmse<min_rmse:
       model.save(model_name+'_'+data_type+'_Model.h5')

results.sort(reverse=False,key=lambda e: e[2])
np.savetxt(model_name+'_'+data_type+'_results.csv', np.asarray(results),fmt='%s', delimiter=',')

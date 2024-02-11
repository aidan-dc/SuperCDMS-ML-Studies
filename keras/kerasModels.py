import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l1

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
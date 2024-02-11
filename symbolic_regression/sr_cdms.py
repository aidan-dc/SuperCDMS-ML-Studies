import numpy as np

train_dataset = np.loadtxt('../datasets/reduced_datasets/training_reduced.csv',delimiter=',')
val_dataset = np.loadtxt('../datasets/reduced_datasets/validation_reduced.csv',delimiter=',')
test_dataset = np.loadtxt('../datasets/reduced_datasets/testing_reduced.csv',delimiter=',')

# train_dataset = np.loadtxt('training_importance20.csv',delimiter=',')
# val_dataset = np.loadtxt('validation_importance20.csv',delimiter=',')
# test_dataset = np.loadtxt('testing_importance20.csv',delimiter=',')

feat_num = 20

X = train_dataset[:,0:feat_num]
y = train_dataset[:,-1]

X_val = val_dataset[:,0:feat_num]
y_val = val_dataset[:,-1]

X_test = test_dataset[:,0:feat_num]
y_test = test_dataset[:,-1]

#dataset = np.concatenate((train_dataset,val_dataset))

# X1 = train_dataset[:, 1]
# X2 = train_dataset[:, 4]
# X3 = train_dataset[:, 3]

# X = []
# for i in range(len(X1)):
#     X.append((X1[i],X2[i],X3[i]))
# X = np.array(X)

# y = train_dataset[:,-1]


# X_val1 = val_dataset[:, 1]
# X_val2 = val_dataset[:, 4]
# X_val3 = val_dataset[:, 3]

# X_val = []
# for i in range(len(X_val1)):
#     X_val.append((X_val1[i],X_val2[i],X_val3[i]))
# X_val = np.array(X_val)


# y_val = val_dataset[:,-1]

# X_test1 = test_dataset[:,1] #PCstart
# X_test2 = test_dataset[:,4] #PArise
# X_test3 = test_dataset[:,3] #PFstart

# X_test = []
# for i in range(len(X_test1)):
#     X_test.append((X_test1[i],X_test2[i],X_test3[i]))#,X_test4[i]))
# X_test = np.array(X_test)

# y_test = test_dataset[:,-1]

# print(np.shape(X_test),np.shape(X_val))
# quit()
from pysr import PySRRegressor

model = PySRRegressor(
    model_selection="best",  # Result is mix of simplicity+accuracy
    niterations=50,
    binary_operators=["+", "*"],
    #timeout_in_seconds=5,
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",
	# ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    loss="loss(x, y) = (x - y)^2",
    # ^ Custom loss function (julia syntax)
)

# print(model.population)
# quit()
model.fit(X, y)

print(model)
# model = PySRRegressor.from_file("hall_of_fame_2023-06-21_064125.297.pkl") #Three Variable
quit()
# model = PySRRegressor.from_file("hall_of_fame_2023-07-20_055838.385.pkl") #Importance 20
# model = PySRRegressor.from_file("hall_of_fame_2023-07-20_060550.557.pkl") #Importance 10
# model = PySRRegressor.from_file("hall_of_fame_2023-07-20_061131.557.pkl") #Importance 5
model = PySRRegressor.from_file("hall_of_fame_2023-07-20_061616.736.pkl") #Importance 3
# model.model_selection = 'best'
model.model_selection = 'accuracy'
print(model)
# quit()
feat_num1 = 3
y_train_preds = model.predict(X[:,0:feat_num1])
y_preds = model.predict(X_test[:,0:feat_num1])
y_val_preds = model.predict(X_val[:,0:feat_num1])

# (((x0 * -10.376375) + (x2 * -3.6399584)) + -21.634624)

# x0=X_test1 #PCstart
# x1=X_test2 #PArise
# x2=X_test3 #PFrise
def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))

# input_data[:,5]
print('Train RMSE: ',rmse(y_train_preds,y))
print('Validation RMSE: ',rmse(y_val_preds,y_val))
print('HOS/Test RMSE: ',rmse(y_preds,y_test))
quit()
# quit()
def one_param_sr(PCstart):
    return (PCstart+1.6859602)*(np.cos(np.exp(PCstart))-12.573013)

def two_param_sr(PCstart,PArise):
    return (((PCstart + 1.8139315) * (PArise + -12.462509)) + (np.exp(np.sin((PCstart + np.exp(PCstart)) * np.sin(np.cos(PArise)))) + 0.22035526))

def three_param_sr(PCstart,PArise,PFstart):
    return -8.584877*PCstart+np.exp(PArise)-4.8735847*PFstart+np.sin(1/  (np.sin(PCstart) -0.661621244621 ) )-22.72335

# def three_param_sr1(PCstart,PArise,PFstart):
#     return (PArise+1.6715223)*((PCstart-13.481543)+np.cos(PFstart))

def all_param_sr(PBstart,PCstart,PDstart,PFwidth):
    return (PCstart+1.6054103)*(np.sin(np.cos(PDstart))-14.085981)+np.sin(PBstart+PFwidth)-0.2991746+PBstart
    #(test_dataset[:,1]+1.6054103)*(np.sin(np.cos(test_dataset[:,2]))-14.085981)+np.sin(test_dataset[:,0]+test_dataset[:,18])-0.2991746+test_dataset[:,0]

def importance_20_test(input_data):
    return (((input_data[:,5] * -14.17462) + input_data[:,18]) + -21.76058)
    # return ((((input_data[:,2] * (((np.sin(np.exp(input_data[:,18])) + 1.6334528) * np.exp(input_data[:,11])) + -15.550054)) + (input_data[:,18] + -21.26865)) + input_data[:,15]) + 0.06945198)


def importance_20(input_data):
    return input_data[:,2] * ( ( (np.sin(np.exp(input_data[:,18])) + 1.6334528) * np.exp(input_data[:,11])) -15.550054) + input_data[:,18] + input_data[:,15] -21.19919802
    # Validation RMSE:  1.618822903073355
    # Test/HOS RMSE:  3.259325002683673

def importance_10(input_data):
    return  input_data[:,2] * ( np.exp(input_data[:,4]) + (1/(input_data[:,0]))-13.2092047 )  -21.918127 + (input_data[:,4] * 3.1107495) + np.sin(input_data[:,9])

print('Validation RMSE: ',rmse(importance_10(X_val),y_val))
print('Test/HOS RMSE: ',rmse(importance_10(X_test),y_test))
quit()
# Single Parameter Function
# y_preds = (X_test1+1.6859602)*(np.cos(np.exp(X_test1))-12.573013)

# Two Parameter Function
#y_preds = (((x0 + 1.8139315) * (x1 + -12.462509)) + (np.exp(np.sin((x0 + np.exp(x0)) * np.sin(np.cos(x1)))) + 0.22035526))

# Three Parameter Function
# y_preds = -8.584877*x0+np.exp(x1)-4.8735847*x2+np.sin(1/  (np.sin(x0) -0.661621244621 ) )-22.72335

# All Parameter Function
# y_preds = (test_dataset[:,1]+1.6054103)*(np.sin(np.cos(test_dataset[:,2]))-14.085981)+np.sin(test_dataset[:,0]+test_dataset[:,18])-0.2991746+test_dataset[:,0]

#PBstart,PCstart,PDstart,PFwidth
# print('Validation RMSE: ',rmse(three_param_sr(X_val1,X_val2,X_val3),y_val))
# print('Test/HOS RMSE: ',rmse(three_param_sr(X_test1,X_test2,X_test3),y_test))

# print('Validation RMSE: ',rmse(all_param_sr(val_dataset[:,0],val_dataset[:,1],val_dataset[:,2],val_dataset[:,18]),y_val))
# print('Test/HOS RMSE: ',rmse(all_param_sr(test_dataset[:,0],test_dataset[:,1],test_dataset[:,2],test_dataset[:,18]),y_test))
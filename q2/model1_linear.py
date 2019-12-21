import numpy as np
import pandas as pd

def init(x):
    #initializes model params (weights, bias) as 0
    weights=np.zeros((x.shape[0],1))
    bias=0
    return weights, bias

def prop(weights, bias, x, y): 
    # cost function - forward propogation
    # calculates activation and cost
    activation=np.dot(weights.T,x)
    cost=(1/x.shape[1])*(np.sum(np.square(y-np.dot(weights.T,x))))
    cost += l2(weights,x,5)
    return activation, cost

def back_prop(activation, x, y): 
    # gradient descent - back propogation
    # returns gradient of the loss for weights and bias
    weights=(2/x.shape[1])*np.dot(x,np.subtract(activation,y).T) 
    bias=(2/x.shape[1])*np.subtract(activation,y)
    return weights, bias

def train(weights, bias, x, y, iterations, learning_rate):
    # learning using forward and back propagation
    for i in range(iterations):
        activation, cost = prop(weights, bias, x, y)
        w,b=back_prop(activation, x, y)
        weights=weights-(w*learning_rate)
        bias=bias-(b*learning_rate)     
    return weights, bias

def predict(weights, bias, x):
    # model predicts whether to classify as A or B using weights and bias
    y_predict=[]
    weights = weights.reshape(x.shape[0],1)
    activation = np.dot(weights.T,x)
    for i in range(activation.shape[1]):
        if abs(activation[0,i]-0)<abs(activation[0,i]-1):
            y_predict.append('A')
        elif abs(activation[0,i]-0)>=abs(activation[0,i]-1):
            y_predict.append('B')
    return y_predict

def l2(weights, x, lamda):
    # L2 regularization to fix overfitting
    loss=lamda/(2*x.shape[1])*(np.sum(weights))
    return loss

#Driver

#loading A, B and mystery 
A=np.loadtxt('A.txt')
B=np.loadtxt('B.txt')
mystery=np.loadtxt('mystery.txt')

#formatting A, B and mysteries
a1=np.reshape(A[0:5],(1,25))
a2=np.reshape(A[5:10],(1,25))
a3=np.reshape(A[10:15],(1,25))
a4=np.reshape(A[15:20],(1,25))
a5=np.reshape(A[20:25],(1,25))
b1=np.reshape(B[0:5],(1,25))
b2=np.reshape(B[5:10],(1,25))
b3=np.reshape(B[10:15],(1,25))
b4=np.reshape(B[15:20],(1,25))
b5=np.reshape(B[20:25],(1,25))
mystery1=np.reshape(mystery[0:5],(1,25))
mystery2=np.reshape(mystery[5:10],(1,25))
mystery3=np.reshape(mystery[10:15],(1,25))
mystery4=np.reshape(mystery[15:20],(1,25))
mystery5=np.reshape(mystery[20:25],(1,25))

#preparing up training and test data
train_x=np.concatenate((a1,a2,a3,a4,a5,b1,b2,b3,b4,b5)).T
train_y=np.array([0,0,0,0,0,1,1,1,1,1])
test_x=np.concatenate((mystery1,mystery2,mystery3,mystery4,mystery5)).T

#initializing weights and bias as 0
weights,bias=init(train_x)

#training using weights and bias and training data 
weights,bias=train(weights,bias,train_x,train_y,2000, 0.05)

#make predictions and display in dataframe
df=pd.DataFrame({'image': ['Mystery 1','Mystery 2','Mystery 3','Mystery 4','Mystery 5'],
                  'prediction': predict(weights, bias, test_x)})
print(df)
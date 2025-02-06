#%%
#importing numpy
import numpy as np

#Buidling a single neuron

#defining the sigmoid function
def sigm(X,W,B):
    return 1/(1+np.exp(-(X.dot(W.T)+B)))

#feedforward function 
def Forward(X, W1, B1, W2, B2):
    #first later
    H = sigm(X, W1, B1)

    Y = sigm(H, W2, B2)

    return Y, H

#random weights and bias
W1 = np.random.randn(3,2)
B1 = np.random.randn(3)
W2 = np.random.randn(1,3)
B2 = np.random.randn(1)

def diff_B2(Z, Y):
    dB = (Z-Y)*Y*(1-Y)
    return dB.sum(axis = 0)

def diff_W2(H, Z, Y):
    dW = (Z-Y)*Y*(1-Y)
    return H.T.dot(dW)

def diff_W1(X,H,Z,Y,W2):
    dZ = (Z-Y).dot(W2)*Y*(1-Y)*H*(1-H)
    return X.T.dot(dZ)

def diff_B1 (Z,Y,W2,H):
    return((Z - Y).dot(W2) * Y * (1 - Y) * H * (1 - H)).sum(axis=0)




#creating training set & test set for bitwise OR operator 
X = np.random.randint(2,size=[100,2])
Z = np.array([X[:,0] ^ X[:,1]]).T

X_test = np.random.randint(2,size=[15,2])
Y_test = np.array([X_test[:,0] ^ X_test[:,1]]).T

learning_rate = 0.01

#learning loop 
for epoch in range(15000): 
    Y, H = Forward(X, W1, B1, W2, B2)

    W2 += learning_rate * diff_W2(H, Z, Y).T
    B2 += learning_rate * diff_B2(Z,Y)
    W1 += learning_rate * diff_W1(X, H, Z, Y, W2).T
    B1 += learning_rate * diff_B1(Z, Y, W2, H)

    if not epoch % 50:
        Accurancy = 1 - np.mean((Z-Y)**2)
        print('epoech: ' , epoch, 'accuracy: ', Accurancy)
# Calculate error in percent

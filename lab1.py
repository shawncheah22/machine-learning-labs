#%%
#importing numpy
import numpy as np

#Buidling a single neuron

#defining the sigmoid function
def sigm(X,W,B):
    return 1/(1+np.exp(-(X.dot(W.T)+B)))

#simulate neuron with n = 2 inputs
W = np.random.randn(2)
B = np.random.randn(1)

#To make the neuron learn, we need to define a loss function. We will be using the square error function and use gradient descent.
# Z is predicted output, Y is actual output
def diff_w(X, Z, Y, B, W):
    dS = sigm(X,W,B)*(1-sigm(X,W,B))
    dW = (Y-Z)*dS
    #mutliply by X transpose, dot product as X is a matrix. X needs to be multplied by x = wx+b dx/dW = X
    return np.sum(X.T.dot(dW))


def diff_B(X,Z,Y,B,W):
    dS = sigm(X,W,B)*(1-sigm(X,W,B))
    dB   = (Y-Z)*dS
    return dB.sum()


#creating training set & test set for bitwise OR operator 
X = np.random.randint(2,size=[100,2])
Y = np.array([X[:,0] | X[:,1]]).T

X_test = np.random.randint(2,size=[15,2])
Y_test = np.array([X_test[:,0] | X_test[:,1]]).T

learning_rate = 0.01

#learning loop 
for epoch in range(15000): 
    output = sigm(X, W, B)
    W += learning_rate * diff_w(X, output, Y, B, W).T
    B += learning_rate * diff_B(X, output, Y, B, W)

    accuracy = np.mean((output > 0.5) == Y_test) * 100
    print(f"Accuracy: {accuracy:.2f}%")

# Calculate error in percent
error = np.mean(np.abs(output - Y_test)) * 100
print(f"Error: {error:.2f}%")

#compare test sets & predictiosn 
print(sigm(X_test, W, B))
print(Y_test.T)
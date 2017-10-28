import numpy as np
import pandas as pd

def sigmoid(z):
    z = z.astype(np.float64)
    return 1 / (1 + np.exp(-z))

def costFunction(X, y, weights):
    m = np.size(X,0)
    hx = sigmoid(X.dot(weights))
    hx = hx.astype(np.float64)
    return (1 / m) * sum(-y * np.log10(hx) - (1 - y) * np.log10(1 - hx))

def main():
    #Load data
    data = pd.read_csv('titanic.csv', skipinitialspace=True, quotechar='"', skiprows=1)
    #Turn Panda into Numpy
    data = pd.DataFrame.as_matrix(data)
    
    #x = passenger class
    X = data[:, 2:3]
    X = np.insert(X,0,1,axis = 1) #Adding column of 1's for the bias units!
    #y = slept with the fishes
    y = data[:, 1:2]

    #Initializing weights at 0!
    weights = np.zeros((np.size(X,1),1))
    
    #Calculate cost function
    print(costFunction(X, y, weights))

if __name__ == '__main__':
    main()


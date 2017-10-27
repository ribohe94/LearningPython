import numpy as np
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def main():
    #Load data
    data = pd.read_csv('titanic.csv', skipinitialspace=True, quotechar='"', skiprows=1)
    #Turn Panda into Numpy
    data = pd.DataFrame.as_matrix(data)
    
    #x = passenger class
    X = data[:, 2]
    #y = slept with the fishes
    y = data[:, 1]

    #Initializing weights at 0!
    weights = np.zeros((np.size(X,1),1))
    X = np.insert(X,0,1,axis = 1) #Adding column of 1's for the bias units!
    #Normalizing data
    #data = (data - np.mean(data)) / np.std(data)

if __name__ == '__main__':
    main()


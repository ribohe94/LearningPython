import numpy as np

def getCost(X, y, weights):
    pass

def main():
   #Load data x=population, y=profit
    data = np.genfromtxt('ex1data1.txt', delimiter=',')


    #Initializing weights at 0!
    weights = np.zeros((2,1))
    
    #Hyperparameters!
    m = np.size(data,0)
    data=np.insert(data,0,1,axis=1) #Adding the bias unit!
    iterations = 1500
    alpha = 0.01



if __name__ == '__main__':
    main()


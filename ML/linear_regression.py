import numpy as np

def getCost(X, y, weights):
    m = np.size(X,0)
    hx = X.dot(weights)
    cost = (1/(2*m)) * np.sum((hx[:,0] - y)**2)
    return cost

def main():
   #Load data x=population, y=profit
    data = np.genfromtxt('ex1data1.txt', delimiter=',')


    #Initializing weights at 0!
    weights = np.zeros((2,1))
    
    #Hyperparameters!
    m = np.size(data,0)
    data=np.insert(data,0,1,axis=1) #Adding the bias unit!
    X = data[:,0:2]
    y = data[:,2]
    iterations = 1500
    alpha = 0.01
    print(getCost(X, y, weights))



if __name__ == '__main__':
    main()


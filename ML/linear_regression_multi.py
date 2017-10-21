import numpy as np

def getCost(X, y, weights):
    m = np.size(X,0)
    hx = X.dot(weights)
    cost = (1/(2*m)) * np.sum((hx[:,0] - y)**2)
    return cost

def derCost(weights, X, y, alpha):
    m = np.size(X,0)
    hx = X.dot(weights)
    bias_unit = weights[0] - alpha * (1/m) * np.sum(hx[:,0] - y)
    temp_weights = weights[1:] - alpha * (1/m) * np.sum((hx[:, 0:1] - y) * X[:,1:])
    temp_weights=np.insert(temp_weights,0,bias_unit,axis=0)
    return temp_weights

def gradientDescent(i_weights, X, y, iterations, alpha):
    print("Initial cost: {0}".format(getCost(X, y, i_weights)))

    weights = i_weights
    for i in range(iterations):
        weights = derCost(weights, X[i, :], y, alpha)
        print("{0}: Cost: {1}, weights: {2}, {3}".format(i, getCost(X, y, weights), weights[0], weights[1]))
    print(weights)

def main():
    #Load data x=population, y=profit
    data = np.genfromtxt('ex1data2.txt', delimiter=',')

    #Hyperparameters!
    m = np.size(data,0)
    X = data[:, :-1]
    X = np.insert(X,0,1,axis=1) #Adding the bias unit!
    y = data[:, np.size(data,1) - 1:]
    iterations = 1500
    alpha = 0.01

    #Initializing weights at 0!
    weights = np.zeros((np.size(X,1),1))

    #gradientDescent(weights, X, y, iterations, alpha)
    print(getCost(X, y, weights))

if __name__ == '__main__':
    main()


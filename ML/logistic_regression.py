import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb

def sigmoid(z):
    z = z.astype(np.float64)
    return 1 / (1 + np.exp(-z))

def costFunction(X, y, weights):
    m = np.size(X,0)
    return (1 / m) * sum(-y * np.log(sigmoid(X.dot(weights))) - (1 - y) * np.log(1 - sigmoid(X.dot(weights))))

def costFunctionDerivative(X, y, weights, alpha):
    #pdb.set_trace()
    m = np.size(X,0)
    activation = sigmoid(X.dot(weights)) - y #Activation function g(z)
    activation = activation.astype(np.float64) #Turn it into float64
    temp_weights = np.copy(weights)
    
    for j in range(0, np.size(weights)):
        temp_weights[j] = weights[j] - alpha * (1/m) * np.sum((sigmoid(X.dot(weights)) - y) * X[:, j:j+1])
    return temp_weights

def gradientDescent(X, y, weights, alpha, iterations):
    print("Initial cost function: {0}".format(costFunction(X, y, weights)))
    for i in range(0, iterations):
        weights = costFunctionDerivative(X, y, weights, alpha)
    print("Final cost function: {0}".format(costFunction(X,y,weights)))
    return weights

def plotData(data, xAxis, yAxis):
    approved = data[data['approved'].isin([1])] #Students that approved
    ded = data[data['approved'].isin([0])] #Slept with the fishes
    
    fig, ax = plt.subplots(figsize=(16,9))  
    ax.scatter(approved[xAxis], approved[yAxis], s=30, c='b', marker='o', label='approved')
    ax.scatter(ded[xAxis],ded[yAxis], s=30, c='r', marker='x', label='Ded')
    ax.legend()
    ax.set_xlabel(xAxis)  
    ax.set_ylabel(yAxis)
    plt.show()


def main():
    #Load data
    headers=['exam1','exam2','approved']
    data = pd.read_csv('ex2data1.txt', names=headers, skipinitialspace=True, quotechar='"')
    
    #x = passenger class, Fare
    X = pd.DataFrame(data[['exam1','exam2']])
    #X = (X - np.mean(X)) / np.std(X) #Normalizing data
    X.insert(loc=0, column='Bias', value=1) #Adding column of 1's for the bias unit!
    #y = slept with the fishes
    y = data['approved']

    #Initializing weights at 0!
    weights = np.zeros((np.size(X,1),1))

    iterations = 1500
    alpha = 0.001
    
    #plotData(data, 'exam1', 'exam2')
        
    #Calculate cost function
    X = np.array(X.values) #Turn X into an ndarray for ease of usage
    y = np.array(y.values) #Turn y into an ndarray for ease of usage
    y = np.reshape(y, [np.size(y),1]) #Reshape it into (size,1) nd array instead of (size,)
    optimized_weights = gradientDescent(X, y, weights, alpha, iterations)
    

    #Load test data
    test_data = pd.read_csv('ex2data1.txt', names=headers, skipinitialspace=True, quotechar='"')
    test_y = pd.DataFrame(test_data['approved'])
    #x = passenger class, Fare
    X_OPT = pd.DataFrame(test_data[['exam1','exam2']])
    X_OPT.insert(loc=0, column='Bias', value=1) #Adding column of 1's for the bias unit!
    predictions = sigmoid(X_OPT.dot(optimized_weights))
    test_y.insert(loc=0, column='Predictions', value=predictions)
    print(test_y)
    #pdb.set_trace()

    index = 0
    for i in test_y['Predictions']:
        count = 0
        if(i > 0.5 and test_y['approved'][index] == 1):
            count += 1
        elif(i <= 0.5 and test_y['approved'][index] == 0):
            count += 1
        index += 1
    print("Accuracy rate: {0}%".format((count * 100)/np.size(test_data,0)))

    test_features = np.array([[1, 45, 85]], np.float64)
    print("Probability for 45 and 85: {0}%".format(sigmoid(test_features.dot(optimized_weights))))


if __name__ == '__main__':
    main()


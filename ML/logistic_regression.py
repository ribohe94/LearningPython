import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
    z = z.astype(np.float64)
    return 1 / (1 + np.exp(-z))

def costFunction(X, y, weights):
    m = np.size(X,0)
    activation = sigmoid(X.dot(weights))
    activation = activation.astype(np.float64)
    return (1 / m) * sum(-y * np.log10(activation) - (1 - y) * np.log10(1 - activation))

def costFunctionGradient(X, y, weights, alpha):
    m = np.size(X,0)
    activation = sigmoid(X.dot(weights)) - y #Activation function g(z)
    activation = activation.astype(np.float64) #Turn it into float64
    temp_weights = weights
    temp_weights[0] = weights[0] - alpha * (1/m) * np.sum(activation) #Bias unit
    
    for j in range(1, np.size(weights)):
        temp_weights[j] = weights[j] - alpha * (1/m) * np.sum(activation * X[:, j:j+1])
    
    return temp_weights

def gradientDescent(X, y, weights, alpha, iterations):
    print("Initial cost function: {0}".format(costFunction(X,y,weights)))
    for i in range(0, iterations):
        weights = costFunctionGradient(X, y, weights, alpha)
    print("Final cost function: {0}".format(costFunction(X,y,weights)))
    return weights

def plotData(data, xAxis, yAxis):
    survived = data[data['Survived'].isin([1])] #Passengers that survived
    ded = data[data['Survived'].isin([0])] #Slept with the fishes
    
    fig, ax = plt.subplots(figsize=(16,9))  
    ax.scatter(survived[xAxis], survived[yAxis], s=10, c='b', marker='o', label='Survived')
    ax.scatter(ded[xAxis],ded[yAxis], s=10, c='r', marker='x', label='Ded')
    ax.legend()
    ax.set_xlabel(xAxis)  
    ax.set_ylabel(yAxis)
    plt.show()


def main():
    #Load data
    headers=['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
    data = pd.read_csv('titanic.csv', header=0, names=headers, skipinitialspace=True, quotechar='"', skiprows=1)
    
    #x = passenger class, Fare
    X = pd.DataFrame(data[['Pclass', 'Fare']])
    X = (X - np.mean(X)) / np.std(X) #Normalizing data
    X.insert(loc=0, column='Bias', value=1) #Adding column of 1's for the bias unit!
    #y = slept with the fishes
    y = data['Survived']

    #Initializing weights at 0!
    weights = np.zeros((np.size(X,1),1))

    iterations = 1500
    alpha = 0.01
    
    #plotData(data, 'PassengerId', 'Pclass')
        
    #Calculate cost function
    X = np.array(X.values) #Turn X into an ndarray for ease of usage
    y = np.array(y.values) #Turn y into an ndarray for ease of usage
    y = np.reshape(y, [np.size(y),1]) #Reshape it into (size,1) nd array instead of (size,)
    optimized_weights = gradientDescent(X, y, weights, alpha, iterations)
    

    #Load test data
    test_data = pd.read_csv('titanic_test.csv', header=0, names=headers, skipinitialspace=True, quotechar='"', skiprows=1)
    test_y = pd.DataFrame(test_data['Survived'])
    #x = passenger class, Fare
    X_OPT = pd.DataFrame(test_data[['Pclass', 'Fare']])
    X_OPT = (X_OPT - np.mean(X_OPT)) / np.std(X_OPT) #Normalizing data
    X_OPT.insert(loc=0, column='Bias', value=1) #Adding column of 1's for the bias unit!
    predictions = sigmoid(X_OPT.dot(optimized_weights))
    test_y.insert(loc=0, column='Predictions', value=predictions)
    print(test_y)

    index = 0
    for i in test_y['Predictions']:
        count = 0
        if(i > 0.5 and test_y['Survived'][index] == 1):
            count += 1
        elif(i <= 0.5 and test_y['Survived'][index] == 0):
            count += 1
        index += 1
    print("Accuracy rate: {0}%".format((count * 100)/np.size(test_data,0)))

if __name__ == '__main__':
    main()


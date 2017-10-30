import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
    z = z.astype(np.float64)
    return 1 / (1 + np.exp(-z))

def costFunction(X, y, weights):
    m = np.size(X,0)
    hx = sigmoid(X.dot(weights))
    hx = hx.astype(np.float64)
    return (1 / m) * sum(-y * np.log10(hx) - (1 - y) * np.log10(1 - hx))

def costFunctionGradient(X, y, weights):
    m = np.size(X,0)
    gradient = sigmoid(X.dot(weights)) - y
    gradient = gradient.astype(np.float64)
    temp_weights[0] = weights[0] - (1/m) * np.sum(gradient)
    
    for j in range(1,n):
        temp_weights[j] = weights[j] - alpha * (1/m) * np.sum(gradient * X[:, j:j+1])


def main():
    #Load data
    data = pd.read_csv('titanic.csv', header=0, names=['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'], skipinitialspace=True, quotechar='"', skiprows=1)
    
    #x = passenger class
    X = pd.DataFrame(data['Pclass'])
    X.insert(loc=0, column='Bias', value=1) #Adding column of 1's for the bias units!
    #y = slept with the fishes
    y = data['Survived']

    #Initializing weights at 0!
    weights = np.zeros((np.size(X,1),1))
    
    survived = data[data['Survived'].isin([1])]
    ded = data[data['Survived'].isin([0])]
    
    fig, ax = plt.subplots(figsize=(16,9))  
    ax.scatter(survived['PassengerId'], survived['Pclass'], s=10, c='b', marker='o', label='Survived')  
    ax.scatter(ded['PassengerId'],ded['Pclass'], s=10, c='r', marker='x', label='Ded')
    ax.legend()
    ax.set_xlabel('Passenger sequence number')  
    ax.set_ylabel('Age')
    plt.show()
    
    #Calculate cost function
    print(costFunction(X, y, weights))

if __name__ == '__main__':
    main()


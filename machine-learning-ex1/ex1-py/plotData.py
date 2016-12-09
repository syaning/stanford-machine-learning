import matplotlib.pyplot as plt


def plotData(x, y):
    plt.scatter(x, y, marker='+')
    plt.xlim(5)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()


def plotData2(theta, X, y):
    plt.scatter(X[:, 1], y, marker='+')
    plt.plot(X, X.dot(theta), 'r')
    plt.xlim(5)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()

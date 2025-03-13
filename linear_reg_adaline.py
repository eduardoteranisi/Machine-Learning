import random
import matplotlib
from matplotlib import pyplot as plt


class AdalineRegressor:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def fit(self, X, Y):
        n_samples, n_features = len(X), len(X[0])
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(n_features)]
        self.bias = 0

        for epoch in range(self.epochs):
            total_error = 0

            for i in range(n_samples):
                Y_pred = self._predict(X[i])
                error = Y[i] - Y_pred

                for j in range(n_features):
                    self.weights[j] += self.learning_rate * error * X[i][j]
                self.bias += self.learning_rate * error

                total_error += error ** 2

            mse = total_error / n_samples  
            if epoch % (self.epochs // 10) == 0:
                print(f"Época {epoch}, Erro MSE: {mse}")

    def _predict(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias

    def predict(self, X):
        return [self._predict(x) for x in X]

####################
#    Leitura do arquivo
####################

X_train = []
Y_train = []

with open("basedeobservacoes_trabalho06.txt", "r") as file:
    next(file) 
    for line in file:
        values = line.split()
        if len(values) == 2:
            x_value, y_value = map(float, values)
            X_train.append([x_value])
            Y_train.append(y_value)

adaline = AdalineRegressor(learning_rate=0.01, epochs=1000) 
adaline.fit(X_train, Y_train)

#Coeficientes
w = adaline.weights[0]
b = adaline.bias


####################
#    GRAFICO
####################
X_line = [min(x[0] for x in X_train), max(x[0] for x in X_train)]
Y_line = [w * x + b for x in X_line]


plt.scatter([x[0] for x in X_train], Y_train, color='blue', label='Dados de Treinamento')
plt.plot(X_line, Y_line, color='red', label='Reta da Regressão')

plt.xlabel("X")
plt.ylabel("y")
plt.title("Regressão Linear com Adaline")
plt.legend()
plt.grid()


matplotlib.use('Agg')
plt.savefig("grafico.png")
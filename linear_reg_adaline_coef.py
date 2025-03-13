import matplotlib
from matplotlib import pyplot as plt
import math


class AdalineRegressor:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = 0
        self.bias = 0

    def train(self, X, Y):
        n_samples = len(X)
        self.weights, coef_pearson, coef_det = self.calc_b(X,Y)
        self.bias = self.calc_a(X,Y)

        for epoch in range(self.epochs):
            total_error = 0

            for i in range(n_samples):
                Y_pred = self._predict(X[i])
                error = Y[i] - Y_pred

                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error

                total_error += error ** 2

            mse = total_error / n_samples  

            #Criterios de parada
            if epoch % 10 == 0:
                print(f"Época {epoch}, Erro MSE: {mse}")
            if mse < 1e-5:
                print(f"Pearson: {coef_pearson}, Coef. determinacao: {coef_det}")
                break

        print(f"Pearson: {coef_pearson}, Coef. determinacao: {coef_det}")

    def _predict(self, x):
        return self.weights * x + self.bias

    def predict(self, X):
        return [self._predict(x) for x in X]
    
    def calc_a(self, X, Y):
        y_sum = 0
        for y in Y:
            y_sum += y
        
        mean_y = y_sum / len(Y)

        x_sum = 0
        for x in X:
            x_sum += x
        
        mean_x = x_sum / len(X)
        
        b, c, e = self.calc_b(X,Y)

        a = mean_y - (b*mean_x)
        return a

    def calc_b(self, X, Y):
        n = len(X)
        
        aux0, aux1, aux2, aux3, aux4, aux5 = 0,0,0,0,0,0
        for i in range(n):
            aux0 += X[i] * Y[i]
            aux1 += X[i]
            aux2 += Y[i]
            aux3 += X[i]**2
            aux4 += X[i]
            aux5 += Y[i]**2
        
        exp_num = (n*aux0) - (aux1*aux2)
        b = (exp_num) / ((n*aux3) - (aux4)**2)

        pearson = (exp_num) / (math.sqrt((n*aux3) - (aux1)**2) * math.sqrt((n*aux5) - (aux2)**2))
        det = pearson**2
        return b, pearson, det
    

####################
#    Leitura do arquivo
####################

X_train = []
Y_train = []

with open("basedeobservacoes_trabalho06.txt", "r") as file:
    next(file)  # Pula a primeira linha (cabeçalho)
    for line in file:
        values = line.split()
        if len(values) == 2:
            x_value, y_value = map(float, values)
            X_train.append(x_value)
            Y_train.append(y_value)

adaline = AdalineRegressor(learning_rate=0.01, epochs=1000)  #learning_rate = 0.09 da problema
adaline.train(X_train, Y_train)

#Coeficientes
w = adaline.weights
b = adaline.bias

####################
#    GRAFICO
####################
X_line = [min(x for x in X_train), max(x for x in X_train)]
Y_line = [w * x + b for x in X_line]


plt.scatter([x for x in X_train], Y_train, color='blue', label='Dados de Treinamento')
plt.plot(X_line, Y_line, color='red', label='Reta da Regressão')

plt.xlabel("X")
plt.ylabel("y")
plt.title("Regressão Linear com Adaline")
plt.legend()
plt.grid()


# plt.tight_layout()  # Ajusta o espaçamento entre os gráficos
matplotlib.use('Agg')
plt.savefig("grafico_coef.png")
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
    next(file)  # Pula a primeira linha (cabeçalho)
    for line in file:
        values = line.split()
        if len(values) == 2:
            x_value, y_value = map(float, values)
            X_train.append([x_value])
            Y_train.append(y_value)

adaline = AdalineRegressor(learning_rate=0.01, epochs=1000)  #learning_rate = 0.09 da problema
adaline.fit(X_train, Y_train)

#Coeficientes
w = adaline.weights[0]
b = adaline.bias

#Encontrar o coeficiente de correlação de Pearson e coeficiente de determinação

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


##Ajustar o codigo para treinar e comparar os dois jeitos
# Treinar dois modelos com taxas de aprendizado diferentes
# adaline1 = AdalineRegressor(learning_rate=0.01, epochs=1000)
# adaline1.fit(X_train, y_train)

# adaline2 = AdalineRegressor(learning_rate=0.001, epochs=1000)
# adaline2.fit(X_train, y_train)

# # Criar valores para a reta da regressão
# X_test = [[x] for x in range(0, 5)]
# y_pred1 = adaline1.predict(X_test)
# y_pred2 = adaline2.predict(X_test)

# # Criar gráficos lado a lado para comparação
# fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # Criar uma figura com 2 gráficos lado a lado

# # Primeiro gráfico
# axs[0].scatter([x[0] for x in X_train], y_train, color='blue', label='Dados de Treinamento')
# axs[0].plot([x[0] for x in X_test], y_pred1, color='red', label='Regressão (LR=0.01)')
# axs[0].set_title("Regressão com LR=0.01")
# axs[0].set_xlabel("X")
# axs[0].set_ylabel("y")
# axs[0].legend()
# axs[0].grid()

# # Segundo gráfico
# axs[1].scatter([x[0] for x in X_train], y_train, color='blue', label='Dados de Treinamento')
# axs[1].plot([x[0] for x in X_test], y_pred2, color='green', label='Regressão (LR=0.001)')
# axs[1].set_title("Regressão com LR=0.001")
# axs[1].set_xlabel("X")
# axs[1].set_ylabel("y")
# axs[1].legend()
# axs[1].grid()

# plt.tight_layout()  # Ajusta o espaçamento entre os gráficos
matplotlib.use('Agg')
plt.savefig("grafico.png")
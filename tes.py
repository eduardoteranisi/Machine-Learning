import random

class MultiPerceptron:
    def __init__(self, input_size, output_size=10, learning_rate=0.1, epochs=100):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Inicializa pesos aleatórios (incluindo bias)
        self.weights = [[random.uniform(-1, 1) for _ in range(input_size + 1)] for _ in range(output_size)]

    def step_function(self, x):
        return 1 if x >= 0 else -1

    def predict(self, x):
        x = [1] + x  # Adiciona bias
        return [self.step_function(sum(w * xi for w, xi in zip(neuron, x))) for neuron in self.weights]

    def train(self, X, Y):
        for _ in range(self.epochs):
            for i in range(len(X)):
                x = [1] + X[i]  # Adiciona bias
                y_pred = self.predict(X[i])
                for j in range(self.output_size):
                    error = Y[i][j] - y_pred[j]
                    if error != 0:  # Apenas ajusta se houver erro
                        self.weights[j] = [w + self.learning_rate * error * xi for w, xi in zip(self.weights[j], x)]


# Exemplo: 4 amostras com 3 entradas cada
X_train = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
]

# 10 neurônios, cada um aprendendo um padrão diferente
Y_train = [
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
    [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

# Criando e treinando o perceptron
perceptron = MultiPerceptron(input_size=3)
perceptron.train(X_train, Y_train)

# Teste
for x in X_train:
    print(f"Entrada: {x} -> Saída: {perceptron.predict(x)}")

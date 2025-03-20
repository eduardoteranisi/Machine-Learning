import random
import math
import matplotlib.pyplot as plt

# Função de ativação sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Derivada da função sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Dados
x_data = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
y_data = [-0.9602, -0.5770, -0.0729, 0.3771, 0.6405, 0.6600, 0.4609, 0.1336, -0.2013, -0.4344, -0.5000]

# Inicialização dos pesos e bias (Camada oculta com 5 neurônios)
random.seed(42)
w1 = [random.uniform(-1, 1) for _ in range(5)]
w2 = [random.uniform(-1, 1) for _ in range(5)]
b1 = [random.uniform(-1, 1) for _ in range(5)]
b2 = random.uniform(-1, 1)

# Taxa de aprendizado
taxa_aprendizado = 0.01

# Treinamento
epochs = 10000
for epoch in range(epochs):
    erro_total = 0
    for i in range(len(x_data)):
        # Forward pass
        h = [sigmoid(x_data[i] * w1[j] + b1[j]) for j in range(5)]
        y_pred = sum(h[j] * w2[j] for j in range(5)) + b2
        
        # Erro
        erro = y_data[i] - y_pred
        erro_total += erro ** 2
        
        # Backpropagation
        d_w2 = [erro * h[j] for j in range(5)]
        d_b2 = erro
        d_h = [erro * w2[j] for j in range(5)]
        d_w1 = [d_h[j] * sigmoid_derivative(h[j]) * x_data[i] for j in range(5)]
        d_b1 = [d_h[j] * sigmoid_derivative(h[j]) for j in range(5)]
        
        # Atualização dos pesos
        for j in range(5):
            w1[j] += taxa_aprendizado * d_w1[j]
            w2[j] += taxa_aprendizado * d_w2[j]
            b1[j] += taxa_aprendizado * d_b1[j]
        b2 += taxa_aprendizado * d_b2
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Erro: {erro_total}')

# Teste e plotagem
x_test = [i / 100.0 for i in range(101)]
y_test = [sum(sigmoid(x * w1[j] + b1[j]) * w2[j] for j in range(5)) + b2 for x in x_test]

plt.scatter(x_data, y_data, color='red', label='Dados Originais')
plt.plot(x_test, y_test, label='MLP Aproximação', color='blue')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('MLP Aproximando Função')
plt.show()

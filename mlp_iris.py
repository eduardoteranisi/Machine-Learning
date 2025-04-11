import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd


# Carregar o novo arquivo enviado
file_path = "iris_data.xlsx"

# Ler o arquivo assumindo colunas A-E (sem cabeçalho)
df = pd.read_excel(file_path, header=None)
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

# Separar dados e rótulos
X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

# Codificar os rótulos
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separar conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Arquitetura do MLP (implementação simples)
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.lr = learning_rate
        self.hidden_size = hidden_size
        
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def softmax(self, z):
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / np.sum(e_z, axis=1, keepdims=True)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def one_hot(self, y, num_classes):
        return np.eye(num_classes)[y]

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y_true):
        m = y_true.shape[0]
        y_one_hot = self.one_hot(y_true, self.W2.shape[1])
        dz2 = self.a2 - y_one_hot
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Atualização
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def compute_loss(self, y_true):
        m = y_true.shape[0]
        y_one_hot = self.one_hot(y_true, self.W2.shape[1])
        log_probs = -np.log(self.a2[range(m), y_true] + 1e-9)
        loss = np.sum(log_probs) / m
        return loss

    def train(self, X, y, epochs=200):
        loss_history = []
        for epoch in range(epochs):
            self.forward(X)
            loss = self.compute_loss(y)
            loss_history.append(loss)
            self.backward(X, y)
        return loss_history

# Instanciar e treinar o modelo
mlp = MLP(input_size=4, hidden_size=10, output_size=3, learning_rate=0.1)
losses = mlp.train(X_train, y_train, epochs=200)

# Gerar a figura com o erro durante as épocas
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Erro (Loss)')
plt.xlabel("Épocas")
plt.ylabel("Erro")
plt.title("Erro durante o treinamento da MLP (Iris Dataset)")
plt.legend()
fig_path = "mlp_iris_training_loss.png"
plt.savefig(fig_path)

# Fazer previsões no conjunto de teste
y_pred_probs = mlp.forward(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)


# Reduzir para 2D com PCA
pca = PCA(n_components=2)
X_test_2D = pca.fit_transform(X_test)

# Cores por classe prevista
colors = ['red', 'green', 'blue']
target_names = label_encoder.classes_

# Gerar o gráfico
plt.figure(figsize=(10, 6))
for i, target_name in enumerate(target_names):
    idx = y_pred == i
    plt.scatter(X_test_2D[idx, 0], X_test_2D[idx, 1],
                alpha=0.6, c=colors[i], label=f"Previsto: {target_name}")

# Marcar os erros com um 'x'
errors = y_pred != y_test
plt.scatter(X_test_2D[errors, 0], X_test_2D[errors, 1],
            facecolors='none', edgecolors='k', linewidths=1.5, marker='x', label='Erros')

plt.title("Classificação MLP no Iris Dataset (PCA 2D)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()
plt.grid(True)

# Salvar a imagem
fig_vis_path = "mlp_iris_classification_2D.png"
plt.savefig(fig_vis_path)
plt.close()
import numpy as np
import pandas as pd

class Adaline:
    def __init__(self, taxa_aprendizado=0.01, epocas=100):
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.pesos = None

    def net_input(self, X):
        return np.dot(X, self.pesos[1:]) + self.pesos[0] 

    def ativacao(self, u):
        return u  

    def fit(self, X, y):
        #Treino usando Gradiente Descendente
        n_amostras, n_features = X.shape
        self.pesos = np.random.uniform(-0.01, 0.01, n_features + 1)  # Pesos iniciais aleatorios

        for epoca in range(self.epocas):
            u = self.net_input(X)
            y_pred = self.ativacao(u) 
            
            erros = y - y_pred  # Erro contínuo
            
            # Atualização dos pesos
            self.pesos[1:] += self.taxa_aprendizado * np.dot(X.T, erros) 
            self.pesos[0] += self.taxa_aprendizado * erros.sum()  
            
            erro_mse = np.mean(erros ** 2)  # Calculo do erro quadrático médio
            if epoca % 10 == 0:
                print(f'Época {epoca} - Erro MSE: {erro_mse:.5f}')
            
            if erro_mse < 1e-5:
                break

    def prever(self, X):
        u = self.net_input(X)
        return np.where(self.ativacao(u) >= 0, 1, -1)  # Regra do sinal


# Lendo base de dados
df = pd.read_excel('Basedados_B2_Adaline.xlsx')

dados = df[["s1", "s2", "t"]].values  

X = dados[:, :-1] #s1 e s2
y = dados[:, -1]  #t


rede = Adaline(taxa_aprendizado=0.01, epocas=100)
print("\nTreinamento\n")
rede.fit(X, y)

novas_entradas = np.array([
    [2.215, 2.063], 
    [1.526, 0.596]
])

entradas_ruido = np.array([
    [2.215, 2.0], 
    [1.500, 0.596]
])

print("\nTeste:\n")
previsoes = rede.prever(novas_entradas)
for i, entrada in enumerate(novas_entradas):
    print(f"Entrada: {entrada} | Classe prevista: {previsoes[i]}")
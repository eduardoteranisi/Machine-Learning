###
# Implementacao do algoritmo perceptron usando 10 neuronios
# para o reconhecimento de dígitos
###

import random

class MultiPerceptron:

	def __init__(self, amostras, saidas, taxa_aprendizado=0.1, epocas=100, bias=-1, neuronios=10):

		self.amostras = amostras
		self.saidas = saidas
		self.taxa_aprendizado = taxa_aprendizado 
		self.epocas = epocas 
		self.bias = bias 
		self.total_amostras = len(amostras)
		self.elementos_amostra = len(amostras[0])
		self.neuronios = neuronios
		 # valores randomicos para cada neuronio (incluindo bias)
		self.pesos = [[random.random() for _ in range(len(amostras[0]) + 1)] for _ in range(neuronios)]


	def treinar(self):
		
		# adiciona -1 para cada uma das amostras
		for amostra in self.amostras:
			amostra.insert(0, -1)

		num_epocas = 0

		while True:

			erro = False

			for i in range(self.total_amostras):
				for n in range(self.neuronios):
					u = 0

					for j in range(self.elementos_amostra + 1):
						u += self.pesos[n][j] * self.amostras[i][j]

					# saída da rede pela função de ativação
					y = self.sinal(u)

					if y != self.saidas[i][n]:

						# subtração entre a saída desejada e a saída da rede
						erro_aux = self.saidas[i][n] - y

						# faz o ajuste dos pesos para cada elemento da amostra
						for k in range(self.elementos_amostra + 1):
							self.pesos[n][k] = self.pesos[n][k] + self.taxa_aprendizado * erro_aux * self.amostras[i][k]

						erro = True

			num_epocas += 1

			if num_epocas > self.epocas or not erro:
				break


	# Func teste
	def testar(self, amostra):
		amostra.insert(0, -1)

		resultados = []
		for i in range(self.neuronios):
			u = sum(self.pesos[i][j] * amostra[j] for j in range(self.elementos_amostra + 1))
			resultados.append(self.sinal(u))

		print("Saídas dos neurônios:", resultados)

		if 1 in resultados:
			numero_reconhecido = resultados.index(1)
			print(f"Número reconhecido: {numero_reconhecido}")
		else:
			print("Número não reconhecido.")


	# função de ativação: degrau bipolar (sinal)
	def sinal(self, u):
		return 1 if u >= 0 else -1


amostras = [[1,1,1,1,1,1,-1],
			[-1,1,1,-1,-1,-1,-1],
			[1,1,-1,1,1,-1,1],
			[1,1,1,1,-1,-1,1],
			[-1,1,1,-1,-1,1,1],
			[1,-1,1,1,-1,1,1],
			[1,-1,1,1,1,1,1],
			[1,1,1,-1,-1,-1,-1],
			[1,1,1,1,1,1,1],
			[1,1,1,1,-1,1,1]
]

saidas = [
    [1,-1,-1,-1,-1,-1,-1,-1,-1,-1],  # Digito 0
    [-1,1,-1,-1,-1,-1,-1,-1,-1,-1],  # Digito 1
    [-1,-1,1,-1,-1,-1,-1,-1,-1,-1],  # Digito 2
    [-1,-1,-1,1,-1,-1,-1,-1,-1,-1],  # Digito 3
    [-1,-1,-1,-1,1,-1,-1,-1,-1,-1],  # Digito 4
    [-1,-1,-1,-1,-1,1,-1,-1,-1,-1],  # Digito 5
    [-1,-1,-1,-1,-1,-1,1,-1,-1,-1],  # Digito 6
    [-1,-1,-1,-1,-1,-1,-1,1,-1,-1],  # Digito 7
    [-1,-1,-1,-1,-1,-1,-1,-1,1,-1],  # Digito 8
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,1],  # Digito 9
]


rede = MultiPerceptron(amostras=amostras, saidas=saidas, taxa_aprendizado=0.9, epocas=100)

rede.treinar()

testes = [
	[1,1,1,1,-1,1,1],         #9
	[1,1,1,1,1,1,1],          #8
	[-1,1,1,-1,-1,-1,-1]      #1
]
#Resultado
for teste in testes:
	rede.testar(teste)
import random

# Sem o limiar, a fronteira de decisão sempre passa pela origem.
# Com o limiar, a fronteira de decisão pode se deslocar, permitindo melhor separação das categoriass.
# No código, o limiar foi implementado como um peso extra e um -1 nas amostras, tornando-o ajustável automaticamente no treinamento.

class Perceptron:

	def __init__(self, amostras, saidas, taxa_aprendizado=0.1, epocas=1000, limiar=-1):

		self.amostras = amostras
		self.saidas = saidas
		self.taxa_aprendizado = taxa_aprendizado 
		self.epocas = epocas 
		self.limiar = limiar 
		self.total_amostras = len(amostras)
		self.elementos_amostra = len(amostras[0])
		self.pesos = [] 


	def treinar(self):
		
		# adiciona -1 para cada uma das amostras
		for amostra in self.amostras:
			amostra.insert(0, -1)

		for i in range(self.elementos_amostra):
			self.pesos.append(random.random())

		# insere o limiar no vetor de pesos
		self.pesos.insert(0, self.limiar)

		num_epocas = 0

		while True:

			erro = False

			for i in range(self.total_amostras):

				u = 0

				for j in range(self.elementos_amostra + 1):
					u += self.pesos[j] * self.amostras[i][j]

				# saída da rede pela função de ativação
				y = self.sinal(u)

				if y != self.saidas[i]:

					# subtração entre a saída desejada e a saída da rede
					erro_aux = self.saidas[i] - y

					# faz o ajuste dos pesos para cada elemento da amostra
					for j in range(self.elementos_amostra + 1):
						self.pesos[j] = self.pesos[j] + self.taxa_aprendizado * erro_aux * self.amostras[i][j]

					erro = True

			num_epocas += 1

			if num_epocas > self.epocas or not erro:
				break


	# Func teste
	def testar(self, amostra, categorias1, categorias2):

		amostra.insert(0, -1)

		# utiliza o vetor de pesos que foi ajustado na fase de treinamento
		u = 0
		for i in range(self.elementos_amostra + 1):
			u += self.pesos[i] * amostra[i]

		# saída da rede
		y = self.sinal(u)

		# verifica a qual categorias pertence
		if y == -1:
			print('A amostra pertence a categorias %s' % categorias1)
		else:
			print('A amostra pertence a categorias %s' % categorias2)


	# função de ativação: degrau bipolar (sinal)
	def sinal(self, u):
		return 1 if u >= 0 else -1


print('\nX ou T?\n')

amostras = [[1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1],
			[1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1]]

saidas = [1, -1]

testes = [[1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1],
		  [1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1]]


rede = Perceptron(amostras=amostras, saidas=saidas,	
						taxa_aprendizado=0.7, epocas=1000)

rede.treinar()

#Resultado
for teste in testes:
	rede.testar(teste, 'T', 'X')
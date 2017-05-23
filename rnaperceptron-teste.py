from rnaperceptron import *

# Operator OR
entradas = [[0, 0], [0, 1], [1, 0], [1, 1]]
saidas = [0, 1, 1, 1]

nn = Perceptron(entradas, saidas)
nn.train()

print(nn.predict([0, 1]))
print(nn.predict([0, 1]))
print(nn.predict([1, 0]))
print(nn.predict([1, 1]))
print(nn.predict([0, 0]))
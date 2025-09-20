import numpy 
from sklearn.linear_model import LinearRegression

#pré-processamento 
#cria conjunto de dados de treinamento 
X_train = numpy.array([
    [50, 1],
    [80, 2],
    [100, 3],
    [120, 3]
])

#conjunto de classes que podem ser classificadas
y_train = numpy.array([200, 300, 400, 450])

#criação do modelo
model = LinearRegression()

#treinamento do modelo
model.fit(X_train, y_train)

#conjunto de dados de teste
X_test = numpy.array([
    [60, 1],
    [110, 3]
])

# uso do modelo para prever as classes para o conjunto de teste
y_pred = model.predict(X_test)

# mostra resultados da predição
for i, pred in enumerate(y_pred):
    print(f"Linha {i+1} - Valor previsto: {pred}")

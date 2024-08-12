# Fecha de realizacion: 26/05/2023
# No. Practica: 05 - ANN
# Grupo: 6CV3
# Materia: Machine Learning
# INTEGRANTES:
#   * Pérez Mondragón Eduardo
#   * Rosiles Hidalgo Emanuel

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import statistics
import matplotlib.pyplot as plt

# Valores a probar para medir el learning_rate del clasificador
learning_r = [0.01, 0.001, 0.0001]

# Cargar el dataset de iris
iris = load_iris()

# Dividir en variables de conjunto de datos y etiquetas
X, y = iris.data, iris.target

# Nombres de rasgos a seleccionar
col_names = ['sepal length', 'sepal width', 'petal length', 'petal width']

# Creacion de dataframe para manejo de los datos 
df = pd.DataFrame(X, columns=col_names)

# Se crea la variable para utilizar el K-Fold cross-validation estratificado con una semilla de 8 y valor de k = 10
skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 8)

# Se crean las listas en donde se van a almacenar los valores de las métricas obtenidas en las 10 ejecuciones del K-Fold para posteriormente obtener su promedio
accuracy = list()
precision = list()
recall = list()
f1 = list()


print("Valores promedio de métricas\n")
for tasa in learning_r:
    
    # Mediante este ciclo se realizan las 10 ejecuciones del K-Fold
    for train_index, test_index in skf.split(iris.data, iris.target):
        # Se obtienen los conjuntos de prueba y entrenamiento
        X_train, X_test = iris.data[train_index], iris.data[test_index]
        y_train, y_test = iris.target[train_index], iris.target[test_index]

        # Se crea la variable para aplicar el clasificador variando el parametro de learning_rate
        clf = MLPClassifier(solver='sgd',
                        hidden_layer_sizes=(50,),
                        activation = 'tanh',
                        learning_rate_init = tasa,
                        validation_fraction = 0.2,
                        max_iter=20)

        
        # Se entrena al clasificador con el conjunto de entrenamiento
        clf = clf.fit(X_train,y_train)
        # Se predicen las clases con el conjunto de prueba 
        y_pred = clf.predict(X_test)

        #Se comparan las clases predichas con las reales y se van almacenando cada una de las métricas obtenidas de esa comparación en su respectiva lista
        accuracy.append(accuracy_score(y_test, y_pred)) #Exactitud
        precision.append(precision_score(y_test, y_pred, average="macro")) #Precisión
        recall.append(recall_score(y_test, y_pred, average="macro")) #Sensibilidad
        f1.append(f1_score(y_test, y_pred, average="macro")) #F1-score


    # Al terminar las 10 ejecuciones se obtiene el promedio de cada métrica y se muestran
    print(f'** {tasa} **')
    print(f'Exactitud: {statistics.mean(accuracy)}')
    print(f'Precisión: {statistics.mean(precision)}')
    print(f'Sensibilidad: {statistics.mean(recall)}')
    print(f'F1-score: {statistics.mean(f1)}\n')

""" plt.plot(clf.loss_curve_)
    plt.title("Loss Curve", fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()"""

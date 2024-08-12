# Fecha de realizacion: 02/05/2023
# No. Practica: 04 - Arboles de decision
# Grupo: 6CV3
# Materia: Machine Learning
# INTEGRANTES:
#   * Pérez Mondragón Eduardo
#   * Rosiles Hidalgo Emanuel

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from io import StringIO
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pandas as pd
import pydotplus
import statistics

# Funciones a probar para medir la calidad de una división (Medicion de impureza).
criterios = ['log_loss', 'gini', 'entropy']

# Cargar el dataset de iris
iris = load_iris()

# Dividir en variables de conjunto de datos y etiquetas
X, y = iris.data, iris.target

# Nombres de rasgos a seleccionar
col_names = ['sepal length', 'sepal width', 'petal length', 'petal width']

# Creacion de dataframe para manejo de los datos y posterior creacion de arbol
df = pd.DataFrame(X, columns=col_names)

# Se crea la variable para utilizar el K-Fold cross-validation estratificado con una semilla de 8 y valor de k = 10
skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 8)

# Se crean las listas en donde se van a almacenar los valores de las métricas obtenidas en las 10 ejecuciones del K-Fold para posteriormente obtener su promedio
accuracy = list()
precision = list()
recall = list()
f1 = list()

# Contador para poder realizar la clasificacion variaando el parametro de criterio
cont_img = 0

print("Valores promedio de métricas\n")
for criterio in criterios:
    
    # Mediante este ciclo se realizan las 10 ejecuciones del K-Fold
    for train_index, test_index in skf.split(iris.data, iris.target):
        # Se obtienen los conjuntos de prueba y entrenamiento
        X_train, X_test = iris.data[train_index], iris.data[test_index]
        y_train, y_test = iris.target[train_index], iris.target[test_index]

        # Se crea la variable para aplicar el Arbol de clasificacion variando el parametro de criterion
        clf = DecisionTreeClassifier(criterion=f'{criterio}')
        # Se entrena al clasificador con el conjunto de entrenamiento
        clf = clf.fit(X_train,y_train)
        # Se predicen las clases con el conjunto de prueba 
        y_pred = clf.predict(X_test)

        #Se comparan las clases predichas con las reales y se van almacenando cada una de las métricas obtenidas de esa comparación en su respectiva lista
        accuracy.append(accuracy_score(y_test, y_pred)) #Exactitud
        precision.append(precision_score(y_test, y_pred, average="macro")) #Precisión
        recall.append(recall_score(y_test, y_pred, average="macro")) #Sensibilidad
        f1.append(f1_score(y_test, y_pred, average="macro")) #F1-score

    # ------------- Imprimir el arbol de decision

    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True,feature_names = col_names,class_names=['0','1', '2'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png(f'{criterio}_{cont_img}.png')
    Image(graph.create_png())

    # Al terminar las 10 ejecuciones se obtiene el promedio de cada métrica y se muestran
    print(f'** {criterio} **')
    print(f'Exactitud: {statistics.mean(accuracy)}')
    print(f'Precisión: {statistics.mean(precision)}')
    print(f'Sensibilidad: {statistics.mean(recall)}')
    print(f'F1-score: {statistics.mean(f1)}\n')

    # ------------- Optimizacion del arbol de decision
    # Establece el maximo nivel de profundidad del arbol de clasificacion a 3
    clf.max_depth = 3  
    # Entrenar el arbol de decision usando los datos de entrenamiento
    clf = clf.fit(X_train, y_train)  
    # Se usa el clasificador entrenado para predecir los datos de prueba
    y_pred = clf.predict(X_test)  

    # Cree un búfer de cadena vacío para almacenar la representación en lenguaje DOT del árbol de decisión
    dot_data = StringIO() 
    export_graphviz(clf, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True, feature_names=col_names, class_names=['0', '1', '2'])  
    # Crear un objeto gráfico a partir de la cadena de lenguaje DOT
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    # Escribir el gráfico en un archivo PNG
    graph.write_png(f'{criterio}_{cont_img}_opt.png')  
    # Mostrar el archivo PNG como una imagen
    Image(graph.create_png())  

    cont_img+=1
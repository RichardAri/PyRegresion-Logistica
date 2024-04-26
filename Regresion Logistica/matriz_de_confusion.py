from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np #Gráficos
from sklearn.linear_model import LogisticRegression 
import matplotlib.pyplot as plt
import pandas as pd

#!configuracion csv
drive.mount('/content/drive')
df = pd.read_csv('/content/drive/My Drive/regresion_logistica.csv')

# Crear una instancia del modelo de regresión logística
model = LogisticRegression()

#! ------------------------------------------ Calcular la matriz de confusión con SKLEARN

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)

# Mostrar la matriz de confusión utilizando un mapa de calor
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.show()

#! --------------------------------------Calcular la matriz de confusion sin SKLEARN

#! Definimos las clases
classes = df['Class'].unique()

# Inicializar la matriz de confusión como una matriz de ceros
conf_matrix_manual = np.zeros((len(classes), len(classes)), dtype=int)

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)


# Iterar sobre las predicciones y las etiquetas reales para llenar la matriz de confusión
for i in range(len(y_test)):
    true_class_index = np.where(classes == y_test.iloc[i])[0][0]
    pred_class_index = np.where(classes == y_pred[i])[0][0]
    conf_matrix_manual[true_class_index][pred_class_index] += 1

# Mostrar la matriz de confusión
print("Matriz de Confusión:")
print(conf_matrix_manual)


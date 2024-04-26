from google.colab import drive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind

# Configurar matplotlib
plt.rcParams['image.cmap'] = "bwr"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

# Configurar advertencias
import warnings
warnings.filterwarnings('ignore')

# Montar Google Drive y cargar el archivo CSV
drive.mount('/content/drive')
df = pd.read_csv('/content/drive/My Drive/regresion_logistica.csv')

# Crear una instancia del modelo de regresión logística
model = LogisticRegression()

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X = df.drop(columns=['Class']) 
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular la precisión del modelo
precision = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", precision)

#! Precisión del modelo: 0.9562043795620438
#* Esto significa que el modelo tiene una precision del 95%

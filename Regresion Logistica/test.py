# Tratamiento de datos
from google.colab import drive #!cargar csv
import pandas as pd
import numpy as np #Gráficos
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns #Preprocesado y modelado
from sklearn.linear_model import LogisticRegression #! se importa la regresion logistica
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score #! funcion para calcular la precision del modelo
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind
#! import 
#! Crear una instancia del modelo de regresión logística
model = LogisticRegression()



#!configuracion csv
drive.mount('/content/drive')
df = pd.read_csv('/content/drive/My Drive/regresion_logistica.csv')

# Configuración matplotlib
plt.rcParams['image.cmap'] = "bwr" #plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

# Configuración warnings
import warnings
warnings.filterwarnings('ignore')

#* ¿Cuál es la distribución de las clases en la variable objetivo (maligna o benigna)?

#! Gráfico de barras de distribución de las clases en la variable objetivo (maligna o benigna)
plt.figure(figsize=(6, 4))
sns.countplot(x="Class", data=df, palette="pastel")
plt.title("Distribución de clases") #! Variable dependiente que indica si la muestra es maligna o benigna
plt.ylabel("Frecuencia") #! la frecuencia se refiere al número de veces que aparece en la data 
plt.xlabel("Clase") #! nombre de variable
plt.ylabel("Frecuencia")
plt.show()

#! Muestra el conteo de cada class(Clase)
print("Distribución de clases:")
print(conteo_clases)


#* ¿Cuál es la precisión de su modelo?
#! Se Divide los datos en conjunto de entrenamiento y conjunto de prueba
X = df.drop(columns=['Class']) # 
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
precision = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", precision)

#* ¿Qué tipo de problema es este: clasificación binaria o regresión?

#* Investigue e implemente la matriz de confusión para el modelo y explique sus resultados

#* Realice una interpretación clínica de los resultados del modelo







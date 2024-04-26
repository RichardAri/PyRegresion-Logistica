# Tratamiento de datos
from google.colab import drive #!cargar csv
import pandas as pd
import numpy as np #Gráficos
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns #Preprocesado y modelado
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind

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
# Datos

conteo_clases = df['Class'].value_counts().sort_index()
print(conteo_clases)

# !Gráfico de distribución por clase
fig, ax = plt.subplots(figsize=(10, 6)) 
sns.violinplot(
    x='Class',  # ! Columna que indica la clase
    y='Clump Thickness',  # ! Variable independiente a analizar
    data=df,  # ! DataFrame (contiene los datos))
    color="white",
    ax=ax)

ax.set_title('Distribución de Clump Thickness|Grosor del monton por clase')
plt.show()  #* Muestra el gráfico

# Test t entre clases
data_class_benigna = df[df['Class'] == 'benign']['Clump Thickness']
data_class_maligna = df[df['Class'] == 'malign']['Clump Thickness']

res_ttest = ttest_ind(x1=data_class_benigna, x2=data_class_maligna, alternative='two-sided')
print(f"t={res_ttest[0]}, pvalue={res_ttest[1]}")





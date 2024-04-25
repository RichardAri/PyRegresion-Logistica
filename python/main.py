# Tratamiento de datos
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
# Configuración matplotlib 
plt.rcParams['image.cmap'] = "bwr" #plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')
# Configuración warnings 
import warnings
warnings.filterwarnings('ignore') 
# Datos
matricula = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,0, 0, 1,
 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
1, 0, 1,
 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
1, 0, 0,
 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
0, 0, 0,
 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0,
1, 0, 0,
 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
0, 0, 1,
 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
0, 1, 1,
 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
0, 0, 0,
 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0,
1, 0, 0,
 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0,
0, 1, 0,
 0, 0, 0, 0, 1, 0, 0, 0, 1, 1]) 

matematicas = np.array([
 41, 53, 54, 47, 57, 51, 42, 45, 54, 52, 51, 51, 71,
57, 50, 43,
 51, 60, 62, 57, 35, 75, 45, 57, 45, 46, 66, 57, 49,
49, 57, 64,
 63, 57, 50, 58, 75, 68, 44, 40, 41, 62, 57, 43, 48,
63, 39, 70,
 63, 59, 61, 38, 61, 49, 73, 44, 42, 39, 55, 52, 45,
61, 39, 41,
 50, 40, 60, 47, 59, 49, 46, 58, 71, 58, 46, 43, 54,
56, 46, 54,
 57, 54, 71, 48, 40, 64, 51, 39, 40, 61, 66, 49, 65,
52, 46, 61,
 72, 71, 40, 69, 64, 56, 49, 54, 53, 66, 67, 40, 46,
69, 40, 41,
 57, 58, 57, 37, 55, 62, 64, 40, 50, 46, 53, 52, 45,
56, 45, 54,
 56, 41, 54, 72, 56, 47, 49, 60, 54, 55, 33, 49, 43,
50, 52, 48,
 58, 43, 41, 43, 46, 44, 43, 61, 40, 49, 56, 61, 50,
51, 42, 67,
 53, 50, 51, 72, 48, 40, 53, 39, 63, 51, 45, 39, 42,
62, 44, 65,
 63, 54, 45, 60, 49, 48, 57, 55, 66, 64, 55, 42, 56,
53, 41, 42,
 53, 42, 60, 52, 38, 57, 58, 65])
datos = pd.DataFrame({'matricula': matricula, 'matematicas': matematicas})
datos.head(3)
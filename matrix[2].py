import numpy as np
import time
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# Tamaño matrices
tamaño_matriz = 3000

# Generamos 2 matrces con valores aleatorios de 3000 x 3000
matriz_A = np.random.rand(tamaño_matriz, tamaño_matriz)
matriz_B = np.random.rand(tamaño_matriz, tamaño_matriz)

# Multiplicación de matrices sin joblib
tiempo_inicio_sin_joblib = time.time() # Iniciamos contador del proceso sin paralelismo
resultado_sin_joblib = np.dot(matriz_A, matriz_B) # Realizamos la multiplicacion de las 2 matrices
tiempo_fin_sin_joblib = time.time() # Detenemos el contador sin paralelimso
tiempo_transcurrido_sin_joblib = tiempo_fin_sin_joblib - tiempo_inicio_sin_joblib # Calculamos tiempo transcurrido

# Multiplicación de matrices con joblib
def producto_punto_paralelo(fila):
    return np.dot(fila, matriz_B)

tiempo_inicio_con_joblib = time.time()  # Iniciamos contador de proceso con paralelismo
# Iniciamos con 
resultado_con_joblib = np.array(Parallel(n_jobs=-1)(delayed(producto_punto_paralelo)(fila) for fila in matriz_A))
tiempo_fin_con_joblib = time.time() # Detenemos el contador del proceso con paralelismo
tiempo_transcurrido_con_joblib = tiempo_fin_con_joblib - tiempo_inicio_con_joblib # Calculamos tiempo transcurrido


# Imprimos tiempo tiempo transcurrido en cada uno de los procesos
print("Tiempo sin joblib:", tiempo_transcurrido_sin_joblib) 
print("Tiempo con joblib:", tiempo_transcurrido_con_joblib)

# Graficamos los tiempos transcurridos
nombres = ['Sin Joblib', 'Con Joblib']
tiempos = [tiempo_transcurrido_sin_joblib, tiempo_transcurrido_con_joblib]

plt.bar(nombres, tiempos, color=['blue', 'green'])
plt.ylabel('Tiempo (segundos)')
plt.title('Comparación de Tiempo de Multiplicación de Matrices')
plt.show()

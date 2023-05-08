# Máster en Ciberseguridad e Inteligencia de Datos
# Técnicas Avanzadas de Análisis de Datos
# David Valverde Gómez
# Práctica 2. Sistemas Recomendadores
# Fichero con las funciones requeridas para la práctica

# Librerías para stemming
import nltk
nltk.download('all-corpora') # Descargar all-corpora
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# Librerías para el tratamiento de los datos y el cálculo del TF-IDF
import numpy as np
import pandas as pd
from math import log

# Librería para calcular la similitud del coseno
from sklearn.metrics.pairwise import cosine_similarity

# Función para leer el fichero indicado y devolver un array con cada línea
def leer_fichero():
  ruta_fichero = input("Introduzca la ruta del fichero: ")
  with open(ruta_fichero, mode='r', encoding='utf-8') as fichero:
    lineas = fichero.readlines()

  return lineas

# Función para el tratamiento del texto:
# - Se tokeniza el texto en un vector de palabras
# - Se eliminan los signos de puntuación y las stop words
# - Se pasan a minúscula todas las palabras
# - Se utiliza el SnowBall Stemmer para lematizar las palabras restantes (reducirlas a su raíz)
def stemming_process(text):
  tokens_lower = [str(word) for word in wordpunct_tokenize(text)]
  stopw = stopwords.words('spanish')

  tokens_lower = [w.lower() for w in tokens_lower]
  punc_lit = [u'.', u'[', ']', u',', u';', u'', u')', u'),', u' ', u'(']
  stopw.extend(punc_lit)
  words = [token for token in tokens_lower if token not in stopw]

  snowball_stemmer = SnowballStemmer('spanish')
  stemmers = [snowball_stemmer.stem(word) for word in words]

  final = [stem for stem in stemmers if stem.isalpha() and len(stem) > 1]
  return final

# Función que calcula la matriz TF-IDF del conjunto de documentos recibido.
# Se espera recibir un array de documentos, donde cada documento se representa por su 
# listado de palabras ya preprocesadas y lematizadas.
# También se construye, para cada documento, una matriz con los campos:
# - Índice del término
# - Término
# - TF
# - IDF
# - TF-IDF
def calculate_tfidf(docs):
    # Calculamos la frecuencia de términos en cada documento
    tfs = []
    for doc in docs:
        tf = {}
        for word in doc:
            tf[word] = tf.get(word, 0) + 1
        for word in tf:
           tf[word] = tf[word] / len(doc)
        tfs.append(tf)

    # Calculamos la frecuencia inversa de documento (IDF) para cada término
    N = len(docs)
    idf = {}
    idfs = {}
    for doc in docs:
        idf = {}
        for word in doc:
            idf[word] = idf.get(word, 0) + 1
            
        for word in idf:
            idfs[word] = idfs.get(word, 0) + 1
              
    for word in idfs:
        idfs[word] = log(N / idfs[word])

    # Calculamos los valores tf-idf a la vez que se construye la matriz con todos
    # los valores de cada término para cada documento
    tfidfs = []
    listado_matrices_documentos = []
    for tf in tfs:
        matriz_documento = []
        row = {}
        i = 0
        for word in tf:
            i+=1
            row[word] = tf[word] * idfs[word]
            matriz_documento.append([i, word, tf[word], idfs[word], row[word]])
        tfidfs.append(row)
        listado_matrices_documentos.append(matriz_documento)

    # Se construyen los dataframes
    words = sorted(list(idfs.keys()))
    data = []
    fila = []

    for i, tfidf in enumerate(tfidfs):
        fila = []
        for word in words:
          if (word in tfidf.keys()):
            fila.append(tfidf[word])
          else:
            fila.append(0)
        data.append(fila)
    columns =words

    df_tfidf = pd.DataFrame(data, columns=columns)

    dfs_termino_valores = []
    for matriz in listado_matrices_documentos:
        prueba = pd.DataFrame(matriz, columns=['Índice del término', 'Término', 'TF', 'IDF', 'TF-IDF'])
        dfs_termino_valores.append(prueba)
  
    return df_tfidf, dfs_termino_valores


# Función que calcula la similitud del coseno entre los documentos de una matriz
def similitud_coseno(df):
    # Calcular la matriz de similitud del coseno
    sim_matrix = cosine_similarity(df)
    # Crear un dataframe con la matriz de similitud del coseno
    sim_df = pd.DataFrame(sim_matrix)
    return sim_df

# Función que calcula la similitud del coseno entre los documentos de 2 matrices
def similitud_coseno_entre_matrices(df1, df2):
    # Calcular la matriz de similitud del coseno
    sim_matrix = cosine_similarity(df1, df2)
    # Crear un dataframe con la matriz de similitud del coseno
    sim_df = pd.DataFrame(sim_matrix)
    return sim_df

# Para calcular la similitud del coseno entre dos matrices, estas
# deben tener las mismas dimensiones y las palabras deben estar
# ordenadas del mimso modo
def igualar_dimensiones_matrices(df1, df2):
  for column1 in df1.columns:
    if column1 not in df2.columns:
      df2[column1] = 0

  for column2 in df2.columns:
    if column2 not in df1.columns:
      df1[column2] = 0


  df1 = df1.sort_index(axis=1)
  df2 = df2.sort_index(axis=1)

  return df1, df2



# Esta función recibe la matriz de similitud del coseno entre el conjunto de documentos de entrenamiento y el 
# conjunto de documentos a recomendar, además de un número de recomendaciones a realizar. Devuelve un array 
# de tuplas formadas por la fila (documento de entrenamiento), la columna (documento a recomendar) y el
# valor de la similitud del coseno entre ambos.
def seleccionar_recomendaciones(matriz, nrecomendaciones):
  matriz = matriz.values
  maximos = []

  # Encontrar los índices de los elementos máximos de cada columna
  for col in range(matriz.shape[1]):
      fila_max = np.argmax(matriz[:, col])
      maximos.append((fila_max, col))
  
  # Ordenar los elementos máximos de todas las columnas y seleccionar los nrecomendaciones mayores
  maximos_ordenados = sorted(maximos, key=lambda x: matriz[x[0], x[1]], reverse=True)[:nrecomendaciones]
  # Crear una lista de tuplas con el índice de fila, índice de columna y valor de la similitud del coseno
  elementos_seleccionados = [(ind[0], ind[1], matriz[ind[0], ind[1]]) for ind in maximos_ordenados]
  return elementos_seleccionados


# Máster en Ciberseguridad e Inteligencia de Datos
# Técnicas Avanzadas de Análisis de Datos
# David Valverde Gómez
# Práctica 2. Sistemas Recomendadores
# Fichero con el programa principal para la práctica

# Este programa recibirá en primer lugar una ruta a un fichero con 
# documentos de entrenamiento, entendiendo estos como los documentos
# que ya se ha demostrado que 'le gustan' al usuario.
# Para este primer conjunto, se construirá una matriz con los siguientes campos:
# - Índice del término
# - Término
# - TF
# - IDF
# - TF-IDF
# Además, se mostrará la similitud del coseno entre cada par de documentos de este primer conjunto.

# Posteriormente, se recibirá un segundo fichero con los documentos entre los cuáles se quiere 
# recomendar al usuario y, además de realizar el mismo proceso que con el primer conjunto,
# se mostrarán los N documentos que se recomienden.

import os 
from funciones import leer_fichero, stemming_process, calculate_tfidf, similitud_coseno, igualar_dimensiones_matrices, similitud_coseno_entre_matrices, seleccionar_recomendaciones

def main():
  # Primero se lee el primer documento
  os.system('cls')
  print('En primer lugar, indique el fichero con los documentos que le gustan al usuario')
  conjunto_entrenamiento = leer_fichero()

  # A continuación, se realiza el proceso de stemming sobre cada documento
  conjunto_entrenamiento_stemming = []
  for doc in conjunto_entrenamiento:
    conjunto_entrenamiento_stemming.append(stemming_process(doc))

  # El siguiente paso es calcular la matriz con los valores TF-IDF de cada término
  conjunto_entrenamiento_tfidf, conjunto_entrenamiento_matriz_terminos = calculate_tfidf(conjunto_entrenamiento_stemming)

  # Se imprime para cada documento la matriz con los datos solicitados
  i = 1
  print('')
  for matriz in conjunto_entrenamiento_matriz_terminos:
    print('')
    print(f'---- Documento {i} ----')
    print(matriz)
    i+=1

  # Se calcula la similitud del coseno entre los documentos del primer conjunto
  conjunto_entrenamiento_similitud = similitud_coseno(conjunto_entrenamiento_tfidf)
  print('')
  print('---- Similitudes entre documentos ----')
  for index, fila in conjunto_entrenamiento_similitud.iterrows():
    for columna in conjunto_entrenamiento_similitud.columns:
      if columna > index:
        print(f'Documentos {index+1} y {columna+1}: {round(conjunto_entrenamiento_similitud.iloc[index,columna], 2)}')
        
  # Se repite el proceso para el conjunto de documentos a recomendar
  print('')
  print('---------------------------------')
  print('Ahora, indique el fichero con los documentos entre los que se quiere recomendar')
  conjunto_recomendacion = leer_fichero()

  # A continuación, se realiza el proceso de stemming sobre cada documento
  conjunto_recomendacion_stemming = []
  for doc in conjunto_recomendacion:
    conjunto_recomendacion_stemming.append(stemming_process(doc))

  # El siguiente paso es calcular la matriz con los valores TF-IDF de cada término
  conjunto_recomendacion_tfidf, conjunto_recomendacion_matriz_terminos = calculate_tfidf(conjunto_recomendacion_stemming)

  # Se imprime para cada documento la matriz con los datos solicitados
  i=1
  print('')
  for matriz in conjunto_recomendacion_matriz_terminos:
    print('')
    print(f'---- Documento {i} ----')
    print(matriz)
    i+=1

  # Se calcula la similitud del coseno entre los documentos del primer conjunto
  conjunto_recomendacion_similitud = similitud_coseno(conjunto_recomendacion_tfidf)
  print('')
  print('---- Similitudes entre documentos ----')
  for index, fila in conjunto_recomendacion_similitud.iterrows():
    for columna in conjunto_recomendacion_similitud.columns:
      if columna > index:
        print(f'Documentos {index+1} y {columna+1}: {round(conjunto_recomendacion_similitud.iloc[index,columna], 2)}')

  # El siguiente paso es calcular la matriz de similitud del coseno entre los documentos de entrenamiento (filas) y 
  # los documentos a recomendar (columnas)
  conjunto_entrenamiento_tfidf, conjunto_recomendacion_tfidf = igualar_dimensiones_matrices(conjunto_entrenamiento_tfidf, conjunto_recomendacion_tfidf)
  matriz_similitud = similitud_coseno_entre_matrices(conjunto_entrenamiento_tfidf, conjunto_recomendacion_tfidf)
  print('')
  print('---- Similitudes entre documentos ----')
  for index, fila in matriz_similitud.iterrows():
    for columna in matriz_similitud.columns:
        print(f'Documento de entrenamiento {index+1} y Documento para recomendación {columna+1}: {round(matriz_similitud.iloc[index,columna], 2)}')


  # Finalmente, se pide al usuario que indique el número de documentos a recomendar y se le muestran en pantalla
  print('')
  nrecomendaciones = int(input('Indique el número de documentos a recomendar:'))
  recomendaciones = seleccionar_recomendaciones(matriz_similitud, nrecomendaciones)
  print('')
  print('---- Documentos Recomendados ----')
  for recomendacion in recomendaciones:
    print(conjunto_recomendacion[recomendacion[1]])

main()
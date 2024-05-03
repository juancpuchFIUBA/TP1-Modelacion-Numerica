
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from gsl import *

ROJO = '\033[91m'
RESET = '\033[0m'
TEMPERATURA_ARRIBA = 63
TEMPERATURA_ABAJO = 44
TEMPERATURA_IZQUIERDA = 30
TEMPERATURA_DERECHA = 72
LARGO_PLANCHA = 5
ANCHO_PLANCHA = 4
DISCRETIZACION = 0.4
TOLERANCIA = 0.000000001
ELEMENTO_INICIAL = 0.1

HORARIOS = [0,5,9,15,20,23]
TEMPERATURAS = [26,21,29,37,31,24]


def imprimir_plancha (vector, ancho, largo):
    for i in range(ancho):
        for j in range (largo):

            print("{:.0f}".format(vector[ j + i * largo]), end=" ")
        print("\n")
def imprimir_matriz (matriz):
    f=0
    for fila in matriz:
        f+=1
        c=0
        for elemento in fila:
            c+=1
            if (c == f): print (ROJO+ str(elemento)+ RESET, end=" " )
            else: print (elemento, end=" ")
        print("\n")

def imprimir_vector (vector):
    for elemento in vector:
        print("{:.2f}".format(elemento), end=" ")
    print("\n")

def multiplicar_vector_constante(vector, constante, dimension):
    for i in range(dimension):
        vector[i] = vector[i] * constante
    return vector
def restar_vectores (vector_minuendo, vector_sustraendo, dimension):
    vector_resultado = []
    for i in range(dimension):
        resultado = vector_minuendo[i] - vector_sustraendo[i]
        vector_resultado.append(resultado)
    return vector_resultado
def calcular_dimencion(longitud, discretizacion):
    return (int ((longitud / discretizacion) + 1))

def crear_placa_inicial (largo, ancho, temperatura_izquierda):
    placa = []
    for i in range(ancho):
        fila_placa = []
        for j in range(largo):
            if (i == 0):
                if (j == 0 or j == largo - 1):
                    fila_placa.append(1)
                else:
                    fila_placa.append(TEMPERATURA_ARRIBA)
            elif (i == (ancho - 1 )):
                if (j == 0 or j == largo - 1):
                    fila_placa.append(1)
                else:
                    fila_placa.append(TEMPERATURA_ABAJO)
            elif (j == 0):
                fila_placa.append(temperatura_izquierda)
            elif (j == (largo - 1)):
                fila_placa.append(TEMPERATURA_DERECHA)
            else:
                fila_placa.append(0)
        placa.append(fila_placa)
    return placa

def crear_matriz_de_ceros (largo, ancho):
    cantidad_nodos = largo * ancho
    matriz = []
    for i in range(cantidad_nodos):
        fila_matriz = []
        for j in range(cantidad_nodos):
            fila_matriz.append(0)
        matriz.append(fila_matriz)
    return matriz

def no_es_borde_placha(posicion, dimension, largo_plancha, ancho_plancha):
    if (posicion < largo_plancha):
        return False

    busqueda = largo_plancha - 1
    while (busqueda < dimension - largo_plancha):
        if (posicion == busqueda or posicion == busqueda + 1):
            return False
        busqueda += largo_plancha

    if (posicion < dimension and posicion > dimension - largo_plancha):
        return False
    return True

def crear_matriz_A (largo, ancho):
    matriz = crear_matriz_de_ceros(largo, ancho)
    dimension = largo * ancho
    for i in range(dimension):
        if (no_es_borde_placha(i, dimension, largo, ancho)):
            matriz[i][i] = 4
            matriz[i][i - largo] = -1
            matriz[i][i + largo] = -1
            matriz[i][i + 1] = -1
            matriz[i][i - 1] = -1
        else :
            matriz[i][i] = 1
    return matriz

def crear_vector_b (largo, ancho, temperatura_izquierda):
    vector = []
    for i in range(ancho):
        for j in range(largo):
            if (i == 0):
                if (j == 0 or j == largo - 1):
                    vector.append(1)
                else:
                    vector.append(TEMPERATURA_ARRIBA)
            elif (i == (ancho - 1)):
                if (j == 0 or j == largo - 1):
                    vector.append(1)
                else:
                    vector.append(TEMPERATURA_ABAJO)
            elif (j == 0):
                vector.append(temperatura_izquierda)
            elif (j == (largo - 1)):
                vector.append(TEMPERATURA_DERECHA)
            else:
                vector.append(0)
    return vector


'''................................................................................'''
'''
def aplicar_eliminacion_gauseana (matriz, dimension, vector):
    A = np.array(matriz)
    b = np.array(vector)
    Ab = np.hstack((A, b.reshape(-1, 1)))
    for i in range(dimension):
        for j in range(i + 1, dimension):
            c = Ab[j][i] / Ab[i][i]
            for k in range(i, dimension+1):
                Ab[j][k] -= c * Ab[i][k]
    return Ab

def sustitucion_inversa(matrix, vector, dimension):
    x = np.zeros(dimension)
    for i in range(dimension - 1, -1, -1):
        b = vector[i]
        variable = np.dot(matrix[i, i+1:], x[i+1:])
        elemento = matrix[i, i]
        x[i] = (b - variable) / elemento
    return x

def resolucion_por_gauss (largo, ancho, temperatura_izquierda):

    dimension = largo * ancho
    A = crear_matriz_A(largo, ancho)
    b = crear_vector_b(largo, ancho, temperatura_izquierda)
    Ab = aplicar_eliminacion_gauseana(A, dimension, b)
    A_con_eliminacion = Ab[:, :dimension]
    b_con_eliminacion = Ab[:, dimension]
    resultado = sustitucion_inversa(A_con_eliminacion, b_con_eliminacion, dimension)
    return resultado
'''
def resolucion_por_gauss (largo, ancho, temperatura_izquierda):

    dimension = largo * ancho
    A = np.array(crear_matriz_A(largo, ancho),dtype=np.float64)
    b = np.array(crear_vector_b(largo, ancho, temperatura_izquierda), dtype=np.float64)

    for i in range(dimension):
        max_index = np.argmax(abs(A[i:, i])) + i
        A[[i, max_index]] = A[[max_index, i]]
        b[[i, max_index]] = b[[max_index, i]]
        for j in range(i + 1, dimension):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    x = np.zeros(dimension)
    for i in range(dimension - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x

'''.....................................................................................'''
def crear_solucion_inicial (dimension, elemento):
    vector = []
    for i in range (dimension):
        vector.append(elemento)
    return vector

def aplicar_metodo_jacobi (matriz, vector, solucion_inicial, tolerancia, max_iteraciones, dimension):

       x_acutal = np.copy(solucion_inicial)
       x_siguiente = np.zeros_like(x_acutal)
       A = np.array(matriz)
       b = np.array(vector)
       for j in range(max_iteraciones):
           for i in range(dimension):
               suma_ax = np.dot(A[i, :i], x_acutal[:i]) + np.dot(A[i, i + 1:], x_acutal[i + 1:])
               x_siguiente[i] = (b[i] - suma_ax) / A[i, i]


           if (np.sqrt(np.sum((x_siguiente - x_acutal) ** 2) / np.sqrt(np.sum(x_siguiente) ** 2)) < tolerancia):
               return x_siguiente, j
           x_acutal = np.copy(x_siguiente)
       return 0, 0
def resolucion_por_jacobi (largo, ancho, temperatura_izquierda):
    dimension = largo * ancho
    A = crear_matriz_A(largo, ancho)
    b = crear_vector_b(largo, ancho, temperatura_izquierda)
    x0 = crear_solucion_inicial (dimension, ELEMENTO_INICIAL)
    tolerancia = TOLERANCIA
    tolerancia_iteraciones = 100000000
    x, iteraciones = aplicar_metodo_jacobi(A, b, x0, tolerancia, tolerancia_iteraciones, dimension)
    return x, iteraciones

'''.....................................................................................'''

def aplicar_metodo_gauss_seidez(matriz, vector, solucion_inicial, tolerancia, max_iteraciones, dimension):
    x_actual = np.copy(solucion_inicial)
    x_siguiente = np.zeros_like(x_actual)
    A = np.array(matriz)
    b = np.array(vector)
    for j in range (max_iteraciones):
        for i in range (dimension):
            x_siguiente[i] = (b[i] - np.dot(A[i, :i], x_siguiente[:i]) - np.dot(A[i, i + 1:], x_actual[i + 1:])) / A[i, i]

        if (np.sqrt(np.sum((x_siguiente - x_actual) ** 2) / np.sqrt(np.sum(x_siguiente) ** 2)) < tolerancia):
            return x_siguiente, j

        x_actual = np.copy(x_siguiente)
    return 0, 0

def resolucion_por_gauss_seidez(largo, ancho, temperatura_izquierda):
    dimension = largo * ancho
    A = crear_matriz_A(largo, ancho)
    b = crear_vector_b(largo, ancho, temperatura_izquierda)
    x0 = crear_solucion_inicial(dimension, ELEMENTO_INICIAL)
    tolerancia = TOLERANCIA
    tolerancia_iteraciones = 10000000000
    x , iteraciones = aplicar_metodo_gauss_seidez(A, b, x0, tolerancia, tolerancia_iteraciones, dimension)
    return x, iteraciones

'''.....................................................................................'''
def dividir_vector(vector, largo):
    return [vector[i:i+largo] for i in range(0, len(vector), largo)]

def graficar_plancha_temperaturas(vector_resultado, largo, temperatura):
    matriz_placa = dividir_vector(vector_resultado, largo)
    matriz = np.array(matriz_placa)
    if (DISCRETIZACION >= 0.5):
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                plt.text(j, i, f'{matriz[i, j]:.0f}', ha='center', va='center', color='white')

    plt.imshow(matriz, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.axis('off')
    plt.text(matriz.shape[1] / 2, -0.7, f'Temperatura ambiente = {temperatura}°C', ha='center')
    plt.text(matriz.shape[1] / 2, -1.5, f'Discretización = {DISCRETIZACION} cm', ha='center')
    plt.show()
    return 0

def crear_graficos_de_plancha (largo, ancho):

    for i in range(len(HORARIOS)):
        x, iteracion = resolucion_por_jacobi(largo, ancho, TEMPERATURAS[i])
        graficar_plancha_temperaturas(x,largo, TEMPERATURAS[i])
    x_temperatura_promedio, iteracion = resolucion_por_jacobi(largo, ancho, TEMPERATURA_IZQUIERDA)
    graficar_plancha_temperaturas(x_temperatura_promedio, largo, TEMPERATURA_IZQUIERDA)
    return 0
def crear_graficos_de_plancha_gauss(largo, ancho):
    for i in range(len(HORARIOS)):
        x = resolucion_por_gauss(largo, ancho, TEMPERATURAS[i])
        graficar_plancha_temperaturas(x, largo, TEMPERATURAS[i])

    x_temperatura_promedio = resolucion_por_gauss(largo, ancho, TEMPERATURA_IZQUIERDA)
    graficar_plancha_temperaturas(x_temperatura_promedio, largo, TEMPERATURA_IZQUIERDA)
    return 0
'''.....................................................................................'''
def crear_funcion_respecto_horario(largo, ancho):
    promedios_temperatura = []
    for i in range(len(HORARIOS)):
        x, iteracion = resolucion_por_gauss_seidez(largo, ancho, TEMPERATURAS[i])
        suma_elementos = sum(x)
        longitud_vector = len(x)
        promedio = suma_elementos / longitud_vector
        promedios_temperatura.append(promedio)

    x_temperatura_promedio, iteracion = resolucion_por_jacobi(largo, ancho, TEMPERATURA_IZQUIERDA)
    promedio_temperatura_x_promedio = (sum(x_temperatura_promedio) / len(x_temperatura_promedio))

    plt.plot(HORARIOS, [promedio_temperatura_x_promedio] * len(HORARIOS), linestyle='--', color='red', label='To')
    plt.plot(HORARIOS, promedios_temperatura, marker='o')
    plt.title('Promedio de temperatura según el horario')
    plt.xlabel('Horario')
    plt.ylabel('Temperatura (celcius)')
    plt.grid(True)
    plt.show()

    return 0

'''.....................................................................................'''
def mostrar_iteraciones_jacobi(largo, ancho):
    discretizacion_1 = 1
    discretizacion_2 = 0.5
    discretizacion_3 = 0.4
    largo_placa_1 = calcular_dimencion(LARGO_PLANCHA, discretizacion_1)
    ancho_placa_1 = calcular_dimencion(ANCHO_PLANCHA, discretizacion_1)
    largo_placa_2 = calcular_dimencion(LARGO_PLANCHA, discretizacion_2)
    ancho_placa_2 = calcular_dimencion(ANCHO_PLANCHA, discretizacion_2)
    largo_placa_3 = calcular_dimencion(LARGO_PLANCHA, discretizacion_3)
    ancho_placa_3 = calcular_dimencion(ANCHO_PLANCHA, discretizacion_3)
    x_1, iteraciones_1 = resolucion_por_jacobi(largo_placa_1, ancho_placa_1, TEMPERATURA_IZQUIERDA)
    x_2, iteraciones_2 = resolucion_por_jacobi(largo_placa_2, ancho_placa_2, TEMPERATURA_IZQUIERDA)
    x_3, iteraciones_3 = resolucion_por_jacobi(largo_placa_3, ancho_placa_3, TEMPERATURA_IZQUIERDA)
    print(f"GS = discretizacion = {discretizacion_1}, iteraciones ={iteraciones_1}")
    print(f"GS = discretizacion = {discretizacion_2}, iteraciones ={iteraciones_2}")
    print(f"GS = discretizacion = {discretizacion_3}, iteraciones ={iteraciones_3}")
    return 0

def mostrar_iteraciones_gauss_seidez(largo, ancho):
    discretizacion_1 = 1
    discretizacion_2 = 0.5
    discretizacion_3 = 0.4
    largo_placa_1 = calcular_dimencion(LARGO_PLANCHA, discretizacion_1)
    ancho_placa_1 = calcular_dimencion(ANCHO_PLANCHA, discretizacion_1)
    largo_placa_2 = calcular_dimencion(LARGO_PLANCHA, discretizacion_2)
    ancho_placa_2 = calcular_dimencion(ANCHO_PLANCHA, discretizacion_2)
    largo_placa_3 = calcular_dimencion(LARGO_PLANCHA, discretizacion_3)
    ancho_placa_3 = calcular_dimencion(ANCHO_PLANCHA, discretizacion_3)
    x_1, iteraciones_1 = resolucion_por_gauss_seidez(largo_placa_1, ancho_placa_1, TEMPERATURA_IZQUIERDA)
    x_2, iteraciones_2 = resolucion_por_gauss_seidez(largo_placa_2, ancho_placa_2,TEMPERATURA_IZQUIERDA)
    x_3, iteraciones_3 = resolucion_por_gauss_seidez(largo_placa_3, ancho_placa_3,TEMPERATURA_IZQUIERDA)
    print(f"GS = discretizacion = {discretizacion_1}, iteraciones ={iteraciones_1}")
    print(f"GS = discretizacion = {discretizacion_2}, iteraciones ={iteraciones_2}")
    print(f"GS = discretizacion = {discretizacion_3}, iteraciones ={iteraciones_3}")
    return 0


'''.....................................................................................'''
def evaluar_tiempo_ejecucion_gauss(largo,ancho):
    start_time = time.time()

    for i in range(len(HORARIOS)):
        x = resolucion_por_gauss(largo, ancho, TEMPERATURAS[i])
    x_temperatura_promedio = resolucion_por_gauss(largo, ancho, TEMPERATURA_IZQUIERDA)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

def evaluar_tiempo_ejecucion_gauss_seidez(largo,ancho):
    start_time = time.time()

    for i in range(len(HORARIOS)):
        x, iteraciones = resolucion_por_gauss_seidez(largo, ancho, TEMPERATURAS[i])
    x_temperatura_promedio, iteraciones = resolucion_por_gauss_seidez(largo, ancho, TEMPERATURA_IZQUIERDA)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

def evaluar_tiempo_ejecucion_jacobi(largo,ancho):
    start_time = time.time()

    for i in range(len(HORARIOS)):
        x, iteraciones = resolucion_por_jacobi(largo, ancho, TEMPERATURAS[i])
    x_temperatura_promedio, iteraciones = resolucion_por_jacobi(largo, ancho, TEMPERATURA_IZQUIERDA)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

def graficar_grafico_costo_computacional_gauss():
    largo_placa_1 = calcular_dimencion(LARGO_PLANCHA, 1)
    ancho_placa_1 = calcular_dimencion(ANCHO_PLANCHA, 1)
    largo_placa_2 = calcular_dimencion(LARGO_PLANCHA, 0.5)
    ancho_placa_2 = calcular_dimencion(ANCHO_PLANCHA, 0.5)
    largo_placa_3 = calcular_dimencion(LARGO_PLANCHA, 0.4)
    ancho_placa_3 = calcular_dimencion(ANCHO_PLANCHA, 0.4)
    tiempo_1 = evaluar_tiempo_ejecucion_gauss(largo_placa_1, ancho_placa_1)
    tiempo_2 = evaluar_tiempo_ejecucion_gauss(largo_placa_2, ancho_placa_2)
    tiempo_3 = evaluar_tiempo_ejecucion_gauss(largo_placa_3, ancho_placa_3)
    tiempos = [tiempo_1, tiempo_2, tiempo_3]
    etiquetas = ['D = 1cm', 'D = 0,5cm', 'D = 0,4cm']
    plt.bar(etiquetas, tiempos)
    plt.xlabel('Discretizacion')
    plt.ylabel('Tiempo de Ejecucion (segs)')
    plt.title('Costo Computacional - Eliminacion de Gauss')

    plt.show()
    return 0

def graficar_grafico_costo_computacional_gauss_seidez():
    largo_placa_1 = calcular_dimencion(LARGO_PLANCHA, 1)
    ancho_placa_1 = calcular_dimencion(ANCHO_PLANCHA, 1)
    largo_placa_2 = calcular_dimencion(LARGO_PLANCHA, 0.5)
    ancho_placa_2 = calcular_dimencion(ANCHO_PLANCHA, 0.5)
    largo_placa_3 = calcular_dimencion(LARGO_PLANCHA, 0.4)
    ancho_placa_3 = calcular_dimencion(ANCHO_PLANCHA, 0.4)
    tiempo_1 = evaluar_tiempo_ejecucion_gauss_seidez(largo_placa_1, ancho_placa_1)
    tiempo_2 = evaluar_tiempo_ejecucion_gauss_seidez(largo_placa_2, ancho_placa_2)
    tiempo_3 = evaluar_tiempo_ejecucion_gauss_seidez(largo_placa_3, ancho_placa_3)
    tiempos = [tiempo_1, tiempo_2, tiempo_3]
    etiquetas = ['D = 1cm', 'D = 0,5cm', 'D = 0,4cm']
    plt.bar(etiquetas, tiempos)
    plt.xlabel('Discretizacion')
    plt.ylabel('Tiempo de Ejecucion (segs)')
    plt.title('Costo Computacional - Gauss Seidez')

    plt.show()
    return 0

def graficar_grafico_costo_computacional_jacobi():
    largo_placa_1 = calcular_dimencion(LARGO_PLANCHA, 1)
    ancho_placa_1 = calcular_dimencion(ANCHO_PLANCHA, 1)
    largo_placa_2 = calcular_dimencion(LARGO_PLANCHA, 0.5)
    ancho_placa_2 = calcular_dimencion(ANCHO_PLANCHA, 0.5)
    largo_placa_3 = calcular_dimencion(LARGO_PLANCHA, 0.4)
    ancho_placa_3 = calcular_dimencion(ANCHO_PLANCHA, 0.4)
    tiempo_1 = evaluar_tiempo_ejecucion_jacobi(largo_placa_1, ancho_placa_1)
    tiempo_2 = evaluar_tiempo_ejecucion_jacobi(largo_placa_2, ancho_placa_2)
    tiempo_3 = evaluar_tiempo_ejecucion_jacobi(largo_placa_3, ancho_placa_3)
    tiempos = [tiempo_1, tiempo_2, tiempo_3]
    etiquetas = ['D = 1cm', 'D = 0,5cm', 'D = 0,4cm']
    plt.bar(etiquetas, tiempos)
    plt.xlabel('Discretizacion')
    plt.ylabel('Tiempo de Ejecucion (segs)')
    plt.title('Costo Computacional - Jacobi')

    plt.show()
    return 0
def main ():
    largo_placa = calcular_dimencion(LARGO_PLANCHA, DISCRETIZACION)
    ancho_placa = calcular_dimencion(ANCHO_PLANCHA, DISCRETIZACION)
    placa_uno = crear_placa_inicial(largo_placa, ancho_placa,21)
    #crear_funcion_respecto_horario(largo_placa, ancho_placa)
    #crear_graficos_de_plancha(largo_placa, ancho_placa)
    #crear_graficos_de_plancha_gauss(largo_placa, ancho_placa)
    #mostrar_iteraciones_gauss_seidez(largo_placa, ancho_placa)
    #mostrar_iteraciones_jacobi(largo_placa, ancho_placa)
    #graficar_grafico_costo_computacional_gauss()
    #graficar_grafico_costo_computacional_jacobi()
    #graficar_grafico_costo_computacional_gauss_seidez()
    #j, i = resolucion_por_jacobi(largo_placa, ancho_placa, TEMPERATURA_IZQUIERDA)
    #gs, i = resolucion_por_gauss_seidez(largo_placa, ancho_placa, TEMPERATURA_IZQUIERDA)
    #eg = resolucion_por_gauss(largo_placa, ancho_placa, TEMPERATURA_IZQUIERDA)
    #imprimir_plancha(j, ancho_placa, largo_placa)
    #imprimir_plancha(eg,ancho_placa, largo_placa)
    return 0

main ()
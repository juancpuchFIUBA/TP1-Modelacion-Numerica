import os

from PIL import Image


def combinar_imagenes(imagenes, salida, ancho_resultante, altura_resultante):
    # Abrir las imágenes
    img1 = Image.open(imagenes[0])
    img2 = Image.open(imagenes[1])
    img3 = Image.open(imagenes[2])

    # Crear una imagen nueva con fondo blanco
    imagen_resultante = Image.new('RGB', (ancho_resultante, altura_resultante), 'white')

    # Pegar la primera imagen arriba centrada
    x_offset = (ancho_resultante - img1.width) // 2
    y_offset = 0
    imagen_resultante.paste(img1, (x_offset, y_offset))

    # Calcular la posición para las otras dos imágenes abajo en cada costado
    img2_width, img2_height = img2.size
    img3_width, img3_height = img3.size

    x_offset2 = (ancho_resultante - img2_width - img3_width) // 3
    y_offset2 = altura_resultante - img2_height
    x_offset3 = x_offset2 * 2 + img2_width

    imagen_resultante.paste(img2, (x_offset2, y_offset2))
    imagen_resultante.paste(img3, (x_offset3, y_offset2))

    # Guardar la imagen resultante
    imagen_resultante.save(salida)


# Obtener la ruta del directorio actual
directorio_actual = os.path.dirname(os.path.abspath(__file__))

# Crear la carpeta 'resultados' si no existe
carpeta_resultados = os.path.join(directorio_actual, 'resultados')
if not os.path.exists(carpeta_resultados):
    os.makedirs(carpeta_resultados)

# Dimensiones deseadas para la imagen resultante
ancho_resultante = 1280
altura_resultante = 960

# Iterar sobre las 24 imágenes
for i in range(25):
    # Rutas de las imágenes de cada carpeta
    ruta_discretizacion1 = os.path.join(directorio_actual, 'discretizacion1', f'im{i}.png')
    ruta_discretizacion2 = os.path.join(directorio_actual, 'discretizacion2', f'im{i}.png')
    ruta_discretizacion3 = os.path.join(directorio_actual, 'discretizacion3', f'im{i}.png')

    # Ruta de salida para la imagen combinada
    ruta_salida = os.path.join(carpeta_resultados, f'res{i}.jpg')

    # Combinar las imágenes y guardar el resultado
    combinar_imagenes([ruta_discretizacion1, ruta_discretizacion2, ruta_discretizacion3], ruta_salida, ancho_resultante,
                      altura_resultante)

print("Imágenes combinadas guardadas en la carpeta 'resultados'.")

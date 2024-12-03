import pandas as pd

# Definir el nombre del archivo de entrada y salida
archivo_entrada = 'data/links/links_nodes.csv'
archivo_salida = 'data/links/links_nodes_clean.csv'

# Leer el archivo CSV
# Se asume que el separador es una coma y que las comillas están correctamente manejadas
df = pd.read_csv(archivo_entrada)



# Procesar la columna 'Label':
# 1. Eliminar las comillas dobles si existen
# 2. Remover el prefijo 'https://en.wikipedia.org/wiki/'
df['Label'] = df['Label'].str.replace('"', '', regex=False)
df['Label'] = df['Label'].str.replace('https://en.wikipedia.org/wiki/', '', regex=False)


# Guardar el DataFrame procesado en un nuevo archivo CSV
df.to_csv(archivo_salida, index=False)

print(f"\nEl archivo procesado se ha guardado como '{archivo_salida}'.")

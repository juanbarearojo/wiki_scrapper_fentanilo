import pandas as pd

def analizar_modularidad(csv_path, top_n=10):
    """
    Analiza la modularidad de las comunidades en un grafo y muestra las comunidades más relevantes.

    Parámetros:
    - csv_path (str): Ruta al archivo CSV que contiene los nodos con las columnas 'modularity_class' y 'degree'.
    - top_n (int): Número máximo de comunidades más relevantes (por tamaño) a mostrar. Por defecto es 10.
    """
    # Leer el archivo CSV
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: El archivo '{csv_path}' no se encontró.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: El archivo '{csv_path}' está vacío.")
        return
    except pd.errors.ParserError:
        print(f"Error: El archivo '{csv_path}' no está bien formateado.")
        return

    # Verificar que las columnas necesarias existan
    columnas_necesarias = {'modularity_class', 'degree', 'Id', 'Label'}
    if not columnas_necesarias.issubset(df.columns):
        print(f"Error: El archivo CSV debe contener las columnas: {columnas_necesarias}")
        return

    # Calcular el total de nodos
    total_nodos = len(df)

    if total_nodos == 0:
        print("El archivo CSV no contiene nodos.")
        return

    # Agrupar por 'modularity_class' y calcular el tamaño de cada grupo
    grupos = df.groupby('modularity_class').size().reset_index(name='size')

    # Ordenar los grupos por tamaño descendente
    grupos = grupos.sort_values(by='size', ascending=False)

    # Seleccionar solo las top_n comunidades
    grupos_top = grupos.head(top_n)

    print(f"\nMostrando las top {min(top_n, len(grupos_top))} comunidades más relevantes por tamaño:\n")

    # Iterar sobre las top_n comunidades
    for idx, row in grupos_top.iterrows():
        clase = row['modularity_class']
        size = row['size']
        grupo = df[df['modularity_class'] == clase]

        print(f"=== Comunidad {idx + 1} ===")
        print(f"Clase de Modularidad: {clase}")
        print(f"Cantidad de nodos en esta clase: {size}")
        porcentaje = (size / total_nodos) * 100
        print(f"Porcentaje de la clase: {porcentaje:.2f}%")

        # Verificar si hay al menos un nodo en el grupo
        if size > 0:
            # Obtener los 10 nodos con mayor grado
            top_nodos = grupo.sort_values(by='degree', ascending=False).head(10)

            print("\nTop 10 nodos con mayor grado:")
            print(top_nodos[['Id', 'Label', 'degree']].to_string(index=False))
        else:
            print("Esta comunidad no contiene nodos.")
        
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    # Ruta al archivo CSV
    ruta_csv = 'data/words/words_metrics_nodes.csv'  # Reemplaza con la ruta real de tu archivo

    # Número máximo de comunidades a mostrar
    top_n = 10

    analizar_modularidad(ruta_csv, top_n=top_n)

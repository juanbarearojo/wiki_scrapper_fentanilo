import pandas as pd

def analizar_modularidad(csv_path):
    # Leer el archivo CSV
    df = pd.read_csv(csv_path)

    # Calcular el total de nodos
    total_nodos = len(df)

    # Agrupar por 'modularity_class'
    grupos = df.groupby('modularity_class')

    # Recorrer cada grupo para obtener la información requerida
    for clase, grupo in grupos:
        print(f"\nClase de Modularidad: {clase}")

        # Número de nodos en la clase
        num_nodos = len(grupo)
        print(f"Cantidad de nodos en esta clase: {num_nodos}")

        # Porcentaje de la clase respecto al total
        porcentaje = (num_nodos / total_nodos) * 100
        print(f"Porcentaje de la clase: {porcentaje:.2f}%")

        # Obtener los 10 nodos con mayor grado
        top_nodos = grupo.sort_values(by='degree', ascending=False).head(10)

        print("Top 10 nodos con mayor grado:")
        print(top_nodos[['Id', 'Label', 'degree']].to_string(index=False))

if __name__ == "__main__":
    # Ruta al archivo CSV
    ruta_csv = 'data/links/links_metrics_nodes.csv'  # Reemplaza con la ruta real de tu archivo

    analizar_modularidad(ruta_csv)

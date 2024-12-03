import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

def cargar_datos(nodos_path, aristas_path):
    """
    Carga los datos de nodos y aristas desde archivos CSV.
    """
    nodos_df = pd.read_csv(nodos_path)
    aristas_df = pd.read_csv(aristas_path)
    return nodos_df, aristas_df

def construir_grafo(nodos_df, aristas_df):
    """
    Construye un grafo no dirigido a partir de los DataFrames de nodos y aristas.
    """
    G = nx.Graph()
    
    # Añadir nodos con sus atributos
    for _, row in nodos_df.iterrows():
        G.add_node(row['Id'], label=row['Label'], group=row['Group'], attribute=row['Attribute'])
    
    # Añadir aristas con sus atributos
    for _, row in aristas_df.iterrows():
        G.add_edge(row['Source'], row['Target'], type=row['Type'], weight=row['Weight'])
    
    return G

def aplicar_modularidad_codiciosa(G):
    """
    Aplica el algoritmo de modularidad codiciosa para detectar comunidades.
    """
    comunidades = greedy_modularity_communities(G)
    
    # Convertir a una lista de listas para mostrar fácilmente las comunidades
    comunidades_lista = [list(comunidad) for comunidad in comunidades]
    
    # Calcular la modularidad de la partición
    modularidad = nx.algorithms.community.quality.modularity(G, comunidades)
    
    return comunidades_lista, modularidad

def main():
    # Rutas a los archivos CSV
    nodos_path = 'data/words/words_bigrams_nodes.csv'
    aristas_path = 'data/words/words_bigrams_edges.csv'
    
    # Cargar datos
    nodos_df, aristas_df = cargar_datos(nodos_path, aristas_path)
    
    # Construir el grafo
    G = construir_grafo(nodos_df, aristas_df)
    
    # Verificar si el grafo tiene al menos una arista
    if G.number_of_edges() == 0:
        print("El grafo no tiene aristas. No se puede aplicar la detección de comunidades.")
        return
    
    # Aplicar el algoritmo de modularidad codiciosa
    print("Aplicando el algoritmo de modularidad codiciosa...\n")
    comunidades, modularidad = aplicar_modularidad_codiciosa(G)
    
    # Mostrar resultados
    print("=== Resultados Finales ===")
    print(f"Mejor modularidad: {modularidad:.4f}")
    print(f"Número de comunidades: {len(comunidades)}")
    print("\nComunidades detectadas:")
    for i, comunidad in enumerate(comunidades):
        print(f" - Comunidad {i+1}: {len(comunidad)} nodos")

if __name__ == "__main__":
    main()

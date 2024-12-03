import requests
from bs4 import BeautifulSoup
import spacy
# import scispacy  # No es necesario si no se usa directamente
import networkx as nx
import csv
from urllib.parse import urljoin
from collections import Counter
import re
import time
import random
import os
from nltk.util import ngrams
import sys  # Importado para usar sys.exit()
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np  # Importar NumPy para cálculos estadísticos

# === Parámetros de Configuración ===
MAX_ARTICLES = 100           # Número máximo de artículos a procesar
MAX_DEPTH = 100              # Profundidad máxima de scraping
MIN_LINK_FREQ = 3            # Frecuencia mínima para retener un enlace
TOP_N_BIGRAMS = 150          # Número de bigramas más frecuentes a integrar
EDGE_POD_PERCENTILE = 45     # Percentil para poda de aristas
NODE_POD_PERCENTILE = 15     # Percentil para poda de nodos
MIN_NODE_FREQ = 5            # Frecuencia mínima para retener un nodo
MIN_EDGE_WEIGHT = 5          # Peso mínimo para retener una arista

# === Configuración de la Sesión HTTP con Reintentos ===
session = requests.Session()
retry_strategy = Retry(
    total=3,                                     # Total de reintentos
    backoff_factor=1,                           # Factor de retroceso
    status_forcelist=[429, 500, 502, 503, 504],  # Códigos de estado que activan reintentos
    allowed_methods=["HEAD", "GET", "OPTIONS"]   # Métodos HTTP permitidos para reintentos
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

# === Cargar el Modelo de SciSpaCy ===
try:
    # Reemplaza 'en_core_sci_md' con el modelo que hayas instalado
    nlp = spacy.load('en_core_sci_md', disable=['parser'])  
except Exception as e:
    print(f"Error al cargar SciSpaCy o el modelo 'en_core_sci_md': {e}")
    print("Intenta instalar el modelo ejecutando:")
    print("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_md-0.5.0.tar.gz")
    sys.exit(1)  # Usar sys.exit() en lugar de exit()

# Obtener las stopwords de spaCy y añadir stopwords médicas personalizadas si es necesario
stopwords_set = nlp.Defaults.stop_words
# Ejemplo: Añadir términos médicos que deseas excluir
# stopwords_set.update({'term1', 'term2'})  # Añade términos personalizados si es necesario

# === Funciones Auxiliares ===

def is_low_information_verb(token):
    """
    Determina si un token es un verbo de baja información.
    """
    return token.pos_ == "VERB" and token.lemma_ in {
        'be', 'have', 'do', 'say', 'go', 'can', 'get', 'would', 'make', 'know',
        'will', 'think', 'take', 'see', 'come', 'could', 'want', 'look', 'use',
        'find', 'give', 'tell', 'work', 'call', 'include'  # Añadido 'include' y 'use'
    }

def clean_text(text, language='english'):
    """
    Limpia y procesa el texto de entrada:
    - Convierte a minúsculas
    - Elimina caracteres especiales
    - Tokeniza el texto
    - Elimina stopwords y verbos de baja información
    - Extrae entidades nombradas
    - Genera bigramas de las palabras más comunes
    """
    # Limpieza básica
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Procesar el texto con SciSpaCy
    doc = nlp(text)

    words = []
    entities = []

    for token in doc:
        if token.is_stop:
            continue
        if is_low_information_verb(token):
            continue
        if token.is_alpha and len(token) > 2:
            words.append(token.lemma_)
    
    # Extraer entidades nombradas
    for ent in doc.ents:
        entities.append(ent.text.lower())

    # Combinar palabras y entidades
    all_words = words + entities

    # Contar frecuencia de palabras
    word_freq = Counter(all_words)
    most_common_words = [word for word, freq in word_freq.most_common(100) if ' ' not in word and word.isalpha()]  # Asegura que son palabras individuales sin espacios

    # Verificar que hay al menos dos palabras antes de generar bigrams
    if len(most_common_words) < 2:
        bigrams = []
    else:
        # Generar bigramas
        bigrams = list(ngrams(most_common_words, 2))
        bigrams = [' '.join(bigram) for bigram in bigrams]

    return all_words, bigrams

def get_cooccurrence_edges(words, window_size=5):
    """
    Genera bordes de co-ocurrencia dentro de una ventana deslizante.
    """
    edges = []
    for i in range(len(words) - window_size + 1):
        window = words[i:i+window_size]
        for j in range(len(window)):
            for k in range(j+1, len(window)):
                edge = tuple(sorted([window[j], window[k]]))
                edges.append(edge)
    return edges

def extract_wikipedia_links(soup, base_url):
    """
    Extrae enlaces válidos de artículos de Wikipedia desde el objeto soup.
    """
    links = set()
    for link in soup.find_all('a', href=True):
        href = link['href']
        # Filtrar enlaces que no sean a otros artículos
        if href.startswith('/wiki/') and ':' not in href and not href.startswith('/wiki/Main_Page') and '#' not in href:
            full_url = urljoin(base_url, href)
            links.add(full_url)
    return links

def prune_links(links_data, min_freq=MIN_LINK_FREQ):
    """
    Elimina enlaces que aparecen menos de `min_freq` veces.
    
    Parámetros:
    - links_data (List[Tuple[str, Set[str]]]): Lista de enlaces por artículo.
    - min_freq (int): Frecuencia mínima requerida para mantener un enlace.
    
    Retorna:
    - Dict[str, Set[str]]: Mapa de artículos con sus enlaces filtrados.
    """
    link_counter = Counter()
    for _, links in links_data:
        link_counter.update(links)
    
    # Determinar enlaces a mantener
    valid_links = {link for link, freq in link_counter.items() if freq >= min_freq}
    
    # Filtrar enlaces en cada artículo
    pruned_links_data = {}
    for source, links in links_data:
        pruned = links.intersection(valid_links)
        if pruned:
            pruned_links_data[source] = pruned
    
    print(f"Enlaces retenidos después de la poda: {len(pruned_links_data)}")
    return pruned_links_data

def prune_edges_by_percentile(graph, percentile=35, min_weight=2):
    """
    Elimina las aristas del grafo que están por debajo del percentil especificado
    o por debajo de un peso mínimo.
    
    Parámetros:
    - graph (networkx.Graph): El grafo a podar.
    - percentile (float): El percentil mínimo requerido para mantener una arista.
    - min_weight (int): El peso mínimo requerido para mantener una arista.
    
    Retorna:
    - networkx.Graph: El grafo podado.
    """
    weights = [data['Weight'] for u, v, data in graph.edges(data=True)]
    if not weights:
        print("No hay aristas para podar.")
        return graph
    threshold = np.percentile(weights, percentile)
    edges_to_remove = [
        (u, v) for u, v, data in graph.edges(data=True) 
        if data.get('Weight', 1) < threshold or data.get('Weight', 1) < min_weight
    ]
    graph.remove_edges_from(edges_to_remove)
    print(f"Umbral de poda de aristas (percentil {percentile} y peso mínimo {min_weight}): {threshold}")
    print(f"Aristas eliminadas durante la poda: {len(edges_to_remove)}")
    return graph

def prune_nodes_by_percentile(graph, percentile=15, min_freq=5):
    """
    Elimina los nodos con frecuencia de palabra por debajo del percentil especificado
    o por debajo de una frecuencia mínima.
    
    Parámetros:
    - graph (networkx.Graph): El grafo a podar.
    - percentile (float): El percentil mínimo requerido para mantener un nodo.
    - min_freq (int): Frecuencia mínima requerida para mantener un nodo.
    
    Retorna:
    - networkx.Graph: El grafo podado.
    """
    # Asumiendo que el atributo 'Attribute' almacena la frecuencia de la palabra
    word_nodes = {node: data['Attribute'] for node, data in graph.nodes(data=True) if data['Group'] == 'word'}
    if not word_nodes:
        print("No hay nodos de palabras para podar.")
        return graph
    
    frequencies = list(word_nodes.values())
    threshold_percentile = np.percentile(frequencies, percentile)
    nodes_to_remove = [
        node for node, freq in word_nodes.items() 
        if freq < threshold_percentile or freq < min_freq
    ]
    
    graph.remove_nodes_from(nodes_to_remove)
    print(f"Umbral de poda de nodos (percentil {percentile} y frecuencia mínima {min_freq}): {threshold_percentile}")
    print(f"Nodos eliminados durante la poda: {len(nodes_to_remove)}")
    return graph

def crawl_wikipedia(article_url, base_url, max_depth, max_articles):
    """
    Rastrear artículos de Wikipedia hasta una profundidad y número máximo especificados.
    Extrae palabras individuales, bigramas y entidades nombradas.
    Implementa una poda de enlaces basada en frecuencia.
    """
    visited_articles = set()
    articles_to_visit = [(article_url, 0)]  # (URL, profundidad)
    words_data = []
    bigrams_data = []
    links_data = []
    total_articles = 0

    while articles_to_visit and total_articles < max_articles:
        current_url, depth = articles_to_visit.pop(0)
        if current_url in visited_articles or depth > max_depth:
            continue

        print(f"Scraping: {current_url} (Depth: {depth})")

        # Retraso respetuoso entre solicitudes
        time.sleep(random.uniform(1, 3))

        # Realizar la solicitud HTTP con manejo de errores
        try:
            response = session.get(current_url, timeout=10)
            response.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print(f"HTTP Error para {current_url}: {errh}")
            continue
        except requests.exceptions.ConnectionError as errc:
            print(f"Connection Error para {current_url}: {errc}")
            continue
        except requests.exceptions.Timeout as errt:
            print(f"Timeout Error para {current_url}: {errt}")
            continue
        except requests.exceptions.RequestException as err:
            print(f"Request Exception para {current_url}: {err}")
            continue

        # Verificar que el contenido es HTML
        if 'html' not in response.headers.get('Content-Type', ''):
            print(f"El contenido no es HTML para {current_url}")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')

        # Extraer y limpiar el texto
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        words, bigrams = clean_text(text, language='english')
        words_data.append((current_url, words))
        bigrams_data.append((current_url, bigrams))
        print(f"Palabras extraídas: {len(words)}")
        print(f"Bigrams extraídos: {len(bigrams)}")

        # Extraer enlaces de Wikipedia
        links = extract_wikipedia_links(soup, base_url)
        links_data.append((current_url, links))
        print(f"Enlaces encontrados: {len(links)}")

        # Añadir nuevos artículos a la cola
        for link in links:
            if link not in visited_articles and depth + 1 <= max_depth:
                articles_to_visit.append((link, depth + 1))

        visited_articles.add(current_url)
        total_articles += 1
        print(f"Total de artículos procesados: {total_articles}/{max_articles}\n")

    # Poda de Enlaces
    pruned_links_data = prune_links(links_data, min_freq=MIN_LINK_FREQ)

    return words_data, bigrams_data, pruned_links_data

def export_graph(graph, output_nodes, output_edges, graph_type='word_bigram'):
    """
    Exporta un grafo a archivos CSV separados para nodos y aristas.
    Soporta tipos de grafo: 'word', 'bigram', 'link', 'word_bigram'.
    """
    # Asignar IDs únicos a todos los nodos
    node_ids = {node: idx+1 for idx, node in enumerate(graph.nodes())}
    nx.set_node_attributes(graph, node_ids, 'Id')

    # Exportar nodos a CSV
    print(f"Exportando nodos a {output_nodes}...")
    try:
        with open(output_nodes, 'w', newline='', encoding='utf-8') as csvfile_nodes:
            fieldnames_nodes = ['Id', 'Label', 'Group', 'Attribute']
            writer_nodes = csv.writer(csvfile_nodes)
            writer_nodes.writerow(fieldnames_nodes)
            for node, data in graph.nodes(data=True):
                writer_nodes.writerow([data['Id'], node, data['Group'], data['Attribute']])
    except Exception as e:
        print(f"Error al exportar nodos a {output_nodes}: {e}")

    # Exportar aristas a CSV
    print(f"Exportando aristas a {output_edges}...")
    try:
        with open(output_edges, 'w', newline='', encoding='utf-8') as csvfile_edges:
            fieldnames_edges = ['Source', 'Target', 'Type', 'Weight']
            writer_edges = csv.writer(csvfile_edges)
            writer_edges.writerow(fieldnames_edges)
            for source, target, data in graph.edges(data=True):
                source_id = graph.nodes[source]['Id']
                target_id = graph.nodes[target]['Id']
                edge_type = data.get('Type', 'Undirected' if graph_type in ['word', 'bigram', 'word_bigram'] else 'Directed')
                weight = data.get('Weight', 1)  # Default weight es 1 si no se especifica
                writer_edges.writerow([source_id, target_id, edge_type, weight])
    except Exception as e:
        print(f"Error al exportar aristas a {output_edges}: {e}")

# === Función Principal ===
def main():
    """
    Función principal para ejecutar el scraping y la generación de las redes.
    """
    try:
        print("=== Wikipedia Scraper and Network Generator ===\n")
        print("=== Configuración Cargada ===")
        print(f"Artículo Inicial: https://en.wikipedia.org/wiki/Fentanyl")
        print(f"URL Base: https://en.wikipedia.org")
        print(f"Profundidad Máxima: {MAX_DEPTH}")  
        print(f"Máximo de Artículos: {MAX_ARTICLES}")  
        print(f"Ruta de Nodos y Aristas (Palabras y Bigrams): data/words_bigrams_nodes.csv, data/words_bigrams_edges.csv")
        print(f"Ruta de Nodos (Hipervínculos): data/links_nodes.csv")
        print(f"Ruta de Aristas (Hipervínculos): data/links_edges.csv\n")

        # Crear el directorio 'data' si no existe
        os.makedirs('data', exist_ok=True)

        # Iniciar el scraping
        print("Iniciando el proceso de scraping...\n")
        words_data, bigrams_data, links_data = crawl_wikipedia(
            article_url="https://en.wikipedia.org/wiki/Fentanyl",
            base_url="https://en.wikipedia.org",
            max_depth=MAX_DEPTH,
            max_articles=MAX_ARTICLES
        )

        if not words_data:
            print("No se recopilaron datos de palabras.")
        if not bigrams_data:
            print("No se recopilaron datos de bigramas.")
        if not links_data:
            print("No se recopilaron datos de enlaces.")

        # Generar la red de palabras y bigramas
        print("\nGenerando la red de palabras y bigramas...")
        G_words = nx.Graph()
        word_freq_global = Counter()
        edge_freq_global = Counter()
        bigram_freq_global = Counter()

        # Actualizar frecuencias de palabras y bigramas
        for url, words in words_data:
            word_freq = Counter(words)
            word_freq_global.update(word_freq)
            cooccurrence_edges = get_cooccurrence_edges(words)
            edge_freq = Counter(cooccurrence_edges)
            edge_freq_global.update(edge_freq)

        for url, bigrams in bigrams_data:
            bigram_freq_global.update(bigrams)

        # Añadir palabras al grafo con sus frecuencias
        for word, freq in word_freq_global.items():
            G_words.add_node(word, Group='word', Attribute=freq)

        # Añadir bigrams al grafo
        top_bigrams = bigram_freq_global.most_common(TOP_N_BIGRAMS)
        for bigram, freq in top_bigrams:
            G_words.add_node(bigram, Group='bigram', Attribute=freq)

        # Conectar palabras entre sí
        for (word1, word2), freq in edge_freq_global.items():
            if word1 in G_words.nodes and word2 in G_words.nodes:
                G_words.add_edge(word1, word2, Type='Undirected', Weight=freq)

        # Conectar bigrams con sus palabras constituyentes
        for bigram, freq in top_bigrams:
            split_bigram = bigram.split()
            if len(split_bigram) != 2:
                print(f"Bigram inválido detectado: '{bigram}'")
                continue
            word1, word2 = split_bigram
            if word1 in G_words.nodes and word2 in G_words.nodes:
                G_words.add_edge(bigram, word1, Type='Contains', Weight=1)
                G_words.add_edge(bigram, word2, Type='Contains', Weight=1)
            else:
                print(f"Una o ambas palabras del bigram '{bigram}' no están en el grafo de palabras.")

        # Conectar bigrams entre sí si comparten una palabra
        print("\nConectando bigrams entre sí basados en palabras compartidas...")
        for i in range(len(top_bigrams)):
            for j in range(i + 1, len(top_bigrams)):
                bigram1, freq1 = top_bigrams[i]
                bigram2, freq2 = top_bigrams[j]
                words1 = set(bigram1.split())
                words2 = set(bigram2.split())
                if words1.intersection(words2):
                    G_words.add_edge(bigram1, bigram2, Type='Co-occurs', Weight=1)

        # **Aplicar la poda de aristas basada en percentil y peso mínimo**
        print("\nAplicando la poda de aristas basada en percentil y peso mínimo...")
        G_words = prune_edges_by_percentile(G_words, percentile=EDGE_POD_PERCENTILE, min_weight=MIN_EDGE_WEIGHT)

        # **Aplicar la poda de nodos basada en percentil y frecuencia mínima**
        print("\nAplicando la poda de nodos basada en percentil y frecuencia mínima...")
        G_words = prune_nodes_by_percentile(G_words, percentile=NODE_POD_PERCENTILE, min_freq=MIN_NODE_FREQ)

        # Generar la red de hipervínculos con poda
        print("\nGenerando la red de hipervínculos con poda...")
        G_links = nx.DiGraph()
        link_nodes = set()

        for source, targets in links_data.items():
            G_links.add_node(source, Group='link', Attribute=0)
            link_nodes.add(source)
            for target in targets:
                G_links.add_node(target, Group='link', Attribute=0)
                link_nodes.add(target)
                G_links.add_edge(source, target, Type='Directed', Weight=1)

        # Exportar la red de palabras y bigramas
        print("\nExportando la red de palabras y bigramas...")
        export_graph(G_words, "data/words/words_bigrams_nodes.csv", "data/words/words_bigrams_edges.csv", graph_type='word_bigram')

        # Exportar la red de hipervínculos
        print("\nExportando la red de hipervínculos...")
        export_graph(G_links, "data/links/links_nodes.csv", "data/links/links_edges.csv", graph_type='link')

        print("\n=== Exportación Completada ===")
        print(f"- Nodos y Aristas de palabras y bigrams guardados en: data/words/words_bigrams_nodes.csv, data/words/words_bigrams_edges.csv")
        print(f"- Nodos de hipervínculos guardados en: data/links/links_nodes.csv")
        print(f"- Aristas de hipervínculos guardadas en: data/links/links_edges.csv")

    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        sys.exit(1)

# Mover la condición principal fuera de la función main()
if __name__ == '__main__':
    main()

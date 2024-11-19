import requests
from bs4 import BeautifulSoup
import nltk
import networkx as nx
import csv
from urllib.parse import urljoin
from collections import Counter
import re
import time
import random
import os
import sys

# Asegurar que los recursos de NLTK estén descargados (solo stopwords)
nltk.download('stopwords')

def clean_text(text, language='english'):
    """
    Limpia y tokeniza el texto de entrada sin usar 'punkt'.
    """
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar puntuación y caracteres especiales
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenizar el texto usando regex
    tokens = re.findall(r'\b\w+\b', text)
    # Eliminar stopwords y palabras cortas
    stopwords_set = set(nltk.corpus.stopwords.words(language))
    words = [word for word in tokens if word not in stopwords_set and len(word) > 2]
    return words

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
        if href.startswith('/wiki/') and not ':' in href and not href.startswith('/wiki/Main_Page'):
            full_url = urljoin(base_url, href)
            links.add(full_url)
    return links

def crawl_wikipedia(article_url, base_url, max_depth, max_articles):
    """
    Rastrear artículos de Wikipedia hasta una profundidad y número máximo especificados.
    """
    visited_articles = set()
    articles_to_visit = [(article_url, 0)]  # (URL, profundidad)
    words_data = []
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
            response = requests.get(current_url, timeout=10)
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

        soup = BeautifulSoup(response.content, 'html.parser')

        # Extraer y limpiar el texto
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        words = clean_text(text, language='english')
        words_data.append((current_url, words))

        # Extraer enlaces de Wikipedia
        links = extract_wikipedia_links(soup, base_url)
        links_data.append((current_url, links))

        # Añadir nuevos artículos a la cola
        for link in links:
            if link not in visited_articles and depth + 1 <= max_depth:
                articles_to_visit.append((link, depth + 1))

        visited_articles.add(current_url)
        total_articles += 1

    return words_data, links_data

def export_graph(graph, output_nodes, output_edges, graph_type='word'):
    """
    Exporta un grafo a archivos CSV separados para nodos y aristas.
    """
    # Asignar IDs únicos a todos los nodos
    node_ids = {node: idx+1 for idx, node in enumerate(graph.nodes())}
    nx.set_node_attributes(graph, node_ids, 'Id')

    # Exportar nodos a CSV
    print(f"Exportando nodos a {output_nodes}...")
    with open(output_nodes, 'w', newline='', encoding='utf-8') as csvfile_nodes:
        if graph_type == 'word':
            fieldnames_nodes = ['Id', 'Label', 'Group', 'Attribute']
        elif graph_type == 'link':
            fieldnames_nodes = ['Id', 'Label', 'Group', 'Attribute']
        writer_nodes = csv.writer(csvfile_nodes)
        writer_nodes.writerow(fieldnames_nodes)
        for node, data in graph.nodes(data=True):
            writer_nodes.writerow([data['Id'], node, data['Group'], data['Attribute']])

    # Exportar aristas a CSV
    print(f"Exportando aristas a {output_edges}...")
    with open(output_edges, 'w', newline='', encoding='utf-8') as csvfile_edges:
        if graph_type == 'word':
            fieldnames_edges = ['Source', 'Target', 'Type', 'Weight']
        elif graph_type == 'link':
            fieldnames_edges = ['Source', 'Target', 'Type', 'Weight']
        writer_edges = csv.writer(csvfile_edges)
        writer_edges.writerow(fieldnames_edges)
        for source, target, data in graph.edges(data=True):
            source_id = graph.nodes[source]['Id']
            target_id = graph.nodes[target]['Id']
            edge_type = data.get('Type', 'Undirected' if graph_type == 'word' else 'Directed')
            if graph_type == 'link':
                weight = 1  # Weight es 1 para hipervínculos
            else:
                weight = data.get('Weight', 1)  # Weight para co-ocurrencia de palabras
            writer_edges.writerow([source_id, target_id, edge_type, weight])

def main():
    """
    Función principal para ejecutar el scraping y la generación de las redes.
    """

    # Definir las variables de configuración
    article_url = "https://en.wikipedia.org/wiki/Fentanyl"  # URL inicial de Wikipedia
    base_url = "https://en.wikipedia.org"                  # URL base de Wikipedia
    max_depth = 1                                           # Profundidad máxima de scraping
    max_articles = 20                                       # Número máximo de artículos a procesar

    # Rutas para los archivos de nodos y aristas de co-ocurrencia de palabras
    output_nodes_words = "/home/juan/Escritorio/facultad/1 CUATRI CUARTO/RSC/wiki_scrapper/data/words_nodes.csv"
    output_edges_words = "/home/juan/Escritorio/facultad/1 CUATRI CUARTO/RSC/wiki_scrapper/data/words_edges.csv"

    # Rutas para los archivos de nodos y aristas de hipervínculos
    output_nodes_links = "/home/juan/Escritorio/facultad/1 CUATRI CUARTO/RSC/wiki_scrapper/data/links_nodes.csv"
    output_edges_links = "/home/juan/Escritorio/facultad/1 CUATRI CUARTO/RSC/wiki_scrapper/data/links_edges.csv"

    print("=== Wikipedia Scraper and Network Generator ===\n")
    print("=== Configuración Cargada ===")
    print(f"Artículo Inicial: {article_url}")
    print(f"URL Base: {base_url}")
    print(f"Profundidad Máxima: {max_depth}")
    print(f"Máximo de Artículos: {max_articles}")
    print(f"Ruta de Nodos (Palabras): {output_nodes_words}")
    print(f"Ruta de Aristas (Palabras): {output_edges_words}")
    print(f"Ruta de Nodos (Hipervínculos): {output_nodes_links}")
    print(f"Ruta de Aristas (Hipervínculos): {output_edges_links}\n")

    # Iniciar el scraping
    print("Iniciando el proceso de scraping...\n")
    words_data, links_data = crawl_wikipedia(article_url, base_url, max_depth, max_articles)

    # Generar la red de co-ocurrencia de palabras
    print("\nGenerando la red de co-ocurrencia de palabras...")
    G_words = nx.Graph()
    word_freq_global = Counter()
    edge_freq_global = Counter()

    for url, words in words_data:
        word_freq = Counter(words)
        word_freq_global.update(word_freq)
        cooccurrence_edges = get_cooccurrence_edges(words)
        edge_freq = Counter(cooccurrence_edges)
        edge_freq_global.update(edge_freq)

    for word, freq in word_freq_global.items():
        G_words.add_node(word, Group='word', Attribute=freq)

    for (word1, word2), freq in edge_freq_global.items():
        G_words.add_edge(word1, word2, Type='Undirected', Weight=freq)

    # Generar la red de hipervínculos
    print("Generando la red de hipervínculos...")
    G_links = nx.DiGraph()
    link_nodes = set()

    for url, links in links_data:
        link_nodes.add(url)
        link_nodes.update(links)

    for link in link_nodes:
        G_links.add_node(link, Group='link', Attribute=0)  # El atributo puede usarse para el in-degree si es necesario

    for source_url, target_links in links_data:
        for target_url in target_links:
            G_links.add_edge(source_url, target_url, Type='Directed', Weight=1)  # Weight es 1 para todos los hipervínculos

    # Exportar la red de co-ocurrencia de palabras
    export_graph(G_words, output_nodes_words, output_edges_words, graph_type='word')

    # Exportar la red de hipervínculos
    export_graph(G_links, output_nodes_links, output_edges_links, graph_type='link')

    print("\n=== Exportación Completada ===")
    print(f"- Nodos de palabras guardados en: {output_nodes_words}")
    print(f"- Aristas de palabras guardadas en: {output_edges_words}")
    print(f"- Nodos de hipervínculos guardados en: {output_nodes_links}")
    print(f"- Aristas de hipervínculos guardadas en: {output_edges_links}")

if __name__ == '__main__':
    main()

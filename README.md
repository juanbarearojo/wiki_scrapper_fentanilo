# Wikipedia Scraper and Network Generator

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Output](#output)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

The **Wikipedia Scraper and Network Generator** is a Python-based tool designed to crawl Wikipedia articles, extract meaningful textual data, and construct comprehensive networks of words, bigrams, and hyperlinks. By analyzing the co-occurrence of words and the structure of Wikipedia's internal links, this tool provides valuable insights into the relationships and significance of terms within a specific domain.

## Repository Structure

The repository structure includes a `data/` directory where the results of the analysis are stored in CSV and TXT formats. These files are ready to be used or processed by other users. Below is a description of the contents:

### Directory Structure

- **`data/links_edges.csv`**: Contains the edges of the hyperlink network.
- **`data/links_nodes.csv`**: Contains the nodes of the hyperlink network.
- **`data/nodos_visitados.txt`**: A list of all the nodes (Wikipedia articles) visited during the scraping process.
- **`data/words_bigrams_edges.csv`**: Contains the edges of the word and bigram network.
- **`data/words_bigrams_nodes.csv`**: Contains the nodes of the word and bigram network.


### Open Data

All files in the `data/` directory are open for use. You can utilize them directly for additional analysis, data visualization, or any other task. Feel free to explore and contribute with new insights or improvements.


## Features

- **Web Scraping**: Efficiently crawls Wikipedia articles up to a specified depth and number of articles.
- **Text Processing**: Cleans and processes textual content to extract significant words and bigrams while filtering out low-information verbs and stopwords.
- **Entity Recognition**: Identifies and incorporates named entities from the text.
- **Network Construction**: Builds co-occurrence networks of words and bigrams, as well as a directed graph of Wikipedia hyperlinks.
- **Graph Pruning**: Applies statistical thresholds and minimum frequency/weight criteria to prune nodes and edges, ensuring meaningful and manageable network sizes.
- **Data Export**: Exports the resulting networks into CSV files for further analysis or visualization.

## Installation

### Prerequisites

- **Python 3.7 or higher**: Ensure you have Python installed. You can download it from the [official website](https://www.python.org/downloads/).

### Clone the Repository

```bash
git clone https://github.com/tu-usuario/wikipedia-scraper-network-generator.git
cd wikipedia-scraper-network-generator
```

### Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

Install the required Python libraries using `pip`:

```bash
pip install -r requirements.txt
```
```

### Download SciSpaCy Model

The project utilizes the `en_core_sci_md` SciSpaCy model for advanced scientific text processing. Download and install the model using the following command:

```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_md-0.5.0.tar.gz
```

Alternatively, you can download other SciSpaCy models based on your requirements from the [SciSpaCy GitHub repository](https://github.com/allenai/scispacy).

## Usage

### Running the Scraper

Execute the main Python script to start the scraping and network generation process:

```bash
python pipeline.py
```

**Note**: Replace `scraper.py` with the actual filename of your Python script if different.

### Configuration

The script allows you to configure various parameters to tailor the scraping and network generation process. These configurations are defined at the beginning of the script:

```python
# === Parámetros de Configuración ===
MAX_ARTICLES = 100           # Número máximo de artículos a procesar
MAX_DEPTH = 100              # Profundidad máxima de scraping
MIN_LINK_FREQ = 3            # Frecuencia mínima para retener un enlace
TOP_N_BIGRAMS = 150          # Número de bigramas más frecuentes a integrar
EDGE_POD_PERCENTILE = 45     # Percentil para poda de aristas
NODE_POD_PERCENTILE = 15     # Percentil para poda de nodos
MIN_NODE_FREQ = 5            # Frecuencia mínima para retener un nodo
MIN_EDGE_WEIGHT = 5          # Peso mínimo para retener una arista
```

You can adjust these parameters directly in the script to meet your specific needs:

- `MAX_ARTICLES`: Maximum number of Wikipedia articles to process.
- `MAX_DEPTH`: Maximum depth for crawling related articles.
- `MIN_LINK_FREQ`: Minimum frequency for retaining a hyperlink.
- `TOP_N_BIGRAMS`: Number of top bigrams to include in the network.
- `EDGE_POD_PERCENTILE`: Percentile threshold for pruning edges.
- `NODE_POD_PERCENTILE`: Percentile threshold for pruning nodes.
- `MIN_NODE_FREQ`: Minimum frequency to retain a node.
- `MIN_EDGE_WEIGHT`: Minimum weight to retain an edge.

### Example

To start scraping from the Wikipedia article on "Fentanyl" with default configurations:

```bash
python scraper.py
```

The script will output progress logs to the console and generate CSV files in the `data/` directory upon completion.

## Dependencies

The project relies on the following Python libraries:

- **requests**: For making HTTP requests.
- **BeautifulSoup4**: For parsing HTML content.
- **spaCy**: For advanced natural language processing.
- **SciSpaCy**: SciPy-based extensions for spaCy, specialized for scientific text.
- **networkx**: For constructing and managing complex networks.
- **numpy**: For numerical computations, especially statistical operations.
- **nltk**: For natural language processing utilities.
- **urllib3**: For handling URL operations.
- **csv**: For reading and writing CSV files.
- **re**: For regular expressions.
- **time & random**: For managing delays between requests.
- **os & sys**: For operating system interactions and system-specific parameters.

Ensure all dependencies are installed as per the [Installation](#installation) section.

## Output

Upon successful execution, the script generates several CSV files within the `data/` directory:

### 1. Words and Bigrams Network

- **`words_bigrams_nodes.csv`**: Contains all nodes (words and bigrams) with their attributes.
  - **Columns**:
    - `Id`: Unique identifier for the node.
    - `Label`: The word or bigram.
    - `Group`: Category (`word` or `bigram`).
    - `Attribute`: Frequency count.

- **`words_bigrams_edges.csv`**: Contains all edges between nodes with their attributes.
  - **Columns**:
    - `Source`: ID of the source node.
    - `Target`: ID of the target node.
    - `Type`: Type of connection (`Undirected`, `Contains`, `Co-occurs`).
    - `Weight`: Weight of the edge representing frequency or co-occurrence strength.

### 2. Hyperlinks Network

- **`links_nodes.csv`**: Contains all hyperlink nodes.
  - **Columns**:
    - `Id`: Unique identifier for the hyperlink node.
    - `Label`: URL of the Wikipedia article.
    - `Group`: Category (`link`).
    - `Attribute`: Placeholder (default `0`).

- **`links_edges.csv`**: Contains all directed edges between hyperlink nodes.
  - **Columns**:
    - `Source`: ID of the source hyperlink node.
    - `Target`: ID of the target hyperlink node.
    - `Type`: Type of connection (`Directed`).
    - `Weight`: Weight of the edge (default `1`).

### Data Visualization

You can utilize these CSV files with network visualization tools such as [Gephi](https://gephi.org/) or [Cytoscape](https://cytoscape.org/) to visualize and analyze the networks.

## How It Works

1. **Initialization**: The script initializes HTTP session settings with retry strategies to handle transient network issues gracefully.

2. **Model Loading**: It loads the SciSpaCy model (`en_core_sci_md`) for processing scientific text. If the model isn't found, it prompts the user to install it.

3. **Crawling**:
   - Starts from a specified Wikipedia article (default: "Fentanyl").
   - Traverses linked Wikipedia articles up to the defined depth and article limit.
   - Extracts textual content from paragraphs (`<p>` tags) of each article.

4. **Text Processing**:
   - Cleans the extracted text by converting it to lowercase and removing non-alphabetic characters.
   - Tokenizes the text and filters out stopwords and low-information verbs.
   - Identifies and extracts named entities.
   - Generates bigrams from the most frequent words.

5. **Network Construction**:
   - **Words and Bigrams Network**: Builds a co-occurrence network where nodes represent words and bigrams, and edges represent their co-occurrence within a sliding window.
   - **Hyperlinks Network**: Constructs a directed graph where nodes are Wikipedia articles and edges represent hyperlinks between them.

6. **Pruning**:
   - Applies statistical thresholds (percentiles) and minimum frequency/weight criteria to prune less significant nodes and edges, ensuring the network remains focused and manageable.

7. **Exporting**:
   - Outputs the resulting networks into CSV files for further analysis or visualization.

## Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, feel free to open an issue or submit a pull request.

### Steps to Contribute

1. **Fork the Repository**: Click the "Fork" button at the top right of this page.

2. **Clone Your Fork**:

   ```bash
   git clone https://github.com/tu-usuario/wikipedia-scraper-network-generator.git
   cd wikipedia-scraper-network-generator
   ```

3. **Create a New Branch**:

   ```bash
   git checkout -b feature/nueva-funcionalidad
   ```

4. **Make Your Changes**: Implement your features or fixes.

5. **Commit Your Changes**:

   ```bash
   git commit -m "Descripción de los cambios realizados"
   ```

6. **Push to Your Fork**:

   ```bash
   git push origin feature/nueva-funcionalidad
   ```

7. **Open a Pull Request**: Navigate to the original repository and open a pull request from your fork.

## License

Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.


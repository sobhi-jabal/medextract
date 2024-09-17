# MedExtract

MedExtract is a clinical datapoint extraction system that uses Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to extract specific information from medical reports.

## Features

- Supports multiple LLM models (llama3, mistral)
- Implements RAG with various embedding models and retriever types
- Configurable processing options and evaluation metrics
- Supports few-shot learning and prompt engineering
- Benchmarking capabilities for model comparison

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/medextract.git
   cd medextract
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Place your input CSV file in the `data/input/` directory.

2. Modify the `config/config.yaml` file to suit your needs or use the default configuration.

3. Run the main script:
   ```
   python medextract.py --config config/config.yaml
   ```

4. Results will be saved in the `data/output/results/` directory, and figures in the `data/output/figures/` directory.

## Configuration

The `config/config.yaml` file contains all the configurable parameters for the pipeline. You can modify this file to change the behavior of the system, including:

- Input/output file paths
- Processing options
- RAG settings
- Model selection
- Prompt engineering options
- Evaluation settings

## Docker

To run MedExtract using Docker:

1. Build the Docker image:
   ```
   docker build -t medextract .
   ```

2. Run the container:
   ```
   docker run -v $(pwd)/data:/app/data medextract
   ```

## Contributing

We welcome contributions to MedExtract! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions or feedback, please open an issue on this GitHub repository.
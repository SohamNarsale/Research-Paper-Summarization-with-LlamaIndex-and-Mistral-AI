Here's a `README.md` template for your GitHub repository:

```markdown
# RAG based Research Paper Summarization using Llama Index and Mistral-AI

This project provides a Python-based Retrieval-Augmented Generation (RAG) pipeline for summarizing research papers using the `llama_index` library. Leveraging advanced language models from Mistral AI and Hugging Face embeddings, this tool allows users to generate comprehensive summaries from academic documents or articles.

## Features

- **Mistral AI LLM Integration**: Utilizes Mistral AI for language model-based document processing.
- **Hugging Face Embeddings**: Embedding model from Hugging Face for document vectorization.
- **Configurable Document Ingestion and Parsing**: Supports ingesting documents from file or directory and parsing with custom settings.
- **Tree-based Asynchronous Summarization**: Summarizes content in a tree structure for efficient and detailed responses.
- **Chunking and Sentence Splitting**: Configurable chunk size and overlap for optimized document processing.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SohamNarsale/Research-Paper-Summarization-with-LlamaIndex-and-Mistral-AI.git
   cd Research-Paper-Summarization-with-LlamaIndex-and-Mistral-AI
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.10 or higher. Install necessary libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys**:
   - Obtain an API key from [Mistral AI](https://console.mistral.ai/api-keys/) and add it to the code:
     ```python
     llm = MistralAI(api_key="YOUR_API_KEY", temperature=0.1, model='open-mixtral-8x7b')
     ```

## Usage
Run the summarization:
```python
python main.py
```
### Code Explanation

- **Initialize Language Model**:
  ```python
  llm = MistralAI(api_key="YOUR_API_KEY", temperature=0.1, model='open-mixtral-8x7b')
  ```
  This sets up MistralAI as the language model for summarization.

- **Initialize Embedding Model**:
  ```python
  embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
  ```
  Configures Hugging Face embedding for document vectorization.

- **Document Ingestion**:
  Documents are read from a specified file or directory:
  ```python
  reader = SimpleDirectoryReader(input_files=["file_path"])
  documents = reader.load_data()
  ```

- **Document Summarization**:
  Configure a tree-based summarization response synthesizer:
  ```python
  response_synthesizer = get_response_synthesizer(response_mode="tree_summarize", use_async=True)
  ```

- **Querying and Displaying the Summary**:
  Execute a summarization query and display in Markdown format:
  ```python
  response = query_engine.query("Provide a full summary of the research paper")
  ```

### Example

To summarize a research paper:
1. Place the document file in the specified path.
2. Adjust settings for `chunk_size`, `chunk_overlap`, and `context_window` as needed.
3. Run the script to generate a summary covering key sections.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/SohamNarsale/Research-Paper-Summarization-with-LlamaIndex-and-Mistral-AI/blob/main/LICENSE) file for details.



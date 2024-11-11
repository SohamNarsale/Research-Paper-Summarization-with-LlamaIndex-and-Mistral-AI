# Import necessary modules from llama_index for configuration, ingestion, and query setup.
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    PromptTemplate
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Initialize the language model (LLM) with MistralAI, specifying model parameters.
llm = MistralAI(api_key="YOUR_API_KEY", temperature=0.1, model='open-mixtral-8x7b')

# Initialize embedding model using Hugging Face's BAAI model for document embeddings.
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Set general settings for the summarization pipeline.
Settings.llm = llm                           # Set the LLM
Settings.embed_model = embed_model           # Set the embedding model
Settings.node_parser = SentenceSplitter(chunk_size=2048, chunk_overlap=256) # Configure text chunking settings
Settings.num_output = 4096                   # Set the number of output tokens
Settings.context_window = 8192               # Set the context window size

# Load documents from a specified file or directory.
reader = SimpleDirectoryReader(input_files=["file_path"])  # Replace 'file_path' with the actual file path
# Alternative: Load all files in a directory
# reader = SimpleDirectoryReader(input_dir="directory_path")

# Read and load data into the pipeline.
documents = reader.load_data()

# Import necessary modules for creating a document summary index.
from llama_index.core import DocumentSummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import get_response_synthesizer

# Initialize a sentence splitter for chunking documents into manageable parts.
splitter = SentenceSplitter(chunk_size=2048, chunk_overlap=32)

# Configure the response synthesizer to use asynchronous tree-based summarization.
response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize", use_async=True
)

# Create a document summary index with the loaded documents, using sentence splitting and the response synthesizer.
doc_summary_index = DocumentSummaryIndex.from_documents(
    documents,
    transformations=[splitter],            # Apply the sentence splitter transformation
    response_synthesizer=response_synthesizer, # Use the response synthesizer
    show_progress=False,                   # Disable progress display (set to True to enable)
    streaming=False                        # Disable streaming (set to True to enable)
)

# Import modules for displaying Markdown content.
from IPython.display import Markdown, display

# Set up a query engine to process summarization queries with asynchronous, tree-based summarization.
query_engine = doc_summary_index.as_query_engine(
    response_mode="tree_summarize", use_async=True
)

# Execute a query to generate a summary of the document, covering various sections.
response = query_engine.query("Provide a full summary of the research paper, including the introduction, methodology, key findings, and conclusion.")

# Display the summary in Markdown format.
display(Markdown(str(response)))
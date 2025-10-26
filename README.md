# local_chatbot
## Project Overview
This Project is an LLM Agent system built with Retrieval-Augmented Generation(RAG).
## System Architecture & Key Components
### Overall Information
+ **LLM Agent Layer**: Handles dialogue, the model is used via Ollama (locally deployed AI model runner)
+ **Retriever Layer**: Performs vector search with ChromaDB
+ **Embedding Model**: sentence-transformers(all-minilm-l6-v2 )
+ **UI Layer**: Provides an interactive GUI with Scores(1-5) and Feedback
### Project Structure
project/<br>
├─ model.py               # LLM and Retrieval Layer、<br>
├─ pdf_verarbeiten.py     # RAG (Retrieval-Augmented Generation) construction from PDFs<br>
├─ panel_test.py          # Frontend UI for testing and demonstration<br>
└─ requirements.txt       # Python dependencies<br>
## Tried Approaches
### Data Ingestion & Chunking
+ **Chunking Strategie**:
  1. Fixed-Length Chunking:
           Splitting text into equal-sized segments based on token count or character length.
           **Pros**: simple and predictable
           **Cons**: may cut sentences, weaker semantic relation, not suitable for tables
  2. Semantic Chunking:
           Using NLP techniques (sentence boundary detection, discourse markers) to ensure coherent splits.
           **Pros**: Preserves meaning- Works better for QA tasks
           **Cons**: Slightly slower- Need overlap tuning
  3. Unstructured.io<br>
      ├─ Using 'Unstructured' to separate texts, tables, images.<br>
      ├─ Store the headers, titles and subtitles as metadata.<br>
      ├─ Convert the page which contains the table into image<br>
      └─ Use img2table to extract the table information and convert into plain text<br>
            (e.g. Error Name: Overload, Description: ..., Error No. : 25)<br>
              **Pros**:  Handles mixed formats (text/table/image)- Keeps document structure & metadata<br>
              **Cons**: OCR quality affects accuracy-Processing cost higher<br>
     
+ **Embedding and Tokenization**:
   **Tooling**: sentence-transformers (all-MiniLM-L6-v2)
   * Tokenization:
     - Splitting text into words, subwords, or tokens using NLP libraries (e.g., Hugging Face, spaCy, NLTK). However with sentence-transformers, the model includes its own tokenizer.
     - custom cases:  Pre-cleaning, Layout-aware text, Truncation control
   * Embdding: Calculates chunks and stores in Vector Databank
     - **Vector Database**
        | Vector DB  | Description |Pros| Cons|
        |-------------|------------|-------------|------------|
        | ChromaDB       | Scalable vector search for fast retrieval. For local/dev and small-to-mid corpora|Local, easy to use- Open source- Fast prototyping| Not ideal for scale-Limited distributed support- Requires manual backup|
        | Pinecone    | Cloud-based vector search with real-time updates. |Managed cloud infra- Scalable + fast- Supports filtering, metadata, hybrid search| Cost for large datasets|
        | Weaviate      | Self-hosted or managed semantic DB (Graph + Vector) | Schema-based (semantic graph + vector)- Hybrid search out of box- Supports modules (OpenAI, Cohere)|More setup overhead- Slightly slower in small datasets|

+ **Indexing**
Indexing converts raw vectorized data into a structured format for efficient retrieval.
It involves embedding each chunk and organizing it in a searchable structure that allows fast similarity queries.<br>
   
   1. **ChromaDB's built-in index**：<br>
   
        **Workflow**:<br>
              * Store embeddings directly in Chroma’s local persistent database.
              * Query using Chroma’s native API for similarity search (e.g., collection.query(...)).
        
        **Pros**:<br>
              * Simple setup, minimal dependencies.
              * Fast for small or medium-sized datasets.
              * Native persistence — automatically stores index files locally.
              
        **Cons**:<br>
              * Tight coupling between the embedding generation and storage layer.
              * Limited flexibility when switching to other vector databases.
              * Less integration with advanced query pipelines (e.g., LLM-based retrieval chains).
   
   2. **Wrapped with LlamaIndex(VectorStoreIndex)**:<br>
   
         **Workflow**:<br>
               *	Use LlamaIndex to manage the end-to-end retrieval pipeline.
               *	LlamaIndex interfaces with Chroma as the underlying vector storage.
               *	The index object is saved locally for faster reloading and reuse.
      
         **Pros**:<br>
               *	Unified abstraction for retrieval, query transformation, and LLM integration.
               * Easier to experiment with hybrid search, reranking, or context filtering.
               * Portable — backend can be switched to Pinecone, Weaviate, etc., without major code changes.
      
         **Cons**:<br>
               * Slightly more setup complexity and dependencies.
               * Local index files may grow large depending on dataset size.


### Retrieval Pipeline
+ **Query Transformations**：
| Approach | Description | Pros |Cons|
 |-------------|------------|-------------|------------|
| Direct Query (Baseline)|Use the raw user query for similarity search.|Fast, simple baseline.|Sensitive to phrasing, may miss relevant context.|
|Query Expansion|Expand query using synonyms or related terms.|Improves recall, especially for domain-specific terms.|May add noise; slower preprocessing.|
|Contextual Reformulation| Use an LLM to rephrase the query into a more precise or historically rich form before retrieval.|Better semantic alignment; more robust to ambiguous queries.|Increases latency and cost.|

+ **Document Retrieval**:
   The retrieval stage performs similarity search on the pre-built vector index (Chroma + LlamaIndex).
   For each query, the system computes vector embeddings using the same model (all-MiniLM-L6-v2) and retrieves the Top-K most similar chunks.

  **Configuration**：
  
     1.	Similarity metric: Cosine similarity
     2.	Top-K: 20 (higher recall to support subsequent reranking)
     3.	Filtering: Optional metadata filtering (e.g., by document source, section, or type)
     4.	Backend: LlamaIndex VectorStoreIndex with Chroma as the underlying vector store
       
+ **Reranking**：
   A reranking model, also known as a cross-encoder, it is for 2-stage retrival system, first stage is to use embedding/retrieve model to extract a set of relevant documents from a larger dataset. Then the reranker will reorder the retrieved documents from the first stage.

### Local Models



### Interactive UI

   

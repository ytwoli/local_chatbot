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
├─ models                 # Local models for embedding and rerank<br>
├─ data                   # The vector database <br>
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
+ **Query Transformations**：<br>

| Approach | Description | Pros |Cons|
|-------------|------------|-------------|------------|
| Direct Query (Baseline)|Use the raw user query for similarity search.|Fast, simple baseline.|Sensitive to phrasing, may miss relevant context.|
|Query Expansion|Expand query using synonyms or related terms.|Improves recall, especially for domain-specific terms.|May add noise; slower preprocessing.|
|Contextual Reformulation| Use an LLM to rephrase the query into a more precise or historically rich form before retrieval.| Better semantic alignment; more robust to ambiguous queries.|Increases latency and cost.|

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

+ **Approach 1**: Using Llama 3.2 via Ollama

  Initially, the project used Llama 3.2 accessed through Ollama, which provides a convenient local API interface for running and managing open-weight models.
  
  * Pros:
    1. Easy setup and integration with local development.
  	2. Fast response time due to Ollama’s optimized serving backend.
  	3.	Supports multiple model sizes and simple switching (e.g., llama3.2:8b vs llama3.2:70b).
  
  * Cons:
    1.	Limited customization for fine-tuning.
  	2.	Depends on Ollama’s runtime environment and model formats.
  
  * Use Case:
    Used in the early stages for rapid prototyping and testing retrieval-augmented generation (RAG) workflows.
  


+ **Approach 2**: Running a Local Fine-tunable Model

  A locally downloaded fine-tunable Llama 3.2 model was adopted to enable custom domain adaptation and deeper control over training and inference.
  The goal was to explore fine-tuning with domain-specific PDFs and RAG contexts.
  
  * Pros:
    1. Full control of model weights and tokenizer.
    2. Enables experimentation with fine-tuning or LoRA adapters.
    3. Works fully offline, no dependency on external services.
  
  * Cons:
    1. Significantly slower inference speed, even with GPU acceleration.
    2. Higher memory consumption.
    3. Requires manual environment management (model loading, quantization, batching).


### Interactive UI
  The UI is implemented with Chainlit to provide a lightweight chat interface for RAG.

  User → Chainlit UI → model retrieve → rerank → answer generate → Stream to UI

  **Key Features**
  
  | Feature|Description | 
  |-------------|------------|
  | Chat interface |Users can ask questions and receive streaming responses from the LLM.|
  | Collection selection |Users can select which vector database collection to query (e.g., different document sets).|
  |Feedback mechanism |After each model response, users can provide feedback(score 1-5 and comments) for quality tracking.|
  | Session memory |Maintains per-user conversation history and active collection.|

  + **Collection Selection**<br>
    	- The active collection is stored in cl.user_session["collection_name"].
      - An initial setting lets the user choose the desired collection.
      - The backend (model.retrieve()) uses this name to query the corresponding Chroma collection.
  + **Feedback Mechanism**
      - Uses @cl.action_callback("feedback") to capture button clicks.
      - Feedback is linked to the corresponding message ID and query.
      - Data can be stored in a local log file, database, or analytics dashboard.


   

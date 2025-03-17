# local_chatbot

## RAG
### Data Preprocessing
1. **Texts**
   + Chunking:
     1. Fixed-Length Chunking: Splitting text into equal-sized segments based on token count or character length.
     2. Semantic Chunking: Using NLP techniques (sentence boundary detection, discourse markers) to ensure coherent splits.
   + Tokenization:
     	1.	Splitting text into words, subwords, or tokens using NLP libraries (e.g., Hugging Face, spaCy, NLTK).
     	2.	Handling special characters, punctuation, and stop words appropriately.
   + Embedding:
     1. Converting text chunks into vector representations
     2. Storing embeddings in a vector database
2. **Tables**
   + Partition:
     1. Using 'Unstructured' to separate texts, tables, images.
     2. Convert the page which contains the table into image
     3. Use img2table to extract the table information 
   + Tokenization:\
     Converting tabular data into structured text representations.
   + Embedding
3. **Images**\
   #TODO

### Indexing
Indexing is a way to convert raw data into structured data so that the retrival process will be more efficient. It typically involves analyzing the loaded content and converting the content into a format suitable for fast retrieval. This often involves creating a vector representation of each document (using techniques like word embeddings or transformer models) and storing these vectors in a way that they can be quickly accessed, often in a vector database or search engine.

+ **Methods**
  | Method       | Description |
  |-------------|------------|
  | FAISS       | Scalable vector search for fast retrieval. |
  | Pinecone    | Cloud-based vector search with real-time updates. |
  | Hybrid      | Combines keyword + vector search for better accuracy. |


### Reranking
A reranking model, also known as a cross-encoder, it is for 2-stage retrival system, first stage is to use embedding/retrieve model to extract a set of relevant documents from a larger dataset. Then the reranker will reorder the retrieved documents from the first stage.
  | Method       | Description |
  |-------------|------------|
  | Cross-Encoders | These models jointly encode the query and each retrieved document, assessing their relevance through a single forward pass. |
  | Bi-Encoders with Interaction Layers    | Bi-encoders independently encode queries and documents, incorporating interaction layers allows for modeling complex relationships, balancing efficiency and accuracy. |
  | Traditional Scoring Techniques |BM25, TF-IDF |

### Query Transformations(multi-queries)



## Model Training

## Interaction with UI

   

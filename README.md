# SupplyChainGPT

## A RAG-Powered Generative AI Co-Pilot for Smart Inventory Planning

SupplyChainGPT is an intelligent AI system that acts as a decision-support co-pilot for supply chain and warehouse managers. It combines **Retrieval-Augmented Generation (RAG)**, **Machine Learning forecasting**, and **Generative AI** to deliver context-aware, explainable, and conversational recommendations for inventory management.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Usage](#usage)
8. [API Reference](#api-reference)
9. [Module Documentation](#module-documentation)
10. [Testing](#testing)
11. [Sample Data](#sample-data)
12. [Performance Metrics](#performance-metrics)
13. [Future Roadmap](#future-roadmap)
14. [Contributing](#contributing)
15. [License](#license)

---

## Overview

### Problem Statement

Traditional supply chain forecasting systems lack business reasoning and context. They provide static reports without explanations, making it difficult for managers to understand *why* certain recommendations are made.

### Solution

SupplyChainGPT bridges this gap by:

- **Reducing manual data interpretation** through AI-powered analysis
- **Integrating forecasts with business context** from internal documents
- **Offering unified knowledge retrieval** across organizational silos
- **Providing explainable outputs** with citations and confidence scores
- **Enabling faster, informed decisions** through conversational AI

### Key Differentiators

| Traditional Systems | SupplyChainGPT |
|---------------------|----------------|
| Static dashboards | Dynamic conversational interface |
| No context explanation | Auto-cited, explainable answers |
| Separate forecasting tools | Integrated ML + RAG + GenAI |
| Manual document search | Semantic document retrieval |
| Fixed reports | Natural language queries |

---

## Features

### Core Capabilities

#### 1. Natural Language Query Interface
- Ask questions in plain English
- Context-aware responses with citations
- Support for follow-up questions

#### 2. Document Intelligence
- **Multi-format support**: PDF, DOCX, PPTX, XLSX, CSV, TXT
- **Automatic parsing** with OCR fallback for scanned documents
- **Entity extraction**: SKUs, warehouses, suppliers, dates
- **Semantic chunking** preserving document structure

#### 3. Hybrid Search & Retrieval
- **Dense retrieval**: Semantic similarity using sentence transformers
- **Sparse retrieval**: BM25 keyword matching
- **Reciprocal Rank Fusion**: Combines both approaches
- **Cross-encoder re-ranking**: Improves precision
- **Business-aware scoring**: Document type and freshness boosts

#### 4. Demand Forecasting
- **Prophet**: Time series forecasting with seasonality
- **XGBoost**: Feature-rich gradient boosting
- **Ensemble**: Combined predictions for accuracy
- **Confidence intervals**: Upper/lower bounds

#### 5. Safety Stock Optimization
- **Statistical method**: Standard deviation based
- **Service level method**: Accounts for lead time variability
- **EOQ calculation**: Economic Order Quantity
- **Actionable recommendations**: Plain language suggestions

#### 6. Guardrails & Compliance
- **PII redaction**: Email, phone, SSN, credit cards
- **Hallucination control**: Citation verification
- **Confidence scoring**: Transparency in responses
- **Warning badges**: Low evidence, outdated docs, etc.

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                              │
│                     (Streamlit Chat + Source Panel)                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              FastAPI BACKEND                             │
│                    (/ask, /search, /forecast, /upload)                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│   LLM ORCHESTRATOR  │ │  HYBRID RETRIEVER   │ │   ML FORECASTER     │
│  - Prompt assembly  │ │  - Dense search     │ │  - Prophet          │
│  - Citation extract │ │  - Sparse search    │ │  - XGBoost          │
│  - Guardrails       │ │  - Re-ranking       │ │  - Safety stock     │
└─────────────────────┘ └─────────────────────┘ └─────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│    VECTOR STORE     │ │    BM25 INDEX       │ │   EMBEDDING MODEL   │
│    (ChromaDB)       │ │  (rank-bm25)        │ │ (all-MiniLM-L6-v2)  │
└─────────────────────┘ └─────────────────────┘ └─────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          INGESTION PIPELINE                              │
│           (Parsers → Chunker → Entity Extraction → Indexing)             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           DOCUMENT SOURCES                               │
│              (PDF, DOCX, XLSX, CSV, SharePoint, S3, etc.)               │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. Document Ingestion
   └─> Parse (PDF/DOCX/XLSX)
       └─> Chunk (semantic/table-aware)
           └─> Extract entities (SKU, warehouse, supplier)
               └─> Generate embeddings
                   └─> Store in ChromaDB + BM25 index

2. Query Processing
   └─> User query
       └─> Detect intent (policy/metrics/inventory)
           └─> Hybrid retrieval (dense + sparse)
               └─> Re-rank results
                   └─> Assemble context + forecast (if needed)
                       └─> LLM synthesis with citations
                           └─> Apply guardrails
                               └─> Return response with sources
```

---

## Project Structure

```
Supply Chain GPT/
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── config.py                     # Configuration management
│   │
│   ├── models/                       # Data models
│   │   ├── __init__.py
│   │   └── schemas.py                # Pydantic schemas
│   │
│   ├── ingestion/                    # Document processing
│   │   ├── __init__.py
│   │   ├── parsers.py                # Multi-format document parsers
│   │   ├── chunker.py                # Text chunking strategies
│   │   └── pipeline.py               # End-to-end ingestion pipeline
│   │
│   ├── embeddings/                   # Vector operations
│   │   ├── __init__.py
│   │   ├── embedding_service.py      # Embedding generation & similarity
│   │   └── vector_store.py           # ChromaDB wrapper
│   │
│   ├── retrieval/                    # Search & retrieval
│   │   ├── __init__.py
│   │   ├── bm25_index.py             # Sparse retrieval index
│   │   └── hybrid_retriever.py       # Combined retrieval system
│   │
│   ├── forecasting/                  # ML predictions
│   │   ├── __init__.py
│   │   ├── demand_forecaster.py      # Prophet + XGBoost forecasting
│   │   └── safety_stock.py           # Inventory optimization
│   │
│   ├── llm/                          # LLM integration
│   │   ├── __init__.py
│   │   ├── prompts.py                # Prompt templates
│   │   └── orchestrator.py           # RAG orchestration
│   │
│   └── api/                          # REST API
│       ├── __init__.py
│       ├── main.py                   # FastAPI application
│       ├── routes.py                 # API endpoints
│       └── dependencies.py           # Dependency injection
│
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── conftest.py                   # Pytest fixtures
│   ├── test_ingestion.py             # Ingestion tests (25 tests)
│   ├── test_embeddings.py            # Embedding tests (18 tests)
│   ├── test_retrieval.py             # Retrieval tests (20 tests)
│   ├── test_forecasting.py           # Forecasting tests (22 tests)
│   ├── test_llm.py                   # LLM tests (15 tests)
│   └── test_api.py                   # API tests (20 tests)
│
├── data/                             # Data directory
│   ├── chroma_db/                    # Vector database (auto-created)
│   ├── documents/                    # Uploaded documents (auto-created)
│   ├── forecasts/                    # Forecast outputs (auto-created)
│   └── sample_data/                  # Sample documents
│       ├── inventory_policy.txt
│       ├── supplier_contracts.txt
│       └── warehouse_sop.txt
│
├── streamlit_app.py                  # Streamlit UI application
├── run_api.py                        # API server runner
├── run_streamlit.py                  # Streamlit runner
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package setup
├── pytest.ini                        # Pytest configuration
├── .env.example                      # Environment template
└── README.md                         # This file
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- 4GB+ RAM recommended
- OpenAI API key (for GPT-4 integration)

### Step-by-Step Installation

#### 1. Clone or navigate to the project

```bash
cd "Supply Chain GPT"
```

#### 2. Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Install the package in development mode (optional)

```bash
pip install -e ".[dev]"
```

### Dependency Overview

| Package | Purpose |
|---------|---------|
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server |
| `streamlit` | Web UI framework |
| `sentence-transformers` | Text embeddings |
| `chromadb` | Vector database |
| `openai` | GPT-4 integration |
| `prophet` | Time series forecasting |
| `xgboost` | Gradient boosting |
| `pdfminer.six` | PDF parsing |
| `python-docx` | DOCX parsing |
| `openpyxl` | Excel parsing |
| `rank-bm25` | Sparse retrieval |
| `plotly` | Visualizations |

---

## Configuration

### Environment Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# OpenAI Configuration (Required for full functionality)
OPENAI_API_KEY=sk-your-api-key-here

# Application Settings
APP_ENV=development
DEBUG=true

# Vector DB Settings
CHROMA_PERSIST_DIR=./data/chroma_db
DOCS_DIR=./data/documents

# Model Settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=gpt-4-turbo-preview
LLM_TEMPERATURE=0.1

# Retrieval Settings
TOP_K_RETRIEVAL=8
CHUNK_SIZE=750
CHUNK_OVERLAP=120
```

### Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | None | OpenAI API key for GPT-4 |
| `APP_ENV` | development | Environment (development/production) |
| `DEBUG` | true | Enable debug logging |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence transformer model |
| `LLM_MODEL` | gpt-4-turbo-preview | OpenAI model to use |
| `LLM_TEMPERATURE` | 0.1 | LLM response temperature |
| `TOP_K_RETRIEVAL` | 8 | Number of chunks to retrieve |
| `CHUNK_SIZE` | 750 | Target tokens per chunk |
| `CHUNK_OVERLAP` | 120 | Overlap between chunks |

---

## Usage

### Starting the Application

#### Option 1: Run both API and UI

**Terminal 1 - Start the API server:**
```bash
python run_api.py
```

**Terminal 2 - Start Streamlit UI:**
```bash
python run_streamlit.py
```

#### Option 2: Run API only

```bash
python run_api.py
# or
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Option 3: Run Streamlit only (requires API running)

```bash
streamlit run streamlit_app.py
```

### Access Points

- **API Documentation**: http://localhost:8000/docs
- **Streamlit UI**: http://localhost:8501
- **API Health Check**: http://localhost:8000/api/v1/health

### Using the Chat Interface

1. **Open the Streamlit UI** at http://localhost:8501

2. **Upload Documents** (sidebar):
   - Click "Choose a file"
   - Select PDF, DOCX, XLSX, CSV, or TXT
   - Choose document type
   - Click "Upload"

3. **Ask Questions**:
   - Type in the chat input
   - Examples:
     - "What is the safety stock policy for SKU-12345?"
     - "What are the SLA terms with supplier SUP-001?"
     - "Show me the receiving procedure for warehouse WH-001"
     - "What is the demand forecast for next month?"

4. **View Sources** (right panel):
   - See retrieved document excerpts
   - Check relevance scores
   - Access source documents

5. **Get Forecasts**:
   - Enter SKU ID in sidebar
   - Click "Get Forecast"
   - View demand chart and safety stock recommendations

---

## API Reference

### Base URL

```
http://localhost:8000/api/v1
```

### Endpoints

#### Query Endpoints

##### POST `/ask`
Full-featured question answering with all options.

**Request Body:**
```json
{
  "query": "What is the reorder point for SKU-12345?",
  "user_context": {
    "user_id": "user123",
    "tenant_id": "default",
    "roles": ["planner", "viewer"]
  },
  "filters": {
    "doc_type": "policy"
  },
  "top_k": 8,
  "include_forecast": true,
  "sku_id": "SKU-12345",
  "warehouse_id": "WH-001"
}
```

**Response:**
```json
{
  "answer": "Based on the inventory policy, the reorder point for SKU-12345 is 900 units...",
  "citations": [
    {
      "doc_title": "Inventory Policy",
      "source_uri": "/docs/inventory_policy.pdf",
      "chunk_text": "The reorder point for SKU-12345 is 900 units...",
      "relevance_score": 0.92
    }
  ],
  "confidence": 0.85,
  "forecast_data": {...},
  "processing_time_ms": 1250.5,
  "warning_badges": []
}
```

##### POST `/ask/simple`
Simplified endpoint for quick queries.

**Query Parameters:**
- `query` (required): The question
- `user_id` (optional): User identifier
- `include_forecast` (optional): Include forecast data
- `sku_id` (optional): SKU for forecast

---

#### Document Endpoints

##### POST `/documents/upload`
Upload and ingest a document.

**Form Data:**
- `file`: The document file
- `doc_type` (optional): Document type (policy, sop, contract, report, manual, export)

**Response:**
```json
{
  "doc_id": "uuid-here",
  "source_uri": "filename.pdf",
  "status": "success",
  "chunks_created": 15,
  "processing_time_ms": 2500.0
}
```

##### DELETE `/documents/{doc_id}`
Delete a document and all its chunks.

##### GET `/documents/stats`
Get statistics about indexed documents.

**Response:**
```json
{
  "collection_name": "supplychain_docs",
  "total_chunks": 1250
}
```

---

#### Search Endpoints

##### POST `/search`
Search for relevant document chunks.

**Query Parameters:**
- `query` (required): Search query
- `k` (optional): Number of results (default: 8)
- `doc_type` (optional): Filter by document type

**Response:**
```json
{
  "query": "safety stock calculation",
  "results": [
    {
      "chunk_id": "uuid",
      "chunk_text": "Safety stock = Z × σd × √L...",
      "doc_title": "Inventory Policy",
      "doc_type": "policy",
      "score": 0.89,
      "dense_score": 0.92,
      "sparse_score": 0.85
    }
  ],
  "total_results": 5
}
```

---

#### Forecast Endpoints

##### POST `/forecast`
Get demand forecast for a SKU.

**Request Body:**
```json
{
  "sku_id": "SKU-12345",
  "warehouse_id": "WH-001",
  "periods": 30,
  "include_safety_stock": true
}
```

**Response:**
```json
{
  "sku_id": "SKU-12345",
  "warehouse_id": "WH-001",
  "forecast": [
    {"date": "2024-01-15", "predicted_demand": 105.5},
    {"date": "2024-01-16", "predicted_demand": 98.2}
  ],
  "safety_stock": 200.0,
  "reorder_point": 500.0,
  "mape": 8.5,
  "model_used": "ensemble"
}
```

##### GET `/forecast/{sku_id}/summary`
Get a text summary of the forecast.

---

#### Feedback Endpoints

##### POST `/feedback`
Submit feedback on a response.

**Request Body:**
```json
{
  "query_id": "uuid",
  "helpful": true,
  "error_type": null,
  "comment": "Very helpful response"
}
```

---

#### Health Endpoints

##### GET `/health`
Basic health check.

##### GET `/health/detailed`
Detailed health check with component status.

---

## Module Documentation

### 1. Ingestion Module (`src/ingestion/`)

#### DocumentParser (`parsers.py`)

Parses documents from various formats into plain text.

**Supported Formats:**
- PDF (with OCR fallback)
- DOCX/DOC
- PPTX
- XLSX/XLS
- CSV
- TXT/MD

**Key Methods:**
```python
parser = DocumentParser()

# Parse a file
text, metadata = parser.parse(Path("document.pdf"))

# Parse from bytes (file upload)
text, metadata = parser.parse_bytes(content, "filename.pdf")

# Extract entities
entities = parser.extract_entities(text)
# Returns: {"sku_ids": [...], "warehouse_ids": [...], "supplier_ids": [...], "dates": [...]}
```

#### TextChunker (`chunker.py`)

Splits text into semantic chunks for optimal retrieval.

**Chunking Strategies:**
- **Heading-based**: For policies, SOPs, manuals
- **Table-aware**: Preserves tables as single units
- **Sliding window**: Default with overlap

**Configuration:**
```python
config = ChunkConfig(
    max_tokens=750,      # Target chunk size
    overlap=120,         # Overlap between chunks
    min_chunk_size=100,  # Minimum chunk size
    preserve_tables=True
)
chunker = TextChunker(config)
chunks = chunker.chunk(text, doc_type="policy")
```

#### IngestionPipeline (`pipeline.py`)

End-to-end document processing pipeline.

```python
pipeline = IngestionPipeline()

# Ingest a single file
document = pipeline.ingest_file(
    file_path=Path("policy.pdf"),
    doc_type=DocumentType.POLICY,
    acl=ACL(roles=["planner"])
)

# Ingest a directory
documents = pipeline.ingest_directory(
    directory=Path("./docs"),
    recursive=True
)
```

---

### 2. Embeddings Module (`src/embeddings/`)

#### EmbeddingService (`embedding_service.py`)

Generates text embeddings using sentence transformers.

```python
service = EmbeddingService(model_name="all-MiniLM-L6-v2")

# Single text
embedding = service.embed_text("Safety stock calculation")

# Multiple texts
embeddings = service.embed_texts(["text1", "text2", "text3"])

# Compute similarity
similarity = service.compute_similarity(emb1, emb2)
```

#### ReRanker (`embedding_service.py`)

Cross-encoder re-ranker for improving retrieval precision.

```python
reranker = ReRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

# Re-rank documents
ranked = reranker.rerank(
    query="safety stock policy",
    documents=["doc1 text", "doc2 text", "doc3 text"],
    top_k=5
)
```

#### VectorStore (`vector_store.py`)

ChromaDB wrapper for vector storage and retrieval.

```python
store = VectorStore(
    persist_directory="./data/chroma_db",
    collection_name="supplychain_docs"
)

# Add chunks
store.add_chunks(chunks)

# Search
results = store.search(
    query="reorder point calculation",
    k=8,
    filters={"doc_type": "policy"}
)

# Search with ACL
results = store.search_with_acl(
    query="confidential policy",
    user_roles=["admin"],
    tenant_id="acme"
)
```

---

### 3. Retrieval Module (`src/retrieval/`)

#### BM25Index (`bm25_index.py`)

Sparse keyword-based retrieval using BM25 algorithm.

```python
index = EnhancedBM25Index()

# Add documents
index.add_documents(documents, text_key="chunk_text")

# Search
results = index.search("SKU-12345 safety stock", k=10)

# Enhanced search with boosting
results = index.search_enhanced(
    query="SKU-12345 inventory policy",
    boost_exact_match=True,
    boost_phrases=True
)
```

#### HybridRetriever (`hybrid_retriever.py`)

Combines dense and sparse retrieval with business logic.

```python
retriever = HybridRetriever(
    vector_store=vector_store,
    bm25_index=bm25_index,
    reranker=reranker,
    alpha=0.6,   # Dense weight
    beta=0.3,    # Sparse weight
    gamma=0.1    # Freshness weight
)

# Basic retrieval
results = retriever.retrieve(
    query="What is the safety stock?",
    k=8,
    filters={"doc_type": "policy"}
)

# With automatic intent detection
results = retriever.retrieve_with_intent(
    query="Show me the SLA penalties",
    k=8,
    user_roles=["planner"]
)
```

**Scoring Formula:**
```
final_score = α × dense_score + β × sparse_score + γ × freshness + doc_type_boost
```

---

### 4. Forecasting Module (`src/forecasting/`)

#### DemandForecaster (`demand_forecaster.py`)

Time series forecasting using Prophet and XGBoost.

```python
forecaster = DemandForecaster(use_prophet=True, use_xgboost=True)

# Prepare historical data
historical_data = pd.DataFrame({
    'ds': date_range,
    'y': demand_values
})

# Generate forecast
result = forecaster.forecast(
    historical_data=historical_data,
    periods=30,
    sku_id="SKU-12345",
    warehouse_id="WH-001"
)

# Get text summary
summary = forecaster.forecast_to_text(result)
```

**Output:**
```python
{
    "sku_id": "SKU-12345",
    "warehouse_id": "WH-001",
    "forecast": [
        {"date": "2024-01-15", "predicted_demand": 105.5, "lower_bound": 85.0, "upper_bound": 126.0},
        ...
    ],
    "model_used": "ensemble",
    "mape": 8.5,
    "all_models": {"prophet": [...], "xgboost": [...], "ensemble": [...]}
}
```

#### SafetyStockCalculator (`safety_stock.py`)

Inventory optimization calculations.

```python
calculator = SafetyStockCalculator(default_service_level=0.95)

# Calculate safety stock
result = calculator.calculate(
    demand_data=demand_df,
    lead_time_days=7,
    lead_time_std=2,
    service_level=0.95,
    method="service_level"
)

# Calculate EOQ
eoq = calculator.calculate_economic_order_quantity(
    annual_demand=10000,
    ordering_cost=50,
    holding_cost_per_unit=2
)
```

**Safety Stock Formula:**
```
Statistical: Safety Stock = Z × σd × √L
Service Level: Safety Stock = Z × √(L × σd² + d² × σL²)
```

---

### 5. LLM Module (`src/llm/`)

#### PromptTemplates (`prompts.py`)

Structured prompt templates for the LLM.

```python
# Build query prompt
prompt = PromptTemplates.build_query_prompt(
    question="What is the safety stock?",
    chunks=retrieved_chunks,
    forecast_result=forecast_data,
    safety_stock_result=safety_data
)

# Extract citations from response
citations = PromptTemplates.extract_citations(response, chunks)
```

#### LLMOrchestrator (`orchestrator.py`)

Orchestrates the complete RAG pipeline.

```python
orchestrator = LLMOrchestrator(
    retriever=hybrid_retriever,
    forecaster=demand_forecaster,
    safety_calculator=safety_calculator,
    llm_model="gpt-4-turbo-preview",
    temperature=0.1,
    openai_api_key="sk-..."
)

# Answer a query
request = QueryRequest(
    query="What is the reorder point for SKU-12345?",
    user_context=UserContext(user_id="user1", roles=["planner"]),
    include_forecast=True,
    sku_id="SKU-12345"
)

response = orchestrator.answer(request)
```

**Pipeline Flow:**
1. Retrieve relevant chunks
2. Get forecast data (if requested)
3. Build prompt with context
4. Generate LLM response
5. Extract citations
6. Apply guardrails
7. Calculate confidence
8. Return structured response

---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_ingestion.py -v

# Run specific test class
pytest tests/test_forecasting.py::TestDemandForecaster -v

# Run specific test
pytest tests/test_api.py::TestQueryEndpoints::test_ask_simple_endpoint -v
```

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| Ingestion | 25 | 90%+ |
| Embeddings | 18 | 85%+ |
| Retrieval | 20 | 85%+ |
| Forecasting | 22 | 90%+ |
| LLM | 15 | 80%+ |
| API | 20 | 85%+ |
| **Total** | **120** | **85%+** |

### Test Categories

#### Unit Tests
- Parser functionality
- Chunking strategies
- Embedding operations
- BM25 indexing
- Forecasting algorithms
- Safety stock calculations

#### Integration Tests
- Hybrid retrieval pipeline
- LLM orchestration flow
- API endpoint behavior

#### Mock Tests
- OpenAI API calls
- Vector store operations
- External dependencies

---

## Sample Data

### Included Sample Documents

#### 1. Inventory Policy (`inventory_policy.txt`)
- ABC classification guidelines
- Reorder point calculations
- Safety stock formulas
- Service level targets
- Warehouse-specific rules

#### 2. Supplier Contracts (`supplier_contracts.txt`)
- SUP-001, SUP-002, SUP-003 details
- Pricing terms
- Delivery SLAs
- Quality standards
- Penalty clauses

#### 3. Warehouse SOP (`warehouse_sop.txt`)
- Receiving procedures
- Picking methods
- Shipping processes
- Cycle counting
- Emergency procedures

### Loading Sample Data

```python
from pathlib import Path
from src.ingestion.pipeline import IngestionPipeline
from src.embeddings.vector_store import VectorStore

# Initialize
pipeline = IngestionPipeline()
store = VectorStore(persist_directory="./data/chroma_db")

# Ingest sample data
sample_dir = Path("./data/sample_data")
documents = pipeline.ingest_directory(sample_dir)

# Index documents
for doc in documents:
    store.add_document(doc)
```

---

## Performance Metrics

### Target Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Forecast MAPE | ≤10% | Mean Absolute Percentage Error |
| Retrieval Recall@10 | ≥85% | Relevant docs in top 10 |
| Response Latency | <5s | End-to-end response time |
| Hallucination Rate | <5% | Unsupported claims |
| User Satisfaction | ≥4.5/5 | Feedback score |

### Evaluation Methods

#### Retrieval Evaluation
- **Recall@k**: Percentage of relevant documents retrieved
- **MRR**: Mean Reciprocal Rank
- **Precision@k**: Accuracy of top-k results

#### LLM Evaluation
- **Faithfulness**: Response grounded in context
- **Factuality**: Correctness of statements
- **Citation Accuracy**: Sources properly attributed

#### Forecast Evaluation
- **MAPE**: Mean Absolute Percentage Error
- **RMSE**: Root Mean Square Error
- **Coverage**: Prediction interval accuracy

---

## Future Roadmap

### Phase 1: Foundation (Current)
- [x] Document ingestion pipeline
- [x] Hybrid retrieval system
- [x] Demand forecasting
- [x] Safety stock optimization
- [x] FastAPI backend
- [x] Streamlit UI
- [x] Comprehensive testing

### Phase 2: Enhancement
- [ ] Multi-language support (Hindi, Spanish)
- [ ] OCR optimization for scanned documents
- [ ] Real-time document sync (SharePoint, S3)
- [ ] Advanced re-ranking models
- [ ] User authentication & RBAC

### Phase 3: Enterprise
- [ ] ERP integration (SAP, Oracle)
- [ ] Voice-based assistant
- [ ] RL-based stock optimization
- [ ] A/B testing framework
- [ ] Production hardening

### Phase 4: Intelligence
- [ ] Anomaly detection
- [ ] Automated reordering
- [ ] Supplier risk scoring
- [ ] Demand sensing
- [ ] What-if simulations

---

## Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Make changes with tests
5. Run tests:
   ```bash
   pytest tests/ -v
   ```
6. Submit pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for public methods
- Maintain test coverage above 80%

### Commit Messages

```
feat: Add new forecasting model
fix: Resolve chunking edge case
docs: Update API documentation
test: Add retrieval tests
refactor: Simplify embedding service
```

---

## License

This project is licensed under the MIT License.

---

## Support

For issues, questions, or contributions:

1. Check existing documentation
2. Search closed issues
3. Open a new issue with:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details

---

## Acknowledgments

- **Sentence Transformers**: For embedding models
- **ChromaDB**: For vector storage
- **Prophet**: For time series forecasting
- **FastAPI**: For the API framework
- **Streamlit**: For the UI framework
- **OpenAI**: For GPT-4 integration

---

*Built with intelligence for intelligent supply chain management.*

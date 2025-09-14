# Atlan Customer Support Copilot

A modern Flask web application showcasing an AI-powered customer support system for Atlan, featuring automated ticket classification and intelligent response generation using RAG (Retrieval-Augmented Generation).

## Problem Summary

Customer support teams often face challenges with:
- **Manual ticket triage**: Time-consuming process of categorizing and prioritizing incoming tickets
- **Knowledge fragmentation**: Support agents need to search through multiple documentation sources
- **Inconsistent responses**: Varying quality and accuracy of responses across different agents
- **Scalability issues**: Difficulty handling increasing ticket volumes efficiently

This copilot addresses these challenges by:
1. **Automated Classification**: Instantly categorizes tickets by topic, sentiment, and priority
2. **Intelligent Routing**: Automatically routes tickets to appropriate teams
3. **RAG-powered Responses**: Generates accurate answers using Atlan's official documentation
4. **Consistent Experience**: Provides standardized, high-quality responses

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   Classifier     │    │   RAG Pipeline  │
│                 │    │                  │    │                 │
│ • Dashboard     │◄──►│ • OpenAI GPT     │    │ • Doc Crawler   │
│ • Chat Interface│    │ • Topic Tags     │    │ • Embeddings    │
│ • Filters       │    │ • Sentiment      │    │ • FAISS Index   │
└─────────────────┘    │ • Priority       │    │ • Answer Gen    │
                       └──────────────────┘    └─────────────────┘
                                │                        │
                                │                        │
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Sample Tickets   │    │ Knowledge Base  │
                       │ (CSV)            │    │                 │
                       │                  │    │ • docs.atlan.com│
                       └──────────────────┘    │ • developer.*   │
                                               └─────────────────┘
```

## Key Features

### 1. Bulk Ticket Classification Dashboard
- Loads and processes tickets from `sample_tickets.csv`
- Classifies each ticket with:
  - **Topic Tags**: How-to, Product, Connector, Lineage, API/SDK, SSO, Glossary, Best practices, Sensitive data
  - **Sentiment**: Frustrated, Curious, Angry, Neutral
  - **Priority**: P0 (urgent), P1 (high), P2 (medium/low)
- Interactive filtering by sentiment, priority, and topic
- Summary statistics and visualizations

### 2. Interactive AI Agent
- Text input for new ticket submission
- **Internal Analysis View**: Shows classification JSON
- **Final Response View**:
  - **RAG Response** for topics: How-to, Product, Best practices, API/SDK, SSO
  - **Routing Message** for other topics (Connector, Lineage, Glossary, Sensitive data)
- Source citation for all RAG-generated answers

## Design Decisions & Trade-offs

### Technology Choices

| Component | Choice | Rationale | Trade-offs |
|-----------|--------|-----------|------------|
| **UI Framework** | Streamlit | Rapid prototyping, built-in components | Limited customization vs React |
| **LLM Provider** | Azure OpenAI GPT-5 | Enterprise-grade, high performance, secure | API dependency, costs |
| **Embeddings** | SentenceTransformers | Local processing, good quality | Slower than hosted solutions |
| **Vector DB** | FAISS | Lightweight, no server needed | Less features than Pinecone/Weaviate |
| **Web Scraping** | BeautifulSoup + Requests | Simple, reliable | Basic crawling vs Scrapy |

### Architecture Decisions

1. **Modular Design**: Separated classifier and RAG into distinct modules for maintainability
2. **Local Vector Store**: FAISS for simplicity; can be replaced with cloud solutions for production
3. **Simple Crawling**: Basic web scraping suitable for demo; production would need more robust crawling
4. **Session State**: Streamlit session state for caching models and data
5. **Fallback Responses**: Graceful degradation when APIs fail

### Limitations & Future Improvements

**Current Limitations:**
- Limited crawling depth (30 pages max)
- No real-time document updates
- Basic text extraction from HTML
- Single-threaded processing
- No user authentication

**Production Enhancements:**
- Scheduled document re-crawling
- Advanced text preprocessing
- Multi-modal support (images, PDFs)
- User feedback loop for model improvement
- A/B testing framework
- Performance monitoring

## Setup Instructions

### Prerequisites
- Python 3.8+
- Azure OpenAI service with GPT-5 deployment
- Azure OpenAI API key and endpoint

### Local Development

1. **Clone and Setup**
   ```bash
   git clone <repository>
   cd atlan-support-copilot
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   # Copy the environment template
   cp env_template.txt .env
   
   # Edit .env and add your Azure OpenAI credentials
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_DEPLOYMENT=gpt-5-chat
   AZURE_OPENAI_API_VERSION=2025-01-01-preview
   ```

3. **Run the Application**
   ```bash
   python run.py
   # OR
   python app.py
   ```

4. **Access the Demo**
   - Open http://localhost:5000
   - Navigate between "Bulk Classification", "AI Agent", and "Analytics" tabs
   - Click "Load and Classify Sample Tickets" to see the dashboard
   - Try the interactive agent with sample queries

### Sample Queries to Test

**How-to Questions:**
- "How do I connect Snowflake to Atlan?"
- "What are the steps to set up data lineage?"

**Product Questions:**
- "What features does Atlan offer for data governance?"
- "How does Atlan handle metadata management?"

**API/SDK Questions:**
- "How do I use the Python SDK to query assets?"
- "What are the rate limits for the Atlan API?"

### Deployment Options

#### Option 1: Streamlit Cloud
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Add environment variables in dashboard
4. Deploy automatically

#### Option 2: Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Option 3: Cloud Platforms
- **AWS**: EC2 + Application Load Balancer
- **GCP**: Cloud Run or App Engine
- **Azure**: Container Instances or App Service

### Environment Variables
- `AZURE_OPENAI_API_KEY`: Required for classification and answer generation
- `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint URL
- `AZURE_OPENAI_DEPLOYMENT`: Deployment name (default: gpt-5-chat)
- `AZURE_OPENAI_API_VERSION`: API version (default: 2025-01-01-preview)

## File Structure

```
atlan-support-copilot/
├── app.py                 # Main Streamlit application
├── classifier.py          # Ticket classification logic
├── rag.py                # RAG pipeline implementation
├── sample_tickets.csv     # Sample ticket data
├── requirements.txt       # Python dependencies
├── env_template.txt       # Environment variables template
├── README.md             # This documentation
└── rag_index.*           # Generated FAISS index files (after first run)
```

## Usage Examples

### Bulk Classification
1. Load the dashboard
2. Click "Load and Classify Sample Tickets"
3. Use filters to explore different ticket categories
4. Review classification accuracy and patterns

### Interactive Agent
1. Navigate to "Interactive AI Agent"
2. Enter a ticket subject and description
3. View both internal analysis and final response
4. Test different types of queries to see RAG vs routing behavior

## Contributing

This is a demo application. For production use, consider:
- Adding comprehensive error handling
- Implementing user authentication
- Adding monitoring and logging
- Scaling the vector database
- Improving the crawling strategy
- Adding unit tests

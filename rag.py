# Atlan RAG (Retrieval-Augmented Generation) Pipeline
# This module implements a comprehensive RAG system for Atlan customer support
# It crawls official Atlan documentation, builds a searchable index, and generates
# context-aware responses using Azure OpenAI and TF-IDF similarity matching.

import requests
from bs4 import BeautifulSoup
import openai
import os
import pickle
from dotenv import load_dotenv
from urllib.parse import urljoin, urlparse
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

class RAGPipeline:
    """
    RAG Pipeline for Atlan Customer Support
    
    This class implements a complete RAG system that:
    1. Crawls official Atlan documentation websites
    2. Builds a TF-IDF based search index
    3. Retrieves relevant documents for user queries
    4. Generates context-aware responses using Azure OpenAI
    """
    
    def __init__(self):
        """Initialize the RAG pipeline with Azure OpenAI configuration"""
        # Load Azure OpenAI configuration from environment variables
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-5-chat')
        api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2025-01-01-preview')
        
        # Initialize Azure OpenAI client if credentials are available
        if not api_key or not endpoint:
            print("⚠️  Warning: Azure OpenAI credentials not set. RAG responses will be limited.")
            self.client = None
        else:
            try:
                self.client = openai.AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=endpoint
                )
                self.deployment = deployment
                print(f"✅ Azure OpenAI RAG client initialized with deployment: {deployment}")
            except Exception as e:
                print(f"❌ Failed to initialize Azure OpenAI RAG client: {e}")
                print("   Please check your Azure OpenAI configuration")
                self.client = None
        
        # Initialize TF-IDF vectorizer for document similarity
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.document_vectors = None  # Will store TF-IDF vectors after indexing
        self.documents = []           # List of crawled document texts
        self.urls = []               # List of corresponding URLs
        
    def crawl_documentation(self, base_urls, max_pages=20):
        """
        Crawl multiple Atlan documentation websites
        
        Args:
            base_urls (list): List of base URLs to crawl (e.g., docs.atlan.com, developer.atlan.com)
            max_pages (int): Maximum number of pages to crawl across all sites
        """
        print("Crawling documentation...")
        all_docs = []
        all_urls = []
        
        # Distribute max_pages across all base URLs
        for base_url in base_urls:
            docs, urls = self._crawl_single_site(base_url, max_pages // len(base_urls))
            all_docs.extend(docs)
            all_urls.extend(urls)
            
        # Store crawled content for indexing
        self.documents = all_docs
        self.urls = all_urls
        print(f"Crawled {len(self.documents)} documents")
        
    def _crawl_single_site(self, base_url, max_pages):
        """
        Enhanced crawler for a single Atlan documentation site
        
        Args:
            base_url (str): Base URL of the documentation site
            max_pages (int): Maximum number of pages to crawl from this site
            
        Returns:
            tuple: (documents, urls) - Lists of extracted content and corresponding URLs
        """
        docs = []
        urls = []
        visited = set()
        
        # Pre-defined high-value pages for comprehensive Atlan documentation coverage
        if 'docs.atlan.com' in base_url:
            to_visit = [
                # Core documentation pages
                'https://docs.atlan.com/',
                'https://docs.atlan.com/getting-started/',
                'https://docs.atlan.com/overview/',
                
                # Connectors - Critical for support tickets (most common issues)
                'https://docs.atlan.com/connectors/',
                'https://docs.atlan.com/connectors/snowflake/',
                'https://docs.atlan.com/connectors/snowflake/setup/',
                'https://docs.atlan.com/connectors/mysql/',
                'https://docs.atlan.com/connectors/mysql/setup/',
                'https://docs.atlan.com/connectors/bigquery/',
                'https://docs.atlan.com/connectors/bigquery/setup/',
                'https://docs.atlan.com/connectors/postgres/',
                'https://docs.atlan.com/connectors/databricks/',
                'https://docs.atlan.com/connectors/tableau/',
                'https://docs.atlan.com/connectors/looker/',
                
                # Core features documentation
                'https://docs.atlan.com/features/',
                'https://docs.atlan.com/features/lineage/',
                'https://docs.atlan.com/features/search/',
                'https://docs.atlan.com/features/discovery/',
                
                # Governance and compliance features
                'https://docs.atlan.com/governance/',
                'https://docs.atlan.com/governance/glossary/',
                'https://docs.atlan.com/governance/classification/',
                'https://docs.atlan.com/governance/policies/',
                
                # Administration and user management
                'https://docs.atlan.com/administration/',
                'https://docs.atlan.com/administration/sso/',
                'https://docs.atlan.com/administration/sso/saml/',
                'https://docs.atlan.com/administration/sso/oidc/',
                'https://docs.atlan.com/administration/users/',
                'https://docs.atlan.com/administration/permissions/',
                
                # Troubleshooting guides (valuable for support)
                'https://docs.atlan.com/troubleshooting/',
                'https://docs.atlan.com/troubleshooting/connectors/',
                'https://docs.atlan.com/troubleshooting/performance/',
                'https://docs.atlan.com/troubleshooting/common-issues/'
            ]
        elif 'developer.atlan.com' in base_url:
            to_visit = [
                # Core API documentation and getting started guides
                'https://developer.atlan.com/',
                'https://developer.atlan.com/getting-started/',
                
                # Comprehensive API endpoint documentation
                'https://developer.atlan.com/api/',
                'https://developer.atlan.com/api/overview/',
                'https://developer.atlan.com/api/authentication/',
                'https://developer.atlan.com/api/rate-limits/',
                'https://developer.atlan.com/api/assets/',
                'https://developer.atlan.com/api/lineage/',
                'https://developer.atlan.com/api/search/',
                'https://developer.atlan.com/api/glossary/',
                
                # SDK documentation for different programming languages
                'https://developer.atlan.com/sdk/',
                'https://developer.atlan.com/sdk/python/',
                'https://developer.atlan.com/sdk/python/quickstart/',
                'https://developer.atlan.com/sdk/python/examples/',
                'https://developer.atlan.com/sdk/java/',
                'https://developer.atlan.com/sdk/javascript/',
                
                # Integration guides and best practices
                'https://developer.atlan.com/guides/',
                'https://developer.atlan.com/guides/integration/',
                'https://developer.atlan.com/guides/best-practices/',
                
                # Code examples and tutorials
                'https://developer.atlan.com/examples/',
                'https://developer.atlan.com/tutorials/'
            ]
        else:
            to_visit = [base_url]
        
        # Set up HTTP session with proper headers to avoid blocking
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Crawl pages using breadth-first approach
        while to_visit and len(docs) < max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue
                
            visited.add(url)
            
            try:
                print(f"Crawling: {url}")
                response = session.get(url, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract structured content optimized for documentation sites
                    text = self._extract_documentation_content(soup, url)
                    if text and len(text) > 200:  # Only include substantial content
                        docs.append(text)
                        urls.append(url)
                        print(f"✓ Extracted {len(text)} characters from {url}")
                        
                    # Discover additional relevant links for comprehensive coverage
                    if len(docs) < max_pages:
                        links = self._find_relevant_links(soup, base_url)
                        for link_url in links[:3]:  # Limit to 3 new links per page to avoid explosion
                            if link_url not in visited and link_url not in to_visit:
                                to_visit.append(link_url)
                                
                time.sleep(1)  # Be respectful to the server with rate limiting
                
            except Exception as e:
                print(f"Error crawling {url}: {e}")
                continue
                
        print(f"Crawled {len(docs)} pages from {base_url}")
        return docs, urls
    
    def _extract_documentation_content(self, soup, url):
        """
        Extract structured content from documentation pages
        
        Args:
            soup (BeautifulSoup): Parsed HTML content
            url (str): Source URL for context
            
        Returns:
            str: Cleaned and structured text content
        """
        # Remove unnecessary elements that don't contain useful content
        for element in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
            element.decompose()
        
        # Try to find main content areas common in documentation sites
        content_selectors = [
            'main', '.main-content', '.content', '.documentation',
            '.docs-content', '.article', '.post-content', '#content',
            '.markdown-body', '.prose', '.doc-content'
        ]
        
        main_content = None
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # Fallback to body if no main content area found
        if not main_content:
            main_content = soup.find('body')
        
        if not main_content:
            return ""
        
        # Extract structured content with hierarchy preservation
        content_parts = []
        
        # Extract page title for context
        title = soup.find('title')
        if title:
            content_parts.append(f"Title: {title.get_text().strip()}")
        
        # Process different HTML elements for structured extraction
        for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'div']):
            if element.name in ['h1', 'h2', 'h3', 'h4']:
                # Extract headings with hierarchy markers
                text = element.get_text().strip()
                if text and len(text) < 200:  # Reasonable heading length
                    content_parts.append(f"\n{element.name.upper()}: {text}")
            elif element.name in ['p', 'li']:
                # Extract paragraphs and list items
                text = element.get_text().strip()
                if text and len(text) > 20 and len(text) < 1000:  # Reasonable paragraph length
                    content_parts.append(text)
            elif element.name == 'div' and element.get('class'):
                # Look for code blocks, examples, and special content
                classes = ' '.join(element.get('class', []))
                if any(keyword in classes.lower() for keyword in ['code', 'example', 'highlight', 'snippet']):
                    text = element.get_text().strip()
                    if text and len(text) > 10:
                        content_parts.append(f"Code/Example: {text}")
        
        # Join and clean content
        full_content = '\n'.join(content_parts)
        
        # Clean up whitespace and empty lines
        lines = [line.strip() for line in full_content.split('\n') if line.strip()]
        cleaned_content = '\n'.join(lines)
        
        # Add URL context for source tracking
        final_content = f"Source URL: {url}\n\n{cleaned_content}"
        
        return final_content[:3000]  # Limit chunk size but allow more for documentation
    
    def _find_relevant_links(self, soup, base_url):
        """
        Find relevant documentation links to crawl based on content relevance
        
        Args:
            soup (BeautifulSoup): Parsed HTML content
            base_url (str): Base URL for relative link resolution
            
        Returns:
            list: List of relevant URLs to crawl
        """
        relevant_links = []
        base_domain = urlparse(base_url).netloc
        
        # Look for links in navigation and content areas
        link_areas = soup.find_all(['nav', 'main', '.sidebar', '.toc', '.navigation'])
        if not link_areas:
            link_areas = [soup]  # Fallback to entire page
        
        for area in link_areas:
            links = area.find_all('a', href=True)
            for link in links:
                href = link.get('href')
                if not href:
                    continue
                    
                # Convert relative URLs to absolute URLs
                full_url = urljoin(base_url, href)
                parsed_url = urlparse(full_url)
                
                # Only crawl same domain to stay within documentation
                if parsed_url.netloc != base_domain:
                    continue
                
                # Skip binary files and media
                if any(full_url.lower().endswith(ext) for ext in ['.pdf', '.zip', '.jpg', '.png', '.gif']):
                    continue
                
                # Clean URL by removing fragments
                clean_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
                if parsed_url.query:
                    clean_url += f"?{parsed_url.query}"
                
                # Filter for documentation-relevant URLs using keywords
                relevant_keywords = [
                    'connector', 'api', 'sdk', 'guide', 'tutorial', 'setup',
                    'configuration', 'authentication', 'lineage', 'governance',
                    'glossary', 'classification', 'troubleshooting', 'sso'
                ]
                
                # Check if URL or link text contains relevant keywords
                if (any(keyword in clean_url.lower() for keyword in relevant_keywords) or
                    any(keyword in link.get_text().lower() for keyword in relevant_keywords)):
                    if clean_url not in relevant_links:
                        relevant_links.append(clean_url)
        
        return relevant_links[:10]  # Limit number of links to prevent explosion
    
    def _extract_text(self, soup):
        """
        Fallback text extraction method
        """
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text and clean it up
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:2000]  # Limit chunk size
    
    def _should_crawl(self, url, base_url):
        """
        Simple check if we should crawl this URL
        """
        parsed_base = urlparse(base_url)
        parsed_url = urlparse(url)
        
        # Only crawl same domain
        return (parsed_url.netloc == parsed_base.netloc and 
                not url.endswith(('.pdf', '.zip', '.jpg', '.png')))
    
    def build_index(self):
        """
        Build TF-IDF index from crawled documents for similarity search
        
        This creates a searchable index that allows finding relevant documents
        based on term frequency and inverse document frequency.
        """
        if not self.documents:
            print("No documents to index. Please crawl first.")
            return
            
        print("Building TF-IDF index...")
        # Create TF-IDF vectors for all documents
        self.document_vectors = self.vectorizer.fit_transform(self.documents)
        
        print(f"Index built with {len(self.documents)} documents")
        
    def save_index(self, path="rag_index"):
        """
        Save the index and documents to disk for persistence
        
        Args:
            path (str): Base path for saving the index files
        """
        if self.document_vectors is None:
            return
            
        # Save all components needed to restore the RAG pipeline
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump({
                'documents': self.documents,           # Original document texts
                'urls': self.urls,                     # Corresponding URLs
                'vectorizer': self.vectorizer,         # TF-IDF vectorizer
                'document_vectors': self.document_vectors  # TF-IDF vectors
            }, f)
            
    def load_index(self, path="rag_index"):
        """
        Load the index and documents from disk
        
        Args:
            path (str): Base path for loading the index files
            
        Returns:
            bool: True if successfully loaded, False otherwise
        """
        try:
            with open(f"{path}.pkl", "rb") as f:
                data = pickle.load(f)
                # Restore all components of the RAG pipeline
                self.documents = data['documents']
                self.urls = data['urls']
                self.vectorizer = data['vectorizer']
                self.document_vectors = data['document_vectors']
            return True
        except:
            return False
    
    def retrieve_relevant_docs(self, query, top_k=3):
        """
        Retrieve most relevant documents for a query using TF-IDF similarity
        
        Args:
            query (str): User query to find relevant documents for
            top_k (int): Number of most relevant documents to return
            
        Returns:
            tuple: (relevant_docs, relevant_urls) - Lists of relevant documents and URLs
        """
        if self.document_vectors is None:
            return [], []
            
        # Transform query using the same TF-IDF vectorizer as documents
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity between query and all documents
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        
        # Get top-k most similar documents (sorted by similarity score)
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Filter out documents with zero similarity
        relevant_docs = [self.documents[i] for i in top_indices if similarities[i] > 0]
        relevant_urls = [self.urls[i] for i in top_indices if similarities[i] > 0]
        
        return relevant_docs, relevant_urls
    
    def generate_answer(self, query, topic_tags):
        """
        Generate answer using enhanced RAG pipeline with better context matching
        
        Args:
            query (str): User's question or issue
            topic_tags (list): AI-identified topic categories for the query
            
        Returns:
            tuple: (answer, sources) - Generated answer and source URLs
        """
        # Enhance query with topic-specific keywords for better retrieval
        enhanced_query = self._enhance_query(query, topic_tags)
        
        # Retrieve relevant documents using TF-IDF similarity
        docs, urls = self.retrieve_relevant_docs(enhanced_query, top_k=5)
        
        if not docs:
            return "I don't have access to the relevant documentation to answer this question. Please visit the official Atlan documentation at https://docs.atlan.com/ or https://developer.atlan.com/ for comprehensive information, or contact support for assistance.", []
        
        # Filter and rank documents by relevance to query and topics
        relevant_docs, relevant_urls = self._filter_relevant_docs(docs, urls, query, topic_tags)
        
        if not relevant_docs:
            return f"I couldn't find specific information about your question. For {', '.join(topic_tags)} related queries, please check https://docs.atlan.com/ or contact our specialist team.", []
        
        # Create structured context from relevant documents
        context = self._create_structured_context(relevant_docs, relevant_urls)
        
        # Create enhanced prompt for better AI responses
        prompt = f"""
        You are an expert Atlan customer support agent. Use the provided documentation to give accurate, helpful answers.
        
        CUSTOMER QUERY: {query}
        TOPIC CATEGORIES: {', '.join(topic_tags)}
        
        DOCUMENTATION CONTEXT:
        {context}
        
        INSTRUCTIONS:
        1. Provide a direct, actionable answer based ONLY on the documentation context
        2. Include specific steps, settings, or configurations when relevant
        3. If the context doesn't fully answer the question, acknowledge limitations
        4. Use clear, professional language appropriate for technical users
        5. Reference specific documentation sections when helpful
        6. For configuration issues, provide exact parameter names and values
        
        ANSWER:"""
        
        # Use fallback method if Azure OpenAI is not available
        if self.client is None:
            return self._generate_fallback_answer(relevant_docs, relevant_urls, query, topic_tags)
                
        try:
            # Generate response using Azure OpenAI
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {'role': 'system', 'content': 'You are an expert Atlan customer support agent. Use the provided documentation to give accurate, helpful answers.'},
                    {'role': 'user', 'content': prompt}
                ],
                temperature=0.2,  # Lower temperature for more consistent, factual answers
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content.strip()
            
            if not answer or answer.strip() == "":
                print("Warning: Empty response from Azure OpenAI API in RAG")
                raise ValueError("Empty response from API")
            
            # Post-process answer for better formatting and readability
            answer = self._format_answer(answer)
            
            return answer, relevant_urls
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            # Fallback to rule-based answer generation
            fallback_answer, fallback_urls = self._generate_fallback_answer(relevant_docs, relevant_urls, query, topic_tags)
            return f"I encountered an issue generating a response, but here's what I found: {fallback_answer}", fallback_urls
    
    def _enhance_query(self, query, topic_tags):
        """
        Enhance query with topic-specific keywords for better document retrieval
        
        Args:
            query (str): Original user query
            topic_tags (list): AI-identified topic categories
            
        Returns:
            str: Enhanced query with additional relevant keywords
        """
        # Topic-specific keyword mappings for better retrieval
        topic_keywords = {
            'Connector': ['connection', 'setup', 'configuration', 'credentials', 'authentication'],
            'API/SDK': ['api', 'endpoint', 'request', 'response', 'rate limit', 'token'],
            'SSO': ['authentication', 'login', 'saml', 'oidc', 'identity provider'],
            'Lineage': ['lineage', 'dependency', 'flow', 'tracking', 'upstream', 'downstream'],
            'How-to': ['setup', 'configure', 'install', 'guide', 'steps'],
            'Best practices': ['recommendation', 'best practice', 'governance', 'optimization'],
            'Sensitive data': ['classification', 'pii', 'security', 'compliance', 'sensitive']
        }
        
        enhanced = query
        for tag in topic_tags:
            if tag in topic_keywords:
                keywords = ' '.join(topic_keywords[tag][:2])  # Add top 2 keywords per topic
                enhanced += f" {keywords}"
        
        return enhanced
    
    def _filter_relevant_docs(self, docs, urls, query, topic_tags):
        """
        Filter and rank documents by relevance to query and topics
        
        Args:
            docs (list): Retrieved documents
            urls (list): Corresponding URLs
            query (str): Original user query
            topic_tags (list): AI-identified topic categories
            
        Returns:
            tuple: (filtered_docs, filtered_urls) - Most relevant documents and URLs
        """
        if not docs:
            return [], []
        
        relevant_pairs = []
        query_lower = query.lower()
        topic_keywords = [tag.lower() for tag in topic_tags]
        
        # Score each document based on multiple relevance factors
        for doc, url in zip(docs, urls):
            doc_lower = doc.lower()
            relevance_score = 0
            
            # Score based on query term frequency
            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 3:  # Skip short words (articles, prepositions)
                    relevance_score += doc_lower.count(word) * 2
            
            # Score based on topic relevance
            for topic in topic_keywords:
                if topic in doc_lower:
                    relevance_score += 5
            
            # Bonus for specific technical terms that indicate actionable content
            technical_terms = ['setup', 'configure', 'error', 'issue', 'solution', 'steps']
            for term in technical_terms:
                if term in query_lower and term in doc_lower:
                    relevance_score += 3
            
            if relevance_score > 0:
                relevant_pairs.append((doc, url, relevance_score))
        
        # Sort by relevance score and return top 3 most relevant documents
        relevant_pairs.sort(key=lambda x: x[2], reverse=True)
        relevant_pairs = relevant_pairs[:3]
        
        if relevant_pairs:
            relevant_docs = [pair[0] for pair in relevant_pairs]
            relevant_urls = [pair[1] for pair in relevant_pairs]
            return relevant_docs, relevant_urls
        
        # Fallback: return original docs if no scoring worked
        return docs[:3], urls[:3]
    
    def _create_structured_context(self, docs, urls):
        """
        Create well-structured context with source references for AI processing
        
        Args:
            docs (list): Relevant documents
            urls (list): Corresponding source URLs
            
        Returns:
            str: Structured context with source attribution
        """
        context_parts = []
        for i, (doc, url) in enumerate(zip(docs, urls), 1):
            # Extract section name from URL for better context
            section_name = url.split('/')[-1].replace('-', ' ').title()
            context_parts.append(f"[Source {i} - {section_name}]:\n{doc}\n")
        
        return "\n".join(context_parts)
    
    def _generate_fallback_answer(self, docs, urls, query, topic_tags):
        """
        Generate structured fallback answer using ONLY real crawled content
        
        This method is used when Azure OpenAI is not available, providing
        rule-based answer generation from crawled documentation.
        
        Args:
            docs (list): Retrieved documents
            urls (list): Corresponding URLs
            query (str): User query
            topic_tags (list): Topic categories
            
        Returns:
            tuple: (answer, sources) - Generated answer and source URLs
        """
        if not docs:
            return "I don't have access to the relevant documentation. Please visit https://docs.atlan.com/ or https://developer.atlan.com/ for comprehensive information.", []
        
        # Verify we have real crawled content (should contain "Source URL:")
        real_docs = [doc for doc in docs if "Source URL:" in doc]
        if not real_docs:
            return "I don't have access to the relevant documentation. Please visit https://docs.atlan.com/ or https://developer.atlan.com/ for comprehensive information.", []
        
        # Extract key information from the most relevant real document
        main_doc = real_docs[0]
        
        # Remove the "Source URL:" line for processing
        content_lines = main_doc.split('\n')
        content_start = 0
        for i, line in enumerate(content_lines):
            if line.startswith('Source URL:'):
                content_start = i + 2  # Skip URL line and empty line
                break
        
        content = '\n'.join(content_lines[content_start:])
        sentences = content.split('. ')
        
        # Find most relevant sentences using word overlap
        query_words = set(query.lower().split())
        relevant_sentences = []
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words.intersection(sentence_words))
            if overlap > 0:
                relevant_sentences.append((sentence.strip(), overlap))
        
        # Sort by relevance and take top sentences
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in relevant_sentences[:3] if s[0]]
        
        if top_sentences:
            answer = ". ".join(top_sentences)
            if not answer.endswith('.'):
                answer += '.'
            return f"Based on the official Atlan documentation: {answer}", urls[:min(2, len(urls))]
        else:
            # Fallback to first meaningful paragraph
            paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]
            if paragraphs:
                return f"Based on the official Atlan documentation: {paragraphs[0][:400]}...", urls[:1]
            else:
                return "I found some documentation but couldn't extract relevant information. Please visit the official Atlan documentation for detailed information.", urls[:1]
    
    def _format_answer(self, answer):
        """
        Format answer for better readability and structure
        
        Args:
            answer (str): Raw answer text
            
        Returns:
            str: Formatted answer with proper structure
        """
        # Ensure proper paragraph breaks
        answer = answer.replace('\n\n', '\n')
        
        # Add bullet points for lists and instructions
        lines = answer.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('•') and not line.startswith('-'):
                # Check if it looks like a step or instruction
                if any(line.startswith(prefix) for prefix in ['1.', '2.', '3.', 'Step', 'First', 'Next', 'Then', 'Finally']):
                    line = f"• {line}"
            formatted_lines.append(line)
        
        return '\n'.join(formatted_lines).strip()
    
    def initialize_knowledge_base(self):
        """
        Initialize knowledge base using ONLY real crawled content from Atlan websites
        
        This method implements a pure RAG-based approach with no sample data fallback.
        It crawls official Atlan documentation and builds a searchable index.
        """
        # Try to load existing knowledge base first
        if self.load_index():
            print("Loaded existing knowledge base")
            # Verify we have real content, not sample data
            if len(self.documents) > 0 and any("Source URL:" in doc for doc in self.documents):
                return
            else:
                print("Existing index may contain sample data, rebuilding with real content...")
        
        print("Building knowledge base from REAL Atlan websites ONLY...")
        
        # Clear any existing data to ensure fresh crawl
        self.documents = []
        self.urls = []
        
        # Define official Atlan documentation websites to crawl
        base_urls = [
            "https://docs.atlan.com/",      # Main documentation
            "https://developer.atlan.com/"  # API and developer documentation
        ]
        
        print("🌐 Starting comprehensive web crawling of Atlan documentation...")
        self.crawl_documentation(base_urls, max_pages=40)  # Comprehensive content coverage
        
        if len(self.documents) == 0:
            print("❌ CRITICAL: No content extracted from websites!")
            print("   RAG responses will not be available.")
            print("   Please check internet connection and website accessibility.")
            print("   Only ticket routing will be available.")
        else:
            print(f"✅ Successfully extracted content from {len(self.documents)} pages")
            print("🎯 Knowledge base contains ONLY real crawled content")
            
            # Build and save index only if we have real content
            self.build_index()
            self.save_index()
            print("🎯 Knowledge base built and saved with real content only")
    

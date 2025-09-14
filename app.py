from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import os
import csv
from classifier import TicketClassifier
from rag import RAGPipeline
from datetime import datetime
from collections import Counter

app = Flask(__name__)
CORS(app)

# Initialize AI components
classifier = TicketClassifier()
rag_pipeline = RAGPipeline()

# Global variable to store classified tickets
classified_tickets = None

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """Check API key status"""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    
    # Only return connection status, never expose API key details
    status = {
        'api_configured': api_key is not None and api_key != 'your_azure_openai_api_key_here' and endpoint is not None,
        'connection_status': 'Connected' if (api_key and api_key != 'your_azure_openai_api_key_here' and endpoint) else 'Not Configured'
    }
    return jsonify(status)

@app.route('/api/classify-bulk', methods=['POST'])
def classify_bulk_tickets():
    """Classify all tickets from CSV"""
    global classified_tickets
    
    try:
        # Load tickets from CSV
        if not os.path.exists('sample_tickets.csv'):
            return jsonify({'error': 'sample_tickets.csv not found'}), 400
            
        # Read CSV file without pandas
        tickets_data = []
        with open('sample_tickets.csv', 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            tickets_data = list(csv_reader)
        
        # Validate required columns
        required_columns = ['ticket_id', 'customer_name', 'subject', 'description']
        if tickets_data:
            first_row_keys = set(tickets_data[0].keys())
            missing_columns = [col for col in required_columns if col not in first_row_keys]
            if missing_columns:
                return jsonify({'error': f'Missing columns: {", ".join(missing_columns)}'}), 400
        else:
            return jsonify({'error': 'No tickets found in CSV file'}), 400
        
        # Classify tickets
        results = []
        for ticket in tickets_data:
            classification = classifier.classify_ticket(ticket['subject'], ticket['description'])
            
            result = {
                'ticket_id': int(ticket['ticket_id']),
                'customer_name': ticket['customer_name'],
                'subject': ticket['subject'],
                'description': ticket['description'],
                'channel': ticket.get('channel', 'email'),
                'timestamp': ticket.get('timestamp', ''),
                'topic_tags': classification['topic_tags'],
                'sentiment': classification['sentiment'],
                'priority': classification['priority'],
                'confidence_score': classification.get('confidence_score', 0.8),
                'reasoning': classification.get('reasoning', 'AI-powered classification')
            }
            results.append(result)
        
        classified_tickets = results
        return jsonify({
            'success': True,
            'tickets': results,
            'total': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/classify-single', methods=['POST'])
def classify_single_ticket():
    """Classify a single ticket"""
    try:
        data = request.get_json()
        subject = data.get('subject', '')
        description = data.get('description', '')
        
        if not subject or not description:
            return jsonify({'error': 'Subject and description are required'}), 400
        
        # Classify the ticket
        classification = classifier.classify_ticket(subject, description)
        
        return jsonify({
            'success': True,
            'classification': classification
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-response', methods=['POST'])
def generate_response():
    """Generate AI response using RAG pipeline"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        topic_tags = data.get('topic_tags', [])
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Check if should use RAG
        rag_topics = {'How-to', 'Product', 'Best practices', 'API/SDK', 'SSO'}
        should_use_rag = any(topic in rag_topics for topic in topic_tags)
        
        if should_use_rag:
            # Generate response using RAG
            answer, sources = rag_pipeline.generate_answer(query, topic_tags)
            return jsonify({
                'success': True,
                'type': 'rag_response',
                'answer': answer,
                'sources': sources
            })
        else:
            # Route to appropriate team
            primary_topic = topic_tags[0] if topic_tags else 'General'
            ticket_id = hash(query) % 10000
            
            routing_message = f"""
            Your ticket has been classified as a **{primary_topic}** issue and routed to our specialist team.
            
            **📋 Ticket Details:**
            - **ID:** #{ticket_id:04d}
            - **Assigned Team:** {primary_topic} Specialists
            
            **⏰ What's Next:**
            - Our {primary_topic} experts will review your ticket
            - Expected response time: 24-48 hours
            - You'll receive email updates as progress is made
            """
            
            return jsonify({
                'success': True,
                'type': 'routing',
                'message': routing_message,
                'ticket_id': f"#{ticket_id:04d}"
            })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug-sentiment')
def debug_sentiment():
    """Debug endpoint to check sentiment data"""
    global classified_tickets
    
    if not classified_tickets:
        return jsonify({'error': 'No classified tickets available'}), 400
    
    try:
        # Calculate sentiment counts without pandas
        sentiment_counts = Counter(ticket['sentiment'] for ticket in classified_tickets)
        
        return jsonify({
            'total_tickets': len(classified_tickets),
            'sentiment_counts': dict(sentiment_counts),
            'sample_tickets': [
                {
                    'ticket_id': ticket['ticket_id'],
                    'sentiment': ticket['sentiment'],
                    'subject': ticket['subject'][:50] + '...' if len(ticket['subject']) > 50 else ticket['subject']
                }
                for ticket in classified_tickets[:5]
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics')
def get_analytics():
    """Get analytics data for dashboard"""
    global classified_tickets
    
    if not classified_tickets:
        return jsonify({'error': 'No classified tickets available'}), 400
    
    try:
        # Calculate metrics without pandas
        total_tickets = len(classified_tickets)
        confidence_scores = [ticket['confidence_score'] for ticket in classified_tickets]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        p0_count = sum(1 for ticket in classified_tickets if ticket['priority'] == 'P0')
        p0_percentage = (p0_count / total_tickets) * 100 if total_tickets > 0 else 0
        
        # Priority distribution
        priority_counts = Counter(ticket['priority'] for ticket in classified_tickets)
        
        # Sentiment distribution
        sentiment_counts = Counter(ticket['sentiment'] for ticket in classified_tickets)
        print(f"Sentiment counts: {sentiment_counts}")  # Debug logging
        
        # Topic distribution
        all_topics = []
        for ticket in classified_tickets:
            tags = ticket['topic_tags']
            if isinstance(tags, list):
                all_topics.extend(tags)
            else:
                all_topics.extend([tag.strip() for tag in str(tags).split(',')])
        topic_counts = dict(Counter(all_topics).most_common(8))
        
        # Channel distribution
        channel_counts = Counter(ticket['channel'] for ticket in classified_tickets)
        
        # Generate simple chart data (no plotly dependency)
        charts = {
            'priority': {
                'type': 'pie',
                'data': dict(priority_counts),
                'colors': {'P0': '#ff4757', 'P1': '#ffa726', 'P2': '#26a69a'}
            },
            'sentiment': {
                'type': 'bar',
                'data': dict(sentiment_counts),
                'colors': {
                    'Angry': '#ff4757', 
                    'Frustrated': '#ffa726', 
                    'Curious': '#26a69a', 
                    'Neutral': '#546e7a'
                }
            },
            'topics': {
                'type': 'horizontal_bar',
                'data': topic_counts
            },
            'channels': {
                'type': 'bar',
                'data': dict(channel_counts)
            }
        }
        
        return jsonify({
            'success': True,
            'metrics': {
                'total_tickets': total_tickets,
                'avg_confidence': round(avg_confidence, 2),
                'p0_count': p0_count,
                'p0_percentage': round(p0_percentage, 1)
            },
            'distributions': {
                'priority': dict(priority_counts),
                'sentiment': dict(sentiment_counts),
                'topics': topic_counts,
                'channels': dict(channel_counts)
            },
            'charts': charts
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/<format>')
def export_data(format):
    """Export classified tickets data"""
    global classified_tickets
    
    if not classified_tickets:
        return jsonify({'error': 'No data to export'}), 400
    
    try:
        if format == 'csv':
            # Generate CSV without pandas
            import io
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=classified_tickets[0].keys())
            writer.writeheader()
            writer.writerows(classified_tickets)
            csv_data = output.getvalue()
            
            return jsonify({
                'success': True,
                'data': csv_data,
                'filename': 'atlan_ticket_analysis.csv'
            })
        elif format == 'json':
            return jsonify({
                'success': True,
                'data': classified_tickets,
                'filename': 'atlan_ticket_analysis.json'
            })
        else:
            return jsonify({'error': 'Unsupported format'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    # Initialize knowledge base (skip if in production/serverless)
    if not os.environ.get('VERCEL'):
        print("Initializing knowledge base...")
        rag_pipeline.initialize_knowledge_base()
        print("Knowledge base ready!")
        
        # Run Flask in development mode
        app.run(host='0.0.0.0', port=9000, debug=True)
    else:
        # In production, try to load existing index or create minimal one
        try:
            rag_pipeline.load_index()
            print("Loaded existing RAG index")
        except:
            print("No existing RAG index found, creating minimal fallback")
            rag_pipeline.documents = ["Atlan is a modern data catalog platform."]
            rag_pipeline.urls = ["https://atlan.com"]
            rag_pipeline.build_index()
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import json
import os
from classifier import TicketClassifier
from rag import RAGPipeline
from datetime import datetime

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
            
        df = pd.read_csv('sample_tickets.csv')
        
        # Validate required columns
        required_columns = ['ticket_id', 'customer_name', 'subject', 'description']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({'error': f'Missing columns: {", ".join(missing_columns)}'}), 400
        
        # Classify tickets
        results = []
        for _, ticket in df.iterrows():
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
        df = pd.DataFrame(classified_tickets)
        sentiment_counts = df['sentiment'].value_counts().to_dict()
        
        return jsonify({
            'total_tickets': len(df),
            'sentiment_counts': sentiment_counts,
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


if __name__ == '__main__':
    # Initialize knowledge base
    print("Initializing knowledge base...")
    rag_pipeline.initialize_knowledge_base()
    print("Knowledge base ready!")

    # Run Flask in production mode (still using built-in server)
    app.run(host='0.0.0.0', port=9000, debug=False, use_reloader=False)
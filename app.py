# Atlan Customer Support Center - Main Flask Application
# This application provides AI-powered ticket classification and response generation
# for Atlan customer support tickets.

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import json
import os
from classifier import TicketClassifier
from rag import RAGPipeline
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Initialize Flask application with CORS support for frontend communication
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for API calls

# Initialize AI components for ticket processing
classifier = TicketClassifier()  # Handles ticket classification (topics, sentiment, priority)
rag_pipeline = RAGPipeline()     # Handles RAG-based response generation from documentation

# Global variable to store classified tickets for analytics and export
classified_tickets = None

@app.route('/')
def index():
    """
    Main dashboard page route
    Serves the HTML template for the customer support dashboard
    """
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """
    Check Azure OpenAI API configuration status
    Returns whether the API is properly configured without exposing sensitive details
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get API configuration from environment variables
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    
    # Security: Only return connection status, never expose API key details
    status = {
        'api_configured': api_key is not None and api_key != 'your_azure_openai_api_key_here' and endpoint is not None,
        'connection_status': 'Connected' if (api_key and api_key != 'your_azure_openai_api_key_here' and endpoint) else 'Not Configured'
    }
    return jsonify(status)

@app.route('/api/classify-bulk', methods=['POST'])
def classify_bulk_tickets():
    """
    Classify all tickets from the sample CSV file
    Processes multiple tickets at once and returns comprehensive classification results
    """
    global classified_tickets
    
    try:
        # Load tickets from CSV file
        if not os.path.exists('sample_tickets.csv'):
            return jsonify({'error': 'sample_tickets.csv not found'}), 400
            
        df = pd.read_csv('sample_tickets.csv')
        
        # Validate that all required columns are present
        required_columns = ['ticket_id', 'customer_name', 'subject', 'description']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({'error': f'Missing columns: {", ".join(missing_columns)}'}), 400
        
        # Process each ticket through the AI classifier
        results = []
        for _, ticket in df.iterrows():
            # Get AI classification for topic tags, sentiment, and priority
            classification = classifier.classify_ticket(ticket['subject'], ticket['description'])
            
            # Structure the result with all ticket data and classification
            result = {
                'ticket_id': int(ticket['ticket_id']),
                'customer_name': ticket['customer_name'],
                'subject': ticket['subject'],
                'description': ticket['description'],
                'channel': ticket.get('channel', 'email'),  # Default to email if not specified
                'timestamp': ticket.get('timestamp', ''),
                'topic_tags': classification['topic_tags'],      # AI-identified topic categories
                'sentiment': classification['sentiment'],        # Customer emotional state
                'priority': classification['priority'],          # P0/P1/P2 priority level
                'confidence_score': classification.get('confidence_score', 0.8),
                'reasoning': classification.get('reasoning', 'AI-powered classification')
            }
            results.append(result)
        
        # Store results globally for analytics and export
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
    """
    Classify a single ticket provided via API request
    Used for real-time ticket classification in the frontend
    """
    try:
        # Extract ticket data from JSON request
        data = request.get_json()
        subject = data.get('subject', '')
        description = data.get('description', '')
        
        # Validate required fields
        if not subject or not description:
            return jsonify({'error': 'Subject and description are required'}), 400
        
        # Process ticket through AI classifier
        classification = classifier.classify_ticket(subject, description)
        
        return jsonify({
            'success': True,
            'classification': classification
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-response', methods=['POST'])
def generate_response():
    """
    Generate AI-powered response using RAG pipeline or route to specialist teams
    Determines whether to use RAG for documentation-based answers or route to human experts
    """
    try:
        # Extract query and topic information from request
        data = request.get_json()
        query = data.get('query', '')
        topic_tags = data.get('topic_tags', [])
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Determine if RAG can handle this query based on topic categories
        # RAG works best for documentation-based topics
        rag_topics = {'How-to', 'Product', 'Best practices', 'API/SDK', 'SSO'}
        should_use_rag = any(topic in rag_topics for topic in topic_tags)
        
        if should_use_rag:
            # Generate response using RAG pipeline with documentation knowledge
            answer, sources = rag_pipeline.generate_answer(query, topic_tags)
            return jsonify({
                'success': True,
                'type': 'rag_response',
                'answer': answer,
                'sources': sources
            })
        else:
            # Route to appropriate specialist team for complex issues
            primary_topic = topic_tags[0] if topic_tags else 'General'
            ticket_id = hash(query) % 10000  # Generate pseudo ticket ID
            
            # Create professional routing message
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
    """
    Debug endpoint to check sentiment analysis data
    Useful for troubleshooting sentiment classification issues
    """
    global classified_tickets
    
    if not classified_tickets:
        return jsonify({'error': 'No classified tickets available'}), 400
    
    try:
        # Convert to DataFrame for analysis
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
                for ticket in classified_tickets[:5]  # Show first 5 tickets as examples
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics')
def get_analytics():
    """
    Generate comprehensive analytics data for the dashboard
    Calculates metrics, distributions, and creates interactive charts
    """
    global classified_tickets
    
    if not classified_tickets:
        return jsonify({'error': 'No classified tickets available'}), 400
    
    try:
        # Convert classified tickets to DataFrame for analysis
        df = pd.DataFrame(classified_tickets)
        
        # Calculate key performance metrics
        total_tickets = len(df)
        avg_confidence = df['confidence_score'].mean()  # Average AI confidence
        p0_count = len(df[df['priority'] == 'P0'])     # Critical tickets count
        p0_percentage = (p0_count / total_tickets) * 100  # Critical tickets percentage
        
        # Calculate distribution metrics for visualization
        priority_counts = df['priority'].value_counts().to_dict()
        sentiment_counts = df['sentiment'].value_counts().to_dict()
        print(f"Sentiment counts: {sentiment_counts}")  # Debug logging
        
        # Process topic tags (can be lists or comma-separated strings)
        all_topics = []
        for tags in df['topic_tags']:
            if isinstance(tags, list):
                all_topics.extend(tags)
            else:
                all_topics.extend([tag.strip() for tag in str(tags).split(',')])
        topic_counts = pd.Series(all_topics).value_counts().head(8).to_dict()  # Top 8 topics
        
        # Channel distribution (email, chat, phone, etc.)
        channel_counts = df['channel'].value_counts().to_dict()
        
        # Generate interactive Plotly charts for dashboard visualization
        charts = {}
        
        # Priority distribution pie chart with color coding
        fig_priority = px.pie(
            values=list(priority_counts.values()),
            names=list(priority_counts.keys()),
            title="Priority Distribution",
            color_discrete_map={'P0': '#ff4757', 'P1': '#ffa726', 'P2': '#26a69a'}  # Red, Orange, Teal
        )
        charts['priority'] = fig_priority.to_json()
        
        # Sentiment analysis bar chart with robust error handling
        try:
            if sentiment_counts:
                print(f"Creating sentiment chart with data: {sentiment_counts}")
                
                # Create a robust bar chart using Plotly Graph Objects
                fig_sentiment = go.Figure()
                
                # Define color scheme for different sentiment types
                color_map = {
                    'Angry': '#ff4757',      # Red for angry customers
                    'Frustrated': '#ffa726',  # Orange for frustrated customers
                    'Curious': '#26a69a',     # Teal for curious customers
                    'Neutral': '#546e7a'      # Gray for neutral customers
                }
                
                # Add individual bars for each sentiment category
                for sentiment, count in sentiment_counts.items():
                    fig_sentiment.add_trace(go.Bar(
                        x=[sentiment],
                        y=[count],
                        name=sentiment,
                        marker_color=color_map.get(sentiment, '#546e7a'),
                        showlegend=False
                    ))
                
                # Configure chart layout and styling
                fig_sentiment.update_layout(
                    title="Sentiment Analysis",
                    xaxis_title="Sentiment",
                    yaxis_title="Count",
                    height=400,
                    margin=dict(l=50, r=50, t=50, b=50),
                    xaxis=dict(
                        type='category',
                        categoryorder='array',
                        categoryarray=list(sentiment_counts.keys())
                    ),
                    yaxis=dict(
                        type='linear',
                        autorange=True
                    )
                )
                
                charts['sentiment'] = fig_sentiment.to_json()
                print("Sentiment chart generated successfully")
            else:
                print("No sentiment data available for chart")
                charts['sentiment'] = None
        except Exception as e:
            print(f"Error generating sentiment chart: {e}")
            charts['sentiment'] = None
        
        # Topic distribution horizontal bar chart (topics on Y-axis)
        fig_topics = px.bar(
            x=list(topic_counts.values()),
            y=list(topic_counts.keys()),
            orientation='h',  # Horizontal orientation for better readability
            title="Top Topics",
            color=list(topic_counts.values()),
            color_continuous_scale='viridis'  # Color gradient for visual appeal
        )
        fig_topics.update_layout(yaxis={'categoryorder':'total ascending'})  # Sort by frequency
        charts['topics'] = fig_topics.to_json()
        
        # Channel distribution bar chart (communication channels)
        fig_channels = px.bar(
            x=list(channel_counts.keys()),
            y=list(channel_counts.values()),
            title="Tickets by Channel",
            color=list(channel_counts.values()),
            color_continuous_scale='blues'  # Blue gradient for professional look
        )
        charts['channels'] = fig_channels.to_json()
        
        # Return comprehensive analytics data for dashboard
        return jsonify({
            'success': True,
            'metrics': {
                'total_tickets': total_tickets,
                'avg_confidence': round(avg_confidence, 2),  # AI classification confidence
                'p0_count': p0_count,                        # Critical tickets count
                'p0_percentage': round(p0_percentage, 1)     # Critical tickets percentage
            },
            'distributions': {
                'priority': priority_counts,    # P0/P1/P2 distribution
                'sentiment': sentiment_counts,  # Customer sentiment breakdown
                'topics': topic_counts,         # Most common topics
                'channels': channel_counts      # Communication channels
            },
            'charts': charts  # Interactive Plotly charts for visualization
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/<format>')
def export_data(format):
    """
    Export classified tickets data in various formats (CSV, JSON)
    Allows users to download analysis results for further processing
    """
    global classified_tickets
    
    if not classified_tickets:
        return jsonify({'error': 'No data to export'}), 400
    
    try:
        df = pd.DataFrame(classified_tickets)
        
        if format == 'csv':
            # Export as CSV for spreadsheet applications
            csv_data = df.to_csv(index=False)
            return jsonify({
                'success': True,
                'data': csv_data,
                'filename': 'atlan_ticket_analysis.csv'
            })
        elif format == 'json':
            # Export as JSON for programmatic access
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
    # Application startup sequence
    print("Initializing knowledge base...")
    rag_pipeline.initialize_knowledge_base()  # Crawl and index Atlan documentation
    print("Knowledge base ready!")

    # Start Flask application server
    # Host 0.0.0.0 allows external connections (useful for deployment)
    # Port 9000 is used to avoid conflicts with common development ports
    app.run(host='0.0.0.0', port=9000, debug=False, use_reloader=False)
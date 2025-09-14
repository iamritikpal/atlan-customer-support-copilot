import openai
import os
from dotenv import load_dotenv
import json

load_dotenv()

class TicketClassifier:
    def __init__(self):
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-5-chat')
        api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2025-01-01-preview')
        
        if not api_key or not endpoint:
            print("⚠️  Warning: Azure OpenAI credentials not set. Please add your API key and endpoint to .env file")
            print("   Required variables: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT")
            print("   Optional variables: AZURE_OPENAI_DEPLOYMENT (default: gpt-5-chat), AZURE_OPENAI_API_VERSION (default: 2025-01-01-preview)")
            self.client = None
        else:
            try:
                self.client = openai.AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=endpoint
                )
                self.deployment = deployment
                print(f"✅ Azure OpenAI client initialized with deployment: {deployment}")
            except Exception as e:
                print(f"❌ Failed to initialize Azure OpenAI client: {e}")
                print("   Please check your Azure OpenAI configuration")
                self.client = None
        
    def classify_ticket(self, subject, description):
        """
        Classify a ticket into topic tags, sentiment, and priority using advanced prompting
        """
        prompt = f"""
        You are an expert customer support analyst for Atlan, a data catalog platform. Analyze this support ticket and provide detailed classification.

        TICKET DETAILS:
        Subject: {subject}
        Description: {description}

        CLASSIFICATION REQUIREMENTS:

        1. TOPIC TAGS (select 1-3 most relevant):
        - How-to: Step-by-step guidance, tutorials, configuration help
        - Product: Feature requests, product capabilities, general product questions
        - Connector: Data source connections (Snowflake, BigQuery, MySQL, etc.)
        - Lineage: Data lineage tracking, dependency mapping, impact analysis
        - API/SDK: REST API usage, SDK integration, programmatic access
        - SSO: Single Sign-On, SAML, authentication, user management
        - Glossary: Business terms, data dictionary, metadata definitions
        - Best practices: Governance, optimization, recommended approaches
        - Sensitive data: PII, compliance, data classification, security

        2. SENTIMENT ANALYSIS:
        - Frustrated: Annoyed, blocked, repeated issues, time pressure
        - Angry: Very upset, escalated tone, demanding immediate action
        - Curious: Learning-oriented, exploratory, seeking understanding
        - Neutral: Professional, matter-of-fact, standard inquiry

        3. PRIORITY ASSESSMENT:
        - P0: Production down, compliance risk, urgent business impact, angry customers
        - P1: Important feature blocked, significant user impact, time-sensitive
        - P2: General questions, feature requests, optimization, learning

        EXAMPLES:
        - "API rate limits blocking deployment" → ["API/SDK"], "Frustrated", "P1"
        - "How to connect Snowflake?" → ["How-to", "Connector"], "Curious", "P2"
        - "SSO completely broken, users can't login!" → ["SSO"], "Angry", "P0"
        - "Best practices for data governance" → ["Best practices"], "Curious", "P2"

        Return ONLY a valid JSON object with EXACTLY these field names and formats:
        {{
            "topic_tags": ["primary_tag", "secondary_tag"],
            "sentiment": "Frustrated" or "Angry" or "Curious" or "Neutral",
            "priority": "P0" or "P1" or "P2",
            "confidence_score": 0.95,
            "reasoning": "Brief explanation of classification logic"
        }}
        
        IMPORTANT: 
        - Return ONLY the JSON object, no markdown formatting, no code blocks, no extra text
        - sentiment must be exactly: "Frustrated", "Angry", "Curious", or "Neutral" (capitalized)
        - priority must be exactly: "P0", "P1", or "P2" (with P prefix)
        - topic_tags must use the exact categories listed above
        """

        # Return intelligent default classification if no API key
        if self.client is None:
            return self._classify_without_api(subject, description)
            
        try:
            full_prompt = f"""You are a customer support ticket classifier. Always respond with valid JSON.

{prompt}"""
            
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {'role': 'system', 'content': 'You are a customer support ticket classifier. Always respond with valid JSON.'},
                    {'role': 'user', 'content': full_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # Debug: Print the raw response
            response_content = response.choices[0].message.content
            print(f"Raw API response: {response_content}")
            
            if not response_content or response_content.strip() == "":
                print("Warning: Empty response from Azure OpenAI API")
                raise ValueError("Empty response from API")
            
            # Clean the response content - remove markdown code blocks if present
            cleaned_content = self._clean_json_response(response_content)
            
            result = json.loads(cleaned_content)
            
            # Validate and normalize the response format
            result = self._validate_and_normalize_response(result)
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response that failed to parse: {response_content if 'response_content' in locals() else 'No response content'}")
            print(f"Cleaned response that failed to parse: {cleaned_content if 'cleaned_content' in locals() else 'No cleaned content'}")
            # Return default classification if JSON parsing fails
            return {
                "topic_tags": ["Product"],
                "sentiment": "Neutral",
                "priority": "P2",
                "confidence_score": 0.5,
                "reasoning": "JSON parsing fallback classification"
            }
        except Exception as e:
            print(f"Error in classification: {e}")
            # Return default classification if API fails
            return {
                "topic_tags": ["Product"],
                "sentiment": "Neutral",
                "priority": "P2",
                "confidence_score": 0.5,
                "reasoning": "API fallback classification"
            }
    
    def _clean_json_response(self, response_content):
        """Clean the API response by removing markdown code blocks and extra whitespace"""
        try:
            # Remove markdown code blocks if present
            if response_content.strip().startswith('```json'):
                # Find the start and end of the JSON content
                start_marker = '```json'
                end_marker = '```'
                
                start_idx = response_content.find(start_marker)
                if start_idx != -1:
                    start_idx += len(start_marker)
                    end_idx = response_content.find(end_marker, start_idx)
                    if end_idx != -1:
                        response_content = response_content[start_idx:end_idx]
                    else:
                        # If no end marker, take everything after start marker
                        response_content = response_content[start_idx:]
            elif response_content.strip().startswith('```'):
                # Handle generic code blocks
                start_marker = '```'
                end_marker = '```'
                
                start_idx = response_content.find(start_marker)
                if start_idx != -1:
                    start_idx += len(start_marker)
                    end_idx = response_content.find(end_marker, start_idx)
                    if end_idx != -1:
                        response_content = response_content[start_idx:end_idx]
                    else:
                        response_content = response_content[start_idx:]
            
            # Clean up whitespace
            response_content = response_content.strip()
            
            return response_content
            
        except Exception as e:
            print(f"Warning: Error cleaning JSON response: {e}")
            return response_content
    
    def _validate_and_normalize_response(self, result):
        """Validate and normalize the API response to match expected format"""
        try:
            # Ensure required fields exist
            if 'topic_tags' not in result:
                result['topic_tags'] = ['Product']
            if 'sentiment' not in result:
                result['sentiment'] = 'Neutral'
            if 'priority' not in result:
                result['priority'] = 'P2'
            if 'confidence_score' not in result:
                result['confidence_score'] = 0.8
            if 'reasoning' not in result:
                result['reasoning'] = 'AI-powered classification'
            
            # Normalize sentiment (capitalize first letter)
            sentiment = result['sentiment']
            if isinstance(sentiment, str):
                result['sentiment'] = sentiment.capitalize()
            
            # Normalize priority (ensure P prefix)
            priority = result['priority']
            if isinstance(priority, str):
                if priority.lower() in ['high', 'medium', 'low']:
                    priority_map = {'high': 'P1', 'medium': 'P2', 'low': 'P2'}
                    result['priority'] = priority_map[priority.lower()]
                elif not priority.startswith('P'):
                    # Try to map numeric priorities
                    if priority.isdigit():
                        priority_map = {'0': 'P0', '1': 'P1', '2': 'P2'}
                        result['priority'] = priority_map.get(priority, 'P2')
                    else:
                        result['priority'] = 'P2'
            
            # Ensure topic_tags is a list
            if not isinstance(result['topic_tags'], list):
                result['topic_tags'] = [str(result['topic_tags'])]
            
            # Validate sentiment values
            valid_sentiments = ['Frustrated', 'Angry', 'Curious', 'Neutral']
            if result['sentiment'] not in valid_sentiments:
                result['sentiment'] = 'Neutral'
            
            # Validate priority values
            valid_priorities = ['P0', 'P1', 'P2']
            if result['priority'] not in valid_priorities:
                result['priority'] = 'P2'
            
            return result
            
        except Exception as e:
            print(f"Warning: Error normalizing response: {e}")
            # Return a safe default
            return {
                "topic_tags": ["Product"],
                "sentiment": "Neutral",
                "priority": "P2",
                "confidence_score": 0.5,
                "reasoning": "Response normalization fallback"
            }
    
    def _classify_without_api(self, subject, description):
        """
        Rule-based classification when API is not available
        """
        text = f"{subject} {description}".lower()
        
        # Topic classification using keywords
        topic_tags = []
        if any(word in text for word in ['connect', 'connection', 'connector', 'snowflake', 'mysql', 'bigquery', 'database']):
            topic_tags.append('Connector')
        if any(word in text for word in ['api', 'sdk', 'rate limit', 'python', 'rest', 'endpoint']):
            topic_tags.append('API/SDK')
        if any(word in text for word in ['sso', 'saml', 'authentication', 'login', 'auth', 'okta']):
            topic_tags.append('SSO')
        if any(word in text for word in ['lineage', 'dependency', 'impact', 'flow']):
            topic_tags.append('Lineage')
        if any(word in text for word in ['glossary', 'terms', 'definition', 'metadata']):
            topic_tags.append('Glossary')
        if any(word in text for word in ['pii', 'sensitive', 'compliance', 'classification', 'security']):
            topic_tags.append('Sensitive data')
        if any(word in text for word in ['how to', 'how do', 'setup', 'configure', 'install']):
            topic_tags.append('How-to')
        if any(word in text for word in ['best practice', 'recommendation', 'governance', 'optimize']):
            topic_tags.append('Best practices')
        
        if not topic_tags:
            topic_tags = ['Product']
        
        # Sentiment analysis using keywords
        sentiment = 'Neutral'
        if any(word in text for word in ['frustrated', 'annoyed', 'slow', 'doesn\'t work', 'not working']):
            sentiment = 'Frustrated'
        elif any(word in text for word in ['urgent', 'critical', 'broken', 'failed', 'error', 'can\'t', 'unable']):
            sentiment = 'Angry'
        elif any(word in text for word in ['how', 'what', 'curious', 'learn', 'understand']):
            sentiment = 'Curious'
        
        # Enhanced priority based on keywords, sentiment, and context with better detection
        priority = 'P2'  # Default
        
        # P0 (Critical) - Production issues, security, compliance, complete failures
        p0_keywords = [
            'urgent', 'critical', 'production', 'down', 'broken', 'compliance', 
            'security', 'breach', 'outage', 'completely', 'entirely', 'audit',
            'failing', 'stopped working', 'cannot access', 'emergency',
            'very urgent', 'immediate assistance', 'can\'t authenticate',
            'can\'t login', 'audit is', 'compliance reporting'
        ]
        
        # P1 (High) - Blocking issues, frustrated customers, time-sensitive, reliability issues
        p1_keywords = [
            'blocking', 'deployment', 'asap', 'time-sensitive', 'frustrated',
            'constantly', 'keeps failing', 'timeout', 'error', 'issue',
            'problem', 'need immediate', 'very', 'extremely', 'unreliable',
            'every single time', 'every time', 'always fails', 'completely unreliable',
            'blocking our', 'automation project', 'production deployment'
        ]
        
        # P2 (Medium/Low) - Questions, how-to, best practices, general inquiries
        p2_keywords = [
            'how', 'what', 'best practice', 'recommend', 'guidance', 'available',
            'when will', 'can atlan', 'looking for', 'need to understand'
        ]
        
        # Advanced priority detection with phrase matching
        text_lower = text.lower()
        
        # Check for P0 conditions with more sophisticated matching
        p0_score = 0
        
        # Direct P0 indicators
        if any(phrase in text_lower for phrase in [
            'very urgent', 'extremely urgent', 'urgent:', 'critical feature',
            'immediate assistance', 'can\'t authenticate', 'can\'t login',
            'audit is next week', 'compliance reporting', 'security breach'
        ]):
            p0_score += 3
            
        # Strong P0 keywords
        for keyword in ['urgent', 'critical', 'compliance', 'audit', 'security', 'emergency']:
            if keyword in text_lower:
                p0_score += 2
                
        # Authentication/access issues
        if ('can\'t' in text_lower or 'cannot' in text_lower) and any(word in text_lower for word in ['authenticate', 'login', 'access']):
            p0_score += 2
            
        # Complete failures
        if any(phrase in text_lower for phrase in ['completely', 'entirely', 'absolutely nothing']):
            p0_score += 2
        
        # Check for P1 conditions
        p1_score = 0
        
        # Direct P1 indicators  
        if any(phrase in text_lower for phrase in [
            'blocking our', 'production deployment', 'automation project',
            'completely unreliable', 'every single time', 'constantly hitting'
        ]):
            p1_score += 3
            
        # Strong P1 keywords
        for keyword in ['blocking', 'unreliable', 'constantly', 'extremely', 'frustrated']:
            if keyword in text_lower:
                p1_score += 2
                
        # Repeated failures
        if any(phrase in text_lower for phrase in ['keeps failing', 'always fails', 'every time', 'constantly']):
            p1_score += 2
        
        # Sentiment-based scoring
        if sentiment == 'Angry':
            p0_score += 2
        elif sentiment == 'Frustrated':
            p1_score += 2
        elif sentiment == 'Curious':
            # Curious questions are typically P2
            pass
            
        # Determine final priority
        if p0_score >= 3:
            priority = 'P0'
        elif p1_score >= 3 or (p1_score >= 2 and sentiment in ['Angry', 'Frustrated']):
            priority = 'P1'
        elif any(word in text_lower for word in p2_keywords):
            priority = 'P2'
        else:
            # Default based on sentiment if no clear indicators
            if sentiment == 'Angry':
                priority = 'P1'
            elif sentiment == 'Frustrated':
                priority = 'P1' 
            else:
                priority = 'P2'
        
        return {
            "topic_tags": topic_tags,
            "sentiment": sentiment,
            "priority": priority,
            "confidence_score": 0.7,
            "reasoning": "Rule-based classification using keyword matching"
        }
    
    def classify_bulk_tickets(self, tickets_list):
        """
        Classify multiple tickets from a list of dictionaries
        """
        results = []
        
        for ticket in tickets_list:
            classification = self.classify_ticket(ticket['subject'], ticket['description'])
            
            result = {
                'ticket_id': ticket['ticket_id'],
                'customer_name': ticket['customer_name'],
                'subject': ticket['subject'],
                'description': ticket['description'],
                'channel': ticket.get('channel', 'email'),
                'timestamp': ticket.get('timestamp', ''),
                'topic_tags': ', '.join(classification['topic_tags']),
                'sentiment': classification['sentiment'],
                'priority': classification['priority'],
                'confidence_score': classification.get('confidence_score', 0.8),
                'reasoning': classification.get('reasoning', 'AI-powered classification')
            }
            results.append(result)
            
        return results

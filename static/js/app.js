// Global variables
let classifiedTickets = [];
let currentTickets = [];

// DOM Elements
const navTabs = document.querySelectorAll('.nav-tab');
const tabContents = document.querySelectorAll('.tab-content');
const loadTicketsBtn = document.getElementById('load-tickets-btn');
const ticketForm = document.getElementById('ticket-form');
const sampleButtons = document.querySelectorAll('.sample-btn');

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    checkApiStatus();
});

function initializeApp() {
    // Show first tab by default
    showTab('bulk-classification');
}

function setupEventListeners() {
    // Navigation tabs
    navTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const tabId = tab.dataset.tab;
            showTab(tabId);
        });
    });

    // Load tickets button
    loadTicketsBtn.addEventListener('click', loadAndClassifyTickets);

    // Ticket form
    ticketForm.addEventListener('submit', handleTicketSubmission);

    // Sample query buttons
    sampleButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            document.getElementById('ticket-subject').value = btn.dataset.subject;
            document.getElementById('ticket-description').value = btn.dataset.description;
        });
    });

    // Filter change handlers
    document.getElementById('sentiment-filter').addEventListener('change', applyFilters);
    document.getElementById('priority-filter').addEventListener('change', applyFilters);
    document.getElementById('topic-filter').addEventListener('change', applyFilters);
    document.getElementById('channel-filter').addEventListener('change', applyFilters);

    // Export buttons
    document.querySelectorAll('.export-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const format = btn.dataset.format;
            exportData(format);
        });
    });
}

function showTab(tabId) {
    // Update nav tabs
    navTabs.forEach(tab => {
        tab.classList.remove('active');
        if (tab.dataset.tab === tabId) {
            tab.classList.add('active');
        }
    });

    // Update tab contents
    tabContents.forEach(content => {
        content.classList.remove('active');
        if (content.id === tabId) {
            content.classList.add('active');
        }
    });

    // Load analytics if switching to analytics tab
    if (tabId === 'analytics' && classifiedTickets.length > 0) {
        loadAnalytics();
    }
}

async function checkApiStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        const indicator = document.getElementById('api-status-indicator');
        const icon = indicator.querySelector('i');
        const text = indicator.querySelector('span');
        
        if (data.api_configured) {
            indicator.className = 'status-indicator connected';
            icon.className = 'fas fa-check-circle';
            text.textContent = 'Azure OpenAI Connected';
        } else {
            indicator.className = 'status-indicator disconnected';
            icon.className = 'fas fa-exclamation-triangle';
            text.textContent = 'Azure OpenAI Not Configured';
        }
    } catch (error) {
        const indicator = document.getElementById('api-status-indicator');
        indicator.className = 'status-indicator error';
        indicator.querySelector('i').className = 'fas fa-times-circle';
        indicator.querySelector('span').textContent = 'API Error';
    }
}

async function loadAndClassifyTickets() {
    const loadingIndicator = document.getElementById('loading-indicator');
    const resultsSection = document.getElementById('results-section');
    const progressFill = document.getElementById('progress-fill');
    
    // Show loading
    loadingIndicator.style.display = 'block';
    resultsSection.style.display = 'none';
    loadTicketsBtn.disabled = true;
    
    // Simulate progress
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90;
        progressFill.style.width = progress + '%';
    }, 200);
    
    try {
        const response = await fetch('/api/classify-bulk', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            classifiedTickets = data.tickets;
            currentTickets = [...classifiedTickets];
            
            // Complete progress
            clearInterval(progressInterval);
            progressFill.style.width = '100%';
            
            setTimeout(() => {
                loadingIndicator.style.display = 'none';
                displayResults();
                resultsSection.style.display = 'block';
            }, 500);
        } else {
            throw new Error(data.error || 'Classification failed');
        }
    } catch (error) {
        clearInterval(progressInterval);
        loadingIndicator.style.display = 'none';
        alert('Error: ' + error.message);
    } finally {
        loadTicketsBtn.disabled = false;
    }
}

function displayResults() {
    // Update metrics
    updateMetrics();
    
    // Populate filters
    populateFilters();
    
    // Display tickets table
    displayTicketsTable();
}

function updateMetrics() {
    const totalTickets = classifiedTickets.length;
    const criticalTickets = classifiedTickets.filter(t => t.priority === 'P0').length;
    const angryCustomers = classifiedTickets.filter(t => t.sentiment === 'Angry').length;
    const avgConfidence = classifiedTickets.reduce((sum, t) => sum + t.confidence_score, 0) / totalTickets;
    
    document.getElementById('total-tickets').textContent = totalTickets;
    document.getElementById('critical-tickets').textContent = criticalTickets;
    document.getElementById('angry-customers').textContent = angryCustomers;
    document.getElementById('avg-confidence').textContent = avgConfidence.toFixed(2);
}

function populateFilters() {
    const sentiments = [...new Set(classifiedTickets.map(t => t.sentiment))];
    const priorities = [...new Set(classifiedTickets.map(t => t.priority))];
    const channels = [...new Set(classifiedTickets.map(t => t.channel))];
    
    // Get all unique topics
    const allTopics = new Set();
    classifiedTickets.forEach(ticket => {
        if (Array.isArray(ticket.topic_tags)) {
            ticket.topic_tags.forEach(tag => allTopics.add(tag));
        } else {
            ticket.topic_tags.split(',').forEach(tag => allTopics.add(tag.trim()));
        }
    });
    
    populateSelect('sentiment-filter', sentiments);
    populateSelect('priority-filter', priorities);
    populateSelect('topic-filter', Array.from(allTopics));
    populateSelect('channel-filter', channels);
}

function populateSelect(selectId, options) {
    const select = document.getElementById(selectId);
    const currentValue = select.value;
    
    // Clear existing options except "All"
    select.innerHTML = '<option value="All">All</option>';
    
    options.forEach(option => {
        const optionElement = document.createElement('option');
        optionElement.value = option;
        optionElement.textContent = option;
        select.appendChild(optionElement);
    });
    
    // Restore previous selection if still valid
    if (options.includes(currentValue)) {
        select.value = currentValue;
    }
}

function applyFilters() {
    const sentimentFilter = document.getElementById('sentiment-filter').value;
    const priorityFilter = document.getElementById('priority-filter').value;
    const topicFilter = document.getElementById('topic-filter').value;
    const channelFilter = document.getElementById('channel-filter').value;
    
    currentTickets = classifiedTickets.filter(ticket => {
        if (sentimentFilter !== 'All' && ticket.sentiment !== sentimentFilter) return false;
        if (priorityFilter !== 'All' && ticket.priority !== priorityFilter) return false;
        if (channelFilter !== 'All' && ticket.channel !== channelFilter) return false;
        
        if (topicFilter !== 'All') {
            const topics = Array.isArray(ticket.topic_tags) 
                ? ticket.topic_tags 
                : ticket.topic_tags.split(',').map(t => t.trim());
            if (!topics.includes(topicFilter)) return false;
        }
        
        return true;
    });
    
    displayTicketsTable();
    updateTableTitle();
}

function updateTableTitle() {
    const title = document.getElementById('table-title');
    title.textContent = `📋 Tickets (${currentTickets.length} of ${classifiedTickets.length})`;
}

function displayTicketsTable() {
    const tbody = document.getElementById('tickets-tbody');
    tbody.innerHTML = '';
    
    currentTickets.forEach(ticket => {
        const row = document.createElement('tr');
        
        const priorityIcon = ticket.priority === 'P0' ? '🚨' : ticket.priority === 'P1' ? '⚡' : '📝';
        const sentimentIcon = getSentimentIcon(ticket.sentiment);
        const topics = Array.isArray(ticket.topic_tags) 
            ? ticket.topic_tags.join(', ') 
            : ticket.topic_tags;
        
        row.innerHTML = `
            <td>${ticket.ticket_id}</td>
            <td>${ticket.customer_name}</td>
            <td title="${ticket.description}">${truncateText(ticket.subject, 50)}</td>
            <td class="priority-${ticket.priority.toLowerCase()}">${priorityIcon} ${ticket.priority}</td>
            <td>${sentimentIcon} ${ticket.sentiment}</td>
            <td>${topics}</td>
            <td>${ticket.channel}</td>
            <td>${ticket.confidence_score.toFixed(2)}</td>
        `;
        
        tbody.appendChild(row);
    });
    
    updateTableTitle();
}

function getSentimentIcon(sentiment) {
    const icons = {
        'Angry': '😠',
        'Frustrated': '😤',
        'Curious': '🤔',
        'Neutral': '😐'
    };
    return icons[sentiment] || '😐';
}

function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

async function handleTicketSubmission(e) {
    e.preventDefault();
    
    const subject = document.getElementById('ticket-subject').value;
    const description = document.getElementById('ticket-description').value;
    const channel = document.getElementById('ticket-channel').value;
    
    if (!subject || !description) {
        alert('Please fill in both subject and description');
        return;
    }
    
    try {
        // Show loading state
        const submitBtn = ticketForm.querySelector('button[type="submit"]');
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        submitBtn.disabled = true;
        
        // Classify the ticket
        const classifyResponse = await fetch('/api/classify-single', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ subject, description })
        });
        
        const classifyData = await classifyResponse.json();
        
        if (!classifyData.success) {
            throw new Error(classifyData.error || 'Classification failed');
        }
        
        const classification = classifyData.classification;
        
        // Generate response
        const responseResponse = await fetch('/api/generate-response', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: `${subject} ${description}`,
                topic_tags: classification.topic_tags
            })
        });
        
        const responseData = await responseResponse.json();
        
        if (!responseData.success) {
            throw new Error(responseData.error || 'Response generation failed');
        }
        
        // Display results
        displayAnalysisResults(classification, responseData);
        
        // Reset form
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
        
    } catch (error) {
        alert('Error: ' + error.message);
        const submitBtn = ticketForm.querySelector('button[type="submit"]');
        submitBtn.innerHTML = '<i class="fas fa-rocket"></i> Analyze Ticket';
        submitBtn.disabled = false;
    }
}

function displayAnalysisResults(classification, responseData) {
    // Show results section with animation
    const resultsSection = document.getElementById('analysis-results');
    resultsSection.style.display = 'block';
    
    // Add a slight delay for smooth animation
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }, 100);
    
    // Update classification summary with enhanced formatting
    const priorityIcon = classification.priority === 'P0' ? '🚨' : classification.priority === 'P1' ? '⚡' : '📝';
    const sentimentIcon = getSentimentIcon(classification.sentiment);
    
    document.getElementById('result-priority').innerHTML = `${priorityIcon} <span class="priority-text">${classification.priority}</span>`;
    document.getElementById('result-sentiment').innerHTML = `${sentimentIcon} <span class="sentiment-text">${classification.sentiment}</span>`;
    document.getElementById('result-confidence').innerHTML = `<span class="confidence-text">${classification.confidence_score?.toFixed(2) || 'N/A'}</span>`;
    document.getElementById('result-topics-count').innerHTML = `<span class="topics-text">${classification.topic_tags.length}</span>`;
    
    // Display classification JSON with syntax highlighting
    const jsonElement = document.getElementById('classification-json');
    jsonElement.textContent = JSON.stringify(classification, null, 2);
    
    // Display enhanced classification breakdown
    const breakdown = document.getElementById('classification-breakdown');
    breakdown.innerHTML = `
        <div class="breakdown-item">
            <div class="breakdown-label">
                <i class="fas fa-tags"></i>
                <strong>Topic Analysis:</strong>
            </div>
            <div class="topic-tags">
                ${classification.topic_tags.map(tag => `
                    <span class="topic-tag">${tag}</span>
                `).join('')}
            </div>
        </div>
        
        <div class="breakdown-item">
            <div class="breakdown-label">
                <i class="fas fa-brain"></i>
                <strong>AI Reasoning:</strong>
            </div>
            <div class="reasoning-text">${classification.reasoning || 'AI-powered classification'}</div>
        </div>
        
        <div class="routing-info">
            ${responseData.type === 'rag_response' ? 
                '<i class="fas fa-robot"></i> <span>Will use RAG pipeline for response</span>' : 
                '<i class="fas fa-envelope"></i> <span>Will route to specialist team</span>'}
        </div>
    `;
    
    // Display enhanced customer response
    const customerResponse = document.getElementById('customer-response');
    const sourcesSection = document.getElementById('knowledge-sources');
    
    if (responseData.type === 'rag_response') {
        // Format the AI response with better styling
        const formattedAnswer = formatAIResponse(responseData.answer);
        customerResponse.innerHTML = `
            <div class="response-type-indicator">
                <i class="fas fa-robot"></i>
                <span>AI-Generated Response</span>
            </div>
            <div class="response-content-text">
                ${formattedAnswer}
            </div>
        `;
        
        if (responseData.sources && responseData.sources.length > 0) {
            sourcesSection.style.display = 'block';
            const sourcesList = document.getElementById('sources-list');
            sourcesList.innerHTML = responseData.sources.map((source, index) => {
                const sourceName = formatSourceName(source);
                // Ensure we always have a valid source name
                const displayName = sourceName && sourceName.trim() !== '' ? sourceName : `Documentation Source ${index + 1}`;
                
                // Debug logging for problematic URLs
                if (!sourceName || sourceName.trim() === '') {
                    console.warn('Empty source name for URL:', source);
                }
                
                return `
                    <a href="${source}" target="_blank" class="source-link" title="${source}">
                        <i class="fas fa-external-link-alt"></i>
                        <span class="source-number">${index + 1}.</span>
                        <span class="source-name">${displayName}</span>
                    </a>
                `;
            }).join('');
        } else {
            sourcesSection.style.display = 'none';
        }
    } else {
        // Format the routing message with better styling
        const formattedMessage = formatRoutingMessage(responseData.message);
        customerResponse.innerHTML = `
            <div class="response-type-indicator routing">
                <i class="fas fa-envelope"></i>
                <span>Ticket Routed Successfully</span>
            </div>
            <div class="response-content-text">
                ${formattedMessage}
            </div>
        `;
        sourcesSection.style.display = 'none';
    }
}

function formatAIResponse(answer) {
    // Format the AI response with better structure
    return answer
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/^/, '<p>')
        .replace(/$/, '</p>');
}

function formatRoutingMessage(message) {
    // Format the routing message with better structure
    return message
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/^/, '<p>')
        .replace(/$/, '</p>');
}

function formatSourceName(source) {
    // Extract and format source name from URL
    try {
        const urlParts = source.split('/');
        let fileName = urlParts[urlParts.length - 1];
        
        // Handle empty or invalid file names
        if (!fileName || fileName.trim() === '') {
            // Try to get the last meaningful part of the URL
            fileName = urlParts[urlParts.length - 2] || urlParts[urlParts.length - 3] || 'Documentation';
        }
        
        // Remove query parameters and fragments
        fileName = fileName.split('?')[0].split('#')[0];
        
        // Handle common URL patterns
        if (fileName === '' || fileName === '/') {
            fileName = 'Documentation';
        }
        
        // Format the name
        let formattedName = fileName
            .replace(/[-_]/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase())
            .replace(/\.(html|htm|php|asp|aspx|md)$/i, '')
            .trim();
        
        // Ensure we have a meaningful name
        if (!formattedName || formattedName.length < 2) {
            formattedName = 'Atlan Documentation';
        }
        
        return formattedName;
    } catch (error) {
        console.warn('Error formatting source name:', error);
        return 'Atlan Documentation';
    }
}

async function loadAnalytics() {
    if (classifiedTickets.length === 0) {
        document.getElementById('analytics-content').innerHTML = `
            <div class="analytics-placeholder">
                <i class="fas fa-chart-bar"></i>
                <p>Load and classify tickets first to see analytics</p>
            </div>
        `;
        return;
    }
    
    try {
        console.log('Loading analytics for', classifiedTickets.length, 'tickets');
        console.log('Sample ticket data:', classifiedTickets[0]);
        
        const response = await fetch('/api/analytics');
        const data = await response.json();
        
        console.log('Analytics API response:', data);
        
        if (data.success) {
            displayAnalyticsCharts(data);
        } else {
            throw new Error(data.error || 'Analytics loading failed');
        }
    } catch (error) {
        console.error('Error loading analytics:', error);
        document.getElementById('analytics-content').innerHTML = `
            <div class="analytics-placeholder">
                <i class="fas fa-exclamation-triangle"></i>
                <p>Error loading analytics: ${error.message}</p>
            </div>
        `;
    }
}

function displayAnalyticsCharts(data) {
    try {
        // Check if Plotly is available
        if (typeof Plotly === 'undefined') {
            console.error('Plotly library not loaded');
            document.getElementById('charts-container').innerHTML = `
                <div class="analytics-placeholder">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>Chart library not loaded. Please refresh the page.</p>
                </div>
            `;
            return;
        }
        
        // Hide placeholder and show charts
        const placeholder = document.querySelector('.analytics-placeholder');
        const chartsContainer = document.getElementById('charts-container');
        
        if (placeholder) {
            placeholder.style.display = 'none';
        }
        if (chartsContainer) {
            chartsContainer.style.display = 'block';
        }
        
        console.log('Analytics data received:', data);
        
        // Render charts using Plotly with error handling
        if (data.charts && data.charts.priority) {
            try {
                const priorityData = JSON.parse(data.charts.priority);
                Plotly.newPlot('priority-chart', priorityData.data, priorityData.layout, {responsive: true});
                console.log('Priority chart rendered successfully');
            } catch (error) {
                console.error('Error rendering priority chart:', error);
            }
        }
        
        if (data.charts && data.charts.sentiment) {
            try {
                const sentimentData = JSON.parse(data.charts.sentiment);
                console.log('Sentiment chart data:', sentimentData);
                
                // Clear any existing content
                document.getElementById('sentiment-chart').innerHTML = '';
                
                // Render the chart with additional configuration
                Plotly.newPlot('sentiment-chart', sentimentData.data, sentimentData.layout, {
                    responsive: true,
                    displayModeBar: true,
                    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
                });
                
                console.log('Sentiment chart rendered successfully');
                
                // Verify the chart was rendered by checking if it has content
                setTimeout(() => {
                    const chartElement = document.getElementById('sentiment-chart');
                    if (chartElement && chartElement.children.length === 0) {
                        console.warn('Chart container is empty after rendering attempt');
                        chartElement.innerHTML = `
                            <div style="padding: 20px; text-align: center; color: #dc3545;">
                                <i class="fas fa-exclamation-triangle"></i>
                                <p>Chart failed to render. Please refresh the page.</p>
                            </div>
                        `;
                    }
                }, 1000);
                
            } catch (error) {
                console.error('Error rendering sentiment chart:', error);
                // Show error message in the chart container
                document.getElementById('sentiment-chart').innerHTML = `
                    <div style="padding: 20px; text-align: center; color: #dc3545;">
                        <i class="fas fa-exclamation-triangle"></i>
                        <p>Error rendering sentiment chart: ${error.message}</p>
                    </div>
                `;
            }
        } else {
            console.warn('No sentiment chart data available');
            // Show placeholder message
            document.getElementById('sentiment-chart').innerHTML = `
                <div style="padding: 20px; text-align: center; color: #6c757d;">
                    <i class="fas fa-chart-bar"></i>
                    <p>No sentiment data available for chart</p>
                </div>
            `;
        }
        
        if (data.charts && data.charts.topics) {
            try {
                const topicsData = JSON.parse(data.charts.topics);
                Plotly.newPlot('topics-chart', topicsData.data, topicsData.layout, {responsive: true});
                console.log('Topics chart rendered successfully');
            } catch (error) {
                console.error('Error rendering topics chart:', error);
            }
        }
        
        if (data.charts && data.charts.channels) {
            try {
                const channelsData = JSON.parse(data.charts.channels);
                Plotly.newPlot('channels-chart', channelsData.data, channelsData.layout, {responsive: true});
                console.log('Channels chart rendered successfully');
            } catch (error) {
                console.error('Error rendering channels chart:', error);
            }
        }
        
    } catch (error) {
        console.error('Error in displayAnalyticsCharts:', error);
        // Show error message
        const chartsContainer = document.getElementById('charts-container');
        if (chartsContainer) {
            chartsContainer.innerHTML = `
                <div class="analytics-placeholder">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>Error displaying charts: ${error.message}</p>
                </div>
            `;
        }
    }
}

async function exportData(format) {
    if (classifiedTickets.length === 0) {
        alert('No data to export. Please classify tickets first.');
        return;
    }
    
    try {
        const response = await fetch(`/api/export/${format}`);
        const data = await response.json();
        
        if (data.success) {
            // Create download link
            const blob = new Blob([format === 'json' ? JSON.stringify(data.data, null, 2) : data.data], {
                type: format === 'json' ? 'application/json' : 'text/csv'
            });
            
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = data.filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        } else {
            throw new Error(data.error || 'Export failed');
        }
    } catch (error) {
        alert('Error exporting data: ' + error.message);
    }
}

// Utility functions
function formatDate(dateString) {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleDateString();
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

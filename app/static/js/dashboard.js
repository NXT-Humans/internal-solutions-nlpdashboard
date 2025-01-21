/**
 * dashboard.js
 * Handles all visualizations and interactivity for the NLP analysis dashboard
 */

document.addEventListener('DOMContentLoaded', function() {
    // Configuration
    const COLORS = {
        POSITIVE: '#22c55e',
        NEGATIVE: '#ef4444',
        NEUTRAL: '#6b7280',
        BACKGROUND: 'rgba(0,0,0,0)',
        GRID: '#f1f5f9'
    };

    const LAYOUT_CONFIG = {
        FONT_FAMILY: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        CHART_MIN_HEIGHT: 400,
        TOPICS_HEIGHT_PER_ITEM: 50,
        WORDCLOUD_HEIGHT: 600
    };

    // Load and validate dashboard data
    let dashboardData = null;
    try {
        const dataElement = document.getElementById('dashboardData');
        if (!dataElement) {
            throw new Error('Dashboard data element not found');
        }
        dashboardData = JSON.parse(dataElement.textContent);
        console.log('Dashboard data loaded successfully');
    } catch (error) {
        console.error('Failed to load dashboard data:', error);
        showError('Failed to load visualization data');
        return;
    }

    function createTopicsChart() {
        const topicsElement = document.getElementById('topics-chart');
        if (!topicsElement || !dashboardData.topics_data?.length) return;

        const trace = {
            y: dashboardData.topics_data.map(topic => topic.text),
            x: dashboardData.topics_data.map(topic => topic.frequency),
            type: 'bar',
            orientation: 'h',
            marker: {
                color: dashboardData.topics_data.map(topic => {
                    if (topic.sentiment === 'positive') return COLORS.POSITIVE;
                    if (topic.sentiment === 'negative') return COLORS.NEGATIVE;
                    return COLORS.NEUTRAL;
                }),
                opacity: 0.8
            },
            hovertemplate: `
                <b>%{y}</b><br>
                Frequency: %{x}<br>
                Sentiment: %{customdata}
                <extra></extra>
            `,
            customdata: dashboardData.topics_data.map(topic => topic.sentiment)
        };

        const layout = {
            font: { family: LAYOUT_CONFIG.FONT_FAMILY },
            title: {
                text: 'Topic Distribution by Sentiment',
                font: { size: 16 }
            },
            showlegend: false,
            xaxis: {
                title: 'Frequency',
                showgrid: true,
                gridcolor: COLORS.GRID,
                zeroline: false
            },
            yaxis: {
                automargin: true,
                tickfont: { size: 12 }
            },
            margin: { l: 10, r: 10, t: 40, b: 40 },
            height: Math.max(
                LAYOUT_CONFIG.CHART_MIN_HEIGHT,
                dashboardData.topics_data.length * LAYOUT_CONFIG.TOPICS_HEIGHT_PER_ITEM
            ),
            plot_bgcolor: COLORS.BACKGROUND,
            paper_bgcolor: COLORS.BACKGROUND
        };

        Plotly.newPlot('topics-chart', [trace], layout, {
            responsive: true,
            displayModeBar: false
        }).then(() => {
            topicsElement.on('plotly_click', data => {
                const topicData = dashboardData.topics_data[data.points[0].pointIndex];
                if (topicData) showTopicDetails(topicData);
            });
        });
    }

    function createWordCloud() {
        const wordcloudElement = document.getElementById('wordcloud-chart');
        if (!wordcloudElement || !dashboardData.wordcloud_data) return;

        const words = Object.entries(dashboardData.wordcloud_data)
            .map(([text, data]) => ({
                text,
                value: data.frequency,
                sentiment: data.sentiment,
                contexts: data.contexts
            }));

        const maxValue = Math.max(...words.map(word => word.value));

        const trace = {
            x: words.map(() => Math.random() * 100),
            y: words.map(() => Math.random() * 100),
            mode: 'text',
            text: words.map(word => word.text),
            textfont: {
                size: words.map(word => 
                    Math.max(12, Math.min((word.value / maxValue) * 60, 60))
                ),
                color: words.map(word => {
                    if (word.sentiment > 0.1) return COLORS.POSITIVE;
                    if (word.sentiment < -0.1) return COLORS.NEGATIVE;
                    return COLORS.NEUTRAL;
                }),
                family: LAYOUT_CONFIG.FONT_FAMILY
            },
            hoverinfo: 'text',
            hovertext: words.map(word => 
                `${word.text}\nFrequency: ${word.value}\nSentiment: ${word.sentiment.toFixed(2)}`
            )
        };

        const layout = {
            font: { family: LAYOUT_CONFIG.FONT_FAMILY },
            title: {
                text: 'Word Frequency & Sentiment Analysis',
                font: { size: 16 }
            },
            showlegend: false,
            xaxis: {
                showgrid: false,
                zeroline: false,
                showticklabels: false
            },
            yaxis: {
                showgrid: false,
                zeroline: false,
                showticklabels: false
            },
            margin: { l: 10, r: 10, t: 40, b: 10 },
            height: LAYOUT_CONFIG.WORDCLOUD_HEIGHT,
            plot_bgcolor: COLORS.BACKGROUND,
            paper_bgcolor: COLORS.BACKGROUND
        };

        Plotly.newPlot('wordcloud-chart', [trace], layout, {
            responsive: true,
            displayModeBar: false
        }).then(() => {
            wordcloudElement.on('plotly_click', data => {
                const word = data.points[0].text;
                const wordData = dashboardData.wordcloud_data[word];
                if (wordData) showWordDetails(word, wordData);
            });
        });
    }

    function showTopicDetails(topicData) {
        const contexts = findContexts(topicData.text);
        const modal = createModal({
            title: `Topic Analysis: "${topicData.text}"`,
            content: `
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-label">Frequency</div>
                        <div class="stat-value">${topicData.frequency} occurrences</div>
                    </div>
                    <div class="stat-item ${getSentimentClass(topicData.sentiment)}">
                        <div class="stat-label">Sentiment</div>
                        <div class="stat-value">${topicData.sentiment}</div>
                    </div>
                </div>

                <div class="contexts-section">
                    <h4>Source Excerpts</h4>
                    ${contexts.map(context => `
                        <div class="context-item">
                            <div class="context-header">
                                <span class="context-document">${context.document}</span>
                                <span class="context-sentiment ${getSentimentClass(context.sentiment)}">
                                    ${context.sentiment.toFixed(2)}
                                </span>
                            </div>
                            <div class="context-text">
                                ${highlightText(context.text, topicData.text)}
                            </div>
                        </div>
                    `).join('')}
                </div>
            `
        });
        
        document.body.appendChild(modal);
    }

    function showWordDetails(word, wordData) {
        const contexts = wordData.contexts || findContexts(word);
        const modal = createModal({
            title: `Word Analysis: "${word}"`,
            content: `
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-label">Frequency</div>
                        <div class="stat-value">${wordData.frequency} occurrences</div>
                    </div>
                    <div class="stat-item ${getSentimentClass(wordData.sentiment)}">
                        <div class="stat-label">Sentiment</div>
                        <div class="stat-value">${wordData.sentiment.toFixed(2)}</div>
                    </div>
                </div>

                <div class="contexts-section">
                    <h4>Examples in Context</h4>
                    ${contexts.map(context => `
                        <div class="context-item">
                            <div class="context-header">
                                <span class="context-document">${context.document}</span>
                                <span class="context-sentiment ${getSentimentClass(context.sentiment)}">
                                    ${context.sentiment.toFixed(2)}
                                </span>
                            </div>
                            <div class="context-text">
                                ${highlightText(context.text, word)}
                            </div>
                        </div>
                    `).join('')}
                </div>
            `
        });
        
        document.body.appendChild(modal);
    }

    function findContexts(searchTerm) {
        const contexts = [];
        if (!dashboardData.documents) return contexts;

        const searchTermLower = searchTerm.toLowerCase();
        
        // Search through all documents
        for (const [docName, docData] of Object.entries(dashboardData.documents)) {
            const paragraphs = docData.cleaned_paragraphs || [];
            const sentiments = docData.paragraph_sentiments || [];
            
            paragraphs.forEach((paragraph, index) => {
                if (paragraph.toLowerCase().includes(searchTermLower)) {
                    contexts.push({
                        document: docName,
                        text: paragraph,
                        sentiment: sentiments[index]?.polarity || 0
                    });
                }
            });
        }
        
        // Sort by sentiment strength and limit results
        return contexts
            .sort((a, b) => Math.abs(b.sentiment) - Math.abs(a.sentiment))
            .slice(0, 5);
    }

    function getSentimentClass(score) {
        if (score < -0.1) return 'negative';
        if (score > 0.1) return 'positive';
        return 'neutral';
    }

    function highlightText(text, searchTerm) {
        if (!text || !searchTerm) return text;
        const regex = new RegExp(`(${searchTerm})`, 'gi');
        return text.replace(regex, '<mark>$1</mark>');
    }

    function createModal({ title, content }) {
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3 class="modal-title">${title}</h3>
                    <button class="modal-close" aria-label="Close">&times;</button>
                </div>
                <div class="modal-body">${content}</div>
            </div>
        `;

        // Event handlers
        const closeButton = modal.querySelector('.modal-close');
        closeButton.onclick = () => modal.remove();
        
        modal.onclick = event => {
            if (event.target === modal) modal.remove();
        };

        // Keyboard handler
        document.addEventListener('keydown', function closeOnEscape(event) {
            if (event.key === 'Escape') {
                modal.remove();
                document.removeEventListener('keydown', closeOnEscape);
            }
        });

        return modal;
    }

    function showError(message) {
        const errorElement = document.createElement('div');
        errorElement.className = 'error-message';
        errorElement.textContent = message;

        document.querySelectorAll('.chart-container').forEach(container => {
            container.innerHTML = '';
            container.appendChild(errorElement.cloneNode(true));
        });
    }

    // Initialize visualizations
    createTopicsChart();
    createWordCloud();

    // Handle window resizing
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            createTopicsChart();
            createWordCloud();
        }, 250);
    });
});
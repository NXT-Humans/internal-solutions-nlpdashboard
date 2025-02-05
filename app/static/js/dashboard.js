/**
 * dashboard.js - Production-Quality Dashboard JavaScript
 *
 * This file handles all visualizations and interactivity for the NLP analysis dashboard.
 * It loads data from the API endpoint, initializes summary panels, a hierarchical treemap for topics,
 * an enhanced word cloud visualizing trigram frequencies with sentiment coloring,
 * and sets up dynamic modal pop-ups for context paragraphs triggered from both the treemap and word cloud.
 */

const COLORS = {
    POSITIVE: '#22c55e',
    NEGATIVE: '#ef4444',
    NEUTRAL: '#6b7280',
    BACKGROUND: 'rgba(0,0,0,0)',
    GRID: '#f1f5f9'
};

const LAYOUT_CONFIG = {
    FONT_FAMILY: 'Segoe UI, Roboto, sans-serif',
    CHART_MIN_HEIGHT: 400,
    WORDCLOUD_HEIGHT: 600,
    TOPICS_HEIGHT_PER_ITEM: 40
};

class ModalManager {
    constructor() {
        this.modal = document.getElementById('insightModal');
        this.modalTitle = this.modal.querySelector('.modal-title');
        this.modalBody = this.modal.querySelector('.modal-body');
        this.setupEventListeners();
    }
    setupEventListeners() {
        const closeButton = this.modal.querySelector('.modal-close');
        if (closeButton) {
            closeButton.addEventListener('click', () => this.hideModal());
        }
        this.modal.addEventListener('click', (e) => {
            if (e.target === this.modal) this.hideModal();
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') this.hideModal();
        });
    }
    showModal(title, content) {
        this.modalTitle.textContent = title;
        this.modalBody.innerHTML = content;
        this.modal.style.display = 'flex';
        document.body.style.overflow = 'hidden';
    }
    hideModal() {
        this.modal.style.display = 'none';
        document.body.style.overflow = '';
    }
}

class DashboardManager {
    constructor() {
        this.data = null;
        this.modalManager = new ModalManager();
        this.contextCache = new Map();
    }
    async initialize() {
        try {
            const response = await fetch('/api/analysis-results');
            if (!response.ok) {
                throw new Error(`API Error: ${response.status}`);
            }
            const dashboardData = await response.json();
            if (!this.validateDashboardData(dashboardData)) {
                throw new Error('Invalid dashboard data structure');
            }
            this.data = dashboardData;
            this.initializeSummaryPanels();
            this.initializeTopicsChart();
            this.initializeWordCloud();
            this.setupEventListeners();
        } catch (error) {
            console.error('Dashboard initialization failed:', error);
            this.handleInitializationError();
        }
    }
    validateDashboardData(data) {
        const requiredKeys = ['global_summary', 'global_topics', 'wordcloud_data', 'documents'];
        return requiredKeys.every(key => key in data);
    }
    handleInitializationError() {
        alert("Failed to load dashboard data after multiple attempts.");
    }
    initializeSummaryPanels() {
        const summaryData = this.data.global_summary || {};
        ['overview', 'findings', 'challenges', 'solutions'].forEach(section => {
            const element = document.getElementById(`${section}Section`);
            if (element) {
                element.innerHTML = this.formatSummaryContent(summaryData[section]);
            }
        });
    }
    formatSummaryContent(content) {
        return content ? content.replace(/\n/g, '<br>') : "No content available.";
    }
    initializeTopicsChart() {
        const topics = this.data.global_topics || [];
        if (!topics.length) return;
        const hierarchy = this.buildTopicHierarchy(topics);
        const trace = {
            type: 'treemap',
            ids: hierarchy.ids,
            labels: hierarchy.labels,
            parents: hierarchy.parents,
            values: hierarchy.values,
            textinfo: "label+value",
            marker: { colors: hierarchy.colors },
            hovertemplate: '<b>%{label}</b><br>Size: %{value}<br>Parent: %{parent}<extra></extra>'
        };
        const layout = {
            title: 'Topic Hierarchy and Relationships',
            font: { family: LAYOUT_CONFIG.FONT_FAMILY },
            height: Math.max(LAYOUT_CONFIG.CHART_MIN_HEIGHT, hierarchy.labels.length * LAYOUT_CONFIG.TOPICS_HEIGHT_PER_ITEM),
            margin: { l: 0, r: 0, b: 0, t: 40 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        };
        const chartElement = document.getElementById('topics-chart');
        Plotly.newPlot(chartElement, [trace], layout, { responsive: true });
        chartElement.on('plotly_click', (data) => {
            const point = data.points[0];
            const topic = this.findTopicByTitle(point.label);
            if (topic) {
                this.showTopicModal(topic);
            }
        });
    }
    buildTopicHierarchy(topicsData) {
        const ids = [];
        const labels = [];
        const parents = [];
        const values = [];
        const colors = [];
        topicsData.forEach(topic => {
            ids.push(topic.title);
            labels.push(topic.title);
            parents.push(topic.parent || "");
            values.push(topic.cluster_size || 1);
            colors.push(COLORS.NEUTRAL);
        });
        return { ids, labels, parents, values, colors };
    }
    initializeWordCloud() {
        const wordData = this.data.wordcloud_data || {};
        if (Object.keys(wordData).length === 0) return;
        const cloudElement = document.getElementById('wordcloud-chart');
        if (!cloudElement) return;
        const wordsArray = Object.entries(wordData)
            .map(([word, data]) => ({
                text: word.length > 30 ? word.substring(0, 30) + '...' : word,
                value: typeof data === 'number' ? data : data.frequency || 1,
                sentiment: (typeof data === 'object' && data.sentiment) || 'neutral',
                category: (typeof data === 'object' && data.category) || 'general'
            }))
            .filter(w => w.text.length > 2);
        const categoryGroups = {};
        wordsArray.forEach(word => {
            if (!categoryGroups[word.category]) {
                categoryGroups[word.category] = [];
            }
            categoryGroups[word.category].push(word);
        });
        const traces = [];
        let yOffset = 0;
        Object.entries(categoryGroups).forEach(([category, words]) => {
            const trace = {
                x: words.map((_, i) => (i % 5) * 20),
                y: words.map((_, i) => yOffset + Math.floor(i / 5) * 20),
                mode: 'text',
                text: words.map(w => w.text),
                textfont: {
                    size: words.map(w => Math.max(12, (w.value / Math.max(...wordsArray.map(w => w.value))) * 40)),
                    color: words.map(w => w.sentiment === 'positive' ? COLORS.POSITIVE :
                                            w.sentiment === 'negative' ? COLORS.NEGATIVE : COLORS.NEUTRAL)
                },
                name: category,
                hoverinfo: 'text',
                hovertext: words.map(w => `${w.text}\nFrequency: ${w.value}\nCategory: ${w.category}`)
            };
            traces.push(trace);
            yOffset += Math.ceil(words.length / 5) * 25;
        });
        const layout = {
            title: 'Word Frequency & Sentiment by Category',
            showlegend: true,
            height: Math.max(LAYOUT_CONFIG.WORDCLOUD_HEIGHT, yOffset + 100),
            margin: { t: 40, b: 40 },
            xaxis: { showgrid: false, zeroline: false, showticklabels: false },
            yaxis: { showgrid: false, zeroline: false, showticklabels: false }
        };
        Plotly.newPlot(cloudElement, traces, layout, { responsive: true }).then(() => {
            cloudElement.on('plotly_click', (data) => {
                const point = data.points[0];
                const topic = this.findTopicByTitle(point.text);
                if (topic) {
                    this.showTopicModal(topic);
                }
            });
        });
    }
    setupEventListeners() {
        const topicsList = document.getElementById('topics-list');
        if (topicsList) {
            topicsList.addEventListener('click', (e) => {
                const topicItem = e.target.closest('.topic-item');
                if (topicItem) {
                    const topicId = topicItem.dataset.topicId;
                    const topic = this.findTopicByTitle(topicId);
                    if (topic) {
                        this.showTopicModal(topic);
                    }
                }
            });
        }
    }
    findTopicByTitle(title) {
        const cleanTitle = title.replace(/<[^>]*>/g, '').split('\n')[0].trim();
        return this.data.global_topics.find(t => t.title === cleanTitle);
    }
    showTopicModal(topic) {
        if (this.contextCache.has(topic.title)) {
            const cachedContent = this.contextCache.get(topic.title);
            this.modalManager.showModal(`Topic: ${topic.title}`, cachedContent);
            return;
        }
        const modalContent = this.formatModalContent(topic);
        this.contextCache.set(topic.title, modalContent);
        this.modalManager.showModal(`Topic: ${topic.title}`, modalContent);
    }
    formatModalContent(topic) {
        const contexts = topic.paragraphs || [];
        const keyPhrases = this.extractKeyPhrases(contexts);
        return `
            <div class="topic-modal-content">
                <div class="topic-header mb-4 border-b pb-4">
                    <h3 class="text-xl font-bold mb-2">${topic.title}</h3>
                    <div class="flex items-center gap-3 text-sm text-gray-600">
                        <span>Type: ${topic.type || 'General'}</span>
                        <span>Size: ${topic.cluster_size || 0}</span>
                    </div>
                </div>
                ${this.formatKeyPhrases(keyPhrases)}
                <div class="context-paragraphs mt-6 space-y-6">
                    <h4 class="text-lg font-semibold mb-3">Related Paragraphs</h4>
                    ${contexts.map((context, idx) => this.formatContextParagraph(context, topic, idx + 1)).join('')}
                </div>
            </div>
        `;
    }
    formatContextParagraph(paragraph, topic, index) {
        const highlightedText = this.highlightTopicMentions(paragraph, topic.title);
        const relevance = this.computeRelevance(paragraph, topic.title);
        return `
            <div class="context-paragraph bg-gray-50 rounded-lg p-4">
                <div class="context-header flex justify-between items-center mb-2">
                    <span class="text-sm font-medium text-gray-700">Paragraph ${index}</span>
                    <span class="text-sm text-gray-500">Relevance: ${(relevance * 100).toFixed(1)}%</span>
                </div>
                <div class="context-text prose max-w-none text-gray-800">
                    ${highlightedText}
                </div>
            </div>
        `;
    }
    highlightTopicMentions(text, topic) {
        const escapedTopic = topic.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const regex = new RegExp(`(${escapedTopic})`, 'gi');
        return text.replace(regex, '<mark class="bg-yellow-100 px-1 rounded">$1</mark>');
    }
    extractKeyPhrases(contexts) {
        const phrases = new Set();
        const patterns = [
            /\b\w+\s+(?:method|approach|algorithm|system|framework)\b/gi,
            /\b(?:propose|present|develop)\s+\w+(?:\s+\w+){0,2}\b/gi
        ];
        contexts.forEach(context => {
            patterns.forEach(pattern => {
                const matches = context.match(pattern);
                if (matches) {
                    matches.forEach(match => phrases.add(match));
                }
            });
        });
        return Array.from(phrases);
    }
    formatKeyPhrases(phrases) {
        if (!phrases.length) return '';
        return `
            <div class="key-phrases mt-4">
                <h4 class="text-sm font-medium text-gray-700 mb-2">Key Phrases</h4>
                <div class="flex flex-wrap gap-2">
                    ${phrases.map(phrase => `
                        <span class="inline-block px-2 py-1 text-sm bg-gray-100 text-gray-700 rounded">
                            ${phrase}
                        </span>
                    `).join('')}
                </div>
            </div>
        `;
    }
    computeRelevance(text, topic) {
        const termCount = (text.match(new RegExp(topic, 'gi')) || []).length;
        const words = text.split(/\s+/).length;
        return Math.min(termCount / (words / 10), 1);
    }
}

const dashboard = new DashboardManager();
document.addEventListener('DOMContentLoaded', () => {
    dashboard.initialize();
});

"""
viz_utils.py - Visualization Utilities

This module contains helper functions for advanced or custom visualizations.
"""

from typing import Dict, Any

def generate_wordcloud_svg(word_frequencies: Dict[str, int]) -> str:
    """
    Generate an SVG representation of a word cloud.

    Args:
        word_frequencies: Dictionary mapping words to their frequency.

    Returns:
        A complete SVG string.
    """
    svg_content = "<svg xmlns='http://www.w3.org/2000/svg' width='600' height='400'>"
    x, y = 10, 20
    for word, freq in word_frequencies.items():
        font_size = 10 + freq
        svg_content += f"<text x='{x}' y='{y}' font-size='{font_size}'>{word}</text>"
        y += font_size + 5
        if y > 380:
            y = 20
            x += 100
    svg_content += "</svg>"
    return svg_content

def generate_mindmap_data(topics: list) -> Dict[str, Any]:
    """
    Generate a node-link structure for a mind map based on topics.

    Args:
        topics: List of topic strings.

    Returns:
        A dictionary with 'nodes' and 'links' for visualization.
    """
    nodes = [{"id": topic, "label": topic} for topic in topics]
    links = []
    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            links.append({"source": topics[i], "target": topics[j], "weight": 1})
    return {"nodes": nodes, "links": links}

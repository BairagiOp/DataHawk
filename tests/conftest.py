"""Shared test fixtures for DataHawk test suite."""

import os
import sys
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_html():
    """Simple HTML page for testing."""
    return """
    <html>
    <head>
        <title>Test Product Page</title>
        <script type="application/ld+json">
        {"@type": "Product", "name": "Test Widget", "price": "29.99"}
        </script>
    </head>
    <body>
        <nav>Navigation Menu | Home | About | Contact</nav>
        <header><h1>Our Store</h1></header>
        <main>
            <article>
                <h2>Test Widget</h2>
                <p class="price">$29.99</p>
                <p class="rating">4.5 out of 5 stars</p>
                <p class="availability">In Stock</p>
                <p>This is a fantastic widget for all your needs.</p>
            </article>
        </main>
        <aside>
            <div class="advertisement">Buy Premium!</div>
            <div class="social-share">Share on Twitter</div>
        </aside>
        <footer>Copyright 2025 Test Store</footer>
        <script>var analytics = {"page": "product"};</script>
        <style>.price { color: red; }</style>
    </body>
    </html>
    """


@pytest.fixture
def sample_dynamic_html():
    """HTML that looks like a JS-heavy SPA."""
    return """
    <html>
    <head><title>SPA App</title></head>
    <body>
        <div id="root"></div>
        <noscript>You need JavaScript enabled to run this app.</noscript>
        <script src="bundle.js"></script>
        <script src="vendor.js"></script>
        <script src="app.js"></script>
        <script>
        var __NEXT_DATA__ = {"props": {"data": []}};
        window.__REACT_DEVTOOLS_GLOBAL_HOOK__ = {};
        </script>
    </body>
    </html>
    """


@pytest.fixture
def sample_table_html():
    """HTML with structured table data."""
    return """
    <html><body>
    <h1>Product Catalog</h1>
    <table>
        <thead><tr><th>Name</th><th>Price</th><th>Rating</th></tr></thead>
        <tbody>
            <tr><td>Widget A</td><td>$19.99</td><td>4.2</td></tr>
            <tr><td>Widget B</td><td>$24.99</td><td>3.8</td></tr>
            <tr><td>Widget C</td><td>$14.99</td><td>4.7</td></tr>
        </tbody>
    </table>
    </body></html>
    """


@pytest.fixture
def sample_schema_dict():
    """Sample schema dictionary."""
    return {
        "description": "Extract product information",
        "fields": {
            "product_name": {"type": "string", "required": True, "description": "Product name"},
            "price": {"type": "number", "required": True, "description": "Product price", "constraints": {"min": 0}},
            "rating": {"type": "number", "required": False, "description": "Rating", "constraints": {"min": 0, "max": 5}},
            "availability": {"type": "categorical", "required": True, "description": "Stock status"},
        },
        "required": ["product_name", "price", "availability"],
    }

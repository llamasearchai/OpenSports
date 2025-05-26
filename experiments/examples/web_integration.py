#!/usr/bin/env python3
"""
Example of integrating the OpenInsight Experiment Service with a web application.

This example shows how to:
1. Create experiments
2. Assign variants to users in a web context
3. Track conversions
4. Retrieve and display experiment results

Run this example with: uvicorn web_integration:app --reload
"""

import sys
import os
import uuid
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, Request, Form, Cookie, Response, Depends, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import json

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from OpenInsight.experiments.experiment_service import (
    ExperimentManager,
    get_experiment_manager,
    ExperimentType
)

# Create FastAPI app
app = FastAPI(title="Experiment Service Web Integration")

# Create a folder for templates and static files in the same directory as this script
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Create directories if they don't exist
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Set up templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Create templates/index.html if it doesn't exist
INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Experiment Service Demo</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .button {
            display: inline-block;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            border: none;
            border-radius: 4px;
            text-align: center;
            text-decoration: none;
            color: white;
            margin: 10px 5px;
        }
        .blue-button {
            background-color: #007bff;
        }
        .green-button {
            background-color: #28a745;
        }
        .container {
            border: 1px solid #ddd;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .price {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        .highlight {
            background-color: #ffffcc;
        }
    </style>
</head>
<body>
    <h1>OpenInsight Experiment Service Demo</h1>
    
    <div class="container">
        <h2>Button Color Experiment</h2>
        <p>We're testing which button color leads to more sign-ups.</p>
        <p>Your variant: <strong>{{ button_variant }}</strong></p>
        
        <a href="/signup?source=button" class="button {{ button_color }}-button">Sign Up Now</a>
    </div>
    
    <div class="container">
        <h2>Pricing Experiment</h2>
        <p>We're testing different price points to optimize conversions.</p>
        <p>Your variant: <strong>{{ price_variant }}</strong></p>
        
        <div class="price">${{ price }}</div>
        <p>Get access to all premium features today!</p>
        
        <a href="/purchase?price={{ price }}" class="button blue-button">Purchase Now</a>
    </div>
    
    {% if show_admin %}
    <div class="container">
        <h2>Admin: Experiment Results</h2>
        
        <h3>Button Color Experiment</h3>
        {% if button_results %}
            <table>
                <tr>
                    <th>Variant</th>
                    <th>Impressions</th>
                    <th>Conversions</th>
                    <th>Rate</th>
                    <th>P-value</th>
                    <th>Improvement</th>
                </tr>
                {% for variant in button_results.variants %}
                <tr {% if variant.is_control %}class="highlight"{% endif %}>
                    <td>{{ variant.name }}{% if variant.is_control %} (Control){% endif %}</td>
                    <td>{{ variant.impressions }}</td>
                    <td>{{ variant.conversions }}</td>
                    <td>{{ "%.2f%%" | format(variant.conversion_rate * 100) }}</td>
                    <td>{{ "%.4f" | format(variant.p_value) if variant.p_value != None else "N/A" }}</td>
                    <td>{{ "%.2f%%" | format(variant.relative_improvement) if not variant.is_control else "N/A" }}</td>
                </tr>
                {% endfor %}
            </table>
            {% if button_results.winner %}
            <p><strong>Winner:</strong> {{ button_results.winner.name }} ({{ "%.2f%%" | format(button_results.winner.relative_improvement) }} improvement, p-value: {{ "%.4f" | format(button_results.winner.p_value) }})</p>
            {% else %}
            <p>No clear winner yet.</p>
            {% endif %}
        {% else %}
            <p>No results available yet.</p>
        {% endif %}
        
        <h3>Pricing Experiment</h3>
        {% if price_results %}
            <table>
                <tr>
                    <th>Variant</th>
                    <th>Impressions</th>
                    <th>Conversions</th>
                    <th>Rate</th>
                    <th>Avg Value</th>
                    <th>P-value</th>
                    <th>Improvement</th>
                </tr>
                {% for variant in price_results.variants %}
                <tr {% if variant.is_control %}class="highlight"{% endif %}>
                    <td>{{ variant.name }}{% if variant.is_control %} (Control){% endif %}</td>
                    <td>{{ variant.impressions }}</td>
                    <td>{{ variant.conversions }}</td>
                    <td>{{ "%.2f%%" | format(variant.conversion_rate * 100) }}</td>
                    <td>${{ "%.2f" | format(variant.avg_conversion_value) }}</td>
                    <td>{{ "%.4f" | format(variant.p_value) if variant.p_value != None else "N/A" }}</td>
                    <td>{{ "%.2f%%" | format(variant.relative_improvement) if not variant.is_control else "N/A" }}</td>
                </tr>
                {% endfor %}
            </table>
            {% if price_results.winner %}
            <p><strong>Winner:</strong> {{ price_results.winner.name }} ({{ "%.2f%%" | format(price_results.winner.relative_improvement) }} improvement, p-value: {{ "%.4f" | format(price_results.winner.p_value) }})</p>
            {% else %}
            <p>No clear winner yet.</p>
            {% endif %}
        {% else %}
            <p>No results available yet.</p>
        {% endif %}
    </div>
    {% endif %}
</body>
</html>
"""

# Write template if it doesn't exist
INDEX_PATH = os.path.join(TEMPLATES_DIR, "index.html")
if not os.path.exists(INDEX_PATH):
    with open(INDEX_PATH, "w") as f:
        f.write(INDEX_TEMPLATE)

# Set up static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Experiment IDs (normally these would be stored in a database)
BUTTON_EXPERIMENT_ID = None
PRICING_EXPERIMENT_ID = None

@app.on_event("startup")
async def startup_event():
    """Initialize experiments on startup."""
    global BUTTON_EXPERIMENT_ID, PRICING_EXPERIMENT_ID
    
    # Get experiment manager
    manager = get_experiment_manager()
    
    # Create button color experiment if it doesn't exist
    if not BUTTON_EXPERIMENT_ID:
        experiment = manager.create_experiment(
            name="Button Color Experiment",
            variants=[
                {"name": "Blue Button", "description": "Standard blue button"},
                {"name": "Green Button", "description": "New green button"}
            ],
            experiment_type="ab_test",
            traffic_allocation=1.0,
            description="Testing whether a green button performs better than our standard blue"
        )
        BUTTON_EXPERIMENT_ID = experiment.experiment_id
    
    # Create pricing experiment if it doesn't exist
    if not PRICING_EXPERIMENT_ID:
        experiment = manager.create_experiment(
            name="Pricing Strategy Experiment",
            variants=[
                {"name": "$19.99", "description": "Standard price"},
                {"name": "$24.99", "description": "Premium price"},
                {"name": "$14.99", "description": "Discount price"}
            ],
            experiment_type="multi_armed_bandit",
            traffic_allocation=1.0,
            description="Testing different price points to maximize revenue"
        )
        PRICING_EXPERIMENT_ID = experiment.experiment_id

def get_or_create_user_id(request: Request, response: Response) -> str:
    """Get or create a user ID from cookies."""
    user_id = request.cookies.get("user_id")
    if not user_id:
        user_id = str(uuid.uuid4())
        response.set_cookie(key="user_id", value=user_id, max_age=31536000)  # 1 year
    return user_id

@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request, 
    response: Response,
    admin: bool = Query(False)
):
    """Homepage with experiments."""
    user_id = get_or_create_user_id(request, response)
    manager = get_experiment_manager()
    
    # Get button variant
    button_variant = manager.get_variant_for_user(BUTTON_EXPERIMENT_ID, user_id)
    button_color = "blue"  # Default
    button_variant_name = "Default"
    
    if button_variant:
        button_variant_name = button_variant["name"]
        button_color = "green" if "Green" in button_variant_name else "blue"
    
    # Get price variant
    price_variant = manager.get_variant_for_user(PRICING_EXPERIMENT_ID, user_id)
    price = "19.99"  # Default
    price_variant_name = "Default"
    
    if price_variant:
        price_variant_name = price_variant["name"]
        price = price_variant_name.replace("$", "")
    
    # Get experiment results for admin view
    button_results = None
    price_results = None
    
    if admin:
        button_results = manager.analyze_experiment(BUTTON_EXPERIMENT_ID)
        price_results = manager.analyze_experiment(PRICING_EXPERIMENT_ID)
    
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "button_color": button_color,
            "button_variant": button_variant_name,
            "price": price,
            "price_variant": price_variant_name,
            "show_admin": admin,
            "button_results": button_results,
            "price_results": price_results
        }
    )

@app.get("/signup")
async def signup(request: Request, response: Response, source: str = "unknown"):
    """Handle sign-up conversion."""
    user_id = get_or_create_user_id(request, response)
    manager = get_experiment_manager()
    
    # Only record conversion if source is button (from our experiment)
    if source == "button":
        button_variant = manager.get_variant_for_user(BUTTON_EXPERIMENT_ID, user_id)
        if button_variant:
            manager.record_conversion(BUTTON_EXPERIMENT_ID, button_variant["variant_id"])
    
    # Normally would redirect to sign-up form, but we'll simulate success
    return {"status": "success", "message": "Signed up successfully"}

@app.get("/purchase")
async def purchase(request: Request, response: Response, price: str = "19.99"):
    """Handle purchase conversion."""
    user_id = get_or_create_user_id(request, response)
    manager = get_experiment_manager()
    
    # Convert price to float
    price_value = float(price)
    
    # Record conversion
    price_variant = manager.get_variant_for_user(PRICING_EXPERIMENT_ID, user_id)
    if price_variant:
        manager.record_conversion(PRICING_EXPERIMENT_ID, price_variant["variant_id"], price_value)
    
    # Normally would redirect to payment processing, but we'll simulate success
    return {"status": "success", "message": f"Purchased successfully at ${price}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
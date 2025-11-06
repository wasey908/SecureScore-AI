"""
Fraud Detection & Risk Scoring System - Main Server Module

This module imports the FastAPI application from app.main to maintain
compatibility with the existing supervisor configuration.
"""

from app.main import app

# The app is now defined in app/main.py
# This file exists for backward compatibility with supervisor config
"""
Gemini AI Utilities for Statistical Analysis Web Portal
This module handles all direct interactions with the Gemini API for generating
AI-powered narrative summaries of statistical analyses.
"""

import streamlit as st
import google.generativeai as genai
from typing import Optional


def set_gemini_api_key(api_key: str) -> bool:
    """
    Configure the Gemini API with the provided API key
    
    Parameters:
    -----------
    api_key : str
        The Gemini API key provided by the user
        
    Returns:
    --------
    bool
        True if configuration was successful, False otherwise
    """
    try:
        if not api_key or api_key.strip() == "":
            st.session_state.gemini_api_key_configured = False
            return False
            
        # Configure the Gemini API
        genai.configure(api_key=api_key.strip())
        
        # Test the API key by making a simple request
        model = genai.GenerativeModel('gemini-1.5-flash')
        test_response = model.generate_content("Hello")
        
        # If we get here, the API key is valid
        st.session_state.gemini_api_key = api_key.strip()
        st.session_state.gemini_api_key_configured = True
        return True
        
    except Exception as e:
        st.session_state.gemini_api_key_configured = False
        st.error(f"Error configuring Gemini API: {str(e)}")
        return False


def get_gemini_summary(text_data_to_analyze: str) -> str:
    """
    Generate an AI-powered narrative summary of statistical analysis results
    
    Parameters:
    -----------
    text_data_to_analyze : str
        Consolidated string containing all the analysis results to be summarized
        
    Returns:
    --------
    str
        AI-generated narrative summary or error message
    """
    try:
        # Check if API key is configured
        if not st.session_state.get('gemini_api_key_configured', False):
            return "Error: Gemini API key is not configured. Please set your API key in the sidebar."
        
        # Get selected model from session state or default to Flash
        selected_model = st.session_state.get('selected_gemini_model', 'gemini-1.5-flash')
        
        # Initialize the Gemini model
        model = genai.GenerativeModel(selected_model)
        
        # Construct a detailed prompt for statistical analysis summary
        prompt = f"""
        As an expert biostatistician and data analyst, please provide a comprehensive narrative summary of the following statistical analysis results. 

        Your summary should be:
        1. Professional and scientifically accurate
        2. Accessible to both technical and non-technical audiences
        3. Structured with clear sections and conclusions
        4. Include key findings, statistical significance, and practical implications
        5. Highlight any notable patterns, trends, or relationships in the data
        6. Provide interpretation of p-values, confidence intervals, and effect sizes where applicable
        7. Include recommendations for further analysis if appropriate

        Please format your response with clear headings and bullet points where appropriate.

        STATISTICAL ANALYSIS DATA:
        {text_data_to_analyze}

        Please provide a detailed narrative summary of these results, focusing on the key findings and their practical significance.
        """
        
        # Generate content using Gemini
        response = model.generate_content(prompt)
        
        if response and response.text:
            return response.text
        else:
            return "Error: No response received from Gemini API. Please try again."
            
    except Exception as e:
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg or "invalid API key" in error_msg.lower():
            return "Error: Invalid Gemini API key. Please check your API key and try again."
        elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
            return "Error: API quota exceeded. Please check your Gemini API usage limits."
        elif "permission" in error_msg.lower():
            return "Error: Permission denied. Please check your API key permissions."
        else:
            return f"Error generating AI summary: {error_msg}"


def check_gemini_api_status() -> dict:
    """
    Check the status of the Gemini API configuration
    
    Returns:
    --------
    dict
        Dictionary containing status information
    """
    status = {
        'configured': st.session_state.get('gemini_api_key_configured', False),
        'api_key_present': bool(st.session_state.get('gemini_api_key', '')),
        'ready': False
    }
    
    status['ready'] = status['configured'] and status['api_key_present']
    
    return status
#!/bin/bash

# Stock Market Analysis App Launcher
# This script runs the Streamlit application in Terminal

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory
cd "$SCRIPT_DIR"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed or not in PATH"
    echo "Please install Python3 and try again"
    read -p "Press any key to exit..."
    exit 1
fi

# Check if streamlit is installed
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "Streamlit is not installed"
    echo "Installing required packages..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Failed to install requirements"
        echo "Please run: pip3 install -r requirements.txt"
        read -p "Press any key to exit..."
        exit 1
    fi
fi

# Clear the terminal
clear

# Display banner
echo "=================================================="
echo "    Stock Market Analysis & Prediction App"
echo "=================================================="
echo ""
echo "Starting Streamlit application..."
echo "The app will open in your default web browser"
echo ""
echo "To stop the application, press Ctrl+C"
echo ""

# Kill any existing Streamlit processes
echo "Checking for existing Streamlit processes..."
pkill -f "streamlit run" 2>/dev/null || true
sleep 2

# Find an available port starting from 8501
PORT=8501
while lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; do
    echo "Port $PORT is in use, trying next port..."
    PORT=$((PORT + 1))
done

echo "Starting Streamlit on port $PORT..."

# Run the Streamlit app on the available port and automatically open browser
streamlit run Main.py --server.port $PORT --server.headless false --browser.gatherUsageStats false &

# Wait a moment for Streamlit to start
sleep 3

# Open the browser automatically
echo "Opening browser..."
open "http://localhost:$PORT"

# Wait for the Streamlit process to finish
wait

# Keep terminal open after the app closes
echo ""
echo "Application has been closed."
read -p "Press any key to exit..."

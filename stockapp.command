#!/bin/bash
cd "/Users/manojkumar/Programs Lang/Stock Market Project(Major pro)"

# Kill any existing streamlit processes
pkill -f streamlit 2>/dev/null

# Wait a moment for processes to stop
sleep 2

# Run the app
python3 -m streamlit run Main.py

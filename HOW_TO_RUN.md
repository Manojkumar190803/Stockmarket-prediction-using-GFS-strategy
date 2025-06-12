# How to Run Your Stock Market Analysis App

You now have **two ways** to run your Stock Market Analysis application with a single click:

## Option 1: Double-click the App (Recommended)
1. **Double-click** on `Stock Market App.app` 
2. This will automatically:
   - Open Terminal
   - Navigate to the correct directory
   - Run your Streamlit application
   - Open the app in your web browser

## Option 2: Run the Shell Script
1. **Double-click** on `run_stock_app.sh`
2. If prompted, choose "Open with Terminal"
3. The script will run your application

## What Happens When You Run It:
- ✅ Checks if Python3 is installed
- ✅ Checks if required packages are installed (installs them if missing)
- ✅ Starts the Streamlit application
- ✅ Opens your web browser to the app
- ✅ Shows a nice banner with instructions

## To Stop the Application:
- Press `Ctrl+C` in the Terminal window
- Or simply close the Terminal window

## Troubleshooting:
- If you get a "Permission denied" error, run: `chmod +x run_stock_app.sh`
- If Python packages are missing, the script will try to install them automatically
- Make sure you have an internet connection for the first run (to install packages)
- If you get "no such file or directory" error, try running: `bash run_stock_app.sh`
- The script automatically handles port conflicts by finding available ports

## Files Created:
- `Stock Market App.app` - The main application (double-click this!)
- `run_stock_app.sh` - The shell script that does the work
- `Stock Market App.applescript` - The source code for the app

**Enjoy your single-click Stock Market Analysis application!** 🚀📈

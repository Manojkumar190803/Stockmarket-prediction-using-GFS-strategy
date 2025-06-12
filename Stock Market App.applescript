-- Stock Market Analysis App Launcher
-- This AppleScript opens Terminal and runs the Streamlit application

tell application "Terminal"
    activate
    set currentTab to do script "cd '" & (POSIX path of (path to me as string)) & "/../' && ./run_stock_app.sh"
end tell

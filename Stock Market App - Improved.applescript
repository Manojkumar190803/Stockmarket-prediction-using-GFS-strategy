-- Stock Market Analysis App Launcher (Improved)
-- This AppleScript opens Terminal and runs the Streamlit application

set projectPath to "/Users/manojkumar/Downloads/Stock Market Project(Major pro)"

tell application "Terminal"
    activate
    set currentTab to do script "cd '" & projectPath & "' && bash run_stock_app.sh"
end tell

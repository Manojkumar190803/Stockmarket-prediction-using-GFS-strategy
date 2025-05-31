# How to Deploy Your Stock Analysis App to Streamlit Cloud

## Step 1: Push to GitHub (Already Done!)
Your code is already on GitHub at: https://github.com/Manojkumar190803/Stockmarket-prediction-using-GFS-strategy

## Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**: Visit https://share.streamlit.io/

2. **Sign in with GitHub**: Use your GitHub account to sign in

3. **Deploy New App**: Click "New app" button

4. **Configure Deployment**:
   - Repository: `Manojkumar190803/Stockmarket-prediction-using-GFS-strategy`
   - Branch: `main`
   - Main file path: `Main.py`
   - App URL: Choose a custom URL like `stockmarket-prediction-gfs`

5. **Click Deploy**: The app will start building and deploying

## Step 3: Share with Your Friend

Once deployed, you'll get a URL like:
`https://stockmarket-prediction-gfs.streamlit.app/`

**Share this URL with your friend** - they can click it and run the app directly in their browser without downloading anything!

## Alternative: Quick Deploy Button

You can also use this one-click deploy button in your README:

[![Deploy to Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/manojkumar190803/stockmarket-prediction-using-gfs-strategy/main/Main.py)

## Troubleshooting

If you encounter issues:

1. **Dependencies**: Make sure all packages in `requirements.txt` are compatible
2. **File Paths**: Update absolute paths in `Main.py` to relative paths for cloud deployment
3. **Memory Limits**: Streamlit Cloud has memory limits - optimize large datasets if needed

## Expected Deployment Time
- Initial deployment: 5-10 minutes
- Subsequent updates: 2-3 minutes

## What Your Friend Will See
- A fully functional web application
- All 4 tabs working (GFS Analysis, LSTM Prediction, Single Stock Analysis, Fundamental Analysis)
- Interactive charts and data tables
- No installation required - runs directly in browser

## Next Steps After Deployment
1. Test all features in the deployed app
2. Share the URL with your friend
3. Monitor app performance and usage
4. Update code as needed (auto-deploys from GitHub)

# HeartGuard — Heart Disease Prediction

Professional, polished Streamlit app for predicting heart disease risk using a Random Forest model trained on the UCI Cleveland dataset.

Features
- Clean, responsive Streamlit UI
- Data loading with fallback sample dataset
- Model training (cached) and prediction
- Multiple interactive Plotly visualizations
- Helpful medical recommendation panels and diagnostic displays

Quick start (Windows / PowerShell)
1. Create a virtual environment and activate it.
   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Run the app:
   ```powershell
   streamlit run .\app.py
   ```

Project structure
- `app.py` — main Streamlit application (entry point)
- `assets/` — logo and styles
- `requirements.txt` — pinned dependencies (kept as-is)
- `run_streamlit.ps1` — convenience runner for Windows PowerShell

Notes
- This repository is intentionally lightweight: `app.py` contains the app logic to make it easy to run and iterate on.
- For production, consider splitting logic into `src/` modules and adding tests and CI.

License
- MIT — see `LICENSE` file.

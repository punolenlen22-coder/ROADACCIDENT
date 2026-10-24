# RACC Streamlit Application

This converts the uploaded **RACCv2** notebook into a Streamlit dashboard while keeping its original analysis functions.

## Files
- `app.py` — Streamlit user interface
- `racc_core.py` — analysis code extracted from the notebook
- `requirements.txt` — Python dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit address, upload a CSV with the required RACC columns, and click **Run complete analysis**.

## Required CSV columns
`DATE`, `TIME`, `LOCATION`, `WEATHER`, `ROAD CONDITION`, `VEHICLE TYPE`, `CAUSE`, `SEVERITY`, `GENDER`

The app exposes cleaned data, model performance, forecasts, generated charts, and downloadable outputs.

from pathlib import Path
import tempfile
import pandas as pd
import streamlit as st
import racc_core as core


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RACC Analytics Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* =========================================================
   GLOBAL PAGE
   ========================================================= */

.stApp {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Main text */
html, body, [class*="css"] {
    font-family: "Segoe UI", Arial, sans-serif;
}

/* =========================================================
   HEADINGS
   ========================================================= */

h1 {
    color: #0f172a;
    font-weight: 800;
    letter-spacing: -1px;
}

h2 {
    color: #172554;
    font-weight: 750;
}

h3 {
    color: #1e3a8a;
    font-weight: 700;
}

/* =========================================================
   SIDEBAR
   ========================================================= */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0f172a 0%,
            #172554 55%,
            #1e3a8a 100%
        );
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: white !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15);
}

/* Sidebar radio navigation */
[data-testid="stSidebar"] div[role="radiogroup"] label {
    background-color: rgba(255,255,255,0.06);
    padding: 9px 12px;
    border-radius: 8px;
    margin-bottom: 5px;
    transition: all 0.2s ease;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background-color: rgba(255,255,255,0.15);
}

/* =========================================================
   METRIC CARDS
   ========================================================= */

[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #e2e8f0;
    border-radius: 15px;
    padding: 18px;
    box-shadow: 0px 4px 14px rgba(15,23,42,0.06);
    transition: all 0.25s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0px 8px 22px rgba(15,23,42,0.10);
}

[data-testid="stMetricLabel"] {
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #1d4ed8;
    font-weight: 800;
}

/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {
    border-radius: 9px;
    border: none;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(
        90deg,
        #2563eb,
        #1d4ed8
    );
    color: white;
}

.stButton > button:hover {
    transform: translateY(-1px);
}

/* =========================================================
   DOWNLOAD BUTTON
   ========================================================= */

.stDownloadButton > button {
    background: linear-gradient(
        90deg,
        #15803d,
        #16a34a
    );
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 600;
}

.stDownloadButton > button:hover {
    color: white;
    border: none;
}

/* =========================================================
   FILE UPLOADER
   ========================================================= */

[data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 5px;
}

/* =========================================================
   DATAFRAME
   ========================================================= */

[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
}

/* =========================================================
   INFO / SUCCESS BOXES
   ========================================================= */

[data-testid="stAlert"] {
    border-radius: 12px;
}

/* =========================================================
   SELECT BOX / MULTISELECT
   ========================================================= */

[data-baseweb="select"] > div {
    border-radius: 9px;
}

/* =========================================================
   CUSTOM CLASSES
   ========================================================= */

.dashboard-header {
    background:
        linear-gradient(
            120deg,
            #0f172a,
            #172554,
            #1d4ed8
        );
    padding: 28px 32px;
    border-radius: 18px;
    margin-bottom: 25px;
    box-shadow: 0px 8px 25px rgba(15,23,42,0.15);
}

.dashboard-header h1 {
    color: white !important;
    margin: 0;
    font-size: 2.2rem;
}

.dashboard-header p {
    color: #dbeafe;
    margin-top: 8px;
    margin-bottom: 0;
    font-size: 1rem;
}

.section-card {
    background-color: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    box-shadow: 0px 3px 12px rgba(15,23,42,0.05);
    margin-bottom: 15px;
}

.small-note {
    opacity: 0.75;
    font-size: 0.9rem;
}

/* =========================================================
   HIDE STREAMLIT DEFAULT UI
   ========================================================= */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Keep header transparent */
header[data-testid="stHeader"] {
    background-color: transparent;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. SESSION STATE
# ============================================================

if "run_dir" not in st.session_state:
    st.session_state.run_dir = tempfile.mkdtemp(
        prefix="racc_streamlit_"
    )

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# ============================================================
# 4. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🚦 RACC")
    st.caption("Road Accident Analytics")

    st.divider()

    # --------------------------------------------------------
    # NAVIGATION
    # --------------------------------------------------------

    st.markdown("### Navigation")

    page = st.radio(
        "Go to",
        [
            "🏠 Dashboard",
            "📋 Cleaned Data",
            "📊 Model Performance",
            "📈 Forecasts",
            "🖼️ Charts",
            "⬇️ Downloads"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    st.markdown("### Dataset")

    uploaded = st.file_uploader(
        "Upload RACC CSV",
        type=["csv"]
    )

    st.caption(
        "Required columns:\n\n"
        "DATE, TIME, LOCATION, WEATHER, "
        "ROAD CONDITION, VEHICLE TYPE, "
        "CAUSE, SEVERITY, GENDER"
    )

    run = st.button(
        "▶ Run Complete Analysis",
        type="primary",
        use_container_width=True,
        disabled=uploaded is None
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.markdown("### Model Defaults")

    st.caption(
        "Severity holdout: **2024**\n\n"
        "Count holdout: **2025**\n\n"
        "Forecast horizon: **12 months**"
    )


# ============================================================
# 5. DASHBOARD HEADER
# ============================================================

st.markdown("""
<div class="dashboard-header">
    <h1>🚦 RACC Analytics Dashboard</h1>
    <p>
        Road Accident Classification, Prediction,
        Forecasting and Analytics System
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 6. RUN COMPLETE ANALYSIS
# ============================================================

if run and uploaded is not None:

    # Each Streamlit session gets its own output directory
    core.OUTPUT_ROOT = (
        Path(st.session_state.run_dir)
        / "racc_outputs_modular"
    )

    core.OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # Reset notebook/module state
    core.STATE.clear()

    # Validate uploaded dataset
    core.UPLOADED_DATASET = (
        core.validate_dataset_selection(
            uploaded.name,
            uploaded.getvalue()
        )
    )

    core.DATASET_UPLOADER = None

    # Progress indicator
    progress = st.progress(
        0,
        text="Preparing analysis..."
    )

    status = st.status(
        "Running RACC modules",
        expanded=True
    )

    try:

        module_ids = list(core.MODULE_SPECS)

        for idx, module_id in enumerate(
            module_ids,
            start=1
        ):

            title = (
                core.MODULE_SPECS[module_id]["title"]
            )

            status.write(
                f"Running **{module_id} — {title}**"
            )

            core.run_task(module_id)

            progress.progress(
                idx / len(module_ids),
                text=(
                    f"Completed {idx}/"
                    f"{len(module_ids)} modules"
                )
            )

        st.session_state.analysis_done = True

        status.update(
            label="Analysis complete",
            state="complete",
            expanded=False
        )

        st.success(
            "✅ Analysis successfully completed."
        )

    except Exception as exc:

        st.session_state.analysis_done = False

        status.update(
            label="Analysis stopped",
            state="error",
            expanded=True
        )

        st.exception(exc)


# ============================================================
# 7. WAIT FOR DATASET
# ============================================================

if not st.session_state.analysis_done:

    st.info(
        "👈 Upload your RACC CSV using the sidebar, "
        "then click **Run Complete Analysis**."
    )

    st.markdown("### How to use the dashboard")

    st.markdown("""
    1. Upload your **RACC CSV dataset**.
    2. Click **Run Complete Analysis**.
    3. Wait for all analysis modules to finish.
    4. Use the **sidebar navigation** to explore the results.
    5. Download the generated results from the **Downloads** page.
    """)

    st.stop()


# ============================================================
# 8. LOAD ANALYSIS RESULTS
# ============================================================

state = core.STATE

clean = state["01_cleaning"]["df"]

perf = (
    state["07_performance"]
    ["performance_dashboard"]
)

source = state["00_setup"]


# ============================================================
# 9. GLOBAL KPI CARDS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "📄 Source Rows",
    f"{len(source['raw']):,}"
)

c2.metric(
    "🧹 Cleaned Rows",
    f"{len(clean):,}"
)

c3.metric(
    "📅 Years Covered",
    (
        f"{int(clean['year'].min())}"
        f"–"
        f"{int(clean['year'].max())}"
    )
)

c4.metric(
    "🤖 Selected Classifier",
    state["04_classification"]
    ["best_classifier_name"]
)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# 10. DASHBOARD / OVERVIEW PAGE
# ============================================================

if page == "🏠 Dashboard":

    st.subheader("🏠 Dashboard Overview")

    st.caption(
        "General summary of the uploaded accident "
        "dataset and completed analysis."
    )

    st.divider()

    left, right = st.columns([1.15, 1])

    with left:

        st.markdown("### ⚙️ Module Status")

        st.dataframe(
            core.module_status(),
            use_container_width=True,
            hide_index=True
        )

    with right:

        st.markdown("### 📁 Uploaded Dataset")

        st.write(
            f"**File:** {source['source_name']}"
        )

        st.write(
            "**SHA-256:** "
            f"`{source['SOURCE_SHA256'][:20]}…`"
        )

        st.info(
            "The application preserves the original "
            "RACC cleaning, classification, numeric "
            "prediction, forecasting, evaluation, "
            "feature-analysis and export logic."
        )

    st.divider()

    st.markdown("### 📊 Performance Summary")

    st.dataframe(
        perf,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 11. CLEANED DATA PAGE
# ============================================================

elif page == "📋 Cleaned Data":

    st.subheader("📋 Cleaned Accident Records")

    st.caption(
        "View and filter the cleaned accident dataset."
    )

    st.divider()

    locations = sorted(
        [
            x
            for x in clean.get(
                "location_clean",
                pd.Series(dtype=str)
            )
            .dropna()
            .astype(str)
            .unique()
        ]
    )

    selected_locations = st.multiselect(
        "📍 Filter by Location",
        locations,
        placeholder="Select one or more locations"
    )

    shown = clean

    if selected_locations:

        shown = shown[
            shown["location_clean"]
            .astype(str)
            .isin(selected_locations)
        ]

    # Data information
    d1, d2 = st.columns(2)

    d1.metric(
        "Displayed Records",
        f"{len(shown):,}"
    )

    d2.metric(
        "Total Cleaned Records",
        f"{len(clean):,}"
    )

    st.dataframe(
        shown,
        use_container_width=True,
        hide_index=True,
        height=520
    )

    st.caption(
        f"Showing {len(shown):,} "
        f"of {len(clean):,} cleaned rows."
    )


# ============================================================
# 12. MODEL PERFORMANCE PAGE
# ============================================================

elif page == "📊 Model Performance":

    st.subheader("📊 Held-Out Model Performance")

    st.caption(
        "Evaluation results of the RACC predictive models."
    )

    st.divider()

    st.markdown("### Overall Performance")

    st.dataframe(
        perf,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown(
        "### 🚨 Binary Severity Classification"
    )

    st.dataframe(
        state["07_performance"]
        ["classification_holdout"],
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown(
        "### 🔢 Daily Count Prediction"
    )

    st.dataframe(
        state["07_performance"]
        ["daily_holdout"],
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown(
        "### 📅 Monthly Forecasting"
    )

    st.dataframe(
        state["07_performance"]
        ["forecast_holdout"],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 13. FORECAST PAGE
# ============================================================

elif page == "📈 Forecasts":

    st.subheader("📈 Forecasts & Predictions")

    st.caption(
        "Future accident-record predictions generated "
        "from the RACC analysis."
    )

    st.divider()

    st.markdown(
        "### 🔮 Next-Day Record Count Prediction"
    )

    st.dataframe(
        state["05_prediction"]
        ["next_day_prediction"],
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown(
        "### 📅 12-Month Forecast"
    )

    future = (
        state["06_forecasting"]
        ["future_forecast"]
    )

    st.dataframe(
        future,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # FORECAST CHART
    # --------------------------------------------------------

    numeric_cols = (
        future
        .select_dtypes(include="number")
        .columns
        .tolist()
    )

    if numeric_cols:

        chart_df = future.copy()

        date_col = next(
            (
                c
                for c in chart_df.columns
                if "month" in c.lower()
                or "date" in c.lower()
            ),
            None
        )

        if date_col:
            chart_df = chart_df.set_index(
                date_col
            )

        st.markdown(
            "### 📉 Forecast Visualization"
        )

        st.line_chart(
            chart_df[numeric_cols],
            use_container_width=True
        )


# ============================================================
# 14. CHARTS PAGE
# ============================================================

elif page == "🖼️ Charts":

    st.subheader("🖼️ Generated Analysis Charts")

    st.caption(
        "View the charts generated by the RACC "
        "analysis modules."
    )

    st.divider()

    root = Path(core.OUTPUT_ROOT)

    pngs = sorted(
        root.rglob("*.png")
    )

    if not pngs:

        st.info(
            "No PNG charts were generated."
        )

    else:

        labels = [
            str(p.relative_to(root))
            for p in pngs
        ]

        chosen = st.selectbox(
            "📊 Choose a Chart",
            labels
        )

        st.markdown(
            f"### {Path(chosen).stem.replace('_', ' ').title()}"
        )

        st.image(
            str(root / chosen),
            use_container_width=True
        )


# ============================================================
# 15. DOWNLOAD PAGE
# ============================================================

elif page == "⬇️ Downloads":

    st.subheader("⬇️ Download Analysis Results")

    st.caption(
        "Export the generated RACC analysis results."
    )

    st.divider()

    # --------------------------------------------------------
    # COMPLETE ZIP
    # --------------------------------------------------------

    st.markdown(
        "### 📦 Complete Analysis"
    )

    archive = (
        state["09_exports"]
        .get("archive")
    )

    if archive and Path(archive).exists():

        data = Path(archive).read_bytes()

        st.download_button(
            "⬇️ Download Complete Analysis ZIP",
            data=data,
            file_name=Path(archive).name,
            mime="application/zip",
            use_container_width=True
        )

    else:

        st.warning(
            "Complete analysis ZIP is not available."
        )

    st.divider()

    # --------------------------------------------------------
    # CLEANED CSV
    # --------------------------------------------------------

    st.markdown(
        "### 📄 Cleaned Dataset"
    )

    cleaned_csv = (
        clean
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "⬇️ Download Cleaned Records CSV",
        data=cleaned_csv,
        file_name="racc_cleaned_records.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.info(
        "The complete ZIP contains the generated tables, "
        "charts, model files, settings and consolidated "
        "analysis results organized by module."
    )


# ============================================================
# 16. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748b;
        font-size:13px;
        padding:10px;
    ">
        🚦 RACC Analytics Dashboard
        <br>
        Road Accident Classification,
        Prediction & Forecasting System
    </div>
    """,
    unsafe_allow_html=True
)

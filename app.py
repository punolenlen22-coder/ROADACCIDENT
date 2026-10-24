from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

import racc_core as core


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RACC Analytics Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD EXTERNAL CSS
# ============================================================

def load_css():
    css_file = Path(__file__).parent / "style.css"

    if css_file.exists():
        with open(css_file, "r", encoding="utf-8") as file:
            st.markdown(
                f"<style>{file.read()}</style>",
                unsafe_allow_html=True
            )
    else:
        st.warning(
            "style.css was not found. "
            "Please make sure style.css is in the same folder as app.py."
        )


load_css()


# ============================================================
# SESSION STATE
# ============================================================

if "run_dir" not in st.session_state:
    st.session_state.run_dir = tempfile.mkdtemp(
        prefix="racc_streamlit_"
    )

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# ============================================================
# SIDEBAR
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
    # MODEL DEFAULTS
    # --------------------------------------------------------

    st.markdown("### Model Defaults")

    st.caption(
        "Severity holdout: **2024**\n\n"
        "Count holdout: **2025**\n\n"
        "Forecast horizon: **12 months**"
    )


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.markdown(
    """
    <div class="dashboard-header">
        <div class="dashboard-title">
            🚦 RACC Analytics Dashboard
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# RUN COMPLETE ANALYSIS
# ============================================================

if run and uploaded is not None:

    core.OUTPUT_ROOT = (
        Path(st.session_state.run_dir)
        / "racc_outputs_modular"
    )

    core.OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    core.STATE.clear()

    core.UPLOADED_DATASET = (
        core.validate_dataset_selection(
            uploaded.name,
            uploaded.getvalue()
        )
    )

    core.DATASET_UPLOADER = None

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

        # Only mark analysis as complete if the important
        # results actually exist.
        required_modules = [
            "00_setup",
            "01_cleaning",
            "04_classification",
            "05_prediction",
            "06_forecasting",
            "07_performance",
            "09_exports"
        ]

        missing_modules = [
            module
            for module in required_modules
            if module not in core.STATE
        ]

        if missing_modules:

            st.session_state.analysis_done = False

            status.update(
                label="Analysis incomplete",
                state="error",
                expanded=True
            )

            st.error(
                "The analysis did not generate all required "
                "results."
            )

            st.write(
                "Missing modules:",
                missing_modules
            )

        else:

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
# WAIT FOR DATASET
# ============================================================

if not st.session_state.analysis_done:

    st.info(
        "👈 Upload your RACC CSV using the sidebar, "
        "then click **Run Complete Analysis**."
    )

    st.markdown(
        "### How to use the dashboard"
    )

    st.markdown(
        """
1. Upload your **RACC CSV dataset**.
2. Click **Run Complete Analysis**.
3. Wait for all analysis modules to finish.
4. Use the **sidebar navigation** to explore the results.
5. Download the generated results from the **Downloads** page.
        """
    )

    st.stop()


# ============================================================
# LOAD ANALYSIS RESULTS
# ============================================================

state = core.STATE


# ------------------------------------------------------------
# SAFETY CHECK
# ------------------------------------------------------------

required_modules = [
    "00_setup",
    "01_cleaning",
    "04_classification",
    "05_prediction",
    "06_forecasting",
    "07_performance",
    "09_exports"
]

missing_modules = [
    module
    for module in required_modules
    if module not in state
]

if missing_modules:

    st.session_state.analysis_done = False

    st.error(
        "⚠️ The analysis results are no longer available."
    )

    st.info(
        "Please upload your RACC CSV again and click "
        "**Run Complete Analysis**."
    )

    st.stop()


# ------------------------------------------------------------
# LOAD RESULTS
# ------------------------------------------------------------

clean = state["01_cleaning"]["df"]

perf = (
    state["07_performance"]
    ["performance_dashboard"]
)

source = state["00_setup"]


# ============================================================
# KPI CARDS
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


# ============================================================
# DASHBOARD
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
# CLEANED DATA
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
# MODEL PERFORMANCE
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
# FORECASTS
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
# CHARTS
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
            f"### "
            f"{Path(chosen).stem.replace('_', ' ').title()}"
        )

        st.image(
            str(root / chosen),
            use_container_width=True
        )


# ============================================================
# DOWNLOADS
# ============================================================

elif page == "⬇️ Downloads":

    st.subheader("⬇️ Download Analysis Results")

    st.caption(
        "Export the generated RACC analysis results."
    )

    st.divider()

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
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="dashboard-footer">
        <strong>🚦 RACC Analytics Dashboard</strong>
        <br>
        Road Accident Classification,
        Prediction & Forecasting System
    </div>
    """,
    unsafe_allow_html=True
)

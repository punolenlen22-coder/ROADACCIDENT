from pathlib import Path
import tempfile
import zipfile
import pandas as pd
import streamlit as st
import racc_core as core

st.set_page_config(page_title='RACC Analytics Dashboard', page_icon='🚦', layout='wide')

st.markdown('''
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); border-radius: 12px; padding: 12px;}
.small-note {opacity: .75; font-size: .9rem;}
</style>
''', unsafe_allow_html=True)

st.title('🚦 RACC Analytics Dashboard')
st.caption('Streamlit version of the modular RACC accident-record analysis notebook')

if 'run_dir' not in st.session_state:
    st.session_state.run_dir = tempfile.mkdtemp(prefix='racc_streamlit_')
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

with st.sidebar:
    st.header('Dataset')
    uploaded = st.file_uploader('Upload RACC CSV', type=['csv'])
    st.caption('Required: DATE, TIME, LOCATION, WEATHER, ROAD CONDITION, VEHICLE TYPE, CAUSE, SEVERITY, GENDER')
    run = st.button('Run complete analysis', type='primary', use_container_width=True, disabled=uploaded is None)
    st.divider()
    st.markdown('**Model defaults**')
    st.caption('Severity holdout: 2024\n\nCount holdout: 2025\n\nForecast horizon: 12 months')

if run and uploaded is not None:
    # Give each Streamlit session its own output directory and reset notebook state.
    core.OUTPUT_ROOT = Path(st.session_state.run_dir) / 'racc_outputs_modular'
    core.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    core.STATE.clear()
    core.UPLOADED_DATASET = core.validate_dataset_selection(uploaded.name, uploaded.getvalue())
    core.DATASET_UPLOADER = None

    progress = st.progress(0, text='Preparing analysis…')
    status = st.status('Running RACC modules', expanded=True)
    try:
        module_ids = list(core.MODULE_SPECS)
        for idx, module_id in enumerate(module_ids, start=1):
            title = core.MODULE_SPECS[module_id]['title']
            status.write(f'Running **{module_id} — {title}**')
            core.run_task(module_id)
            progress.progress(idx / len(module_ids), text=f'Completed {idx}/{len(module_ids)} modules')
        st.session_state.analysis_done = True
        status.update(label='Analysis complete', state='complete', expanded=False)
        st.success('Analysis finished. Use the tabs below to inspect the results.')
    except Exception as exc:
        st.session_state.analysis_done = False
        status.update(label='Analysis stopped', state='error', expanded=True)
        st.exception(exc)

if not st.session_state.analysis_done:
    st.info('Upload your RACC CSV in the sidebar, then click **Run complete analysis**.')
    st.stop()

state = core.STATE
clean = state['01_cleaning']['df']
perf = state['07_performance']['performance_dashboard']
source = state['00_setup']

# Top-level KPI cards
c1, c2, c3, c4 = st.columns(4)
c1.metric('Source rows', f"{len(source['raw']):,}")
c2.metric('Cleaned rows', f"{len(clean):,}")
c3.metric('Years covered', f"{int(clean['year'].min())}–{int(clean['year'].max())}")
c4.metric('Selected classifier', state['04_classification']['best_classifier_name'])

overview_tab, data_tab, performance_tab, forecast_tab, charts_tab, download_tab = st.tabs(
    ['Overview', 'Cleaned Data', 'Model Performance', 'Forecasts', 'Charts', 'Downloads']
)

with overview_tab:
    st.subheader('Analysis overview')
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown('**Module status**')
        st.dataframe(core.module_status(), use_container_width=True, hide_index=True)
    with right:
        st.markdown('**Uploaded dataset**')
        st.write(f"**File:** {source['source_name']}")
        st.write(f"**SHA-256:** `{source['SOURCE_SHA256'][:20]}…`")
        st.write('This application preserves the notebook’s original cleaning, classification, numeric prediction, forecasting, evaluation, feature-analysis, and export logic.')
    st.markdown('**Performance dashboard**')
    st.dataframe(perf, use_container_width=True, hide_index=True)

with data_tab:
    st.subheader('Cleaned accident records')
    locations = sorted([x for x in clean.get('location_clean', pd.Series(dtype=str)).dropna().astype(str).unique()])
    selected_locations = st.multiselect('Filter by location', locations)
    shown = clean
    if selected_locations:
        shown = shown[shown['location_clean'].astype(str).isin(selected_locations)]
    st.dataframe(shown, use_container_width=True, hide_index=True, height=520)
    st.caption(f'Showing {len(shown):,} of {len(clean):,} cleaned rows.')

with performance_tab:
    st.subheader('Held-out model performance')
    st.dataframe(perf, use_container_width=True, hide_index=True)
    st.markdown('**Binary severity classification**')
    st.dataframe(state['07_performance']['classification_holdout'], use_container_width=True, hide_index=True)
    st.markdown('**Daily count prediction**')
    st.dataframe(state['07_performance']['daily_holdout'], use_container_width=True, hide_index=True)
    st.markdown('**Monthly forecasting**')
    st.dataframe(state['07_performance']['forecast_holdout'], use_container_width=True, hide_index=True)

with forecast_tab:
    st.subheader('Forecasts and next prediction')
    st.markdown('**Next-day record-count prediction after dataset end**')
    st.dataframe(state['05_prediction']['next_day_prediction'], use_container_width=True, hide_index=True)
    st.markdown('**12-month forecast after dataset end**')
    future = state['06_forecasting']['future_forecast']
    st.dataframe(future, use_container_width=True, hide_index=True)
    numeric_cols = future.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        chart_df = future.copy()
        date_col = next((c for c in chart_df.columns if 'month' in c.lower() or 'date' in c.lower()), None)
        if date_col:
            chart_df = chart_df.set_index(date_col)
        st.line_chart(chart_df[numeric_cols])

with charts_tab:
    st.subheader('Generated analysis charts')
    root = Path(core.OUTPUT_ROOT)
    pngs = sorted(root.rglob('*.png'))
    if not pngs:
        st.info('No PNG charts were generated.')
    else:
        labels = [str(p.relative_to(root)) for p in pngs]
        chosen = st.selectbox('Choose a chart', labels)
        st.image(str(root / chosen), use_container_width=True)

with download_tab:
    st.subheader('Download outputs')
    archive = state['09_exports'].get('archive')
    if archive and Path(archive).exists():
        data = Path(archive).read_bytes()
        st.download_button('Download complete analysis ZIP', data=data,
                           file_name=Path(archive).name, mime='application/zip', use_container_width=True)
    cleaned_csv = clean.to_csv(index=False).encode('utf-8')
    st.download_button('Download cleaned records CSV', data=cleaned_csv,
                       file_name='racc_cleaned_records.csv', mime='text/csv', use_container_width=True)
    st.caption('The complete ZIP contains the notebook-generated tables, charts, model files, settings, and consolidated results organized by module.')

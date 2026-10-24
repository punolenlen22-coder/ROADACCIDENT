# Source notebook cell 6
from pathlib import Path
from io import BytesIO
import hashlib, importlib.metadata, itertools, json, math, platform, shutil, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier, RandomForestRegressor, HistGradientBoostingRegressor)
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, average_precision_score,
    brier_score_loss, log_loss, classification_report, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay, mean_absolute_error,
    mean_squared_error, r2_score, mean_poisson_deviance)
from sklearn.calibration import calibration_curve
import joblib
try:
    from IPython.display import display, Markdown
except ImportError:
    def display(obj):
        print(obj.to_string() if isinstance(obj, (pd.DataFrame, pd.Series)) else obj)
    def Markdown(s):
        return s

SEED = 42
rng = np.random.default_rng(SEED)
DATA_INPUT_MODE = 'auto'  # Colab chooser; Jupyter upload button outside Colab.
DATA_PATH = Path('RACC.csv')  # Used only when DATA_INPUT_MODE='path'.
CSV_ENCODING = 'utf-8-sig'
CSV_SEPARATOR = ','
OUTPUT_ROOT = Path('racc_outputs_modular')
OUTPUT_ROOT.mkdir(exist_ok=True)
APPLY_CONTEXT_DATE_FIXES = True  # Provisional, explicit corrections below; owner confirmation is needed.
DROP_EXACT_DUPLICATES = False   # Repeated rows may be separate people/vehicles; retain and flag.
INCLUDE_INVESTIGATED_CAUSE = False  # Include only if available at the intended prediction time.
SEVERITY_TEST_YEAR = 2024
COUNT_TEST_YEAR = 2025
FIRST_MONTH = '2021-01-01'
LAST_MONTH = '2025-12-01'
ASSUME_COMPLETE_REPORTING_CALENDAR = True  # Unlisted dates become zero records, NOT proven zero accidents.
FORECAST_HORIZON = 12
N_PERMUTATIONS = 999
N_BOOTSTRAP = 500
N_IMPORTANCE_REPEATS = 10
MIN_COMBINATION_RECORDS = 12
ALPHA = 0.05
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 24)
sns.set_theme(style='whitegrid', context='notebook', palette='colorblind')
plt.rcParams.update({'figure.figsize': (10, 4.5), 'figure.dpi': 110, 'savefig.dpi': 160})

versions = {p: importlib.metadata.version(p) for p in
            ['numpy','pandas','scipy','scikit-learn','matplotlib','seaborn','joblib']}
print('Python:', platform.python_version(), '\nPackage versions:', versions)

def binary_metrics(y,pred,prob):
    return {'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),
        'macro_f1':f1_score(y,pred,average='macro',zero_division=0),
        'severe_precision':precision_score(y,pred,zero_division=0),
        'severe_recall':recall_score(y,pred,zero_division=0),
        'ROC_AUC':roc_auc_score(y,prob) if np.unique(y).size==2 else np.nan,
        'PR_AUC_average_precision':average_precision_score(y,prob) if np.unique(y).size==2 else np.nan,
        'Brier_score':brier_score_loss(y,prob),
        'log_loss':log_loss(y,np.column_stack([1-prob,prob]),labels=[0,1])}

def count_metrics(actual,predicted):
    a=np.asarray(actual,dtype=float);p=np.clip(np.asarray(predicted,dtype=float),0,None)
    denominator=np.abs(a).sum()
    smape_denominator=np.abs(a)+np.abs(p)
    smape_terms=np.divide(2*np.abs(a-p),smape_denominator,out=np.zeros_like(a),where=smape_denominator>0)
    return {'MAE':mean_absolute_error(a,p),'RMSE':np.sqrt(mean_squared_error(a,p)),
            'R2':r2_score(a,p) if len(a)>1 else np.nan,
            'WAPE_pct':100*np.abs(a-p).sum()/denominator if denominator>0 else np.nan,
            'sMAPE_pct':100*np.mean(smape_terms),
            'Poisson_deviance':mean_poisson_deviance(a,np.maximum(p,1e-9))}

def output_helpers(directory):
    """Create per-module table/figure writers without changing other modules' paths."""
    directory.mkdir(parents=True,exist_ok=True)
    def save_table(table,name,index=False):
        table.to_csv(directory/f'{name}.csv',index=index)
        return table
    def finish_figure(name):
        plt.tight_layout()
        plt.savefig(directory/f'{name}.png',bbox_inches='tight')
        plt.show()
        plt.close('all')
    return save_table,finish_figure

STATE = {}
MODULE_SPECS = {'00_setup': {'title': 'Setup and dataset loading', 'function': 'module_00_load_dataset', 'folder': '00_setup', 'requires': []}, '01_cleaning': {'title': 'Data cleaning and preparation', 'function': 'module_01_clean_data', 'folder': '01_data_cleaning', 'requires': ['00_setup']}, '02_statistics': {'title': 'Statistical analysis', 'function': 'module_02_statistical_analysis', 'folder': '02_statistical_analysis', 'requires': ['01_cleaning']}, '03_eda': {'title': 'Exploratory data analysis', 'function': 'module_03_exploratory_analysis', 'folder': '03_exploratory_analysis', 'requires': ['01_cleaning']}, '04_classification': {'title': 'Classification algorithms', 'function': 'module_04_classification', 'folder': '04_classification', 'requires': ['01_cleaning']}, '05_prediction': {'title': 'Numeric prediction algorithms', 'function': 'module_05_numeric_prediction', 'folder': '05_numeric_prediction', 'requires': ['01_cleaning']}, '06_forecasting': {'title': 'Forecasting algorithms', 'function': 'module_06_forecasting', 'folder': '06_forecasting', 'requires': ['01_cleaning']}, '07_performance': {'title': 'Model performance evaluation', 'function': 'module_07_model_performance', 'folder': '07_model_performance', 'requires': ['01_cleaning', '04_classification', '05_prediction', '06_forecasting']}, '08_features': {'title': 'Features and variable combinations', 'function': 'module_08_feature_analysis', 'folder': '08_features_and_combinations', 'requires': ['01_cleaning', '04_classification', '05_prediction', '07_performance']}, '09_exports': {'title': 'Exports and consolidated results', 'function': 'module_09_export_results', 'folder': '09_exports', 'requires': ['00_setup', '01_cleaning', '02_statistics', '03_eda', '04_classification', '05_prediction', '06_forecasting', '07_performance', '08_features']}}

def module_status():
    """Show prerequisites and whether each task has a current result."""
    return pd.DataFrame([{'module':key,'task':spec['title'],
        'requires':', '.join(spec['requires']) or 'none','completed':key in STATE}
        for key,spec in MODULE_SPECS.items()])

def run_task(module_id):
    """Run one defined task; invalidate downstream in-memory results if it is rerun."""
    if module_id not in MODULE_SPECS:
        raise KeyError(f'Unknown module {module_id}. Choose one of {list(MODULE_SPECS)}')
    spec=MODULE_SPECS[module_id]
    missing=[key for key in spec['requires'] if key not in STATE]
    if missing:
        raise RuntimeError(f'{module_id} needs these completed modules first: {missing}')
    runner=globals().get(spec['function'])
    if runner is None:
        raise RuntimeError(f'Run the function-definition cell for {module_id} first.')
    print(f"Running {module_id}: {spec['title']}")
    result=runner(STATE)
    if not isinstance(result,dict):
        raise TypeError('Each task must return a dictionary of named outputs.')
    invalid={module_id}
    changed=True
    while changed:
        changed=False
        for key,other in MODULE_SPECS.items():
            if key not in invalid and set(other['requires']).intersection(invalid):
                invalid.add(key);changed=True
    stale=[key for key in invalid if key in STATE and key!=module_id]
    for key in invalid: STATE.pop(key,None)
    STATE[module_id]=result
    if stale: print('Rerun these dependent modules before using their results:',sorted(stale))
    print(f"Completed {module_id}; outputs in {OUTPUT_ROOT/spec['folder']}")

def run_all_modules():
    """Rerun every task in dependency order after all definition cells have been executed."""
    for module_id in MODULE_SPECS: run_task(module_id)
    return module_status()


# Source notebook cell 8
# Upload controls: no built-in dataset is used.
UPLOADED_DATASET = None
DATASET_UPLOADER = None

def validate_dataset_selection(name,content):
    """Validate a single selected CSV and normalize its bytes."""
    name=Path(str(name)).name
    if Path(name).suffix.lower()!='.csv':
        raise ValueError('Select a CSV dataset (.csv). This notebook expects the RACC column structure.')
    if not isinstance(content,(bytes,bytearray,memoryview)):
        raise TypeError('The selected file did not provide readable bytes. Select the CSV again.')
    content=bytes(content)
    if not content:
        raise ValueError('The selected CSV is empty. Select a file containing a header and data rows.')
    return {'name':name,'content':content}

def parse_upload_value(value):
    """Read Colab uploads and Jupyter FileUpload values from ipywidgets 7 or 8."""
    if not value:
        raise RuntimeError('No CSV has been selected. Choose one file, wait for the upload to finish, then run Module 0.B.')
    if isinstance(value,dict):
        if len(value)!=1:
            raise ValueError('Upload exactly one CSV dataset at a time.')
        name,item=next(iter(value.items()))
        if isinstance(item,dict):
            name=item.get('name') or item.get('metadata',{}).get('name') or name
            content=item.get('content')
        else:
            content=item  # google.colab.files.upload(): filename -> bytes
    elif isinstance(value,(tuple,list)):
        if len(value)!=1:
            raise ValueError('Upload exactly one CSV dataset at a time.')
        item=value[0]  # ipywidgets 8: a tuple/list of file dictionaries
        name=item.get('name','dataset.csv')
        content=item.get('content')
    else:
        raise TypeError('Unsupported upload response. Select the CSV again or use path mode.')
    return validate_dataset_selection(name,content)

def request_dataset_upload():
    """Display a file chooser in Colab/Jupyter, or select an explicit local path."""
    global UPLOADED_DATASET,DATASET_UPLOADER
    mode=DATA_INPUT_MODE.strip().lower()
    if mode not in {'auto','colab','jupyter','path'}:
        raise ValueError("DATA_INPUT_MODE must be 'auto', 'colab', 'jupyter', or 'path'.")
    UPLOADED_DATASET=None
    if DATASET_UPLOADER is not None and hasattr(DATASET_UPLOADER,'close'):
        DATASET_UPLOADER.close()
    DATASET_UPLOADER=None
    STATE.clear()  # Prevent a newly selected file from being mixed with old model results.
    if mode=='path':
        path=Path(DATA_PATH).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'CSV not found: {path}. Set DATA_PATH to the uploaded file path.')
        UPLOADED_DATASET=validate_dataset_selection(path.name,path.read_bytes())
        print(f'Selected {UPLOADED_DATASET["name"]}. Run Module 0.B to validate and load it.')
        return
    try:
        from google.colab import files as colab_files
    except ImportError:
        colab_files=None
    if mode=='colab' or (mode=='auto' and colab_files is not None):
        if colab_files is None:
            raise RuntimeError("Google Colab is unavailable here. Use DATA_INPUT_MODE='jupyter' or 'path'.")
        print('Choose one CSV dataset from your computer.')
        UPLOADED_DATASET=parse_upload_value(colab_files.upload())
        print(f'Upload complete: {UPLOADED_DATASET["name"]}. Run Module 0.B next.')
        return
    try:
        import ipywidgets as widgets
    except ImportError as exc:
        raise RuntimeError("For a Jupyter upload button, install ipywidgets (%pip install ipywidgets), "
                           "or set DATA_INPUT_MODE='path' and DATA_PATH to your CSV.") from exc
    DATASET_UPLOADER=widgets.FileUpload(accept='.csv',multiple=False,description='Upload CSV')
    def invalidate_on_new_file(change):
        STATE.clear()
        if change.get('new'):
            print('File selected. Run Module 0.B to validate it, then run Modules 1–9.')
    DATASET_UPLOADER.observe(invalidate_on_new_file,names='value')
    display(DATASET_UPLOADER)
    print('Click Upload CSV, wait for completion, then run Module 0.B. In Jupyter, pause here before running later cells.')

def get_selected_dataset():
    """Read only the explicit selection; never fall back to another dataset."""
    if UPLOADED_DATASET is not None:
        return UPLOADED_DATASET.copy()
    if DATASET_UPLOADER is not None:
        return parse_upload_value(DATASET_UPLOADER.value)
    raise RuntimeError('Run the Upload your CSV dataset cell first, choose one CSV, then run Module 0.B.')


# Source notebook cell 12
def module_00_load_dataset(state):
    """Load and validate the CSV chosen in the upload step.

    Input: one explicitly uploaded CSV, or a CSV chosen through path mode.
    Returns: source rows, required columns, source filename/bytes and SHA-256.
    """
    OUTPUT_DIR=OUTPUT_ROOT/'00_setup'
    save_table,finish_figure=output_helpers(OUTPUT_DIR)
    selected=get_selected_dataset()
    source_bytes=selected['content']
    source_name=selected['name']
    SOURCE_SHA256=hashlib.sha256(source_bytes).hexdigest()
    try:
        raw=pd.read_csv(BytesIO(source_bytes),dtype='string',encoding=CSV_ENCODING,sep=CSV_SEPARATOR)
    except UnicodeDecodeError as exc:
        raise ValueError('CSV encoding could not be read. Update CSV_ENCODING, then rerun Module 0.B.') from exc
    except (pd.errors.EmptyDataError,pd.errors.ParserError) as exc:
        raise ValueError('The file is not a readable CSV. Check its contents and CSV_SEPARATOR.') from exc
    raw.columns=raw.columns.str.strip().str.upper()
    if raw.columns.duplicated().any():
        raise ValueError('Duplicate column names after header normalization. Give each source column a unique name.')
    required=['DATE','TIME','LOCATION','WEATHER','ROAD CONDITION','VEHICLE TYPE','CAUSE','SEVERITY','GENDER']
    missing=[c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}. Found: {list(raw.columns)}. '
                         'Upload a CSV with the documented RACC structure or adjust the column mapping.')
    if raw.empty:
        raise ValueError('The CSV contains a header but no records.')
    extra=[c for c in raw.columns if c not in required]
    if extra:
        print('Additional source columns are not analyzed by the current RACC modules:',extra)
    raw=raw[required].copy(deep=True)
    source_info=pd.DataFrame([{'filename':source_name,'bytes':len(source_bytes),'rows':len(raw),
        'columns':len(raw.columns),'sha256':SOURCE_SHA256,'encoding':CSV_ENCODING,
        'separator':CSV_SEPARATOR}])
    display(save_table(source_info,'00_uploaded_dataset_info'))
    print(f'Loaded uploaded dataset: {source_name} | rows: {len(raw)} | columns: {len(raw.columns)}')
    print('SHA-256:',SOURCE_SHA256)
    display(raw.head(6))
    return {'raw':raw,'required':required,'SOURCE_SHA256':SOURCE_SHA256,
            'source_name':source_name,'source_bytes':source_bytes}


# Source notebook cell 19
def module_01_clean_data(state):
    """Audit data quality, normalize categories, review date repairs, retain duplicate flags and build analysis-ready count series.

    Inputs: Raw CSV rows from Module 0.
    Returns: a dictionary of named module outputs."""
    OUTPUT_DIR = OUTPUT_ROOT / '01_data_cleaning'
    save_table, finish_figure = output_helpers(OUTPUT_DIR)
    raw = state['00_setup']['raw']
    required = state['00_setup']['required']

    raw_audit = pd.DataFrame({
        'dtype':raw.dtypes.astype(str), 'missing_n':raw.isna().sum(),
        'missing_pct':100*raw.isna().mean(), 'unique_nonmissing':raw.nunique(),
        'whitespace_affected_n':[int((raw[c].notna() & raw[c].ne(raw[c].str.strip())).sum()) for c in raw]
    }).rename_axis('variable').reset_index()
    display(save_table(raw_audit, '01_raw_data_audit'))
    print('Exact repeated rows after their first occurrence:', int(raw.duplicated().sum()))
    display(raw.loc[raw.duplicated(keep=False)].head(14))
    raw_counts = pd.concat([raw[c].value_counts(dropna=False).rename('n').reset_index()
        .rename(columns={c:'value'}).assign(variable=c) for c in raw], ignore_index=True)
    save_table(raw_counts[['variable','value','n']], '02_raw_category_frequencies')

    # Explicit reviewable corrections, not a general automatic year repair.
    DATE_FIXES = {
        '8/18/0201': ('2021-08-18','Between 15/08/2021 and 23/08/2021; US-style date and year typo'),
        '1/4/202':   ('2022-01-04','Between 01/01/2022 and 10/01/2022; truncated year and US-style date'),
        '1/14/202':  ('2022-01-14','January 2022 neighbors; truncated year and unambiguous US-style date'),
        '1/16/202':  ('2022-01-16','January 2022 neighbors; truncated year and unambiguous US-style date'),
        '11/2/1014': ('2024-11-02','Between 28/10/2024 and 02/11/2024; assumed month/day and year typo'),
        '8/3/1025':  ('2025-08-03','Between 01/08/2025 and later August records; assumed US-style date/year typo'),
        '8/10/1025': ('2025-08-10','Before 14/08/2025; assumed US-style date/year typo'),
    }

    def normalize_text(s):
        s = s.astype('string').str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
        return s.replace({'':pd.NA, 'n/a':pd.NA, 'na':pd.NA, 'null':pd.NA, 'none':pd.NA})

    def parse_date(value):
        if pd.isna(value):
            return pd.NaT, 'missing', None
        value = str(value).strip()
        if value in DATE_FIXES:
            proposed, reason = DATE_FIXES[value]
            return (pd.Timestamp(proposed) if APPLY_CONTEXT_DATE_FIXES else pd.NaT,
                    'provisional_context_fix' if APPLY_CONTEXT_DATE_FIXES else 'quarantined_context_fix', proposed)
        compact = __import__('re').sub(r'/+', '/', value)
        parts = compact.split('/')
        try:
            a, b, y = map(int, parts)
            if not (2021 <= y <= 2025):
                return pd.NaT, 'invalid_or_out_of_scope_year', None
            month_first = b > 12 and a <= 12
            date = pd.Timestamp(year=y, month=a if month_first else b, day=b if month_first else a)
            status = 'unambiguous_month_first' if month_first else 'dominant_day_first'
            if compact != value:
                status += '_repeated_slash_removed'
            return date, status, None
        except (ValueError, TypeError):
            return pd.NaT, 'invalid_date', None

    def parse_hour(value):
        if pd.isna(value): return np.nan
        for fmt in ['%I:%M %p','%I:%M:%S %p','%I:%M%p','%H:%M','%H:%M:%S']:
            try:
                t = __import__('datetime').datetime.strptime(str(value).strip().upper(), fmt)
                return t.hour + t.minute/60 + t.second/3600
            except ValueError:
                continue
        return np.nan

    def clean_location(s):
        s = normalize_text(s).str.replace(r'\bp\s*-?\s*(?=\d)', 'p', regex=True)
        aliases = {'bunawna':'bunawan','san trodoro':'san teodoro',
                   'pobalcion':'poblacion','mamabalili':'mambalili'}
        for a,b in aliases.items(): s = s.str.replace(a,b,regex=False)
        return s

    def area_from_location(v):
        if pd.isna(v): return 'unknown'
        # Match named areas explicitly; do not infer a barangay from a sitio alone.
        for area in ['san teodoro','san andres','san marcos','bunawan brook',
                     'consuelo','libertad','poblacion','mambalili','imelda','nueva era']:
            if area in v: return area
        if any(t in v for t in ['bun. brook','bun, brook','b-brook']): return 'bunawan brook'
        if __import__('re').search(r'\bpob\.?\b',v): return 'poblacion'
        return 'other / unresolved'

    def vehicle_group(v):
        if pd.isna(v): return 'unknown'
        if __import__('re').search(r'\bvs\.?|&|\band\b|^2\b',v): return 'multiple vehicles recorded'
        if v in ['motorcycle','single motor','motor single','motorcyle','motorcycle aid']: return 'motorcycle'
        if 'pedestrian' in v: return 'pedestrian'
        if v in ['tricycle','side car','sidecar']: return 'tricycle / sidecar'
        if v in ['truck','wing van','wingvan','bus']: return 'truck / bus'
        if v in ['car','suv','van','mini van','wagon','pick up','multicab','l300'] or '4 wheeled' in v:
            return 'car / utility / van'
        if v in ['vehicle','vehicular']: return 'vehicle unspecified'
        return 'other / ambiguous'

    def cause_group(v):
        if pd.isna(v): return 'unknown'
        if any(t in v for t in ['alcohol','aid']): return 'alcohol mentioned / abbreviation (unverified)'
        if any(t in v for t in ['dog','animal']): return 'animal related'
        if 'pedestrian' in v: return 'pedestrian related'
        if any(t in v for t in ['tire','break','brake']): return 'mechanical / braking mentioned'
        if any(t in v for t in ['road condition','weather','visibility']): return 'road / weather / visibility'
        if any(t in v for t in ['reckless','overspeed','human error','asleep']): return 'driving behavior mentioned'
        if any(t in v for t in ['self','sels','overshoot']): return 'self accident / overshoot'
        if any(t in v for t in ['collision','swip','bump','hit','bangga','vs.','vs ']): return 'collision / contact'
        return 'other / unspecified'

    def gender_composition(v):
        if pd.isna(v): return 'unknown'
        tokens = __import__('re').findall(r'\b(?:female|male)\b',v)
        sexes = set(tokens)
        if 'others' in v: return 'mixed + unspecified' if len(sexes)==2 else 'partly unspecified'
        if sexes=={'male','female'}: return 'mixed'
        if sexes=={'male'}: return 'male only'
        if sexes=={'female'}: return 'female only'
        return 'unknown'

    def clean_records(frame):
        d = frame.copy(deep=True).reset_index(drop=True)
        for c in required:
            if c not in d: d[c] = pd.NA
        d['source_row'] = np.arange(len(d)) + 2
        d['exact_duplicate_flag'] = d[required].duplicated(keep=False)
        for c in required:
            d[c.lower().replace(' ','_')+'_raw'] = d[c]
        for c in required: d[c] = normalize_text(d[c])
        dates = [parse_date(v) for v in d['date_raw']]
        d['date'] = pd.to_datetime([v[0] for v in dates])
        d['date_status'] = [v[1] for v in dates]
        d['proposed_date'] = [v[2] for v in dates]
        d['provisional_date_flag'] = d['date_raw'].isin(DATE_FIXES)
        d['hour'] = d['TIME'].map(parse_hour)
        d['invalid_time_flag'] = d['TIME'].notna() & d['hour'].isna()
        d['location_clean'] = clean_location(d['LOCATION'])
        d['area'] = d['location_clean'].map(area_from_location)
        d['weather'] = d['WEATHER']
        d['road_condition'] = d['ROAD CONDITION'].replace({'clear':'clear (unverified)'})
        d['vehicle_clean'] = d['VEHICLE TYPE'].str.replace(r'pick\s+up','pick up',regex=True)
        d['vehicle_group'] = d['vehicle_clean'].map(vehicle_group)
        d['cause_group'] = d['CAUSE'].map(cause_group)
        d['gender_composition'] = d['GENDER'].map(gender_composition)
        d['severity_clean'] = d['SEVERITY'].replace({'minor injry':'minor injury'})
        injury_labels = ['no injury','minor injury','major injury','fatal']
        d['severity_grade'] = d['severity_clean'].where(d['severity_clean'].isin(injury_labels))
        d['severe'] = d['severity_grade'].map({'no injury':0,'minor injury':0,'major injury':1,'fatal':1})
        d['year'] = d['date'].dt.year
        d['month'] = d['date'].dt.month
        d['day_of_week'] = d['date'].dt.dayofweek
        d['is_weekend'] = d['day_of_week'].isin([5,6]).astype(float).where(d['date'].notna())
        for col,period in [('hour',24),('month',12),('day_of_week',7)]:
            d[col+'_sin'] = np.sin(2*np.pi*d[col]/period)
            d[col+'_cos'] = np.cos(2*np.pi*d[col]/period)
        # Provisional key: same date + exact parsed time + normalized location. Never use severity.
        complete = d['date'].notna() & d['hour'].notna() & d['location_clean'].notna()
        d['incident_proxy'] = ('row_'+d['source_row'].astype(str))
        d.loc[complete,'incident_proxy'] = (d.loc[complete,'date'].dt.strftime('%Y-%m-%d')+'|'+
            d.loc[complete,'hour'].round(6).astype(str)+'|'+d.loc[complete,'location_clean'])
        if DROP_EXACT_DUPLICATES:
            d = d.loc[~frame[required].reset_index(drop=True).duplicated()].copy()
        return d

    df = clean_records(raw)
    date_review = df.loc[~df['date_status'].eq('dominant_day_first'),
        ['source_row','date_raw','date','date_status','proposed_date','location_raw']].copy()
    date_review['reason'] = date_review['date_raw'].map({k:v[1] for k,v in DATE_FIXES.items()})
    display(save_table(date_review,'03_date_review'))
    print('Provisional context-dependent dates:', int(df.provisional_date_flag.sum()),
          '| Unusable dates:', int(df.date.isna().sum()))
    display(df.groupby('date_status',dropna=False).size().rename('records').to_frame())

    mapping_tables = []
    for original,cleaned in [('LOCATION','location_clean'),('ROAD CONDITION','road_condition'),
        ('VEHICLE TYPE','vehicle_group'),('CAUSE','cause_group'),('GENDER','gender_composition'),('SEVERITY','severity_clean')]:
        t = pd.DataFrame({'original':raw[original].astype(object),'cleaned':df[cleaned].reindex(raw.index).astype(object)})
        mapping_tables.append(t.groupby(['original','cleaned'],dropna=False).size().rename('n')
                              .reset_index().assign(variable=original))
    cleaning_mappings = pd.concat(mapping_tables,ignore_index=True)
    save_table(cleaning_mappings,'04_cleaning_mapping_audit')
    display(cleaning_mappings.loc[cleaning_mappings.variable.eq('SEVERITY')])
    print('Unparseable nonmissing times:', int(df.invalid_time_flag.sum()))
    print('Records with no usable injury grade:', int(df.severity_grade.isna().sum()))
    print('Record count:',len(df),'| distinct provisional incident keys:',df.incident_proxy.nunique())
    save_table(df,'05_cleaned_records_with_raw_values')

    eligibility = pd.DataFrame({
        'condition':['all records','usable date','usable injury grade','severity development (2021–2023)',
                     'severity holdout (2024)','2025 status/schema quarantine'],
        'n':[len(df),df.date.notna().sum(),df.severity_grade.notna().sum(),
             (df.severity_grade.notna() & df.year.lt(SEVERITY_TEST_YEAR)).sum(),
             (df.severity_grade.notna() & df.year.eq(SEVERITY_TEST_YEAR)).sum(),df.year.ge(2025).sum()]
    })
    display(save_table(eligibility,'06_model_eligibility'))

    injury_era = df.loc[df.severity_grade.notna() & df.year.between(2021,2024)].copy()
    association_features=['weather','road_condition','vehicle_group','area','gender_composition','cause_group']
    def consistent_or_mixed(s):
        values=s.fillna('missing').unique()
        return values[0] if len(values)==1 else 'mixed across records'
    agg={'severe':'max','year':'first',**{c:consistent_or_mixed for c in association_features}}
    stat_df=injury_era.groupby('incident_proxy',sort=False).agg(agg).reset_index()

    if not ASSUME_COMPLETE_REPORTING_CALENDAR:
        raise ValueError('Confirm coverage or provide a reporting-coverage calendar before count modeling.')
    calendar = pd.date_range(pd.Timestamp(FIRST_MONTH), pd.Timestamp(LAST_MONTH)+pd.offsets.MonthEnd(0),freq='D')
    dated = df.loc[df.date.notna()].copy()
    outside_calendar = dated.loc[~dated.date.between(calendar.min(),calendar.max())]
    assert outside_calendar.empty, 'Update FIRST_MONTH/LAST_MONTH: dates fall outside configured calendar.'
    daily = pd.DataFrame(index=calendar)
    daily.index.name = 'date'
    daily['record_count'] = dated.groupby('date').size().reindex(calendar,fill_value=0).astype(int)
    daily['proxy_count'] = dated.groupby('date').incident_proxy.nunique().reindex(calendar,fill_value=0).astype(int)
    strict_dates = dated.loc[~dated.provisional_date_flag]
    daily['strict_date_record_count'] = strict_dates.groupby('date').size().reindex(calendar,fill_value=0).astype(int)
    monthly = daily.resample('MS').sum()
    save_table(daily.reset_index(),'09_daily_record_counts')
    save_table(monthly.reset_index(),'10_monthly_count_sensitivity')

    exported_names = ['df', 'DATE_FIXES', 'clean_records', 'raw_audit', 'raw_counts', 'date_review', 'cleaning_mappings', 'eligibility', 'calendar', 'daily', 'monthly', 'injury_era', 'stat_df', 'association_features']
    return {name:value for name,value in locals().items() if name in exported_names}


# Source notebook cell 26
def module_02_statistical_analysis(state):
    """Calculate descriptive statistics, categorical associations, adjusted p-values and severity-rate intervals.

    Inputs: Cleaned records, daily counts and provisional incident groups from Module 1.
    Returns: a dictionary of named module outputs."""
    OUTPUT_DIR = OUTPUT_ROOT / '02_statistical_analysis'
    save_table, finish_figure = output_helpers(OUTPUT_DIR)
    association_features = state['01_cleaning']['association_features']
    daily = state['01_cleaning']['daily']
    df = state['01_cleaning']['df']
    stat_df = state['01_cleaning']['stat_df']

    analysis_categories = ['weather','road_condition','vehicle_group','area','gender_composition','severity_clean']
    frequency_tables = []
    for c in analysis_categories:
        t = df[c].fillna('missing').value_counts().rename('n').to_frame()
        t['percent'] = 100*t.n/len(df)
        frequency_tables.append(t.rename_axis('value').reset_index().assign(variable=c))
    frequencies = pd.concat(frequency_tables,ignore_index=True)
    save_table(frequencies[['variable','value','n','percent']],'07_cleaned_frequencies')
    display(frequencies.loc[frequencies.variable.eq('severity_clean')])

    display(daily.describe().T)
    count_summary = pd.DataFrame({'mean':daily.mean(),'median':daily.median(),
        'variance':daily.var(),'zero_day_fraction':daily.eq(0).mean(),'max':daily.max()})
    count_summary['variance_to_mean'] = count_summary.variance/count_summary['mean']
    display(save_table(count_summary.reset_index(names='count_definition'),'11_daily_count_summary'))

    def wilson_interval(k,n,alpha=0.05):
        if n==0: return np.nan,np.nan
        z=stats.norm.ppf(1-alpha/2);p=k/n;den=1+z*z/n
        mid=(p+z*z/(2*n))/den
        half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        return max(0,mid-half),min(1,mid+half)

    def bh_adjust(pvalues):
        p=np.asarray(pvalues,float);order=np.argsort(p);m=len(p)
        adjusted=np.empty(m);adjusted[order]=np.minimum.accumulate((p[order]*m/np.arange(1,m+1))[::-1])[::-1]
        return np.minimum(adjusted,1)

    test_rows=[]
    for c in association_features:
        values=stat_df[c].fillna('missing');freq=values.value_counts()
        values=values.where(values.map(freq).ge(10),'other (rare pooled)')
        x,levels=pd.factorize(values);y=stat_df.severe.to_numpy(dtype=int);r=len(levels)
        observed=np.bincount(x*2+y,minlength=r*2).reshape(r,2)
        chi,p_asym,dof,expected=stats.chi2_contingency(observed,correction=False)
        def chi_stat(labels):
            counts=np.bincount(x*2+labels,minlength=r*2).reshape(r,2)
            return float(np.sum((counts-expected)**2/expected))
        permutation_stats=[]
        local_rng=np.random.default_rng(SEED)
        year_indices=[np.flatnonzero(stat_df.year.to_numpy()==yr) for yr in sorted(stat_df.year.unique())]
        for _ in range(N_PERMUTATIONS):
            perm=y.copy()
            for idx in year_indices: perm[idx]=local_rng.permutation(y[idx])
            permutation_stats.append(chi_stat(perm))
        perm_p=(1+np.sum(np.asarray(permutation_stats)>=chi-1e-12))/(N_PERMUTATIONS+1)
        test_rows.append({'feature':c,'proxy_incidents':len(y),'levels_after_pooling':r,
            'chi_square':chi,'cramers_v':np.sqrt(chi/(len(y)*min(r-1,1))),
            'min_expected_count':expected.min(),'expected_below_5_pct':100*np.mean(expected<5),
            'p_chi_square_asymptotic':p_asym,'p_permutation_within_year':perm_p})
    association_tests=pd.DataFrame(test_rows)
    association_tests['p_BH_adjusted']=bh_adjust(association_tests.p_permutation_within_year)
    association_tests['passes_FDR_0_05']=association_tests.p_BH_adjusted.lt(ALPHA)
    display(save_table(association_tests.sort_values('p_BH_adjusted'),'12_exploratory_association_tests'))

    area_rates=stat_df.groupby('area').severe.agg(['size','sum','mean']).rename(columns={'size':'proxy_n','sum':'severe_n','mean':'severe_fraction'})
    area_rates[['wilson_low','wilson_high']]=pd.DataFrame([wilson_interval(r.severe_n,r.proxy_n) for r in area_rates.itertuples()],index=area_rates.index)
    display(save_table(area_rates.reset_index().sort_values('proxy_n',ascending=False),'13_area_severity_intervals'))

    exported_names = ['frequencies', 'count_summary', 'association_tests', 'area_rates']
    return {name:value for name,value in locals().items() if name in exported_names}


# Source notebook cell 33
def module_03_exploratory_analysis(state):
    """Visualize category distributions, missingness, changes in labels, time patterns and associations.

    Inputs: Cleaned records and daily/monthly count series from Module 1.
    Returns: a dictionary of named module outputs."""
    OUTPUT_DIR = OUTPUT_ROOT / '03_exploratory_analysis'
    save_table, finish_figure = output_helpers(OUTPUT_DIR)
    association_features = state['01_cleaning']['association_features']
    df = state['01_cleaning']['df']
    monthly = state['01_cleaning']['monthly']
    stat_df = state['01_cleaning']['stat_df']

    analysis_categories = ['weather','road_condition','vehicle_group','area','gender_composition','severity_clean']

    fig,axes = plt.subplots(2,3,figsize=(16,10))
    for ax,c in zip(axes.flat,analysis_categories):
        counts = df[c].fillna('missing').value_counts().head(10).sort_values()
        counts.plot.barh(ax=ax,color='#147d92')
        ax.set(title=c.replace('_',' ').title(),xlabel='Reported records',ylabel='')
    finish_figure('01_category_distributions')

    severity_by_year = pd.crosstab(df.year,df.severity_clean.fillna('missing'))
    display(save_table(severity_by_year.reset_index(),'08_severity_vocabulary_by_year'))
    severity_by_year.div(severity_by_year.sum(axis=1),axis=0).plot.bar(stacked=True,figsize=(12,5))
    plt.ylabel('Proportion of records');plt.title('Severity vocabulary changes: consciousness is not an injury grade')
    plt.legend(bbox_to_anchor=(1.02,1),loc='upper left')
    finish_figure('02_severity_schema_shift')

    missing = df[['date','hour','location_clean','weather','road_condition','vehicle_clean','severity_grade']].isna()
    missing.mean().mul(100).sort_values().plot.barh(color='#ad5b28')
    plt.xlabel('Missing / unusable (%)');plt.title('Missingness after deterministic cleaning')
    finish_figure('03_missingness')

    hour_day = pd.crosstab(df.day_of_week, np.floor(df.hour))
    hour_day = hour_day.reindex(index=range(7),columns=range(24),fill_value=0)
    plt.figure(figsize=(14,4));sns.heatmap(hour_day,cmap='YlOrRd',yticklabels=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
    plt.xlabel('Hour of day');plt.ylabel('Day');plt.title('Reported records by weekday and hour; missing times omitted')
    finish_figure('04_weekday_hour_heatmap')

    monthly[['record_count','proxy_count','strict_date_record_count']].plot(marker='o',markersize=3,figsize=(13,5))
    plt.title('Monthly recording volume and count-definition sensitivity');plt.ylabel('Records / provisional keys')
    finish_figure('05_monthly_record_counts')
    month_matrix = monthly.assign(year=monthly.index.year,month=monthly.index.month).pivot(index='year',columns='month',values='record_count')
    sns.heatmap(month_matrix,annot=True,fmt='g',cmap='Blues');plt.title('Monthly reported rows by year')
    finish_figure('06_year_month_heatmap')

    association_matrix=pd.DataFrame(index=association_features,columns=association_features,dtype=float)
    for a,b in itertools.product(association_features,repeat=2):
        table=pd.crosstab(stat_df[a],stat_df[b]);n=table.to_numpy().sum()
        if a==b: value=1.0
        elif min(table.shape)>1:
            chi=stats.chi2_contingency(table,correction=False)[0]
            phi2=max(0,chi/n-((table.shape[0]-1)*(table.shape[1]-1))/(n-1))
            rr=table.shape[0]-(table.shape[0]-1)**2/(n-1)
            cc=table.shape[1]-(table.shape[1]-1)**2/(n-1)
            value=np.sqrt(phi2/min(rr-1,cc-1)) if min(rr-1,cc-1)>0 else np.nan
        else: value=np.nan
        association_matrix.loc[a,b]=value
    save_table(association_matrix.reset_index(names='feature'),'14_predictor_association_matrix')
    sns.heatmap(association_matrix,annot=True,vmin=0,vmax=1,cmap='Blues',fmt='.2f')
    plt.title('Bias-corrected categorical association (Cramér’s V)')
    finish_figure('07_predictor_associations')

    exported_names = ['severity_by_year', 'hour_day', 'association_matrix', 'month_matrix']
    return {name:value for name,value in locals().items() if name in exported_names}


# Source notebook cell 40
def module_04_classification(state):
    """Train binary and four-class severity classifiers, select on development folds and generate holdout predictions.

    Inputs: Reliable injury-grade observations from Module 1.
    Returns: a dictionary of named module outputs."""
    OUTPUT_DIR = OUTPUT_ROOT / '04_classification'
    save_table, finish_figure = output_helpers(OUTPUT_DIR)
    injury_era = state['01_cleaning']['injury_era']

    calendar_features=['hour_sin','hour_cos','month_sin','month_cos','day_of_week_sin','day_of_week_cos','is_weekend']
    environment_features=['weather','road_condition','area']
    profile_features=['vehicle_group','gender_composition']
    FEATURES=calendar_features+environment_features+profile_features
    if INCLUDE_INVESTIGATED_CAUSE: FEATURES=FEATURES+['cause_group']
    numeric_features=calendar_features

    def classifier_pipeline(estimator,features=FEATURES):
        num=[c for c in features if c in numeric_features]
        cat=[c for c in features if c not in numeric_features]
        transformers=[]
        if num:
            transformers.append(('numeric',Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
                                                    ('scale',StandardScaler())]),num))
        if cat:
            transformers.append(('categorical',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='missing')),
                ('encode',OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=10,sparse_output=False))]),cat))
        return Pipeline([('preprocess',ColumnTransformer(transformers,remainder='drop')),
                         ('model',clone(estimator))])

    # Convert pandas missing strings to NumPy NaN for consistent sklearn imputation.
    model_data=injury_era.sort_values(['date','source_row']).copy()
    for c in FEATURES:
        if c not in numeric_features:
            model_data[c]=model_data[c].astype(object).where(model_data[c].notna(),np.nan)
    dev=model_data.loc[model_data.year.lt(SEVERITY_TEST_YEAR)].copy()
    test=model_data.loc[model_data.year.eq(SEVERITY_TEST_YEAR)].copy()
    assert len(dev)>0 and len(test)>0
    assert dev.date.max()<test.date.min()
    assert set(dev.incident_proxy).isdisjoint(test.incident_proxy)
    X_dev,y_dev=dev[FEATURES],dev.severe.astype(int)
    X_test,y_test=test[FEATURES],test.severe.astype(int)
    severity_folds=[]
    for year in [2022,2023]:
        ti=np.flatnonzero(dev.year.to_numpy()<year);vi=np.flatnonzero(dev.year.to_numpy()==year)
        if len(ti) and len(vi):
            assert dev.iloc[ti].date.max()<dev.iloc[vi].date.min()
            assert set(dev.iloc[ti].incident_proxy).isdisjoint(dev.iloc[vi].incident_proxy)
            assert y_dev.iloc[ti].nunique()==2 and y_dev.iloc[vi].nunique()==2
            severity_folds.append((ti,vi))
    assert len(severity_folds)>=2, 'Need at least two valid development folds.'
    split_table=pd.DataFrame([{'split':'development','start':dev.date.min(),'end':dev.date.max(),'n':len(dev)},
                             {'split':'holdout','start':test.date.min(),'end':test.date.max(),'n':len(test)}])
    display(save_table(split_table,'15_classification_split'))
    display(pd.crosstab(model_data.year,model_data.severity_grade))

    classifier_specs={
        'Prior baseline':DummyClassifier(strategy='prior'),
        'Logistic C=0.3':LogisticRegression(C=0.3,class_weight='balanced',max_iter=3000,random_state=SEED),
        'Logistic C=3':LogisticRegression(C=3,class_weight='balanced',max_iter=3000,random_state=SEED),
        'Random forest shallow':RandomForestClassifier(n_estimators=180,max_depth=5,min_samples_leaf=8,class_weight='balanced_subsample',random_state=SEED,n_jobs=1),
        'Random forest flexible':RandomForestClassifier(n_estimators=180,min_samples_leaf=5,class_weight='balanced_subsample',random_state=SEED,n_jobs=1),
        'Extra trees':ExtraTreesClassifier(n_estimators=180,max_depth=8,min_samples_leaf=5,class_weight='balanced',random_state=SEED,n_jobs=1),
        'Gradient boosting':HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,l2_regularization=5,
                             min_samples_leaf=15,early_stopping=False,class_weight='balanced',random_state=SEED)
    }

    cv_rows=[]
    for name,estimator in classifier_specs.items():
        for fold,(ti,vi) in enumerate(severity_folds,1):
            pipe=classifier_pipeline(estimator)
            pipe.fit(X_dev.iloc[ti],y_dev.iloc[ti])
            pred=pipe.predict(X_dev.iloc[vi]);prob=pipe.predict_proba(X_dev.iloc[vi])[:,1]
            cv_rows.append({'model':name,'fold':fold,**binary_metrics(y_dev.iloc[vi],pred,prob)})
    classifier_cv=pd.DataFrame(cv_rows)
    save_table(classifier_cv,'16_binary_classification_fold_scores')
    classifier_cv_summary=classifier_cv.groupby('model').agg(
        macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),
        balanced_accuracy_mean=('balanced_accuracy','mean'),ROC_AUC_mean=('ROC_AUC','mean'),
        severe_recall_mean=('severe_recall','mean'),Brier_mean=('Brier_score','mean')).sort_values('macro_f1_mean',ascending=False)
    display(save_table(classifier_cv_summary.reset_index(),'17_binary_classification_cv_summary'))
    best_classifier_name=classifier_cv_summary.index[0]  # Includes baseline: weak learned models need not win.
    best_classifier=classifier_pipeline(classifier_specs[best_classifier_name]).fit(X_dev,y_dev)
    print('Selected on development macro-F1:',best_classifier_name)

    test_pred=best_classifier.predict(X_test)
    test_prob=best_classifier.predict_proba(X_test)[:,1]
    prior_classifier=classifier_pipeline(DummyClassifier(strategy='prior')).fit(X_dev,y_dev)

    MC_LABELS=['no injury','minor injury','major injury','fatal']
    mc_candidates={k:classifier_specs[k] for k in ['Prior baseline','Logistic C=0.3','Random forest shallow']}
    mc_rows=[]
    for name,estimator in mc_candidates.items():
        for fold,(ti,vi) in enumerate(severity_folds,1):
            model=classifier_pipeline(estimator).fit(X_dev.iloc[ti],dev.severity_grade.iloc[ti])
            pred=model.predict(X_dev.iloc[vi])
            mc_rows.append({'model':name,'fold':fold,
                'macro_f1':f1_score(dev.severity_grade.iloc[vi],pred,labels=MC_LABELS,average='macro',zero_division=0),
                'accuracy':accuracy_score(dev.severity_grade.iloc[vi],pred)})
    mc_cv=pd.DataFrame(mc_rows)
    mc_summary=mc_cv.groupby('model')[['macro_f1','accuracy']].mean().sort_values('macro_f1',ascending=False)
    display(save_table(mc_summary.reset_index(),'22_multiclass_cv_summary'))
    mc_name=mc_summary.index[0]
    multiclass_model=classifier_pipeline(mc_candidates[mc_name]).fit(X_dev,dev.severity_grade)

    mc_pred=multiclass_model.predict(X_test)

    exported_names = ['FEATURES', 'numeric_features', 'calendar_features', 'environment_features', 'profile_features', 'classifier_pipeline', 'model_data', 'dev', 'test', 'X_dev', 'y_dev', 'X_test', 'y_test', 'severity_folds', 'classifier_specs', 'classifier_cv', 'classifier_cv_summary', 'best_classifier_name', 'best_classifier', 'prior_classifier', 'test_pred', 'test_prob', 'MC_LABELS', 'mc_candidates', 'mc_cv', 'mc_summary', 'mc_name', 'multiclass_model', 'mc_pred']
    return {name:value for name,value in locals().items() if name in exported_names}


# Source notebook cell 47
def module_05_numeric_prediction(state):
    """Build past-only daily features, compare regression models and baselines, and generate count predictions.

    Inputs: Daily record counts and the reporting calendar from Module 1.
    Returns: a dictionary of named module outputs."""
    OUTPUT_DIR = OUTPUT_ROOT / '05_numeric_prediction'
    save_table, finish_figure = output_helpers(OUTPUT_DIR)
    calendar = state['01_cleaning']['calendar']
    daily = state['01_cleaning']['daily']

    def make_daily_features(series):
        z=pd.DataFrame({'target':series.astype(float)})
        idx=z.index
        z['weekday_sin']=np.sin(2*np.pi*idx.dayofweek/7)
        z['weekday_cos']=np.cos(2*np.pi*idx.dayofweek/7)
        z['month_sin']=np.sin(2*np.pi*idx.month/12)
        z['month_cos']=np.cos(2*np.pi*idx.month/12)
        z['is_weekend']=(idx.dayofweek>=5).astype(int)
        z['trend_day']=(idx-calendar.min()).days
        for lag in [1,7,14,28]: z[f'lag_{lag}']=series.shift(lag)
        previous=series.shift(1)
        z['rolling_mean_7']=previous.rolling(7,min_periods=7).mean()
        z['rolling_mean_28']=previous.rolling(28,min_periods=28).mean()
        z['rolling_sd_7']=previous.rolling(7,min_periods=7).std()
        return z.iloc[28:].copy()  # Remove warm-up dates; do not fabricate prior history.

    daily_model=make_daily_features(daily.record_count)
    DAILY_FEATURES=[c for c in daily_model if c!='target']
    daily_dev=daily_model.loc[daily_model.index.year<COUNT_TEST_YEAR]
    daily_test=daily_model.loc[daily_model.index.year==COUNT_TEST_YEAR]
    assert daily_dev.index.max()<daily_test.index.min()
    assert np.allclose(daily_model['lag_1'],daily.record_count.shift(1).reindex(daily_model.index))
    daily_folds=[]
    for year in [2022,2023,2024]:
        ti=np.flatnonzero(daily_dev.index.year<year);vi=np.flatnonzero(daily_dev.index.year==year)
        if len(ti)>100 and len(vi)>30: daily_folds.append((ti,vi))
    assert len(daily_folds)>=2

    def regression_pipeline(estimator):
        return Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
                         ('scale',StandardScaler()),('model',clone(estimator))])

    regressor_specs={
        'Mean-count baseline':DummyRegressor(strategy='mean'),
        'Median-count baseline':DummyRegressor(strategy='median'),
        'Ridge regression':Ridge(alpha=10),
        'Poisson alpha=0.1':PoissonRegressor(alpha=.1,max_iter=1500),
        'Poisson alpha=1':PoissonRegressor(alpha=1,max_iter=1500),
        'Poisson alpha=10':PoissonRegressor(alpha=10,max_iter=1500),
        'Random forest count':RandomForestRegressor(n_estimators=160,max_depth=6,min_samples_leaf=15,random_state=SEED,n_jobs=1),
        'Poisson gradient boosting':HistGradientBoostingRegressor(loss='poisson',max_iter=100,
            max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=10,early_stopping=False,random_state=SEED)
    }

    reg_rows=[]
    for name,estimator in regressor_specs.items():
        for fold,(ti,vi) in enumerate(daily_folds,1):
            pipe=regression_pipeline(estimator).fit(daily_dev[DAILY_FEATURES].iloc[ti],daily_dev.target.iloc[ti])
            pred=np.clip(pipe.predict(daily_dev[DAILY_FEATURES].iloc[vi]),0,None)
            reg_rows.append({'model':name,'fold':fold,**count_metrics(daily_dev.target.iloc[vi],pred)})
    regression_cv=pd.DataFrame(reg_rows)
    save_table(regression_cv,'29_daily_prediction_cv_fold_scores')
    regression_summary=regression_cv.groupby('model').agg(MAE_mean=('MAE','mean'),MAE_sd=('MAE','std'),
        RMSE_mean=('RMSE','mean'),Poisson_deviance_mean=('Poisson_deviance','mean')).sort_values('MAE_mean')
    display(save_table(regression_summary.reset_index(),'30_daily_prediction_cv_summary'))
    best_regressor_name=regression_summary.index[0]
    best_regressor=regression_pipeline(regressor_specs[best_regressor_name]).fit(daily_dev[DAILY_FEATURES],daily_dev.target)
    daily_pred=np.clip(best_regressor.predict(daily_test[DAILY_FEATURES]),0,None)
    daily_baseline=regression_pipeline(DummyRegressor(strategy='mean')).fit(daily_dev[DAILY_FEATURES],daily_dev.target)
    daily_baseline_pred=np.clip(daily_baseline.predict(daily_test[DAILY_FEATURES]),0,None)
    daily_seasonal_pred=daily_test.lag_7.to_numpy()  # Yesterday's week: legitimate one-day-ahead reference.
    learned_regressor_name=next(name for name in regression_summary.index if 'baseline' not in name.lower())
    learned_regressor=regression_pipeline(regressor_specs[learned_regressor_name]).fit(daily_dev[DAILY_FEATURES],daily_dev.target)
    learned_daily_pred=np.clip(learned_regressor.predict(daily_test[DAILY_FEATURES]),0,None)

    print('Selected on development MAE:',best_regressor_name)

    next_day=daily.index.max()+pd.Timedelta(days=1)
    history_with_next=pd.concat([daily.record_count,pd.Series([np.nan],index=[next_day])])
    next_features=make_daily_features(history_with_next).loc[[next_day],DAILY_FEATURES]
    daily_deployment_model=regression_pipeline(regressor_specs[best_regressor_name]).fit(
        daily_model[DAILY_FEATURES],daily_model.target)
    next_day_prediction=pd.DataFrame({'date':[next_day],
        'predicted_record_count':np.clip(daily_deployment_model.predict(next_features),0,None)})
    display(save_table(next_day_prediction,'34_next_day_prediction_after_dataset_end'))
    print('This date follows the dataset end; it may precede the current calendar date.')

    exported_names = ['make_daily_features', 'daily_model', 'DAILY_FEATURES', 'daily_dev', 'daily_test', 'daily_folds', 'regression_pipeline', 'regressor_specs', 'regression_cv', 'regression_summary', 'best_regressor_name', 'best_regressor', 'daily_pred', 'daily_baseline', 'daily_baseline_pred', 'daily_seasonal_pred', 'learned_regressor_name', 'learned_regressor', 'learned_daily_pred', 'daily_deployment_model', 'next_day_prediction']
    return {name:value for name,value in locals().items() if name in exported_names}


# Source notebook cell 54
def module_06_forecasting(state):
    """Backtest monthly forecasting methods, generate fixed-origin predictions and extrapolate beyond the dataset.

    Inputs: Monthly record-count series from Module 1.
    Returns: a dictionary of named module outputs."""
    OUTPUT_DIR = OUTPUT_ROOT / '06_forecasting'
    save_table, finish_figure = output_helpers(OUTPUT_DIR)
    monthly = state['01_cleaning']['monthly']

    def smooth_forecast(values,horizon,method):
        y=np.asarray(values,float)
        if method=='ses':
            def run(alpha):
                level=y[0];errors=[]
                for value in y[1:]:
                    errors.append(value-level);level=alpha*value+(1-alpha)*level
                return level,np.asarray(errors)
            fit=optimize.minimize_scalar(lambda a:np.mean(run(a)[1]**2),bounds=(.001,.999),method='bounded')
            if not fit.success: raise RuntimeError('SES optimization failed.')
            return np.repeat(run(float(fit.x))[0],horizon)
        if method=='holt_damped':
            def run(parameters):
                alpha,beta,phi=parameters;level=y[0]
                trend=float(np.mean(np.diff(y[:min(6,len(y))])))
                errors=[]
                for value in y[1:]:
                    prediction=level+phi*trend;errors.append(value-prediction)
                    old_level=level;level=alpha*value+(1-alpha)*prediction
                    trend=beta*(level-old_level)+(1-beta)*phi*trend
                return level,trend,np.asarray(errors)
            fits=[optimize.minimize(lambda p:np.mean(run(p)[2]**2),x0=start,
                bounds=[(.001,.999),(.001,.999),(.8,1)],method='L-BFGS-B')
                for start in [(0.2,0.1,0.9),(0.6,0.2,0.98)]]
            fit=min(fits,key=lambda z:z.fun)
            if not fit.success: warnings.warn('Holt optimizer did not fully converge; inspect fitted forecast.')
            level,trend,_=run(fit.x);phi=fit.x[2]
            return np.array([level+trend*np.sum(phi**np.arange(1,h+1)) for h in range(1,horizon+1)])
        raise ValueError(method)

    def monthly_lag_features(series):
        z=pd.DataFrame({'target':series.astype(float)})
        z['month_sin']=np.sin(2*np.pi*z.index.month/12)
        z['month_cos']=np.cos(2*np.pi*z.index.month/12)
        z['trend_month']=(z.index.year-2021)*12+z.index.month
        for lag in [1,3,12]: z[f'lag_{lag}']=series.shift(lag)
        z['rolling_3']=series.shift(1).rolling(3).mean()
        z['rolling_12']=series.shift(1).rolling(12).mean()
        return z.iloc[12:]

    def forecast_monthly(series,horizon,method):
        y=np.asarray(series,float)
        if len(y)<24: raise ValueError('At least 24 monthly training observations required.')
        if method=='last_value': pred=np.repeat(y[-1],horizon)
        elif method=='seasonal_naive': pred=np.array([y[-12+(i%12)] for i in range(horizon)])
        elif method=='drift': pred=y[-1]+np.arange(1,horizon+1)*(y[-1]-y[0])/(len(y)-1)
        elif method in ['ses','holt_damped']: pred=smooth_forecast(y,horizon,method)
        elif method=='recursive_random_forest':
            train=monthly_lag_features(series)
            cols=[c for c in train if c!='target']
            model=RandomForestRegressor(n_estimators=140,max_depth=3,min_samples_leaf=3,random_state=SEED,n_jobs=1)
            model.fit(train[cols],train.target)
            history=series.copy().astype(float);pred=[]
            for _ in range(horizon):
                date=history.index[-1]+pd.offsets.MonthBegin(1)
                augmented=pd.concat([history,pd.Series([np.nan],index=[date])])
                value=float(np.clip(model.predict(monthly_lag_features(augmented).loc[[date],cols])[0],0,None))
                pred.append(value);history.loc[date]=value  # Append prediction, never future actuals.
        else: raise ValueError(method)
        return np.clip(np.asarray(pred,float),0,None)

    FORECAST_METHODS=['last_value','seasonal_naive','drift','ses','holt_damped','recursive_random_forest']
    monthly_series=monthly.record_count.astype(float)
    forecast_dev=monthly_series.loc[monthly_series.index.year<COUNT_TEST_YEAR]
    forecast_test=monthly_series.loc[monthly_series.index.year==COUNT_TEST_YEAR]
    assert len(forecast_dev)==48 and len(forecast_test)==12, 'Review hard-coded calendar/backtest windows for a new dataset.'
    monthly_backtest_rows=[];monthly_backtest_predictions=[]
    for origin in [24,36]:
        train=forecast_dev.iloc[:origin];validation=forecast_dev.iloc[origin:origin+12]
        assert train.index.max()<validation.index.min()
        for method in FORECAST_METHODS:
            pred=forecast_monthly(train,len(validation),method)
            monthly_backtest_rows.append({'model':method,'train_months':len(train),
                'validation_year':validation.index[0].year,**count_metrics(validation,pred)})
            monthly_backtest_predictions.extend({'model':method,'origin':str(train.index[-1].date()),
                'date':date,'horizon':h+1,'actual':float(a),'predicted':float(p),'error':float(a-p)}
                for h,(date,a,p) in enumerate(zip(validation.index,validation,pred)))
    forecast_backtests=pd.DataFrame(monthly_backtest_rows)
    forecast_backtest_predictions=pd.DataFrame(monthly_backtest_predictions)
    save_table(forecast_backtests,'35_monthly_forecast_backtest_scores')
    save_table(forecast_backtest_predictions,'36_monthly_forecast_backtest_predictions')
    forecast_cv_summary=forecast_backtests.groupby('model').agg(MAE_mean=('MAE','mean'),MAE_sd=('MAE','std'),
        RMSE_mean=('RMSE','mean'),WAPE_pct_mean=('WAPE_pct','mean')).sort_values('MAE_mean')
    display(save_table(forecast_cv_summary.reset_index(),'37_monthly_forecast_cv_summary'))
    best_forecast_method=forecast_cv_summary.index[0]
    print('Selected on development MAE:',best_forecast_method)

    monthly_test_pred=forecast_monthly(forecast_dev,len(forecast_test),best_forecast_method)
    seasonal_test_pred=forecast_monthly(forecast_dev,len(forecast_test),'seasonal_naive')
    naive_test_pred=forecast_monthly(forecast_dev,len(forecast_test),'last_value')

    development_errors=np.abs(forecast_backtest_predictions.loc[
        forecast_backtest_predictions.model.eq(best_forecast_method),'error'].to_numpy())
    INTERVAL_ALPHA=.10
    adjusted_quantile=min(1,math.ceil((len(development_errors)+1)*(1-INTERVAL_ALPHA))/len(development_errors))
    interval_half_width=float(np.quantile(development_errors,adjusted_quantile,method='higher'))
    monthly_test_lower=np.maximum(0,monthly_test_pred-interval_half_width)
    monthly_test_upper=monthly_test_pred+interval_half_width

    future_index=pd.date_range(monthly_series.index[-1]+pd.offsets.MonthBegin(1),periods=FORECAST_HORIZON,freq='MS')
    future_pred=forecast_monthly(monthly_series,FORECAST_HORIZON,best_forecast_method)
    future_forecast=pd.DataFrame({'date':future_index,'forecast_expected_record_count':future_pred,
        'approx_90pct_lower':np.maximum(0,future_pred-interval_half_width),
        'approx_90pct_upper':future_pred+interval_half_width})
    display(save_table(future_forecast,'40_forecast_after_dataset_end'))
    plt.figure(figsize=(13,5));plt.plot(monthly_series.index,monthly_series,label='Observed reported rows')
    plt.plot(future_index,future_pred,'o-',label=f'Forecast: {best_forecast_method}')
    plt.fill_between(future_index,future_forecast.approx_90pct_lower,future_forecast.approx_90pct_upper,alpha=.2,
                     label='Approximate 90% interval')
    plt.axvline(future_index[0],color='gray',ls='--');plt.ylabel('Recorded rows');plt.legend()
    plt.title('Forecast beyond the dataset end; refresh the data before current operational use')
    finish_figure('16_forecast_after_dataset_end')

    exported_names = ['smooth_forecast', 'monthly_lag_features', 'forecast_monthly', 'FORECAST_METHODS', 'monthly_series', 'forecast_dev', 'forecast_test', 'forecast_backtests', 'forecast_backtest_predictions', 'forecast_cv_summary', 'best_forecast_method', 'monthly_test_pred', 'seasonal_test_pred', 'naive_test_pred', 'development_errors', 'interval_half_width', 'monthly_test_lower', 'monthly_test_upper', 'future_index', 'future_pred', 'future_forecast']
    return {name:value for name,value in locals().items() if name in exported_names}


# Source notebook cell 61
def module_07_model_performance(state):
    """Evaluate the selected models on reserved years and report task-specific metrics, diagnostics and uncertainty.

    Inputs: Fitted development models, untouched holdout predictions and count series from Modules 1, 4, 5 and 6.
    Returns: a dictionary of named module outputs."""
    OUTPUT_DIR = OUTPUT_ROOT / '07_model_performance'
    save_table, finish_figure = output_helpers(OUTPUT_DIR)
    DAILY_FEATURES = state['05_prediction']['DAILY_FEATURES']
    FEATURES = state['04_classification']['FEATURES']
    MC_LABELS = state['04_classification']['MC_LABELS']
    X_test = state['04_classification']['X_test']
    best_classifier_name = state['04_classification']['best_classifier_name']
    best_forecast_method = state['06_forecasting']['best_forecast_method']
    best_regressor_name = state['05_prediction']['best_regressor_name']
    daily = state['01_cleaning']['daily']
    daily_baseline_pred = state['05_prediction']['daily_baseline_pred']
    daily_pred = state['05_prediction']['daily_pred']
    daily_seasonal_pred = state['05_prediction']['daily_seasonal_pred']
    daily_test = state['05_prediction']['daily_test']
    dev = state['04_classification']['dev']
    development_errors = state['06_forecasting']['development_errors']
    forecast_dev = state['06_forecasting']['forecast_dev']
    forecast_monthly = state['06_forecasting']['forecast_monthly']
    forecast_test = state['06_forecasting']['forecast_test']
    future_pred = state['06_forecasting']['future_pred']
    learned_daily_pred = state['05_prediction']['learned_daily_pred']
    learned_regressor_name = state['05_prediction']['learned_regressor_name']
    make_daily_features = state['05_prediction']['make_daily_features']
    mc_name = state['04_classification']['mc_name']
    mc_pred = state['04_classification']['mc_pred']
    monthly = state['01_cleaning']['monthly']
    monthly_test_lower = state['06_forecasting']['monthly_test_lower']
    monthly_test_pred = state['06_forecasting']['monthly_test_pred']
    monthly_test_upper = state['06_forecasting']['monthly_test_upper']
    naive_test_pred = state['06_forecasting']['naive_test_pred']
    prior_classifier = state['04_classification']['prior_classifier']
    seasonal_test_pred = state['06_forecasting']['seasonal_test_pred']
    severity_folds = state['04_classification']['severity_folds']
    test = state['04_classification']['test']
    test_pred = state['04_classification']['test_pred']
    test_prob = state['04_classification']['test_prob']
    y_test = state['04_classification']['y_test']

    classification_holdout=pd.DataFrame([
        {'model':best_classifier_name,**binary_metrics(y_test,test_pred,test_prob)},
        {'model':'Prior baseline (reference)',**binary_metrics(y_test,prior_classifier.predict(X_test),prior_classifier.predict_proba(X_test)[:,1])}])
    display(save_table(classification_holdout,'18_binary_classification_holdout_metrics'))
    binary_report=pd.DataFrame(classification_report(y_test,test_pred,labels=[0,1],
        target_names=['non-serious recorded grade','major / fatal'],output_dict=True,zero_division=0)).T
    display(save_table(binary_report.reset_index(names='class_or_average'),'19_binary_classification_report'))
    holdout_predictions=test[['source_row','date','incident_proxy','severity_grade']].copy()
    holdout_predictions['actual_severe']=y_test.to_numpy();holdout_predictions['predicted_severe']=test_pred
    holdout_predictions['probability_major_or_fatal']=test_prob
    save_table(holdout_predictions,'20_binary_holdout_predictions')

    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    ConfusionMatrixDisplay.from_predictions(y_test,test_pred,display_labels=['non-serious','major/fatal'],ax=axes[0],colorbar=False)
    RocCurveDisplay.from_predictions(y_test,test_prob,ax=axes[1],name='Selected model')
    axes[1].plot([0,1],[0,1],'k--',alpha=.5)
    PrecisionRecallDisplay.from_predictions(y_test,test_prob,ax=axes[2],name='Selected model')
    axes[2].axhline(y_test.mean(),ls='--',color='gray',label='Severe prevalence');axes[2].legend()
    finish_figure('08_binary_classifier_diagnostics')

    observed,predicted=calibration_curve(y_test,test_prob,n_bins=6,strategy='quantile')
    plt.plot(predicted,observed,'o-',label=best_classifier_name);plt.plot([0,1],[0,1],'k--',label='Perfect calibration')
    plt.xlabel('Mean predicted probability');plt.ylabel('Observed severe fraction');plt.legend()
    plt.title('Holdout calibration check; class weighting does not guarantee calibrated probabilities')
    finish_figure('09_probability_calibration')

    # Resample whole provisional incidents, keeping all associated test rows together.
    group_indices=[np.flatnonzero(test.incident_proxy.to_numpy()==g) for g in test.incident_proxy.unique()]
    bootstrap_rows=[];boot_rng=np.random.default_rng(SEED)
    for _ in range(N_BOOTSTRAP):
        idx=np.concatenate([group_indices[i] for i in boot_rng.integers(0,len(group_indices),len(group_indices))])
        yy=y_test.to_numpy()[idx];pp=test_pred[idx];pr=test_prob[idx]
        bootstrap_rows.append({'accuracy':accuracy_score(yy,pp),'macro_f1':f1_score(yy,pp,average='macro',zero_division=0),
                              'ROC_AUC':roc_auc_score(yy,pr) if len(np.unique(yy))==2 else np.nan})
    bootstrap_ci=pd.DataFrame(bootstrap_rows).quantile([.025,.975]).T.rename(columns={.025:'95pct_low',.975:'95pct_high'})
    display(save_table(bootstrap_ci.reset_index(names='metric'),'21_incident_proxy_bootstrap_intervals'))
    print('Intervals condition on the chosen model, split and proxy grouping; they do not include model-selection uncertainty.')

    mc_report=pd.DataFrame(classification_report(test.severity_grade,mc_pred,labels=MC_LABELS,
                                               output_dict=True,zero_division=0)).T
    display(save_table(mc_report.reset_index(names='class_or_average'),'23_multiclass_holdout_report'))
    ConfusionMatrixDisplay.from_predictions(test.severity_grade,mc_pred,labels=MC_LABELS,
                                            xticks_rotation=25,colorbar=False)
    plt.title(f'Four-class holdout: {mc_name}');finish_figure('10_multiclass_confusion_matrix')

    daily_holdout=pd.DataFrame([
        {'model':best_regressor_name,**count_metrics(daily_test.target,daily_pred)},
        {'model':learned_regressor_name+' (best learned candidate by development MAE)',**count_metrics(daily_test.target,learned_daily_pred)},
        {'model':'Mean-count baseline (reference)',**count_metrics(daily_test.target,daily_baseline_pred)},
        {'model':'Same weekday previous week',**count_metrics(daily_test.target,daily_seasonal_pred)}])
    display(save_table(daily_holdout,'31_daily_prediction_holdout_metrics'))
    daily_predictions=pd.DataFrame({'date':daily_test.index,'actual_record_count':daily_test.target,
        'predicted_record_count':daily_pred,'best_learned_candidate_prediction':learned_daily_pred,'mean_baseline':daily_baseline_pred,
        'previous_week_baseline':daily_seasonal_pred}).reset_index(drop=True)
    save_table(daily_predictions,'32_daily_holdout_predictions')

    fig,axes=plt.subplots(1,2,figsize=(14,4.5))
    axes[0].plot(daily_test.index,daily_test.target,label='Observed',alpha=.6,lw=.8)
    axes[0].plot(daily_test.index,daily_pred,label='Selected one-day-ahead point prediction',lw=1)
    axes[0].set(title='Daily holdout predictions',ylabel='Recorded rows');axes[0].legend()
    residual=daily_test.target.to_numpy()-daily_pred
    axes[1].scatter(daily_pred,residual,alpha=.3,s=15)
    axes[1].axhline(0,color='black',lw=.8)
    axes[1].set(xlabel='Predicted count',ylabel='Actual minus predicted',title='Holdout residuals')
    finish_figure('12_daily_prediction_diagnostics')

    # A constant baseline cannot attribute predictions to variables. If it wins, inspect the
    # strongest learned candidate separately, without replacing the selected baseline.

    mase_scale=np.mean(np.abs(forecast_dev.to_numpy()[12:]-forecast_dev.to_numpy()[:-12]))
    forecast_holdout_rows=[]
    for name,pred in [(best_forecast_method,monthly_test_pred),('seasonal_naive (reference)',seasonal_test_pred),
                      ('last_value (reference)',naive_test_pred)]:
        row={'model':name,**count_metrics(forecast_test,pred)}
        row['seasonal_MASE']=row['MAE']/mase_scale if mase_scale>0 else np.nan
        forecast_holdout_rows.append(row)
    forecast_holdout=pd.DataFrame(forecast_holdout_rows)
    display(save_table(forecast_holdout,'38_monthly_forecast_holdout_metrics'))

    interval_coverage=float(np.mean((forecast_test.to_numpy()>=monthly_test_lower)&(forecast_test.to_numpy()<=monthly_test_upper)))
    print(f'Approximate 90% interval: {len(development_errors)} pooled development errors; holdout coverage={interval_coverage:.1%}')
    monthly_holdout_predictions=pd.DataFrame({'date':forecast_test.index,'actual_record_count':forecast_test.to_numpy(),
        'predicted_expected_count':monthly_test_pred,'approx_90pct_lower':monthly_test_lower,
        'approx_90pct_upper':monthly_test_upper,'seasonal_naive':seasonal_test_pred})
    save_table(monthly_holdout_predictions,'39_monthly_holdout_predictions')

    fig,ax=plt.subplots(figsize=(13,5))
    ax.plot(forecast_dev.index,forecast_dev,label='Development observations',color='#4b6570')
    ax.plot(forecast_test.index,forecast_test,label='Holdout actual',color='black',marker='o')
    ax.plot(forecast_test.index,monthly_test_pred,label=f'Fixed-origin forecast: {best_forecast_method}',marker='o')
    ax.fill_between(forecast_test.index,monthly_test_lower,monthly_test_upper,alpha=.2,label='Approximate 90% interval')
    ax.axvline(forecast_test.index[0],ls='--',color='gray');ax.set(ylabel='Reported rows',title='12-month fixed-origin forecast evaluation')
    ax.legend(loc='upper left',fontsize=9);finish_figure('14_monthly_forecast_holdout')

    monthly_residual=forecast_test.to_numpy()-monthly_test_pred
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    axes[0].bar(forecast_test.index,monthly_residual,width=20);axes[0].axhline(0,color='black',lw=.8)
    axes[0].set(title='Monthly holdout residuals',ylabel='Actual minus forecast')
    lags=np.arange(1,6)
    autocorrelations=[pd.Series(monthly_residual).autocorr(lag=int(k)) for k in lags]
    axes[1].stem(lags,autocorrelations);axes[1].axhline(0,color='black',lw=.8)
    axes[1].set(xlabel='Lag in months',ylabel='Residual autocorrelation',title='Descriptive only: just 12 holdout months')
    finish_figure('15_monthly_forecast_residuals')

    sensitivity_rows=[]
    for col in ['record_count','proxy_count','strict_date_record_count']:
        s=monthly[col].astype(float);train=s.loc[s.index.year<COUNT_TEST_YEAR];actual=s.loc[s.index.year==COUNT_TEST_YEAR]
        pred=forecast_monthly(train,len(actual),best_forecast_method)
        sensitivity_rows.append({'count_definition':col,'all_year_total':int(s.sum()),'holdout_total':int(actual.sum()),
            'fixed_method':best_forecast_method,**count_metrics(actual,pred)})
    sensitivity_summary=pd.DataFrame(sensitivity_rows)
    display(save_table(sensitivity_summary,'41_date_and_count_definition_sensitivity'))

    # Demonstrate that count feature construction cannot see the current/future target.
    probe_date=pd.Timestamp('2025-06-15')
    perturbed=daily.record_count.copy();perturbed.loc[probe_date:]+=100
    original_features=make_daily_features(daily.record_count).loc[probe_date,DAILY_FEATURES]
    perturbed_features=make_daily_features(perturbed).loc[probe_date,DAILY_FEATURES]
    assert np.allclose(original_features,perturbed_features,equal_nan=True), 'Current/future count leaked into predictor.'
    assert 'severe' not in FEATURES and 'severity_grade' not in FEATURES and 'incident_proxy' not in FEATURES
    assert not set(dev.incident_proxy).intersection(test.incident_proxy)
    assert all(dev.iloc[ti].date.max()<dev.iloc[vi].date.min() for ti,vi in severity_folds)
    assert np.isfinite(daily_pred).all() and np.isfinite(future_pred).all()
    assert (daily_pred>=0).all() and (future_pred>=0).all()
    print('Checks passed: chronological separation, grouped records, target exclusion, past-only daily predictors and finite nonnegative forecasts.')

    performance_dashboard=pd.DataFrame([
        {'task':'binary severity classification','selected_model':best_classifier_name,
         'holdout_year':SEVERITY_TEST_YEAR,'metric':'macro-F1','selected_value':classification_holdout.iloc[0]['macro_f1'],
         'reference_value':classification_holdout.iloc[1]['macro_f1'],'larger_is_better':True},
        {'task':'one-day-ahead count prediction','selected_model':best_regressor_name,
         'holdout_year':COUNT_TEST_YEAR,'metric':'MAE','selected_value':daily_holdout.iloc[0]['MAE'],
         'reference_value':daily_holdout.loc[daily_holdout.model.eq('Mean-count baseline (reference)'),'MAE'].iloc[0],
         'larger_is_better':False},
        {'task':'12-month fixed-origin forecasting','selected_model':best_forecast_method,
         'holdout_year':COUNT_TEST_YEAR,'metric':'MAE','selected_value':forecast_holdout.iloc[0]['MAE'],
         'reference_value':forecast_holdout.iloc[1]['MAE'],'larger_is_better':False}
    ])
    display(save_table(performance_dashboard,'43_task_performance_dashboard'))

    exported_names = ['classification_holdout', 'binary_report', 'holdout_predictions', 'bootstrap_ci', 'mc_report', 'daily_holdout', 'daily_predictions', 'forecast_holdout', 'interval_coverage', 'monthly_holdout_predictions', 'sensitivity_summary', 'performance_dashboard']
    return {name:value for name,value in locals().items() if name in exported_names}


# Source notebook cell 68
def module_08_feature_analysis(state):
    """Inspect original-variable importance, predefined feature combinations, supported observed patterns, rules and scenarios.

    Inputs: Cleaned data, fitted models, temporal folds and evaluation context from Modules 1, 4, 5 and 7.
    Returns: a dictionary of named module outputs."""
    OUTPUT_DIR = OUTPUT_ROOT / '08_features_and_combinations'
    save_table, finish_figure = output_helpers(OUTPUT_DIR)
    DAILY_FEATURES = state['05_prediction']['DAILY_FEATURES']
    FEATURES = state['04_classification']['FEATURES']
    X_dev = state['04_classification']['X_dev']
    X_test = state['04_classification']['X_test']
    best_classifier = state['04_classification']['best_classifier']
    best_classifier_name = state['04_classification']['best_classifier_name']
    best_regressor = state['05_prediction']['best_regressor']
    best_regressor_name = state['05_prediction']['best_regressor_name']
    calendar_features = state['04_classification']['calendar_features']
    classifier_pipeline = state['04_classification']['classifier_pipeline']
    classifier_specs = state['04_classification']['classifier_specs']
    clean_records = state['01_cleaning']['clean_records']
    daily_dev = state['05_prediction']['daily_dev']
    daily_folds = state['05_prediction']['daily_folds']
    daily_test = state['05_prediction']['daily_test']
    dev = state['04_classification']['dev']
    environment_features = state['04_classification']['environment_features']
    learned_regressor = state['05_prediction']['learned_regressor']
    learned_regressor_name = state['05_prediction']['learned_regressor_name']
    numeric_features = state['04_classification']['numeric_features']
    profile_features = state['04_classification']['profile_features']
    regression_pipeline = state['05_prediction']['regression_pipeline']
    regressor_specs = state['05_prediction']['regressor_specs']
    severity_folds = state['04_classification']['severity_folds']
    y_dev = state['04_classification']['y_dev']
    y_test = state['04_classification']['y_test']

    importance=permutation_importance(best_classifier,X_test,y_test,scoring='f1_macro',
                                    n_repeats=N_IMPORTANCE_REPEATS,random_state=SEED,n_jobs=1)
    severity_importance=pd.DataFrame({'feature':FEATURES,'macro_f1_drop_mean':importance.importances_mean,
                                    'repeat_sd':importance.importances_std}).sort_values('macro_f1_drop_mean',ascending=False)
    display(save_table(severity_importance,'24_severity_permutation_importance'))
    plot_importance=severity_importance.sort_values('macro_f1_drop_mean')
    plt.barh(plot_importance.feature,plot_importance.macro_f1_drop_mean,xerr=plot_importance.repeat_sd,color='#147d92')
    plt.axvline(0,color='black',lw=.8);plt.xlabel('Holdout macro-F1 decrease after permutation')
    plt.title('Severity model: original-variable importance (repeat SD bars)')
    finish_figure('11_severity_variable_importance')

    feature_sets={'calendar':calendar_features,'environment':environment_features,'recorded vehicle / gender':profile_features,
        'calendar + environment':calendar_features+environment_features,
        'environment + vehicle / gender':environment_features+profile_features,'all configured features':FEATURES}
    ablation_rows=[]
    for name,cols in feature_sets.items():
        scores=[]
        for ti,vi in severity_folds:
            p=classifier_pipeline(classifier_specs[best_classifier_name],cols).fit(dev[cols].iloc[ti],y_dev.iloc[ti])
            scores.append(f1_score(y_dev.iloc[vi],p.predict(dev[cols].iloc[vi]),average='macro',zero_division=0))
        ablation_rows.append({'feature_combination':name,'n_original_variables':len(cols),
            'variables':', '.join(cols),'CV_macro_f1_mean':np.mean(scores),'CV_macro_f1_sd':np.std(scores,ddof=1)})
    ablation_results=pd.DataFrame(ablation_rows).sort_values('CV_macro_f1_mean',ascending=False)
    display(save_table(ablation_results,'25_feature_combination_ablation'))

    oof_frames=[]
    for ti,vi in severity_folds:
        p=classifier_pipeline(classifier_specs[best_classifier_name]).fit(X_dev.iloc[ti],y_dev.iloc[ti])
        part=dev.iloc[vi][['weather','road_condition','vehicle_group','incident_proxy','severe']].copy()
        part['oof_probability']=p.predict_proba(X_dev.iloc[vi])[:,1]
        oof_frames.append(part)
    oof=pd.concat(oof_frames,ignore_index=True)
    combinations=oof.groupby(['weather','road_condition','vehicle_group'],dropna=False).agg(
        records=('severe','size'),unique_proxy_support=('incident_proxy','nunique'),
        observed_severe_record_fraction=('severe','mean'),mean_out_of_fold_probability=('oof_probability','mean')).reset_index()
    combinations=combinations.loc[combinations.records.ge(MIN_COMBINATION_RECORDS)].sort_values('records',ascending=False)
    display(save_table(combinations,'26_supported_weather_road_vehicle_combinations'))

    rule_model=classifier_pipeline(DecisionTreeClassifier(max_depth=3,min_samples_leaf=30,random_state=SEED)).fit(X_dev,y_dev)
    rule_names=rule_model.named_steps['preprocess'].get_feature_names_out()
    rule_text=export_text(rule_model.named_steps['model'],feature_names=list(rule_names),decimals=2)
    print(rule_text)
    (OUTPUT_DIR/'severity_reference_tree_rules.txt').write_text(rule_text,encoding='utf-8')
    tree=rule_model.named_steps['model'].tree_;tree_classes=rule_model.named_steps['model'].classes_
    rule_rows=[]
    def walk_rules(node=0,conditions=None):
        conditions=[] if conditions is None else conditions
        if tree.children_left[node]==tree.children_right[node]:
            values=tree.value[node][0];prob=values/values.sum()
            rule_rows.append({'conditions':' AND '.join(conditions) or '(root)',
                'development_record_support':int(tree.n_node_samples[node]),
                'tree_probability_severe':float(prob[list(tree_classes).index(1)]),
                'predicted_class':int(tree_classes[np.argmax(prob)])})
            return
        name=rule_names[tree.feature[node]];threshold=tree.threshold[node]
        walk_rules(tree.children_left[node],conditions+[f'{name} <= {threshold:.3f}'])
        walk_rules(tree.children_right[node],conditions+[f'{name} > {threshold:.3f}'])
    walk_rules()
    display(save_table(pd.DataFrame(rule_rows),'27_reference_decision_tree_combinations'))
    print('Reference tree holdout macro-F1:',round(f1_score(y_test,rule_model.predict(X_test),average='macro'),3))

    def predict_record_severity(record,model=best_classifier):
        # Supply original CSV column names; deterministic cleaning matches the training notebook.
        incoming=clean_records(pd.DataFrame([record]))
        for c in FEATURES:
            if c not in numeric_features:
                incoming[c]=incoming[c].astype(object).where(incoming[c].notna(),np.nan)
        if incoming.date.isna().any():
            raise ValueError('Supply a usable DATE for calendar features.')
        p=float(model.predict_proba(incoming[FEATURES])[0,1])
        return {'predicted_class':int(model.predict(incoming[FEATURES])[0]),
                'probability_major_or_fatal':p,'model':best_classifier_name,
                'cleaned_area':incoming.area.iloc[0],'vehicle_group':incoming.vehicle_group.iloc[0]}

    example={'DATE':'15/08/2024','TIME':'8:00 pm','LOCATION':'P1 San Teodoro',
             'WEATHER':'Rainy','ROAD CONDITION':'Wet','VEHICLE TYPE':'Motorcycle','GENDER':'Male'}
    display(pd.DataFrame([predict_record_severity(example)]))
    scenario_rows=[]
    for weather,road in [('Clear','Dry'),('Rainy','Wet'),('Cloudy','Dry')]:
        rec={**example,'WEATHER':weather,'ROAD CONDITION':road}
        scenario_rows.append({'weather':weather,'road_condition':road,**predict_record_severity(rec)})
    display(save_table(pd.DataFrame(scenario_rows),'28_illustrative_prediction_scenarios'))
    print('Illustrative model scenarios; not causal interventions or calibrated operational risk estimates.')

    inspection_regressor=learned_regressor if 'baseline' in best_regressor_name.lower() else best_regressor
    inspection_regressor_name=learned_regressor_name if 'baseline' in best_regressor_name.lower() else best_regressor_name
    print('Daily feature importance describes:',inspection_regressor_name)
    daily_pi=permutation_importance(inspection_regressor,daily_test[DAILY_FEATURES],daily_test.target,
        scoring='neg_mean_absolute_error',n_repeats=N_IMPORTANCE_REPEATS,random_state=SEED,n_jobs=1)
    daily_importance=pd.DataFrame({'feature':DAILY_FEATURES,'MAE_increase_mean':daily_pi.importances_mean,
                                 'repeat_sd':daily_pi.importances_std}).sort_values('MAE_increase_mean',ascending=False)
    daily_importance['inspected_model']=inspection_regressor_name
    display(save_table(daily_importance,'33_daily_prediction_variable_importance'))
    p=daily_importance.sort_values('MAE_increase_mean')
    plt.barh(p.feature,p.MAE_increase_mean,xerr=p.repeat_sd,color='#6e638f')
    plt.axvline(0,color='black',lw=.8);plt.xlabel('Holdout MAE increase after permutation')
    plt.title(f'Daily count variable importance: {inspection_regressor_name}')
    finish_figure('13_daily_prediction_variable_importance')

    # Next-day prediction after the last observed dataset date, using only known past counts.

    daily_calendar=[c for c in DAILY_FEATURES if c.startswith(('weekday','month')) or c in ['is_weekend','trend_day']]
    daily_lags=[c for c in DAILY_FEATURES if c.startswith('lag_')]
    daily_rolling=[c for c in DAILY_FEATURES if c.startswith('rolling_')]
    daily_feature_sets={'calendar only':daily_calendar,'past lags only':daily_lags,
        'rolling summaries only':daily_rolling,'calendar + lags':daily_calendar+daily_lags,
        'calendar + rolling summaries':daily_calendar+daily_rolling,'all daily features':DAILY_FEATURES}
    daily_ablation_rows=[]
    for name,cols in daily_feature_sets.items():
        scores=[]
        for ti,vi in daily_folds:
            model=regression_pipeline(regressor_specs[learned_regressor_name]).fit(
                daily_dev[cols].iloc[ti],daily_dev.target.iloc[ti])
            pred=np.clip(model.predict(daily_dev[cols].iloc[vi]),0,None)
            scores.append(mean_absolute_error(daily_dev.target.iloc[vi],pred))
        daily_ablation_rows.append({'feature_combination':name,'inspected_model':learned_regressor_name,
            'variables':', '.join(cols),'CV_MAE_mean':np.mean(scores),'CV_MAE_sd':np.std(scores,ddof=1)})
    daily_ablation=pd.DataFrame(daily_ablation_rows).sort_values('CV_MAE_mean')
    display(save_table(daily_ablation,'33b_daily_feature_combination_ablation'))
    if isinstance(learned_regressor.named_steps['model'],PoissonRegressor):
        coefficients=learned_regressor.named_steps['model'].coef_
        scale=learned_regressor.named_steps['scale'].scale_
        poisson_coefficients=pd.DataFrame({'feature':DAILY_FEATURES,'standardized_log_count_coefficient':coefficients,
            'log_count_coefficient_per_raw_unit':coefficients/scale,
            'count_ratio_per_raw_unit':np.exp(coefficients/scale)})
        display(save_table(poisson_coefficients,'33c_poisson_count_coefficients'))
        print('Coefficients describe the fitted log-link model holding other inputs fixed; no causal or significance claim.')

    exported_names = ['severity_importance', 'ablation_results', 'combinations', 'rule_model', 'rule_text', 'predict_record_severity', 'scenario_rows', 'daily_importance', 'daily_ablation', 'poisson_coefficients']
    return {name:value for name,value in locals().items() if name in exported_names}


# Source notebook cell 75
def module_09_export_results(state):
    """Save fitted models, settings, summary and one ZIP organized by task module.

    Inputs: Completed outputs from all preceding modules.
    Returns: a dictionary of named module outputs."""
    OUTPUT_DIR = OUTPUT_ROOT / '09_exports'
    save_table, finish_figure = output_helpers(OUTPUT_DIR)
    DAILY_FEATURES = state['05_prediction']['DAILY_FEATURES']
    DATE_FIXES = state['01_cleaning']['DATE_FIXES']
    FEATURES = state['04_classification']['FEATURES']
    MC_LABELS = state['04_classification']['MC_LABELS']
    SOURCE_SHA256 = state['00_setup']['SOURCE_SHA256']
    best_classifier = state['04_classification']['best_classifier']
    best_classifier_name = state['04_classification']['best_classifier_name']
    best_forecast_method = state['06_forecasting']['best_forecast_method']
    best_regressor = state['05_prediction']['best_regressor']
    best_regressor_name = state['05_prediction']['best_regressor_name']
    classification_holdout = state['07_performance']['classification_holdout']
    classifier_pipeline = state['04_classification']['classifier_pipeline']
    classifier_specs = state['04_classification']['classifier_specs']
    daily = state['01_cleaning']['daily']
    daily_deployment_model = state['05_prediction']['daily_deployment_model']
    daily_holdout = state['07_performance']['daily_holdout']
    df = state['01_cleaning']['df']
    forecast_holdout = state['07_performance']['forecast_holdout']
    interval_coverage = state['07_performance']['interval_coverage']
    interval_half_width = state['06_forecasting']['interval_half_width']
    mc_name = state['04_classification']['mc_name']
    model_data = state['04_classification']['model_data']
    monthly_series = state['06_forecasting']['monthly_series']
    multiclass_model = state['04_classification']['multiclass_model']
    raw = state['00_setup']['raw']

    severity_deployment_model=classifier_pipeline(classifier_specs[best_classifier_name]).fit(
        model_data[FEATURES],model_data.severe.astype(int))
    joblib.dump({'pipeline':best_classifier,'features':FEATURES,'selected_model':best_classifier_name,
                 'training_period':'2021–2023','target':'major injury or fatal vs minor or no injury'},
                OUTPUT_DIR/'severity_evaluation_model.joblib')
    joblib.dump({'pipeline':severity_deployment_model,'features':FEATURES,'selected_model':best_classifier_name,
                 'training_period':'reliable injury grades 2021–2024','target':'major injury or fatal vs minor or no injury'},
                OUTPUT_DIR/'severity_deployment_model.joblib')
    joblib.dump({'pipeline':multiclass_model,'features':FEATURES,'classes':MC_LABELS},
                OUTPUT_DIR/'multiclass_evaluation_model.joblib')
    joblib.dump({'pipeline':best_regressor,'features':DAILY_FEATURES,'selected_model':best_regressor_name,
                 'forecast_type':'one-day-ahead with observed lag updates','training_period':'through 2024'},
                OUTPUT_DIR/'daily_count_evaluation_model.joblib')
    joblib.dump({'pipeline':daily_deployment_model,'features':DAILY_FEATURES,'selected_model':best_regressor_name,
                 'last_observed_date':str(daily.index.max().date())},OUTPUT_DIR/'daily_count_deployment_model.joblib')
    joblib.dump({'method':best_forecast_method,'training_series':monthly_series,
                 'approx_interval_half_width':interval_half_width,'forecast_function':'forecast_monthly in this notebook'},
                OUTPUT_DIR/'monthly_forecast_state.joblib')

    settings={'seed':SEED,'source_sha256':SOURCE_SHA256,'rows':len(raw),'versions':versions,
        'apply_context_date_fixes':APPLY_CONTEXT_DATE_FIXES,'drop_exact_duplicates':DROP_EXACT_DUPLICATES,
        'include_investigated_cause':INCLUDE_INVESTIGATED_CAUSE,'assume_complete_reporting_calendar':ASSUME_COMPLETE_REPORTING_CALENDAR,
        'severity_test_year':SEVERITY_TEST_YEAR,'count_test_year':COUNT_TEST_YEAR,
        'severity_features':FEATURES,'daily_features':DAILY_FEATURES,'forecast_horizon':FORECAST_HORIZON,
        'date_corrections':DATE_FIXES,'selected_binary_classifier':best_classifier_name,
        'selected_multiclass_classifier':mc_name,'selected_daily_regressor':best_regressor_name,
        'selected_monthly_forecaster':best_forecast_method}
    (OUTPUT_DIR/'analysis_settings.json').write_text(json.dumps(settings,indent=2,default=str),encoding='utf-8')
    (OUTPUT_DIR/'requirements_tested.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items())+'\n',encoding='utf-8')

    selected_classification=classification_holdout.iloc[0]
    selected_daily=daily_holdout.iloc[0]
    selected_forecast=forecast_holdout.iloc[0]
    summary_lines=[
        '# RACC analysis run summary',
        f'Source SHA-256: {SOURCE_SHA256}',
        f'Raw records: {len(raw)}; retained records: {len(df)}; provisional incident keys: {df.incident_proxy.nunique()}.',
        f'Provisional context-dependent dates: {int(df.provisional_date_flag.sum())}; usable injury labels: {int(df.severity_grade.notna().sum())}.',
        '', '## Chronological holdout results',
        f'Binary classifier: {best_classifier_name}; accuracy={selected_classification.accuracy:.3f}; macro-F1={selected_classification.macro_f1:.3f}; ROC-AUC={selected_classification.ROC_AUC:.3f}.',
        f'Daily prediction: {best_regressor_name}; MAE={selected_daily.MAE:.3f}; RMSE={selected_daily.RMSE:.3f}; R2={selected_daily.R2:.3f}.',
        f'Monthly forecasting: {best_forecast_method}; MAE={selected_forecast.MAE:.3f}; RMSE={selected_forecast.RMSE:.3f}; WAPE={selected_forecast.WAPE_pct:.1f}%.',
        f'Approximate 90% monthly interval empirical holdout coverage: {interval_coverage:.1%}.',
        '', '## What the results support',
        ('- Severity discrimination is close to chance. Do not describe ranked features or combinations as validated injury predictors.'
            if selected_classification.ROC_AUC<.55 else
            '- Interpret severity discrimination with the reported holdout metrics, calibration and confidence intervals.'),
        ('- A constant daily-count baseline wins development MAE. Variable importance is shown for a separate learned reference; it does not demonstrate improved prediction over that baseline.'
            if 'baseline' in best_regressor_name.lower() else
            '- Compare daily regression against the mean and previous-week baselines; inspect bias and error metrics together.'),
        ('- The monthly forecaster has negative holdout R2. Any improvement over a naive forecast is relative, not evidence of high absolute forecast accuracy.'
            if selected_forecast.R2<0 else
            '- Monthly forecast performance is specific to this short history and recording process.'),
        '', '## Interpretation limits',
        '- Counts measure reported records. True accident rates require exposure and verified incident IDs.',
        '- Most 2025 severity statuses are consciousness descriptions, not injury grades.',
        '- Date corrections are provisional and require source-owner confirmation.',
        '- Missing calendar dates are assumed to have zero recorded rows; completeness is unverified.',
        '- Feature importance and combinations describe predictions and associations, not causation.',
        '- Dataset-relative extrapolations may include dates already elapsed when this notebook is run.',
        '- Limited data, rare fatal cases and temporal reporting changes limit generalization.',
    ]
    run_summary='\n'.join(summary_lines)
    (OUTPUT_DIR/'RUN_SUMMARY.md').write_text(run_summary,encoding='utf-8')
    display(Markdown(run_summary))

    module_index=pd.DataFrame([
        {'module':key,'task':spec['title'],'prerequisites':', '.join(spec['requires']) or 'none',
         'output_folder':spec['folder']} for key,spec in MODULE_SPECS.items()
    ])
    save_table(module_index,'44_module_directory')
    manifest=pd.DataFrame([{'filename':str(p.relative_to(OUTPUT_ROOT)),'bytes':p.stat().st_size}
        for p in sorted(OUTPUT_ROOT.rglob('*')) if p.is_file() and p.name!='42_output_manifest.csv'])
    save_table(manifest,'42_output_manifest')
    archive=Path(shutil.make_archive(str(OUTPUT_ROOT.resolve()),'zip',root_dir=OUTPUT_ROOT))
    print(f'Created {len(manifest)+1} categorized output files and archive: {archive.name}')

    exported_names = ['severity_deployment_model', 'settings', 'run_summary', 'manifest', 'archive', 'module_index']
    return {name:value for name,value in locals().items() if name in exported_names}

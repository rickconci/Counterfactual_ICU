import pandas as pd
from pathlib import Path
try:
    import pyarrow as pa
except Exception:  # pragma: no cover - fallback to satisfy static analyzers
    pa = None  # type: ignore


BASE_DIR = Path('/n/netscratch/mzitnik_lab/Lab/rconci/BIOMM')
MIMIC_DATA_DIR = BASE_DIR/'input_data'
WAVEFORM_DIR = BASE_DIR/'numerics'
HELPFUL_DF_DIR = BASE_DIR/'sCF_helpful_csvs'
PROCESSED_DATA_DIR = BASE_DIR/'processed_data'


### Treatment processing
treatment_cols_to_keep = {
    'inputevents_cv': ['subject_id', 'hadm_id', 'itemid', 'charttime', 'amount', 'amountuom', 'originalroute'],
    'inputevents_mv': ['subject_id', 'hadm_id', 'itemid', 'starttime', 'endtime', 'amount', 'amountuom','patientweight', 'ORDERCATEGORYNAME',  'ORDERCATEGORYDESCRIPTION', 'rate', 'rateuom', 'CANCELREASON'],
    'emar': ['subject_id', 'hadm_id', 'emar_id', 'charttime', 'medication', 'event_txt']
}



### Chartevents processing
physio_cols_to_keep = {
    'chartevents': ['subject_id', 'hadm_id', 'itemid', 'charttime','valuenum', 'valueuom'],
    'outputevents': ['subject_id', 'hadm_id', 'itemid', 'charttime', 'value', 'valueuom'],
    'labevents': ['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum', 'valueuom'],
    'micro_events': ['subject_id', 'hadm_id', 'charttime','SPEC_TYPE_DESC', 'ORG_NAME', 'ab_name', 'INTERPRETATION'],
}



### Waveform processing

TARGET_COLUMNS = [
    'hadm_id','record_name','absolute_timestamp',
    'ABP MEAN','NBP MEAN','CVP','HR','RESP',
    'record_start_time','record_end_time','icu_admission_time','time_seconds'
]

TARGET_SCHEMA = pa.schema([
    ("hadm_id", pa.int64()),
    ("record_name", pa.string()),
    ("absolute_timestamp", pa.timestamp("ns", tz="UTC")),
    ("ABP MEAN", pa.float64()),
    ("NBP MEAN", pa.float64()),
    ("CVP", pa.float64()),
    ("HR", pa.float64()),
    ("RESP", pa.float64()),
    ("record_start_time", pa.timestamp("us")),
    ("record_end_time", pa.timestamp("us")),
    ("icu_admission_time", pa.timestamp("us")),
    ("time_seconds", pa.int64()),
])

from pathlib import Path
import sys

import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_path, get_value, load_config

ROOT = PROJECT_ROOT
CONFIG_PATH = ROOT / 'config' / 'config.yaml'
config = load_config(CONFIG_PATH)

MODELS = get_path(config, 'models_dir', base_dir=ROOT)
PROCESSED = get_path(config, 'processed_dir', base_dir=ROOT)
REPORTS = get_path(config, 'reports_dir', base_dir=ROOT)
REPORTS.mkdir(exist_ok=True)

MODEL_PATH = MODELS / 'xgboost.pkl'
XVAL_PATH = PROCESSED / 'X_val.parquet'
RANDOM_STATE = int(get_value(config, 'project', 'random_state'))
SHAP_SAMPLE_SIZE = get_value(config, 'analysis', 'shap_sample_size')
SHAP_CHUNK_SIZE = int(get_value(config, 'analysis', 'shap_chunk_size'))
SHAP_BEESWARM_SAMPLE_SIZE = int(get_value(config, 'analysis', 'shap_beeswarm_sample_size'))
SHAP_MAX_DISPLAY = int(get_value(config, 'analysis', 'shap_max_display'))
SHAP_FIGURE_DPI = int(get_value(config, 'analysis', 'shap_figure_dpi'))

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)
if not XVAL_PATH.exists():
    raise FileNotFoundError(XVAL_PATH)

print('Loading model and data...')
model = joblib.load(MODEL_PATH)
X_val = pd.read_parquet(XVAL_PATH)

output_suffix = ''
if SHAP_SAMPLE_SIZE is not None:
    n_sample = min(int(SHAP_SAMPLE_SIZE), len(X_val))
    X_val = X_val.sample(n=n_sample, random_state=RANDOM_STATE)
    print(f'Using SHAP sample of {n_sample} rows')
    output_suffix = '_sample'

import shap
import matplotlib.pyplot as plt

try:
    booster = model.get_booster()
    expected = list(booster.feature_names) if booster.feature_names is not None else None
except Exception:
    expected = None
if expected is not None:
    missing = [f for f in expected if f not in X_val.columns]
    extra = [c for c in X_val.columns if c not in expected]
    if missing:
        print('Warning: model expects features not present in X_val:', missing[:10])
    if extra:
        print('Warning: X_val has extra features not in model:', extra[:10])
    X_val = X_val.reindex(columns=expected)

explainer = shap.TreeExplainer(model)

shap_chunks = []
for start in range(0, len(X_val), SHAP_CHUNK_SIZE):
    end = min(start + SHAP_CHUNK_SIZE, len(X_val))
    print(f'Computing SHAP for rows {start}:{end}')
    X_chunk = X_val.iloc[start:end]
    sv = explainer.shap_values(X_chunk)
    shap_chunks.append(sv)

first = shap_chunks[0]
if isinstance(first, list):
    n_outputs = len(first)
    shap_values = [np.concatenate([chunk[i] for chunk in shap_chunks], axis=0) for i in range(n_outputs)]
    sv = shap_values[-1]
else:
    sv = np.concatenate(shap_chunks, axis=0)

abs_mean = np.abs(sv).mean(axis=0)
feat_imp = pd.DataFrame({'feature': X_val.columns, 'mean_abs_shap': abs_mean})
feat_imp = feat_imp.sort_values('mean_abs_shap', ascending=False)
feat_imp.to_csv(REPORTS / f'shap_summary{output_suffix}.csv', index=False)

top20 = feat_imp.head(SHAP_MAX_DISPLAY)
plt.figure(figsize=(8,6))
plt.barh(top20['feature'][::-1], top20['mean_abs_shap'][::-1])
plt.xlabel('mean |SHAP|')
plt.tight_layout()
plt.savefig(REPORTS / f'shap_top20{output_suffix}.png', dpi=SHAP_FIGURE_DPI)

beeswarm_n = min(SHAP_BEESWARM_SAMPLE_SIZE, len(X_val))
beeswarm_idx = np.random.default_rng(RANDOM_STATE).choice(len(X_val), size=beeswarm_n, replace=False)
beeswarm_sv = sv[beeswarm_idx]
beeswarm_X = X_val.iloc[beeswarm_idx]
plt.figure(figsize=(10, 7))
shap.summary_plot(beeswarm_sv, beeswarm_X, show=False, max_display=SHAP_MAX_DISPLAY)
plt.tight_layout()
plt.savefig(REPORTS / f'shap_beeswarm{output_suffix}.png', dpi=SHAP_FIGURE_DPI, bbox_inches='tight')
print('Wrote', REPORTS / f'shap_summary{output_suffix}.csv', 'and', f'shap_top20{output_suffix}.png')
print('Wrote', REPORTS / f'shap_beeswarm{output_suffix}.png')

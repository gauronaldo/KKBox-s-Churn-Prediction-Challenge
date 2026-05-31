import pickle
import joblib
from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
MODELS = ROOT / 'models'
PROCESSED = ROOT / 'data' / 'processed'
REPORTS = ROOT / 'reports'
REPORTS.mkdir(exist_ok=True)

X_train = pd.read_parquet(PROCESSED / 'X_train.parquet')
feature_names = list(X_train.columns)

def get_importances(model):
    if hasattr(model, 'feature_importances_'):
        vals = model.feature_importances_
        return list(vals)
    if hasattr(model, 'coef_'):
        coef = model.coef_
        if coef.ndim == 2:
            coef = coef[0]
        return list(abs(coef))
    return [0.0] * len(feature_names)

models = {}
for name in ['xgboost.pkl','lightgbm.pkl','random_forest.pkl','logistic_regression.pkl']:
    path = MODELS / name
    if path.exists():
        try:
            models[name] = joblib.load(path)
        except Exception:
            with open(path,'rb') as f:
                models[name] = pickle.load(f)

rows = []
for mname, model in models.items():
    imps = get_importances(model)
    if len(imps) != len(feature_names):
        try:
            import numpy as np
            arr = np.array(imps)
            arr = arr[:len(feature_names)]
            imps = list(arr)
        except Exception:
            imps = [0.0]*len(feature_names)
    for feat, val in zip(feature_names, imps):
        rows.append({'model': mname.replace('.pkl',''), 'feature': feat, 'importance': float(val)})

df = pd.DataFrame(rows)
out_frames = []
for m, g in df.groupby('model'):
    s = g['importance'].sum()
    if s > 0:
        g = g.copy()
        g['importance_norm'] = g['importance'] / s
    else:
        g = g.copy()
        g['importance_norm'] = 0.0
    out_frames.append(g)

out = pd.concat(out_frames).sort_values(['model','importance_norm'], ascending=[True, False])
out.to_csv(REPORTS / 'feature_importances.csv', index=False)
print('Wrote', REPORTS / 'feature_importances.csv')

if 'xgboost' in out['model'].unique():
    top = out[out['model']=='xgboost'].sort_values('importance_norm', ascending=False).head(10)
    print('\nTop 10 features for xgboost:')
    print(top[['feature','importance_norm']].to_string(index=False))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

en_reps = np.load("representations/bloom_english_reps.npy")
en_meta = pd.read_csv("representations/bloom_english_meta.csv")
N_LAYERS = en_reps.shape[1]

def probe_binary(X_layers, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    out = []
    for L in range(X_layers.shape[1]):
        X = X_layers[:, L, :]
        f1s = []
        for tr, te in skf.split(X, y):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, C=1.0, solver='liblinear')
            )
            clf.fit(X[tr], y[tr])
            f1s.append(f1_score(y[te], clf.predict(X[te]), average="macro"))
        out.append(np.mean(f1s))
    return np.array(out)

def probe_regression(X_layers, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    out = []
    for L in range(X_layers.shape[1]):
        X = X_layers[:, L, :]
        r2s = []
        for tr, te in kf.split(X):
            reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            reg.fit(X[tr], y[tr])
            r2s.append(r2_score(y[te], reg.predict(X[te])))
        out.append(np.mean(r2s))
    return np.array(out)

# voicing
m = en_meta['voiced_unvoiced'].isin(['voiced','unvoiced'])
y = (en_meta.loc[m,'voiced_unvoiced']=='voiced').astype(int).values
v = probe_binary(en_reps[m.values], y)
print(f"voicing  n={m.sum()} peak L{v.argmax()} F1={v.max():.3f}")

# real word
m = en_meta['is_real_word'].notna()
y = en_meta.loc[m,'is_real_word'].astype(str).map({'True':1,'False':0}).values
r = probe_binary(en_reps[m.values], y)
print(f"realword n={m.sum()} peak L{r.argmax()} F1={r.max():.3f}")

# ortho
m = en_meta['orthographic_transparency'].isin(['transparent','opaque'])
y = (en_meta.loc[m,'orthographic_transparency']=='transparent').astype(int).values
o = probe_binary(en_reps[m.values], y)
print(f"ortho    n={m.sum()} peak L{o.argmax()} F1={o.max():.3f}")

# syllable
m = en_meta['syllable_count'].notna() & (en_meta['task_category']!='sentence')
y = en_meta.loc[m,'syllable_count'].astype(float).values
s = probe_regression(en_reps[m.values], y)
print(f"syllable n={m.sum()} peak L{s.argmax()} R²={s.max():.3f}")

pd.DataFrame({
    'layer': range(N_LAYERS),
    'f1_voicing': v, 'f1_realword': r, 'f1_ortho': o, 'r2_syllable': s,
}).to_csv("results_english_probing_bloom.csv", index=False)
print("saved.")
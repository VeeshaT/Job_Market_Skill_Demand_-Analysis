"""
================================================================
 Model 1 : Linear Regression — Salary Prediction
 File    : model_backend.py  (Backend — Flask REST API)
 Project : Job Market & Skill Demand Analysis
================================================================
 
 HOW TO RUN
 ----------
      pip install pandas scikit-learn flask flask-cors
      python model_backend.py
 
 Place  combined_cleaned_jobs (1).csv  in the same folder.
 The script auto-detects column names so it works with
 any version of the CSV.
================================================================
"""
 
import warnings
warnings.filterwarnings('ignore')
 
import numpy  as np
import pandas as pd
 
from sklearn.linear_model    import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import (mean_absolute_error,
                                     mean_squared_error,
                                     r2_score)
from flask      import Flask, request, jsonify
from flask_cors import CORS
 
# ================================================================
#  CONFIGURATION
# ================================================================
INPUT_CSV    = "combined_cleaned_jobs (1).csv"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
PORT         = 5000
 
FEATURE_NAMES = [
    'experience_required',
    'skills_count',
    'location_enc',
    'job_category_enc',
]
 
# ── globals filled by train_pipeline() ───────────────────────
model         = None
le_cat        = None
train_metrics = {}
test_metrics  = {}
 
 
# ================================================================
#  HELPERS
# ================================================================
 
def parse_salary(value) -> float:
    try:
        s = str(value).strip().upper()
        s = s.replace('L','').replace('₹','').replace(',','').replace(' ','')
        return float(s)
    except Exception:
        return np.nan
 
 
def extract_experience_from_title(title: str) -> float:
    t = str(title).lower()
    senior = ['senior','sr.','lead','principal','staff','head',
              'director','vp','manager','chief']
    entry  = ['junior','jr.','entry','intern','trainee',
              'fresher','graduate','associate']
    if any(k in t for k in senior): return 7.0
    if any(k in t for k in entry):  return 1.0
    return 3.0
 
 
def get_category_from_exp(exp: float) -> str:
    if exp <= 2.0: return 'Entry-Level Analyst'
    if exp >= 6.0: return 'Senior Leadership'
    return 'Mid-Level Specialist'
 
 
# ================================================================
#  TRAINING PIPELINE
# ================================================================
 
def train_pipeline():
    global model, le_cat, train_metrics, test_metrics
 
    print("=" * 65)
    print("  Model 1 : Linear Regression — Salary Prediction")
    print("=" * 65)
 
    # ── Load ─────────────────────────────────────────────────
    print(f"\n[LOAD]  Reading  '{INPUT_CSV}' ...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"\n[ERROR] '{INPUT_CSV}' not found!")
        print("        Put the CSV file in the same folder as model_backend.py")
        raise
 
    # Normalise column names to lowercase + strip spaces
    df.columns = [c.strip().lower().replace(' ','_') for c in df.columns]
    print(f"[LOAD]  {len(df):,} rows  |  columns: {df.columns.tolist()}")
 
    # ── salary_num ───────────────────────────────────────────
    for c in ['salary','salary_num','salary_inr','pay','ctc','package']:
        if c in df.columns:
            df['salary_num'] = df[c].apply(parse_salary)
            print(f"[FEAT]  Salary  ← '{c}'")
            break
    else:
        raise ValueError(f"No salary column found in {df.columns.tolist()}")
 
    before = len(df)
    df = df.dropna(subset=['salary_num']).reset_index(drop=True)
    print(f"[FEAT]  Salary rows kept : {before:,} → {len(df):,}")
    print(f"[FEAT]  Salary range     : {df['salary_num'].min():.2f} – {df['salary_num'].max():.2f}")
 
    # ── skills_count ─────────────────────────────────────────
    for c in ['skills','skill','skill_set','key_skills','skills_list']:
        if c in df.columns:
            df['skills_count'] = (
                df[c].fillna('').apply(
                    lambda s: max(1, len([x for x in str(s).split(',') if x.strip()]))
                )
            )
            print(f"[FEAT]  Skills ← '{c}'  |  range: {df['skills_count'].min()}–{df['skills_count'].max()}")
            break
    else:
        df['skills_count'] = 3
        print("[FEAT]  No skills column found — defaulting to 3")
 
    # ── experience_required ───────────────────────────────────
    for c in ['experience_required','experience','exp','years_exp','min_experience']:
        if c in df.columns:
            df['experience_required'] = (
                pd.to_numeric(df[c], errors='coerce').fillna(3.0).clip(0, 20)
            )
            print(f"[FEAT]  Experience ← '{c}'  |  range: {df['experience_required'].min():.0f}–{df['experience_required'].max():.0f}")
            break
    else:
        for c in ['title','job_title','position','role','jobtitle']:
            if c in df.columns:
                df['experience_required'] = df[c].apply(extract_experience_from_title)
                print(f"[FEAT]  Experience extracted from title col '{c}'")
                break
        else:
            df['experience_required'] = 3.0
            print("[FEAT]  No experience column — defaulting to 3.0")
 
    # ── location_enc ─────────────────────────────────────────
    if 'source' in df.columns:
        le_loc = LabelEncoder()
        df['location_enc'] = le_loc.fit_transform(df['source'].fillna('india').astype(str))
        print(f"[FEAT]  Location ← 'source'  |  {dict(enumerate(le_loc.classes_))}")
    else:
        df['location_enc'] = 0   # india = 0
        print("[FEAT]  No source column — location_enc = 0 (india)")
 
    # ── job_category_enc ─────────────────────────────────────
    if 'cluster_label' in df.columns:
        df['job_category'] = df['cluster_label'].fillna('Mid-Level Specialist')
    else:
        df['job_category'] = df['experience_required'].apply(get_category_from_exp)
 
    le_cat = LabelEncoder()
    df['job_category_enc'] = le_cat.fit_transform(df['job_category'])
    print(f"[FEAT]  Category encoding: {dict(enumerate(le_cat.classes_))}")
 
    # ── Remove salary outliers >99th pct ─────────────────────
    p99 = df['salary_num'].quantile(0.99)
    df  = df[df['salary_num'] <= p99].reset_index(drop=True)
    print(f"[FEAT]  After outlier removal: {len(df):,} rows")
 
    # ── Build X, y ────────────────────────────────────────────
    X = df[FEATURE_NAMES].astype(float)
    y = df['salary_num'].astype(float)
 
    # ── Split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\n[SPLIT] Train={len(X_train):,}  Test={len(X_test):,}")
 
    # ── Train ────────────────────────────────────────────────
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"\n[TRAIN] sklearn LinearRegression fitted.")
    for name, coef in zip(FEATURE_NAMES, model.coef_):
        print(f"        {name:<28}: {coef:+.4f}")
    print(f"        {'intercept':<28}: {model.intercept_:+.4f}")
 
    # ── Evaluate ─────────────────────────────────────────────
    def calc_metrics(y_true, y_pred):
        return {
            'mae' : round(float(mean_absolute_error(y_true, y_pred)), 4),
            'rmse': round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            'r2'  : round(float(r2_score(y_true, y_pred)), 4),
        }
 
    train_metrics = calc_metrics(y_train, model.predict(X_train))
    test_metrics  = calc_metrics(y_test,  model.predict(X_test))
 
    print(f"\n[EVAL]  Train → R²={train_metrics['r2']}  MAE={train_metrics['mae']}  RMSE={train_metrics['rmse']}")
    print(f"[EVAL]  Test  → R²={test_metrics['r2']}  MAE={test_metrics['mae']}  RMSE={test_metrics['rmse']}")
    print(f"\n[READY] Flask API ready at http://localhost:{PORT}\n")
 
 
# ================================================================
#  FLASK APP
# ================================================================
app = Flask(__name__)
CORS(app)
 
 
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'LinearRegression'})
 
 
@app.route('/model_info', methods=['GET'])
def model_info():
    if model is None:
        return jsonify({'error': 'Model not trained yet'}), 503
 
    coefs = {n: round(float(c), 4) for n, c in zip(FEATURE_NAMES, model.coef_)}
    coefs['intercept'] = round(float(model.intercept_), 4)
 
    return jsonify({
        'algorithm'            : 'Linear Regression',
        'library'              : 'sklearn.linear_model.LinearRegression',
        'features'             : FEATURE_NAMES,
        'coefficients'         : coefs,
        'train_metrics'        : train_metrics,
        'test_metrics'         : test_metrics,
        'valid_locations'      : ['india', 'global'],
        'valid_job_categories' : list(le_cat.classes_),
    })
 
 
@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    {
      "experience"   : 3.0,
      "skills_count" : 5,
      "location"     : "india",
      "job_category" : "Mid-Level Specialist"
    }
    """
    if model is None:
        return jsonify({'error': 'Model not trained yet'}), 503
 
    data = request.get_json(force=True)
 
    try:
        experience = float(data.get('experience', 3.0))
        experience = max(0.0, min(20.0, experience))
    except Exception:
        experience = 3.0
 
    try:
        skills_count = int(data.get('skills_count', 3))
        skills_count = max(1, min(30, skills_count))
    except Exception:
        skills_count = 3
 
    location     = str(data.get('location', 'india')).strip().lower()
    location_enc = 1 if location == 'global' else 0
 
    job_category = str(data.get('job_category', 'Mid-Level Specialist')).strip()
    if le_cat is not None and job_category not in le_cat.classes_:
        job_category = le_cat.classes_[1] if len(le_cat.classes_) > 1 else le_cat.classes_[0]
    job_category_enc = int(le_cat.transform([job_category])[0])
 
    X_in      = np.array([[experience, skills_count, location_enc, job_category_enc]])
    predicted = float(model.predict(X_in)[0])
    predicted = max(0.0, round(predicted, 2))
 
    interp = (
        f"With {experience:.1f} yr(s) experience, {skills_count} skill(s), "
        f"{'India' if location=='india' else 'Global'} market, "
        f"{job_category} → Predicted Salary: ₹{predicted:.2f} Lakhs p.a."
    )
 
    return jsonify({
        'predicted_salary': predicted,
        'inputs_used': {
            'experience'       : experience,
            'skills_count'     : skills_count,
            'location'         : location,
            'location_enc'     : location_enc,
            'job_category'     : job_category,
            'job_category_enc' : job_category_enc,
        },
        'interpretation': interp,
        'model_equation': (
            f"salary = ({model.coef_[0]:+.4f})×{experience} "
            f"+ ({model.coef_[1]:+.4f})×{skills_count} "
            f"+ ({model.coef_[2]:+.4f})×{location_enc} "
            f"+ ({model.coef_[3]:+.4f})×{job_category_enc} "
            f"+ ({model.intercept_:+.4f})"
        ),
    })
 
 
# ================================================================
#  ENTRY POINT
# ================================================================
if __name__ == '__main__':
    train_pipeline()
    app.run(host='0.0.0.0', port=PORT, debug=False)
# ============================================================
#  Decision Tree — Job Category Classifier
#  Model 3 | Dataset: combined_cleaned_jobs.csv (30,347 rows)
#  Question: Based on skills listed, can we predict if a job
#            is Data, Engineering, or Management?
#  Input  : Skills keywords, Salary range, Experience level
#  Output : Predicted job category with confidence score
#  Library: sklearn.tree.DecisionTreeClassifier
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
os.startfile('decision_tree_results.png')
os.startfile('decision_tree_structure.png')
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# ─────────────────────────────────────────────────────────────
# STEP 1 — Load Dataset
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  Decision Tree — Job Category Classifier")
print("=" * 60)

df = pd.read_csv('combined_cleaned_jobs.csv')
print(f"\n✅ Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"   Columns: {df.columns.tolist()}")

# ─────────────────────────────────────────────────────────────
# STEP 2 — Map job titles → 3 categories
# ─────────────────────────────────────────────────────────────
def assign_category(title):
    title = str(title).lower()

    data_keywords = [
        'data scientist', 'data analyst', 'data engineer', 'data architect',
        'machine learning', 'ml engineer', 'ai engineer', 'nlp', 'deep learning',
        'analytics', 'statistician', 'bi analyst', 'business intelligence',
        'data science', 'data mining', 'research analyst', 'quantitative'
    ]
    engineering_keywords = [
        'software engineer', 'developer', 'programmer', 'devops', 'cloud engineer',
        'backend', 'frontend', 'full stack', 'android', 'ios', 'java', 'python developer',
        'php', 'flutter', 'react', 'nodejs', 'sqa', 'qa engineer', 'test engineer',
        'system engineer', 'network engineer', 'infrastructure', 'site reliability',
        'embedded', 'firmware', 'hardware', 'architect', 'solution architect',
        'security engineer', 'cyber', 'database administrator', 'dba'
    ]
    management_keywords = [
        'manager', 'management', 'director', 'vp ', 'vice president', 'head of',
        'chief', 'cto', 'ceo', 'cfo', 'lead', 'supervisor', 'executive', 'president',
        'operations', 'product manager', 'project manager', 'program manager',
        'hr manager', 'talent', 'recruiter', 'account manager', 'sales manager',
        'marketing manager', 'business analyst', 'consultant', 'strategy', 'admin'
    ]

    for kw in data_keywords:
        if kw in title:
            return 'Data'
    for kw in engineering_keywords:
        if kw in title:
            return 'Engineering'
    for kw in management_keywords:
        if kw in title:
            return 'Management'
    return None  # drop ambiguous rows

df['category'] = df['title'].apply(assign_category)
df = df[df['category'].notna()].copy()
print(f"\n📂 Category distribution (after mapping):")
print(df['category'].value_counts().to_string())
print(f"\n   Total usable rows: {len(df):,}")

# ─────────────────────────────────────────────────────────────
# STEP 3 — Parse Salary → numeric (e.g. "12.8L" → 1280000)
# ─────────────────────────────────────────────────────────────
def parse_salary(s):
    try:
        s = str(s).strip().upper()
        if 'L' in s:
            return float(s.replace('L', '')) * 100000
        elif 'K' in s:
            return float(s.replace('K', '')) * 1000
        else:
            return float(s)
    except:
        return np.nan

df['salary_num'] = df['salary'].apply(parse_salary)
df['salary_num'].fillna(df['salary_num'].median(), inplace=True)

# ─────────────────────────────────────────────────────────────
# STEP 4 — Skill-Score Feature Engineering
# ─────────────────────────────────────────────────────────────
data_skills = [
    'python', 'sql', 'machine learning', 'deep learning', 'tableau',
    'power bi', 'pandas', 'numpy', 'statistics', 'data visualization',
    'spark', 'hadoop', 'tensorflow', 'keras', 'r ', 'matplotlib',
    'data analysis', 'scikit', 'nlp', 'data mining', 'azure ml'
]
engineering_skills = [
    'java', 'javascript', 'react', 'nodejs', 'angular', 'html', 'css',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'devops', 'git',
    'spring', 'microservices', 'rest api', 'mongodb', 'mysql', 'postgresql',
    'c++', 'flutter', 'android', 'ios', 'algorithms', 'data structures',
    'linux', 'networking', 'cloud', 'ci/cd', 'terraform', 'selenium'
]
management_skills = [
    'communication', 'leadership', 'project management', 'excel',
    'teamwork', 'problem solving', 'negotiation', 'sales', 'recruitment',
    'hr management', 'stakeholder', 'agile', 'scrum', 'budgeting',
    'strategic planning', 'business development', 'client management',
    'presentations', 'team management', 'process improvement', 'crm'
]

def skill_score(skills_text, keyword_list):
    if pd.isna(skills_text):
        return 0
    text = skills_text.lower()
    return sum(1 for kw in keyword_list if kw in text)

df['data_score']        = df['skills'].apply(lambda x: skill_score(x, data_skills))
df['engineering_score'] = df['skills'].apply(lambda x: skill_score(x, engineering_skills))
df['management_score']  = df['skills'].apply(lambda x: skill_score(x, management_skills))
df['total_skills']      = df['skills'].apply(
    lambda x: len(str(x).split(',')) if pd.notna(x) else 0
)

print(f"\n🔧 Feature engineering complete.")
print(df[['data_score','engineering_score','management_score','salary_num']].describe().round(2))

# ─────────────────────────────────────────────────────────────
# STEP 5 — Prepare X, y
# ─────────────────────────────────────────────────────────────
feature_cols = [
    'data_score', 'engineering_score', 'management_score',
    'total_skills', 'salary_num'
]

X = df[feature_cols]
le = LabelEncoder()
y = le.fit_transform(df['category'])   # Data=0, Engineering=1, Management=2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📊 Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

# ─────────────────────────────────────────────────────────────
# STEP 6 — Train Decision Tree
# ─────────────────────────────────────────────────────────────
clf = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=50,
    min_samples_leaf=20,
    class_weight='balanced',
    random_state=42
)
clf.fit(X_train, y_train)
print("\n✅ Model trained!")

# ─────────────────────────────────────────────────────────────
# STEP 7 — Evaluate
# ─────────────────────────────────────────────────────────────
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print(f"  MODEL ACCURACY: {acc*100:.2f}%")
print("=" * 60)
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ─────────────────────────────────────────────────────────────
# STEP 8 — Predict New Job (with confidence score)
# ─────────────────────────────────────────────────────────────
def predict_job_category(skills_text: str, salary_lakh: float):
    """
    Input : skills keywords (comma-separated string), salary in Lakhs
    Output: Predicted job category + confidence score
    """
    salary_num = salary_lakh * 100000

    features = {
        'data_score':        skill_score(skills_text, data_skills),
        'engineering_score': skill_score(skills_text, engineering_skills),
        'management_score':  skill_score(skills_text, management_skills),
        'total_skills':      len(skills_text.split(',')),
        'salary_num':        salary_num,
    }
    input_df = pd.DataFrame([features])
    pred_enc  = clf.predict(input_df)[0]
    probs     = clf.predict_proba(input_df)[0]

    label      = le.inverse_transform([pred_enc])[0]
    confidence = round(probs[pred_enc] * 100, 1)

    print(f"\n{'─'*50}")
    print(f"  🔍 Skills    : {skills_text}")
    print(f"  💰 Salary    : {salary_lakh}L (₹{salary_num:,.0f})")
    print(f"  🏷️  Predicted : {label}  ({confidence}% confidence)")
    all_probs = {le.classes_[i]: f"{p*100:.1f}%" for i, p in enumerate(probs)}
    print(f"  📊 All Probs : {all_probs}")
    return label, confidence

print("\n\n🧪 EXAMPLE PREDICTIONS:")
predict_job_category("python, sql, machine learning, pandas, numpy, tensorflow", 12.5)
predict_job_category("java, spring, kubernetes, docker, aws, algorithms, data structures", 15.0)
predict_job_category("leadership, excel, communication, project management, agile, scrum", 18.0)
predict_job_category("python, aws, spark, hadoop, data visualization, tableau", 14.0)

# ─────────────────────────────────────────────────────────────
# STEP 9 — Feature Importance
# ─────────────────────────────────────────────────────────────
importances = clf.feature_importances_
feat_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=False)
print("\n\n🎯 Feature Importances:")
print(feat_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# STEP 10 — Visualisations
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle("Decision Tree — Job Category Classifier\n(Dataset: combined_cleaned_jobs.csv | 30,347 rows)",
             fontsize=14, fontweight='bold', y=1.01)

# — Confusion Matrix —
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0])
axes[0].set_title(f'Confusion Matrix\nAccuracy: {acc*100:.1f}%', fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# — Feature Importance —
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
axes[1].barh(feat_df['Feature'], feat_df['Importance'], color=colors)
axes[1].set_title('Feature Importances', fontweight='bold')
axes[1].set_xlabel('Importance Score')
axes[1].invert_yaxis()

# — Category Distribution —
cat_counts = df['category'].value_counts()
axes[2].pie(cat_counts.values, labels=cat_counts.index, autopct='%1.1f%%',
            colors=['#2196F3', '#4CAF50', '#FF9800'], startangle=90,
            textprops={'fontsize': 11})
axes[2].set_title('Category Distribution\nin Dataset', fontweight='bold')

plt.tight_layout()
plt.savefig('decision_tree_results.png', dpi=150, bbox_inches='tight')
print("\n\n📊 Saved: decision_tree_results.png")

# — Decision Tree Diagram —
plt.figure(figsize=(24, 10))
plot_tree(clf, feature_names=feature_cols, class_names=le.classes_,
          filled=True, rounded=True, fontsize=9, max_depth=3)
plt.title("Decision Tree Structure (max_depth=3 shown)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('decision_tree_structure.png', dpi=150, bbox_inches='tight')
print("📊 Saved: decision_tree_structure.png")
print("\n✅ All done!")

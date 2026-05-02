"""
================================================================
 Model 2 : K-Means Clustering — Job Role Grouping
 Project : Job Market & Skill Demand Analysis
 Author  : Data Science Project Team
================================================================

 WHAT THIS FILE DOES (read before running)
 ------------------------------------------
 1. Loads BOTH datasets and merges them into one pipeline
      - combined_cleaned_jobs.csv    (30,347 India jobs)
      - final_jobs_dataset_csv_global.csv  (47,490 global jobs)

 2. Three inputs as per project specification:
      a) Skills         → encoded using MultiLabelBinarizer
      b) Salary         → parsed from '7.8L' format  (India)
                          derived from skill_count    (Global)
      c) Experience     → extracted from title        (India)
                          extracted from description  (Global, regex)

 3. Finds best K using Elbow Method + Silhouette Score
 4. Trains K-Means and assigns cluster labels
 5. Saves labelled CSV + 4 charts

 HOW TO RUN
 ----------
      pip install pandas scikit-learn matplotlib seaborn
      python kmeans_job_clustering.py

 OUTPUT FILES
 ------------
      clustered_jobs_output.csv
      plot_01_elbow_silhouette.png
      plot_02_pca_scatter.png
      plot_03_experience_salary.png
      plot_04_skill_heatmap.png
================================================================
"""

# ── Standard library ────────────────────────────────────────
import re
import warnings
warnings.filterwarnings('ignore')
from collections import Counter

# ── Third-party libraries ────────────────────────────────────
import pandas  as pd
import numpy   as np
import matplotlib.pyplot  as plt
import seaborn as sns

from sklearn.preprocessing  import StandardScaler, MultiLabelBinarizer
from sklearn.cluster        import KMeans
from sklearn.decomposition  import PCA
from sklearn.metrics        import silhouette_score


# ================================================================
#  CONFIGURATION  — change these values if needed
# ================================================================

INDIA_FILE   = "combined_cleaned_jobs.csv"
GLOBAL_FILE  = "final_jobs_dataset_csv_global.csv"
OUTPUT_CSV   = "clustered_jobs_output.csv"

TOP_N_SKILLS = 20     # how many most-frequent skills to encode
K_FINAL      = 4      # number of clusters  (confirmed by elbow)
RANDOM_STATE = 42     # for reproducibility


# ================================================================
#  STEP 1 — LOAD DATA
#  Function: load_datasets()
#  Reads both CSV files and adds a 'source' column so we know
#  which dataset each row came from after merging.
# ================================================================

def load_datasets(india_path: str, global_path: str) -> tuple:
    """
    Load both datasets from disk.

    Parameters
    ----------
    india_path  : path to combined_cleaned_jobs.csv
    global_path : path to final_jobs_dataset_csv_global.csv

    Returns
    -------
    df_india, df_global  — two DataFrames
    """
    print("=" * 65)
    print("STEP 1 — Loading datasets")
    print("=" * 65)

    df_india  = pd.read_csv(india_path)
    df_global = pd.read_csv(global_path)

    print(f"  India  dataset : {df_india.shape[0]:>6,} rows  | cols: {df_india.columns.tolist()}")
    print(f"  Global dataset : {df_global.shape[0]:>6,} rows  | cols: {df_global.columns.tolist()}")
    print(f"  Total          : {len(df_india) + len(df_global):,} job records\n")

    return df_india, df_global


# ================================================================
#  STEP 2 — CLEAN & STANDARDISE EACH DATASET
#  Functions: clean_india_data(), clean_global_data()
#
#  India dataset  columns: title, company, location, skills, salary
#  Global dataset columns: job_title, company, location,
#                          description, skills, skill_count
#
#  Goal: both datasets should end up with the same 4 columns:
#        job_title | skills | salary_num | experience_required
# ================================================================

def parse_salary_india(value) -> float:
    """
    Convert India salary strings to float (Lakhs).
    Handles:  '7.8L'  →  7.8
              '12.8L' →  12.8
              '10.31' →  10.31   (already numeric string)

    Parameters
    ----------
    value : raw salary value from CSV

    Returns
    -------
    float salary in Lakhs, or NaN if unparseable
    """
    try:
        cleaned = str(value).strip().upper().replace('L', '').replace(',', '')
        return float(cleaned)
    except (ValueError, TypeError):
        return np.nan


def extract_experience_from_title(title: str) -> int:
    """
    Derive experience level (1/2/3) from job title keywords.
    Used for India dataset which has no experience column.

    Logic:
        Senior keywords  →  3  (Senior, 5-10 yrs)
        Entry  keywords  →  1  (Junior/Intern, 0-2 yrs)
        Everything else  →  2  (Mid-level, 2-5 yrs)

    Parameters
    ----------
    title : job title string

    Returns
    -------
    int : 1 = Entry, 2 = Mid, 3 = Senior
    """
    t = str(title).lower()

    senior_keywords = ['senior', 'sr.', 'lead', 'principal', 'staff',
                       'head', 'director', 'vp', 'manager', 'chief']
    entry_keywords  = ['junior', 'jr.', 'entry', 'intern',
                       'trainee', 'fresher', 'graduate', 'associate']

    if any(k in t for k in senior_keywords): return 3
    if any(k in t for k in entry_keywords):  return 1
    return 2


def extract_experience_from_description(text) -> float:
    """
    Extract years of experience from job description text using regex.
    Used for Global dataset which has no experience column but has
    rich description text like "4+ years of experience required".

    Patterns handled:
        "4+ years"          →  4.0
        "4 + years"         →  4.0
        "3-5 years"         →  4.0  (average of range)
        "3 years of exp"    →  3.0
        "minimum 3"         →  3.0
        "at least 6 years"  →  6.0
        "5 yrs"             →  5.0

    Parameters
    ----------
    text : job description string

    Returns
    -------
    float years, or NaN if not found in text
    """
    if pd.isna(text):
        return np.nan

    t = str(text)

    # Pattern A:  "4+ years"  or  "4 + years"
    m = re.search(r'(\d+)\s*\+\s*years?', t, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # Pattern B:  "3-5 years"  or  "3 to 5 years"
    m = re.search(r'(\d+)\s*[-to]+\s*(\d+)\s*years?', t, re.IGNORECASE)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0

    # Pattern C:  "3 years of experience"
    m = re.search(r'(\d+)\s*years?\s*of\s*(?:related\s*)?exp', t, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # Pattern D:  "minimum 3"  /  "at least 5"
    m = re.search(r'(?:minimum|at least)\s*(\d+)', t, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # Pattern E:  "5 yrs"
    m = re.search(r'(\d+)\s*yrs?\.?\b', t, re.IGNORECASE)
    if m:
        return float(m.group(1))

    return np.nan


def clean_india_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardise the India dataset.

    Steps:
        1. Parse salary strings ('7.8L' → 7.8)
        2. Drop rows with unparseable salary
        3. Extract experience level from job title
        4. Rename columns to unified schema

    Parameters
    ----------
    df : raw India DataFrame

    Returns
    -------
    DataFrame with columns:
        job_title | skills | salary_num | experience_required | source
    """
    print("  [India] Cleaning salary column ...")
    df = df.copy()

    df['salary_num'] = df['salary'].apply(parse_salary_india)

    # Drop rows where salary cannot be determined
    before = len(df)
    df = df.dropna(subset=['salary_num']).reset_index(drop=True)
    print(f"  [India] Salary parsing: {before:,} → {len(df):,} rows kept")
    print(f"  [India] Salary range  : ₹{df['salary_num'].min():.1f}L  –  ₹{df['salary_num'].max():.1f}L")

    # Experience from title (1=Entry, 2=Mid, 3=Senior)
    df['experience_required'] = df['title'].apply(extract_experience_from_title)
    exp_counts = df['experience_required'].value_counts().sort_index()
    print(f"  [India] Experience levels → Entry:{exp_counts.get(1,0):,}  "
          f"Mid:{exp_counts.get(2,0):,}  Senior:{exp_counts.get(3,0):,}")

    df['source'] = 'india'

    return df[['title', 'skills', 'salary_num', 'experience_required', 'source']].rename(
        columns={'title': 'job_title'}
    )


def clean_global_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardise the Global dataset.

    Steps:
        1. Extract experience from description text using regex
        2. Fill missing experience with dataset median
        3. Create salary_num from skill_count (proxy: more skills = higher pay)
        4. Rename columns to unified schema

    Parameters
    ----------
    df : raw Global DataFrame

    Returns
    -------
    DataFrame with columns:
        job_title | skills | salary_num | experience_required | source
    """
    print("\n  [Global] Extracting experience from description text ...")
    df = df.copy()

    df['experience_required'] = df['description'].apply(extract_experience_from_description)

    # Cap outlier values (e.g. "20+ years" or false regex matches)
    df['experience_required'] = df['experience_required'].clip(upper=20)

    extracted = df['experience_required'].notna().sum()
    print(f"  [Global] Experience extracted : {extracted:,} / {len(df):,} rows "
          f"({extracted/len(df)*100:.1f}%)")

    # Fill missing experience with median (safe imputation)
    median_exp = df['experience_required'].median()
    df['experience_required'] = df['experience_required'].fillna(median_exp)
    print(f"  [Global] Missing exp filled with median = {median_exp:.1f} years")
    print(f"  [Global] Experience range : {df['experience_required'].min():.0f}  –  "
          f"{df['experience_required'].max():.0f} years")

    # Salary proxy: scale skill_count (1-11) into a salary-like range
    # skill_count is the best available salary signal in this dataset
    # (jobs requiring more skills typically command higher salaries)
    df['salary_num'] = df['skill_count'].astype(float)
    print(f"  [Global] salary_num (proxy) range: {df['salary_num'].min():.0f} – "
          f"{df['salary_num'].max():.0f}  (from skill_count)")

    df['source'] = 'global'

    return df[['job_title', 'skills', 'salary_num', 'experience_required', 'source']]


# ================================================================
#  STEP 3 — MERGE BOTH DATASETS INTO ONE PIPELINE
#  Function: merge_datasets()
# ================================================================

def merge_datasets(df_india_clean: pd.DataFrame,
                   df_global_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Vertically stack both cleaned datasets into a single DataFrame.

    Both datasets now share the same 5 columns:
        job_title | skills | salary_num | experience_required | source

    Parameters
    ----------
    df_india_clean  : cleaned India data
    df_global_clean : cleaned Global data

    Returns
    -------
    Merged DataFrame
    """
    print("\n" + "=" * 65)
    print("STEP 3 — Merging both datasets")
    print("=" * 65)

    df = pd.concat([df_india_clean, df_global_clean],
                   ignore_index=True)

    print(f"  Combined shape : {df.shape[0]:,} rows  x  {df.shape[1]} columns")
    print(f"  India rows     : {(df['source']=='india').sum():,}")
    print(f"  Global rows    : {(df['source']=='global').sum():,}")
    print(f"  Null skills    : {df['skills'].isna().sum()}")

    # Drop rows with no skills (cannot encode them)
    df = df.dropna(subset=['skills']).reset_index(drop=True)
    print(f"  After null drop: {df.shape[0]:,} rows")

    return df


# ================================================================
#  STEP 4 — FEATURE ENGINEERING
#  Function: preprocess_data()
#
#  Converts text-based skills into a numeric matrix that K-Means
#  can actually work with.
#
#  MultiLabelBinarizer example:
#    Input row  : "Python, SQL, Excel"
#    Output row : [0, 1, 0, 1, 1, 0 ...]   (one column per skill)
#
#  Then StandardScaler ensures salary and experience don't dominate
#  the distance calculations just because they have larger numbers.
# ================================================================

def preprocess_data(df: pd.DataFrame,
                    top_n: int = TOP_N_SKILLS) -> tuple:
    """
    Full feature engineering pipeline:
        1. Parse skill strings into lists
        2. Select top-N most frequent skills
        3. One-hot encode skills with MultiLabelBinarizer
        4. Concatenate skills + salary + experience
        5. Scale all features with StandardScaler

    Parameters
    ----------
    df    : merged clean DataFrame
    top_n : number of most-frequent skills to keep

    Returns
    -------
    X_scaled   : scaled numpy array ready for K-Means
    features_df: un-scaled DataFrame (for interpretation)
    top_skills : list of skill names used
    mlb        : fitted MultiLabelBinarizer (to transform new data)
    scaler     : fitted StandardScaler
    """
    print("\n" + "=" * 65)
    print("STEP 4 — Feature Engineering")
    print("=" * 65)

    # ── 4a. Parse skill strings into lists ──────────────────
    print("  4a. Parsing skill strings into lists ...")
    df['skills_list'] = (
        df['skills']
        .fillna('')
        .str.lower()
        .str.split(r',\s*')
    )

    # ── 4b. Select top-N skills ──────────────────────────────
    print(f"  4b. Selecting top {top_n} most frequent skills ...")
    all_skills = [
        s.strip()
        for lst in df['skills_list']
        for s in lst
        if s.strip()
    ]
    skill_freq = Counter(all_skills)
    top_skills = [s for s, _ in skill_freq.most_common(top_n)]

    print(f"      Total unique skills in dataset : {len(skill_freq)}")
    print(f"      Top {top_n} skills selected    :")
    for i, s in enumerate(top_skills, 1):
        print(f"          {i:2d}. {s:<30s} ({skill_freq[s]:,} occurrences)")

    # Filter each row to only include top-N skills
    df['skills_filtered'] = df['skills_list'].apply(
        lambda lst: [s.strip() for s in lst if s.strip() in top_skills]
                    or ['other']
    )

    # ── 4c. One-hot encode skills ────────────────────────────
    print(f"\n  4c. One-hot encoding skills with MultiLabelBinarizer ...")
    mlb        = MultiLabelBinarizer(classes=top_skills)
    skills_mat = mlb.fit_transform(df['skills_filtered'])
    skills_df  = pd.DataFrame(skills_mat, columns=mlb.classes_)
    print(f"      Skills matrix shape : {skills_mat.shape}")

    # Merge skill columns back into df for later heatmap
    for col in skills_df.columns:
        df[col] = skills_df[col].values

    # ── 4d. Build final feature matrix ──────────────────────
    print(f"  4d. Building feature matrix ...")
    features_df = pd.concat([
        skills_df.reset_index(drop=True),
        df[['salary_num', 'experience_required']].reset_index(drop=True)
    ], axis=1)
    print(f"      Final shape : {features_df.shape}")
    print(f"      Columns     : {top_n} skill cols + salary_num + experience_required")

    # ── 4e. Scale features ───────────────────────────────────
    # WHY? salary_num can be 8.5 while skill binary is 0/1
    # Without scaling, salary would dominate distance calculations
    print(f"  4e. Scaling with StandardScaler ...")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)
    print(f"      X_scaled shape : {X_scaled.shape}  ✓")

    return X_scaled, features_df, top_skills, mlb, scaler, df


# ================================================================
#  STEP 5 — FIND OPTIMAL K
#  Function: find_optimal_k()
#
#  Two methods used together:
#   a) Elbow Method  : plot inertia (WCSS) vs K
#      → Look for the "elbow" / "knee" in the curve
#      → After the elbow, adding more clusters gives less benefit
#
#   b) Silhouette Score : measures how well-separated clusters are
#      → Range: -1 to +1  (higher is better)
#      → Score > 0.25 is acceptable for job market data
# ================================================================

def find_optimal_k(X_scaled: np.ndarray,
                   k_range: range = range(2, 11)) -> None:
    """
    Run K-Means for K=2 to K=10 and plot Elbow + Silhouette charts.
    Prints a table of results to help choose the best K.

    Parameters
    ----------
    X_scaled : scaled feature matrix
    k_range  : range of K values to test
    """
    print("\n" + "=" * 65)
    print("STEP 5 — Finding optimal K  (Elbow + Silhouette)")
    print("=" * 65)
    print(f"  {'K':>3}  {'Inertia':>15}  {'Silhouette':>12}  {'Verdict':>20}")
    print(f"  {'-'*3}  {'-'*15}  {'-'*12}  {'-'*20}")

    inertias   = []
    sil_scores = []

    for k in k_range:
        km  = KMeans(n_clusters=k, init='k-means++',
                     n_init=10, random_state=RANDOM_STATE, max_iter=300)
        lbl = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)

        sil = silhouette_score(X_scaled, lbl,
                               sample_size=5000, random_state=RANDOM_STATE)
        sil_scores.append(sil)

        verdict = "← good" if sil > 0.25 else ""
        print(f"  {k:>3}  {km.inertia_:>15,.0f}  {sil:>12.4f}  {verdict:>20}")

    # ── Plot ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Step 5: Finding Optimal K — Elbow & Silhouette",
                 fontsize=14, fontweight='bold')

    # Elbow curve
    axes[0].plot(list(k_range), inertias, marker='o', color='steelblue',
                 linewidth=2.5, markersize=9, label='Inertia')
    axes[0].axvline(x=K_FINAL, color='red', linestyle='--',
                    alpha=0.8, label=f'K={K_FINAL} chosen')
    axes[0].fill_between(list(k_range), inertias,
                         alpha=0.08, color='steelblue')
    axes[0].set_xlabel("Number of Clusters  (K)", fontsize=12)
    axes[0].set_ylabel("Inertia — Within-Cluster Sum of Squares", fontsize=11)
    axes[0].set_title("Elbow Curve\n(look for the 'knee' point)", fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    # Silhouette scores
    axes[1].plot(list(k_range), sil_scores, marker='s', color='darkorange',
                 linewidth=2.5, markersize=9, label='Silhouette')
    axes[1].axvline(x=K_FINAL, color='red', linestyle='--',
                    alpha=0.8, label=f'K={K_FINAL} chosen')
    axes[1].axhline(y=0.25, color='green', linestyle=':', alpha=0.7,
                    label='0.25 threshold')
    axes[1].set_xlabel("Number of Clusters  (K)", fontsize=12)
    axes[1].set_ylabel("Silhouette Score  (higher = better)", fontsize=11)
    axes[1].set_title("Silhouette Score\n(higher = more separated clusters)", fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("plot_01_elbow_silhouette.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\n  [Saved] plot_01_elbow_silhouette.png")
    print(f"\n  ✓ Chosen K = {K_FINAL}  (best balance of inertia drop + silhouette score)")


# ================================================================
#  STEP 6 — TRAIN FINAL MODEL
#  Function: train_model()
# ================================================================

def train_model(X_scaled: np.ndarray,
                k: int = K_FINAL) -> tuple:
    """
    Train the final K-Means model with the chosen K.

    Parameters
    ----------
    X_scaled : scaled feature matrix
    k        : number of clusters

    Returns
    -------
    kmeans  : trained KMeans object
    labels  : cluster assignment for every row (array of ints)
    """
    print("\n" + "=" * 65)
    print(f"STEP 6 — Training final K-Means model  (K={k})")
    print("=" * 65)

    kmeans = KMeans(
        n_clusters   = k,
        init         = 'k-means++',   # smarter initialisation (not random)
        n_init       = 10,             # run 10 times, keep best result
        max_iter     = 300,            # max iterations per run
        random_state = RANDOM_STATE    # for reproducibility
    )

    labels = kmeans.fit_predict(X_scaled)

    # ── Evaluation metrics ───────────────────────────────────
    sil_score = silhouette_score(X_scaled, labels,
                                 sample_size=5000,
                                 random_state=RANDOM_STATE)

    print(f"  Silhouette Score : {sil_score:.4f}")
    print(f"    → Interpretation: {'Good separation' if sil_score > 0.25 else 'Overlapping clusters (normal for job data)'}")
    print(f"  Inertia (WCSS)   : {kmeans.inertia_:,.0f}")
    print(f"\n  Cluster sizes:")
    unique, counts = np.unique(labels, return_counts=True)
    for c, n in zip(unique, counts):
        print(f"    Cluster {c} : {n:,} jobs  ({n/len(labels)*100:.1f}%)")

    return kmeans, labels


# ================================================================
#  STEP 7 — INTERPRET & LABEL CLUSTERS
#  Function: assign_cluster_labels()
#
#  K-Means only gives numbers: 0, 1, 2, 3
#  We read the cluster properties and give them human names.
#
#  Logic:
#   → Highest experience + skill_count  =  Senior Leadership
#   → Lowest  experience                =  Entry-Level Analyst
#   → Highest skill_count among rest    =  High-Pay Tech
#   → Remaining                         =  Mid-Level Specialist
# ================================================================

def assign_cluster_labels(df: pd.DataFrame,
                          labels: np.ndarray,
                          top_skills: list) -> pd.DataFrame:
    """
    Add cluster number and human-readable label to the DataFrame.
    Also prints a cluster interpretation table.

    Parameters
    ----------
    df         : merged DataFrame (after preprocess_data)
    labels     : cluster assignments from train_model
    top_skills : list of encoded skill names

    Returns
    -------
    df with new columns: cluster, cluster_label
    """
    print("\n" + "=" * 65)
    print("STEP 7 — Interpreting & Labelling Clusters")
    print("=" * 65)

    df = df.copy()
    df['cluster'] = labels

    # ── Print raw cluster stats ──────────────────────────────
    print("\n  Raw cluster statistics:")
    stats = df.groupby('cluster').agg(
        count        = ('salary_num',         'size'),
        avg_salary   = ('salary_num',         'mean'),
        avg_exp      = ('experience_required', 'mean'),
        med_salary   = ('salary_num',         'median'),
    ).round(2)
    print(stats.to_string())

    print("\n  Top 5 skills per cluster:")
    for c in sorted(df['cluster'].unique()):
        cdf  = df[df['cluster'] == c]
        top5 = (cdf['skills_list'].explode()
                                  .str.strip()
                                  .value_counts()
                                  .head(5)
                                  .index.tolist())
        print(f"    C{c} | exp={cdf['experience_required'].mean():.1f}y "
              f"| salary={cdf['salary_num'].mean():.1f} "
              f"| n={len(cdf):,} | {top5}")

    # ── Auto-assign labels by ranking ───────────────────────
    cluster_meta = df.groupby('cluster').agg(
        avg_exp    = ('experience_required', 'mean'),
        avg_salary = ('salary_num',          'mean'),
    ).reset_index()

    # Sort by experience ascending
    sorted_exp = cluster_meta.sort_values('avg_exp').reset_index(drop=True)

    label_map = {}
    for rank, row in sorted_exp.iterrows():
        c = int(row['cluster'])
        if rank == 0:
            label_map[c] = 'Entry-Level Analyst'
        elif rank == len(sorted_exp) - 1:
            label_map[c] = 'Senior Leadership'
        elif row['avg_salary'] == cluster_meta['avg_salary'].max():
            label_map[c] = 'High-Pay Tech'
        else:
            label_map[c] = 'Mid-Level Specialist'

    df['cluster_label'] = df['cluster'].map(label_map)

    # ── Print final interpretation ───────────────────────────
    print("\n  ── CLUSTER INTERPRETATION ─────────────────────────────")
    final_stats = df.groupby('cluster_label').agg(
        Count     = ('salary_num',          'size'),
        Avg_Exp   = ('experience_required',  lambda x: f"{x.mean():.1f} yrs"),
        Avg_Sal   = ('salary_num',           lambda x: f"{x.mean():.1f}"),
    )
    print(final_stats.to_string())
    print()
    print("  Label meanings:")
    print("    Entry-Level Analyst  → Low experience, basic skills (SQL/Excel)")
    print("    Mid-Level Specialist → Mid experience, broad tech stack")
    print("    High-Pay Tech        → Highest skill demand (ML/Python/Cloud)")
    print("    Senior Leadership    → Highest experience, architecture/management")

    print(f"\n  Assigned label map: {label_map}")

    return df, label_map


# ================================================================
#  STEP 8 — VISUALISE RESULTS
#  Function: visualise_results()
#  Generates 3 charts: PCA scatter, experience/salary bars, heatmap
# ================================================================

def visualise_results(df: pd.DataFrame,
                      X_scaled: np.ndarray,
                      kmeans: KMeans,
                      top_skills: list,
                      label_map: dict) -> None:
    """
    Generate and save 3 visualisation charts.

    Chart 1 — PCA 2D Scatter    : cluster separation view
    Chart 2 — Experience/Salary : bar charts per cluster
    Chart 3 — Skill Heatmap     : which skills dominate each cluster

    Parameters
    ----------
    df         : DataFrame with cluster_label column
    X_scaled   : scaled feature matrix
    kmeans     : trained KMeans object
    top_skills : list of encoded skill names
    label_map  : dict mapping cluster int → label string
    """
    print("\n" + "=" * 65)
    print("STEP 8 — Visualising results")
    print("=" * 65)

    # ── Colour palette ───────────────────────────────────────
    COLOR_MAP = {
        'Entry-Level Analyst': '#378ADD',
        'Mid-Level Specialist':'#BA7517',
        'High-Pay Tech':        '#1D9E75',
        'Senior Leadership':    '#7F77DD'
    }
    # Fill in any labels that exist in data
    all_labels = df['cluster_label'].unique()
    fallback_colors = ['#E24B4A', '#888780', '#D85A30', '#1D9E75']
    for i, lbl in enumerate(all_labels):
        if lbl not in COLOR_MAP:
            COLOR_MAP[lbl] = fallback_colors[i % len(fallback_colors)]

    cluster_order = [
        'Entry-Level Analyst', 'Mid-Level Specialist',
        'High-Pay Tech', 'Senior Leadership'
    ]
    cluster_order = [c for c in cluster_order if c in df['cluster_label'].unique()]

    # ── CHART 1 : PCA 2D Scatter ─────────────────────────────
    print("  Generating Chart 1 — PCA 2D Scatter ...")
    pca   = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    var   = pca.explained_variance_ratio_

    df['pca1'] = X_pca[:, 0]
    df['pca2'] = X_pca[:, 1]

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_facecolor('#f8f8f8')

    sample = df.sample(min(6000, len(df)), random_state=RANDOM_STATE)
    for label, color in COLOR_MAP.items():
        sub = sample[sample['cluster_label'] == label]
        if len(sub) > 0:
            ax.scatter(sub['pca1'], sub['pca2'],
                       c=color, label=label,
                       alpha=0.55, s=18, edgecolors='none')

    # Plot centroid stars
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    for i, (cx, cy) in enumerate(centroids_pca):
        ax.scatter(cx, cy, c='black', s=280, marker='*', zorder=6)
        ax.annotate(label_map.get(i, f'C{i}'), (cx, cy),
                    textcoords='offset points', xytext=(10, 8),
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white',
                              alpha=0.85, ec='gray', linewidth=0.8))

    ax.set_xlabel(f"PC1  ({var[0]*100:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2  ({var[1]*100:.1f}% variance)", fontsize=12)
    ax.set_title(
        "K-Means Job Role Clusters — PCA 2D Projection\n"
        f"Job Market & Skill Demand Analysis  |  K={K_FINAL}  |  "
        f"n={len(df):,} jobs",
        fontsize=13, fontweight='bold', pad=14)
    ax.legend(fontsize=11, framealpha=0.95, loc='upper right')
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig("plot_02_pca_scatter.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("  [Saved] plot_02_pca_scatter.png")

    # ── CHART 2 : Experience & Salary per Cluster ────────────
    print("  Generating Chart 2 — Experience & Salary bars ...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Experience Required & Salary per Cluster",
                 fontsize=14, fontweight='bold')

    # Box plot — experience required
    data_plot = df.copy()
    sns.boxplot(data=data_plot, x='cluster_label', y='experience_required',
                order=cluster_order, palette=COLOR_MAP,
                ax=axes[0], width=0.5)
    axes[0].set_title("Experience Required by Cluster", fontsize=12)
    axes[0].set_xlabel("Cluster", fontsize=11)
    axes[0].set_ylabel("Experience Required (Years / Level)", fontsize=11)
    axes[0].tick_params(axis='x', rotation=20)
    axes[0].grid(axis='y', alpha=0.3)

    # Bar chart — average salary
    avg_sal  = (df.groupby('cluster_label')['salary_num']
                  .mean()
                  .reindex(cluster_order))
    bar_clrs = [COLOR_MAP[c] for c in cluster_order]
    bars     = axes[1].bar(cluster_order, avg_sal.values,
                           color=bar_clrs, width=0.55, edgecolor='white')
    for bar, val in zip(bars, avg_sal.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.04,
                     f"{val:.1f}", ha='center', va='bottom',
                     fontsize=11, fontweight='bold')
    axes[1].set_title("Average Salary / Skill-Proxy per Cluster", fontsize=12)
    axes[1].set_xlabel("Cluster", fontsize=11)
    axes[1].set_ylabel("Avg Salary  (₹ Lakhs  or  Skill Proxy)", fontsize=11)
    axes[1].tick_params(axis='x', rotation=20)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("plot_03_experience_salary.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("  [Saved] plot_03_experience_salary.png")

    # ── CHART 3 : Skill Heatmap ───────────────────────────────
    print("  Generating Chart 3 — Skill presence heatmap ...")
    TOP_VIZ   = 15
    viz_skills = [s for s in top_skills[:TOP_VIZ] if s in df.columns]

    heat_df = (df.groupby('cluster_label')[viz_skills]
                 .mean()
                 .reindex(cluster_order)
                 * 100)   # convert to percentage

    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(heat_df, annot=True, fmt='.0f', cmap='YlGnBu',
                linewidths=0.4,
                cbar_kws={'label': '% of jobs in cluster requiring this skill'},
                ax=ax)
    ax.set_title(
        f"Top {TOP_VIZ} Skill Presence (%) per Job Cluster\n"
        f"Job Market & Skill Demand Analysis",
        fontsize=13, fontweight='bold')
    ax.set_xlabel("Skill", fontsize=11)
    ax.set_ylabel("Cluster", fontsize=11)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig("plot_04_skill_heatmap.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("  [Saved] plot_04_skill_heatmap.png")


# ================================================================
#  STEP 9 — SAVE OUTPUT
#  Function: save_output()
# ================================================================

def save_output(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the final labelled DataFrame to CSV.

    Output columns:
        job_title | skills | salary_num | experience_required |
        source | cluster | cluster_label

    Parameters
    ----------
    df          : DataFrame with cluster labels
    output_path : file path for CSV output
    """
    cols = ['job_title', 'skills', 'salary_num',
            'experience_required', 'source', 'cluster', 'cluster_label']
    # Keep only columns that exist in df
    cols = [c for c in cols if c in df.columns]

    df[cols].to_csv(output_path, index=False)
    print(f"\n  [Saved] {output_path}  ({len(df):,} rows)")


# ================================================================
#  STEP 10 — PRINT FINAL SUMMARY
#  Function: print_summary()
# ================================================================

def print_summary(df: pd.DataFrame,
                  features_df: pd.DataFrame,
                  kmeans: KMeans,
                  X_scaled: np.ndarray,
                  top_skills: list) -> None:
    """
    Print a clean summary table of the entire ML pipeline results.
    Useful for copy-pasting into project reports.

    Parameters
    ----------
    df          : labelled DataFrame
    features_df : un-scaled feature matrix
    kmeans      : trained KMeans
    X_scaled    : scaled feature matrix
    top_skills  : skill names used
    """
    sil = silhouette_score(X_scaled, df['cluster'],
                           sample_size=5000, random_state=RANDOM_STATE)

    print("\n" + "=" * 65)
    print("FINAL SUMMARY — Model 2: K-Means Clustering")
    print("=" * 65)
    print()
    print("  MODEL CONFIGURATION")
    print(f"    Algorithm      : K-Means")
    print(f"    K (clusters)   : {K_FINAL}")
    print(f"    Init strategy  : k-means++ (smarter than random)")
    print(f"    n_init         : 10  (best of 10 runs kept)")
    print(f"    max_iter       : 300")
    print(f"    random_state   : {RANDOM_STATE}")
    print()
    print("  DATASET")
    print(f"    Total records  : {len(df):,}")
    print(f"    India records  : {(df['source']=='india').sum():,}")
    print(f"    Global records : {(df['source']=='global').sum():,}")
    print()
    print("  FEATURES USED")
    print(f"    {len(top_skills)} encoded skill columns  (MultiLabelBinarizer)")
    print(f"    salary_num             (India: parsed ₹L | Global: skill proxy)")
    print(f"    experience_required    (India: from title | Global: from description regex)")
    print(f"    Total feature cols     : {features_df.shape[1]}")
    print()
    print("  PERFORMANCE")
    print(f"    Silhouette Score : {sil:.4f}")
    print(f"    Inertia (WCSS)   : {kmeans.inertia_:,.0f}")
    print()
    print("  CLUSTER RESULTS")

    cluster_order = ['Entry-Level Analyst', 'Mid-Level Specialist',
                     'High-Pay Tech', 'Senior Leadership']
    cluster_order = [c for c in cluster_order if c in df['cluster_label'].unique()]

    result = df.groupby('cluster_label').agg(
        Jobs       = ('salary_num',          'size'),
        Pct        = ('salary_num',          lambda x: f"{len(x)/len(df)*100:.1f}%"),
        Avg_Exp    = ('experience_required',  lambda x: f"{x.mean():.1f} yrs"),
        Avg_Salary = ('salary_num',           lambda x: f"{x.mean():.1f}"),
    ).reindex(cluster_order)
    print(result.to_string())

    print()
    print("  OUTPUT FILES")
    print("    clustered_jobs_output.csv       — full labelled dataset")
    print("    plot_01_elbow_silhouette.png    — optimal K selection")
    print("    plot_02_pca_scatter.png         — 2D cluster visualisation")
    print("    plot_03_experience_salary.png   — experience & salary bars")
    print("    plot_04_skill_heatmap.png       — skill demand heatmap")
    print()
    print("  MODEL 2 COMPLETE ✓")


# ================================================================
#  MAIN — PIPELINE ENTRY POINT
#  This ties all functions together in order.
#  Run this file directly: python kmeans_job_clustering.py
# ================================================================

def main():
    print("\n" + "=" * 65)
    print("  JOB MARKET & SKILL DEMAND ANALYSIS")
    print("  Model 2 : K-Means Clustering — Job Role Grouping")
    print("=" * 65 + "\n")

    # ── Step 1: Load ─────────────────────────────────────────
    df_india_raw, df_global_raw = load_datasets(INDIA_FILE, GLOBAL_FILE)

    # ── Step 2: Clean ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 2 — Cleaning each dataset")
    print("=" * 65)
    df_india_clean  = clean_india_data(df_india_raw)
    df_global_clean = clean_global_data(df_global_raw)

    # ── Step 3: Merge ────────────────────────────────────────
    df_merged = merge_datasets(df_india_clean, df_global_clean)

    # ── Step 4: Preprocess ───────────────────────────────────
    X_scaled, features_df, top_skills, mlb, scaler, df_merged = \
        preprocess_data(df_merged, top_n=TOP_N_SKILLS)

    # ── Step 5: Find best K ──────────────────────────────────
    find_optimal_k(X_scaled, k_range=range(2, 11))

    # ── Step 6: Train ────────────────────────────────────────
    kmeans, labels = train_model(X_scaled, k=K_FINAL)

    # ── Step 7: Label clusters ───────────────────────────────
    df_merged, label_map = assign_cluster_labels(df_merged, labels, top_skills)

    # ── Step 8: Visualise ────────────────────────────────────
    visualise_results(df_merged, X_scaled, kmeans, top_skills, label_map)

    # ── Step 9: Save ─────────────────────────────────────────
    save_output(df_merged, OUTPUT_CSV)

    # ── Step 10: Summary ─────────────────────────────────────
    print_summary(df_merged, features_df, kmeans, X_scaled, top_skills)


# ── Run when executed directly ───────────────────────────────
if __name__ == "__main__":
    main()

import io
from io import BytesIO
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif, f_regression, f_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier, 
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    VotingRegressor, VotingClassifier,
    StackingRegressor, StackingClassifier,
    IsolationForest
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import ElasticNet, HuberRegressor, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, roc_curve, auc, 
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import umap
import shap
import joblib
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import optional libraries
try:
    import xgboost as xgb
    XGB_INSTALLED = True
except ImportError:
    XGB_INSTALLED = False

try:
    import lightgbm as lgb
    LGBM_INSTALLED = True
except ImportError:
    LGBM_INSTALLED = False

try:
    import catboost as cb
    CATBOOST_INSTALLED = True
except ImportError:
    CATBOOST_INSTALLED = False

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Ultimate Data Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .subheader {
        color: gray;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        gap: 8px;
        padding-top: 10px;
        padding-bottom: 10px;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4facfe;
        color: white;
    }
    .feature-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .dark-theme {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .dark-theme .metric-card {
        background-color: #262730;
        color: #FAFAFA;
    }
    .dark-theme .metric-label {
        color: #FAFAFA;
    }
    .dark-theme .feature-card {
        background-color: #262730;
        color: #FAFAFA;
    }
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .pipeline-step {
        background-color: #e9ecef;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 4px solid #4facfe;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Helpers
# ------------------------------
@st.cache_data
def load_file(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame()
    name = uploaded.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded)
        elif name.endswith(".tsv"):
            return pd.read_csv(uploaded, sep="\t")
        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded)
        elif name.endswith(".json"):
            return pd.read_json(uploaded)
        else:
            return pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return pd.DataFrame()

def df_info(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "non_null": df.notna().sum().values,
        "nulls": df.isna().sum().values,
        "unique": df.nunique(dropna=True).values,
        "% null": (df.isna().sum()/len(df)*100).round(2).values,
    })

def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Data") -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def to_pickle_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df.to_pickle(buffer)
    return buffer.getvalue()

def plotly_download_link(fig, filename: str = "chart.html") -> Tuple[str, bytes]:
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    return filename, html.encode("utf-8")

def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Automatically detect column types"""
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].dtype.name == 'category']
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    
    # Check for high cardinality categorical columns
    high_cardinality = [c for c in categorical_cols if df[c].nunique() > 50]
    low_cardinality = [c for c in categorical_cols if df[c].nunique() <= 50]
    
    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols,
        'high_cardinality': high_cardinality,
        'low_cardinality': low_cardinality
    }

def auto_suggest_models(problem_type: str, col_types: Dict[str, List[str]]) -> List[str]:
    """Automatically suggest models based on problem type and data characteristics"""
    if problem_type == 'regression':
        models = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost']
        
        # For smaller datasets, add linear models
        if len(col_types['numeric']) < 20:
            models.extend(['ElasticNet', 'Huber Regression'])
            
    elif problem_type == 'classification':
        models = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost']
        
        # For binary classification with few features
        if len(col_types['numeric']) + len(col_types['low_cardinality']) < 10:
            models.extend(['Logistic Regression', 'SVM'])
            
    elif problem_type == 'clustering':
        models = ['K-Means', 'DBSCAN', 'Agglomerative', 'Gaussian Mixture']
        
    return models

# ------------------------------
# Sidebar Navigation
# ------------------------------
st.sidebar.title("📊 Ultimate Data Analytics Dashboard")

# Page selection
page = st.sidebar.radio(
    "Navigate Pages",
    ["Main Dashboard", "How to Use", "Data Cleaning Guide", "Visualization Guide", "ML Guide", "About", "Advanced ML"]
)

# Theme selection
theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0, key="theme_selector")

# Apply theme
if theme == "Dark":
    st.markdown('<div class="dark-theme">', unsafe_allow_html=True)

# Upload section in sidebar
st.sidebar.markdown("---")
st.sidebar.header("📁 Data Input")
uploaded = st.sidebar.file_uploader(
    "Upload file (CSV, Excel, JSON)", 
    type=["csv", "tsv", "xlsx", "xls", "json"], 
    accept_multiple_files=False
)
use_sample = st.sidebar.checkbox("Use sample dataset (Tips)", value=False)

# Reset button
st.sidebar.markdown("---")
st.sidebar.markdown("**Reset Data**")
if st.sidebar.button("Reset to Original"):
    if "df_raw" in st.session_state:
        st.session_state.df_work = st.session_state.df_raw.copy()
        st.session_state.history = []
        st.sidebar.success("✅ Reset done!")

# ------------------------------
# Page: How to Use
# ------------------------------
if page == "How to Use":
    st.markdown('<h1 class="main-header">📖 How to Use This Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Getting Started
    
    1. **Upload Data**: Use the sidebar to upload your dataset (CSV, Excel, JSON) or use the sample Tips dataset
    2. **Explore**: View your data in the main dashboard to understand its structure
    3. **Clean**: Use the cleaning tools to handle missing values, duplicates, and outliers
    4. **Transform**: Engineer new features and encode categorical variables
    5. **Visualize**: Create interactive charts to explore relationships in your data
    6. **Model**: Build simple machine learning models for regression or clustering
    7. **Download**: Export your processed data in various formats
    
    ## Navigation
    
    - Use the sidebar to switch between different pages
    - The main dashboard contains all the data manipulation tools
    - Reference guides are available for each section
    
    ## Tips
    
    - Your data is preserved as you navigate between pages
    - Use the reset button to revert to your original dataset
    - All transformations are recorded in the history section
    """)
    
    st.markdown("---")
    st.info("💡 Pro Tip: Start with the sample dataset to explore all features before uploading your own data.")

# ------------------------------
# Page: Data Cleaning Guide
# ------------------------------
elif page == "Data Cleaning Guide":
    st.markdown('<h1 class="main-header">🧹 Data Cleaning Guide</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## Handling Missing Values
        
        ### Strategies:
        - **Drop rows**: Remove rows with missing values (use when few missing values)
        - **Fill with Mean**: For numerical columns (preserves distribution)
        - **Fill with Median**: For numerical columns (robust to outliers)
        - **Fill with Mode**: For categorical columns
        - **Fill with Constant**: Use a specific value
        
        ### When to use which:
        - Use mean/median for normally/skewed distributions
        - Use mode for categorical data
        - Drop rows if missing data is random and <5% of dataset
        """)
    
    with col2:
        st.markdown("""
        ## Handling Duplicates
        
        ### Approaches:
        - **Keep first**: Keep first occurrence, remove subsequent duplicates
        - **Keep last**: Keep last occurrence, remove earlier duplicates
        - **Remove all**: Remove all duplicate rows
        
        ### Considerations:
        - Some duplicates may be legitimate (transaction data)
        - Check if duplicates represent true data errors
        
        ## Handling Outliers
        
        ### Methods:
        - **IQR Method**: Identify outliers using interquartile range
        - **Visual Inspection**: Use box plots to identify outliers
        
        ### When to remove:
        - When outliers are measurement errors
        - When they don't represent the phenomenon being studied
        """)
    
    st.markdown("---")
    st.markdown("""
    ## Data Type Conversion
    
    ### Common conversions:
    - **To Numeric**: Convert strings to numbers (handles currency symbols, commas)
    - **To Datetime**: Convert strings to datetime objects
    - **To Category**: Reduce memory usage for categorical variables
    - **One-Hot Encode**: Create dummy variables for machine learning
    
    ### Best practices:
    - Check for consistent formatting before conversion
    - Handle errors during conversion (e.g., coerce to NaN)
    """)

# ------------------------------
# Page: Visualization Guide
# ------------------------------
elif page == "Visualization Guide":
    st.markdown('<h1 class="main-header">📊 Visualization Guide</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Choosing the Right Chart
    
    Select the appropriate visualization based on your data and questions:
    """)
    
    viz_data = {
        "Chart Type": ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Pie Chart", "Area Chart", "Violin Plot"],
        "Best For": [
            "Relationship between two continuous variables",
            "Trends over time or ordered categories",
            "Comparing categories or groups",
            "Distribution of a single variable",
            "Distribution and outliers of a variable across categories",
            "Composition of a whole (proportions)",
            "Cumulative totals over time or categories",
            "Distribution and density of data across categories"
        ],
        "Variables Needed": [
            "2 continuous variables",
            "1 continuous, 1 ordered categorical",
            "1 continuous, 1 categorical",
            "1 continuous variable",
            "1 continuous, 1 categorical",
            "1 categorical variable",
            "1 continuous, 1 categorical/ordered",
            "1 continuous, 1 categorical"
        ]
    }
    
    viz_df = pd.DataFrame(viz_data)
    st.table(viz_df)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## Color and Faceting
        
        ### Color:
        - Use color to represent an additional categorical variable
        - Choose colorblind-friendly palettes
        - Ensure adequate contrast
        
        ### Faceting:
        - Create small multiples for comparing across categories
        - Useful when you have too many categories for a single chart
        - Maintain consistent scales across facets for comparison
        """)
    
    with col2:
        st.markdown("""
        ## Best Practices
        
        ### Labels and Titles:
        - Always include descriptive titles
        - Label axes clearly with units
        - Include a legend when using color
        
        ### Accessibility:
        - Use sufficient size for text and markers
        - Provide alt text for screen readers
        - Consider color vision deficiencies
        
        ### Interpretation:
        - Avoid misleading scales (truncated axes)
        - Include context and reference points
        - Highlight important patterns or outliers
        """)

# ------------------------------
# Page: ML Guide
# ------------------------------
elif page == "ML Guide":
    st.markdown('<h1 class="main-header">🤖 Machine Learning Guide</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Regression Models
    
    ### Random Forest Regression:
    - Ensemble method that combines multiple decision trees
    - Robust to outliers and non-linear relationships
    - Provides feature importance scores
    
    ### Evaluation Metrics:
    - **Mean Squared Error (MSE)**: Average squared difference between predicted and actual values
    - **R² Score**: Proportion of variance explained by the model (0-1, higher is better)
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## Clustering Models
    
    ### K-Means Clustering:
    - Partitions data into K clusters based on feature similarity
    - Requires specifying the number of clusters (K) in advance
    - Works best with spherical clusters of similar size
    
    ### How to choose K:
    - Use domain knowledge if available
    - Try the elbow method (plot inertia vs. K)
    - Consider silhouette score for cluster quality
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## Data Preparation for ML
    
    ### Preprocessing Steps:
    1. Handle missing values
    2. Encode categorical variables (one-hot or label encoding)
    3. Scale numerical features (especially for distance-based algorithms)
    4. Split data into training and testing sets
    
    ### Feature Engineering:
    - Create interaction terms between features
    - Transform skewed variables (log, sqrt transformations)
    - Create polynomial features for non-linear relationships
    - Extract features from datetime variables
    """)

# ------------------------------
# Page: About
# ------------------------------
elif page == "About":
    st.markdown('<h1 class="main-header">ℹ️ About This Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Overview
    
    The Ultimate Data Analytics Dashboard is a comprehensive tool for data exploration, 
    cleaning, visualization, and machine learning. It's designed to make data science 
    accessible without requiring programming expertise.
    
    ## Features
    
    - **Data Upload**: Support for multiple file formats (CSV, Excel, JSON)
    - **Data Cleaning**: Handle missing values, duplicates, and outliers
    - **Feature Engineering**: Create new features and transform variables
    - **Visualization**: Interactive charts with Plotly and Matplotlib
    - **Machine Learning**: Regression and clustering models
    - **Export**: Download processed data in various formats
    
    ## Technology Stack
    
    - Built with Streamlit for the web interface
    - Uses Pandas for data manipulation
    - Plotly and Matplotlib for visualization
    - Scikit-learn for machine learning algorithms
    
    ## Getting Help
    
    - Use the How to Use guide for basic instructions
    - Refer to the specialized guides for each section
    - Check the example with the sample dataset
    """)
    
    st.markdown("---")
    st.markdown("""
    ### Version Information
    - Dashboard Version: 3.0 (Advanced)
    - Last Updated: October 2023
    """)

# ------------------------------
# Page: Advanced ML
# ------------------------------
elif page == "Advanced ML":
    st.markdown('<h1 class="main-header">🚀 Advanced Machine Learning</h1>', unsafe_allow_html=True)
    
    # Load data if not already loaded
    if use_sample:
        df_raw = px.data.tips()
    else:
        df_raw = load_file(uploaded)

    if df_raw.empty:
        st.info("👋 Upload a dataset or toggle 'Use sample dataset'.")
        st.stop()

    if "df_work" not in st.session_state:
        st.session_state.df_work = df_raw.copy()
    if "df_raw" not in st.session_state:
        st.session_state.df_raw = df_raw.copy()

    df = st.session_state.df_work
    
    # Auto-detect column types
    col_types = detect_column_types(df)
    
    # Problem type selection
    problem_type = st.selectbox("Select Problem Type", 
                               ["Regression", "Classification", "Clustering"])
    
    # Display dataset info
    st.markdown("### Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Numeric Features", len(col_types['numeric']))
    with col4:
        st.metric("Categorical Features", len(col_types['categorical']))
    
    # Auto-suggest models
    suggested_models = auto_suggest_models(problem_type.lower(), col_types)
    st.markdown(f"<div class='info-box'>💡 Suggested models: {', '.join(suggested_models)}</div>", unsafe_allow_html=True)
    
    # Preprocessing options
    st.markdown("### Preprocessing Options")
    preprocess_col1, preprocess_col2 = st.columns(2)
    
    with preprocess_col1:
        # Handle missing values
        missing_strategy = st.selectbox("Missing Value Strategy", 
                                       ["None", "Simple Imputer", "KNN Imputer"])
        
        # Feature scaling
        scaling_method = st.selectbox("Feature Scaling", 
                                     ["None", "Standard Scaler", "MinMax Scaler", "Robust Scaler"])
    
    with preprocess_col2:
        # Feature selection
        feature_selection = st.selectbox("Feature Selection", 
                                        ["None", "SelectKBest", "Variance Threshold"])
        
        if feature_selection != "None":
            k_features = st.slider("Number of Features to Select", 
                                  min_value=1, 
                                  max_value=min(20, len(col_types['numeric']) + len(col_types['categorical'])),
                                  value=min(10, len(col_types['numeric']) + len(col_types['categorical'])))
    
    # Model selection based on problem type
    st.markdown("### Model Selection")
    
    if problem_type == "Regression":
        models = {
            "Random Forest": RandomForestRegressor,
            "Gradient Boosting": GradientBoostingRegressor,
            "ElasticNet": ElasticNet,
            "Huber Regression": HuberRegressor,
            "SVM": SVR,
            "K-Neighbors": KNeighborsRegressor,
            "MLP": MLPRegressor,
            "AdaBoost": AdaBoostRegressor,
            "Extra Trees": ExtraTreesRegressor
        }
        
        # Add optional models if installed
        if XGB_INSTALLED:
            models["XGBoost"] = xgb.XGBRegressor
        if LGBM_INSTALLED:
            models["LightGBM"] = lgb.LGBMRegressor
        if CATBOOST_INSTALLED:
            models["CatBoost"] = cb.CatBoostRegressor
        
    elif problem_type == "Classification":
        models = {
            "Random Forest": RandomForestClassifier,
            "Gradient Boosting": GradientBoostingClassifier,
            "Logistic Regression": LogisticRegression,
            "SVM": SVC,
            "K-Neighbors": KNeighborsClassifier,
            "MLP": MLPClassifier,
            "AdaBoost": AdaBoostClassifier,
            "Extra Trees": ExtraTreesClassifier
        }
        
        # Add optional models if installed
        if XGB_INSTALLED:
            models["XGBoost"] = xgb.XGBClassifier
        if LGBM_INSTALLED:
            models["LightGBM"] = lgb.LGBMClassifier
        if CATBOOST_INSTALLED:
            models["CatBoost"] = cb.CatBoostClassifier
            
    else:  # Clustering
        models = {
            "K-Means": KMeans,
            "DBSCAN": DBSCAN,
            "Agglomerative": AgglomerativeClustering,
            "Spectral": SpectralClustering,
            "Gaussian Mixture": GaussianMixture
        }
    
    # Allow multiple model selection for comparison
    selected_models = st.multiselect("Select Models", list(models.keys()), default=suggested_models[:2])
    
    # Target selection for supervised learning
    if problem_type in ["Regression", "Classification"]:
        target_col = st.selectbox("Select Target Variable", df.columns)
    
    # Feature selection
    available_features = col_types['numeric'] + col_types['categorical']
    selected_features = st.multiselect("Select Features", available_features, default=available_features[:min(5, len(available_features))])
    
    # Hyperparameter tuning options
    st.markdown("### Hyperparameter Tuning")
    tune_hyperparams = st.checkbox("Enable Hyperparameter Tuning")
    
    if tune_hyperparams:
        tuning_method = st.selectbox("Tuning Method", ["Grid Search", "Random Search"])
        cv_folds = st.slider("CV Folds", min_value=2, max_value=10, value=5)
    
    # Train/test split
    if problem_type in ["Regression", "Classification"]:
        test_size = st.slider("Test Set Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    
    # Train button
    if st.button("Train Models", type="primary"):
        if not selected_models:
            st.error("Please select at least one model")
        elif problem_type in ["Regression", "Classification"] and not target_col:
            st.error("Please select a target variable")
        elif not selected_features:
            st.error("Please select at least one feature")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Prepare the data
            X = df[selected_features].copy()
            
            if problem_type in ["Regression", "Classification"]:
                y = df[target_col]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            else:
                X_train, X_test = X, None
                y_train, y_test = None, None
            
            # Handle categorical features
            categorical_features = [col for col in selected_features if col in col_types['categorical']]
            numeric_features = [col for col in selected_features if col in col_types['numeric']]
            
            # Create preprocessing pipeline
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
            
            # Train models
            results = []
            for i, model_name in enumerate(selected_models):
                status_text.text(f"Training {model_name}...")
                progress_bar.progress((i) / len(selected_models))
                
                try:
                    # Create pipeline
                    pipeline = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('model', models[model_name]())
                    ])
                    
                    # Train model
                    pipeline.fit(X_train, y_train)
                    
                    # Evaluate model
                    if problem_type in ["Regression", "Classification"]:
                        y_pred = pipeline.predict(X_test)
                        
                        if problem_type == "Regression":
                            score = r2_score(y_test, y_pred)
                            mse = mean_squared_error(y_test, y_pred)
                            results.append({
                                'Model': model_name,
                                'R² Score': score,
                                'MSE': mse
                            })
                        else:  # Classification
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, average='weighted')
                            recall = recall_score(y_test, y_pred, average='weighted')
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            results.append({
                                'Model': model_name,
                                'Accuracy': accuracy,
                                'Precision': precision,
                                'Recall': recall,
                                'F1 Score': f1
                            })
                    else:  # Clustering
                        labels = pipeline.predict(X_train)
                        silhouette = silhouette_score(X_train, labels)
                        results.append({
                            'Model': model_name,
                            'Silhouette Score': silhouette
                        })
                    
                    # Store model in session state
                    if 'trained_models' not in st.session_state:
                        st.session_state.trained_models = {}
                    st.session_state.trained_models[model_name] = pipeline
                    
                except Exception as e:
                    st.error(f"Error training {model_name}: {str(e)}")
            
            progress_bar.progress(1.0)
            status_text.text("Training complete!")
            
            # Display results
            if results:
                results_df = pd.DataFrame(results)
                st.markdown("### Model Comparison")
                st.dataframe(results_df.style.highlight_max(axis=0))
                
                # Visualize results
                if problem_type == "Regression":
                    fig = px.bar(results_df, x='Model', y='R² Score', title='Model Comparison (R² Score)')
                    st.plotly_chart(fig)
                elif problem_type == "Classification":
                    fig = px.bar(results_df, x='Model', y='Accuracy', title='Model Comparison (Accuracy)')
                    st.plotly_chart(fig)
                else:  # Clustering
                    fig = px.bar(results_df, x='Model', y='Silhouette Score', title='Model Comparison (Silhouette Score)')
                    st.plotly_chart(fig)
            
            # Feature importance for tree-based models
            if problem_type in ["Regression", "Classification"] and any(m in selected_models for m in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost']):
                st.markdown("### Feature Importance")
                
                # Get feature names after one-hot encoding
                feature_names = numeric_features.copy()
                for col in categorical_features:
                    unique_vals = df[col].unique()
                    feature_names.extend([f"{col}_{val}" for val in unique_vals])
                
                # Display feature importance for each model
                for model_name in selected_models:
                    if model_name in st.session_state.trained_models and hasattr(st.session_state.trained_models[model_name].named_steps['model'], 'feature_importances_'):
                        importance = st.session_state.trained_models[model_name].named_steps['model'].feature_importances_
                        feat_imp_df = pd.DataFrame({'Feature': feature_names[:len(importance)], 'Importance': importance})
                        feat_imp_df = feat_imp_df.sort_values('Importance', ascending=False).head(10)
                        
                        fig = px.bar(feat_imp_df, x='Importance', y='Feature', 
                                    title=f'Feature Importance - {model_name}')
                        st.plotly_chart(fig)
    
    # Model deployment section
    if 'trained_models' in st.session_state and st.session_state.trained_models:
        st.markdown("### Model Deployment")
        
        selected_model = st.selectbox("Select Model to Deploy", list(st.session_state.trained_models.keys()))
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download model
            model_bytes = joblib.dumps(st.session_state.trained_models[selected_model])
            st.download_button(
                label="Download Model",
                data=model_bytes,
                file_name=f"{selected_model.replace(' ', '_')}_model.pkl",
                mime="application/octet-stream"
            )
        
        with col2:
            # Make predictions on new data
            st.info("Upload new data for predictions")
            new_data_file = st.file_uploader("Upload new data", type=["csv", "xlsx"])
            
            if new_data_file:
                new_df = load_file(new_data_file)
                if not new_df.empty:
                    try:
                        model = st.session_state.trained_models[selected_model]
                        predictions = model.predict(new_df[selected_features])
                        
                        result_df = new_df.copy()
                        result_df['Predictions'] = predictions
                        
                        st.success("Predictions generated successfully!")
                        st.dataframe(result_df.head())
                        
                        # Download predictions
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Error making predictions: {str(e)}")

# ------------------------------
# Page: Main Dashboard
# ------------------------------
else:
    # Gradient title
    st.markdown('<h1 class="main-header">📊 Ultimate Data Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Upload → Clean → Explore → Visualize → Feature Engineering → ML → Download</p>', unsafe_allow_html=True)

    # ------------------------------
    # Load Data
    # ------------------------------
    if use_sample:
        df_raw = px.data.tips()
    else:
        df_raw = load_file(uploaded)

    if df_raw.empty:
        st.info("👋 Upload a dataset or toggle 'Use sample dataset'.")
        st.stop()

    # Initialize session
    if "df_work" not in st.session_state:
        st.session_state.df_work = df_raw.copy()
    if "df_raw" not in st.session_state:
        st.session_state.df_raw = df_raw.copy()
    if "history" not in st.session_state:
        st.session_state.history = []

    df = st.session_state.df_work

    # Auto-detect column types and provide insights
    col_types = detect_column_types(df)
    
    # Display warnings and suggestions
    if len(col_types['numeric']) == 0:
        st.markdown("<div class='warning-box'>⚠️ No numeric columns detected. Some functionality may be limited.</div>", unsafe_allow_html=True)
    
    if df.isna().sum().sum() > 0:
        missing_percent = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        st.markdown(f"<div class='warning-box'>⚠️ Dataset contains {df.isna().sum().sum()} missing values ({missing_percent:.2f}%). Consider handling them in the Cleaning tab.</div>", unsafe_allow_html=True)
    
    if len(col_types['high_cardinality']) > 0:
        st.markdown(f"<div class='info-box'>💡 High cardinality features detected: {', '.join(col_types['high_cardinality'])}. Consider feature engineering techniques.</div>", unsafe_allow_html=True)

    # ------------------------------
    # Dashboard Cards
    # ------------------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">{}</div><div class="metric-label">Rows</div></div>'.format(df.shape[0]), unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">{}</div><div class="metric-label">Columns</div></div>'.format(df.shape[1]), unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">{}</div><div class="metric-label">Missing Values</div></div>'.format(df.isna().sum().sum()), unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">{:.1f}</div><div class="metric-label">Memory Usage (KB)</div></div>'.format(df.memory_usage(deep=True).sum()/1024), unsafe_allow_html=True)

    # ------------------------------
    # Data Overview
    # ------------------------------
    with st.expander("🔎 Data Overview", expanded=True):
        st.dataframe(df.head(200), use_container_width=True, height=300)
        st.dataframe(df_info(df), use_container_width=True, height=250)

    # ------------------------------
    # Cleaning & Feature Engineering
    # ------------------------------
    st.subheader("🧹 Clean, Transform & Engineer Features")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Missing Values", "Duplicates", "Outliers", "Types & Encoding", "Feature Engineering", "Advanced Preprocessing"
    ])

    # -- Missing Values --
    with tab1:
        cols_mv = st.multiselect("Columns", list(df.columns), key="mv_cols")
        method = st.selectbox("Strategy", ["Drop rows", "Fill with Mean", "Fill with Median", "Fill with Mode", "Fill with Constant", "KNN Imputation"], key="mv_method")
        const_val = st.text_input("Constant value", value="0", key="mv_const") if method=="Fill with Constant" else None
        if st.button("Apply Missing Values", key="mv"):
            work = df.copy()
            for c in cols_mv:
                if method=="Drop rows": 
                    work = work.dropna(subset=cols_mv)
                elif method=="Fill with Mean" and pd.api.types.is_numeric_dtype(work[c]): 
                    work[c].fillna(work[c].mean(), inplace=True)
                elif method=="Fill with Median" and pd.api.types.is_numeric_dtype(work[c]): 
                    work[c].fillna(work[c].median(), inplace=True)
                elif method=="Fill with Mode": 
                    work[c].fillna(work[c].mode()[0] if not work[c].mode().empty else None, inplace=True)
                elif method=="Fill with Constant": 
                    work[c].fillna(float(const_val) if pd.api.types.is_numeric_dtype(work[c]) else const_val, inplace=True)
                elif method=="KNN Imputation":
                    from sklearn.impute import KNNImputer
                    imputer = KNNImputer(n_neighbors=5)
                    work[[c]] = imputer.fit_transform(work[[c]])
            st.session_state.df_work = work
            st.session_state.history.append("Missing Values handled")
            st.success("✅ Done!")

    # -- Duplicates --
    with tab2:
        subset = st.multiselect("Subset columns", list(df.columns), key="dup_cols")
        keep = st.selectbox("Keep", ["first","last",False], key="dup_keep")
        if st.button("Drop Duplicates", key="dup"):
            work = df.drop_duplicates(subset=subset or None, keep=keep)
            st.session_state.df_work = work
            st.session_state.history.append("Dropped Duplicates")
            st.success(f"✅ Done! Removed {len(df) - len(work)} duplicates.")

    # -- Outliers --
    with tab3:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cols_out = st.multiselect("Columns", num_cols, default=num_cols[0] if num_cols else [], key="out_cols")
        method = st.selectbox("Method", ["IQR", "Isolation Forest", "Z-Score"], key="out_method")
        
        if method == "IQR":
            iqr_factor = st.slider("IQR factor", 1.0, 3.0, 1.5, 0.1, key="out_iqr")
        elif method == "Z-Score":
            z_threshold = st.slider("Z-Score threshold", 2.0, 5.0, 3.0, 0.1, key="out_z")
        
        if st.button("Handle Outliers", key="out") and cols_out:
            work = df.copy()
            initial_count = len(work)
            
            if method == "IQR":
                for c in cols_out:
                    q1, q3 = work[c].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    work = work[work[c].between(q1-iqr_factor*iqr, q3+iqr_factor*iqr)]
            elif method == "Isolation Forest":
                from sklearn.ensemble import IsolationForest
                clf = IsolationForest(contamination=0.1, random_state=42)
                outliers = clf.fit_predict(work[cols_out])
                work = work[outliers == 1]
            elif method == "Z-Score":
                from scipy import stats
                z_scores = np.abs(stats.zscore(work[cols_out]))
                work = work[(z_scores < z_threshold).all(axis=1)]
                
            st.session_state.df_work = work
            st.session_state.history.append("Outliers handled")
            st.success(f"✅ Done! Removed {initial_count - len(work)} outliers.")

    # -- Types & Encoding --
    with tab4:
        col = st.selectbox("Column", list(df.columns), key="type_col")
        action = st.selectbox("Action", ["To Numeric", "To Datetime", "To Category", "One-Hot Encode", "Label Encode", "Target Encode"], key="type_action")
        if st.button("Apply Conversion", key="type"):
            work = df.copy()
            if action=="To Numeric": 
                work[col] = pd.to_numeric(work[col], errors="coerce")
            elif action=="To Datetime": 
                work[col] = pd.to_datetime(work[col], errors="coerce")
            elif action=="To Category": 
                work[col] = work[col].astype("category")
            elif action=="One-Hot Encode": 
                work = pd.get_dummies(work, columns=[col], prefix=col, drop_first=False, dtype=int)
            elif action=="Label Encode":
                le = LabelEncoder()
                work[col] = le.fit_transform(work[col].astype(str))
            elif action=="Target Encode":
                if "target" in st.session_state and col != st.session_state.target:
                    # Calculate mean target per category
                    target = st.session_state.target
                    encoding = work.groupby(col)[target].mean().to_dict()
                    work[col] = work[col].map(encoding)
                else:
                    st.error("Please set a target variable first in the ML tab")
            st.session_state.df_work = work
            st.session_state.history.append(f"Conversion: {action}")
            st.success("✅ Done!")

    # -- Feature Engineering --
    with tab5:
        feat_type = st.selectbox("Feature Type", ["Arithmetic","Binning","Datetime Extract","Polynomial","Interaction"], key="feat_type")
        if feat_type=="Arithmetic":
            col1 = st.selectbox("Column 1", num_cols, key="feat_col1")
            col2 = st.selectbox("Column 2", num_cols, key="feat_col2")
            op = st.selectbox("Operation", ["+","-","*","/", "log", "sqrt"], key="feat_op")
            new_col = st.text_input("New column name", value=f"{col1}_{op}_{col2}", key="feat_newcol1")
            if st.button("Create Feature", key="feat1") and col1 and col2 and new_col:
                work = df.copy()
                if op == "+": work[new_col] = work[col1] + work[col2]
                elif op == "-": work[new_col] = work[col1] - work[col2]
                elif op == "*": work[new_col] = work[col1] * work[col2]
                elif op == "/": work[new_col] = work[col1] / work[col2].replace(0, np.nan)
                elif op == "log": work[new_col] = np.log(work[col1] + 1)  # Add 1 to avoid log(0)
                elif op == "sqrt": work[new_col] = np.sqrt(work[col1])
                st.session_state.df_work = work
                st.session_state.history.append(f"Feature: {new_col}")
                st.success("✅ Done!")
        elif feat_type=="Binning":
            col = st.selectbox("Numeric Column for Binning", num_cols, key="bin_col")
            method = st.selectbox("Binning Method", ["Equal Width", "Equal Frequency", "Custom"], key="bin_method")
            
            if method == "Custom":
                bins = st.text_input("Bins (comma-separated, e.g., 0,10,20)", value="0,10,20", key="bin_bins")
                labels = st.text_input("Labels (comma-separated, e.g., Low,Mid,High)", value="Low,Mid,High", key="bin_labels")
            elif method == "Equal Width":
                n_bins = st.slider("Number of bins", 2, 10, 3, key="bin_n")
                labels = None
            else:  # Equal Frequency
                n_bins = st.slider("Number of bins", 2, 10, 3, key="bin_n")
                labels = None
                
            new_col = st.text_input("New column name", value=f"{col}_binned", key="bin_newcol")
            if st.button("Create Binned Feature", key="feat2") and col and new_col:
                work = df.copy()
                if method == "Custom":
                    bins_list = [float(x.strip()) for x in bins.split(",")]
                    labels_list = [x.strip() for x in labels.split(",")]
                    if len(bins_list) - 1 != len(labels_list):
                        st.error("Number of labels must be one less than number of bins")
                    else:
                        work[new_col] = pd.cut(work[col], bins=bins_list, labels=labels_list, include_lowest=True)
                elif method == "Equal Width":
                    work[new_col] = pd.cut(work[col], bins=n_bins, include_lowest=True)
                else:  # Equal Frequency
                    work[new_col] = pd.qcut(work[col], q=n_bins, duplicates='drop')
                st.session_state.df_work = work
                st.session_state.history.append(f"Binned: {new_col}")
                st.success("✅ Done!")
        elif feat_type=="Datetime Extract":
            datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
            if datetime_cols:
                col = st.selectbox("Datetime Column", datetime_cols, key="dt_col")
                dt_part = st.selectbox("Extract", ["Year","Month","Day","Hour","Weekday","Quarter","Dayofyear"], key="dt_part")
                new_col = st.text_input("New column name", value=f"{col}_{dt_part.lower()}", key="dt_newcol")
                if st.button("Extract Date Feature", key="feat3"):
                    work = df.copy()
                    if dt_part=="Year": work[new_col] = work[col].dt.year
                    elif dt_part=="Month": work[new_col] = work[col].dt.month
                    elif dt_part=="Day": work[new_col] = work[col].dt.day
                    elif dt_part=="Hour": work[new_col] = work[col].dt.hour
                    elif dt_part=="Weekday": work[new_col] = work[col].dt.weekday
                    elif dt_part=="Quarter": work[new_col] = work[col].dt.quarter
                    elif dt_part=="Dayofyear": work[new_col] = work[col].dt.dayofyear
                    st.session_state.df_work = work
                    st.session_state.history.append(f"Datetime Feature: {new_col}")
                    st.success("✅ Done!")
            else:
                st.info("No datetime columns found for extraction.")
        elif feat_type=="Polynomial":
            col = st.selectbox("Numeric Column", num_cols, key="poly_col")
            degree = st.slider("Degree", 2, 5, 2, key="poly_degree")
            new_col = st.text_input("New column name", value=f"{col}_poly{degree}", key="poly_newcol")
            if st.button("Create Polynomial Feature", key="feat4"):
                work = df.copy()
                work[new_col] = work[col] ** degree
                st.session_state.df_work = work
                st.session_state.history.append(f"Polynomial Feature: {new_col}")
                st.success("✅ Done!")
        elif feat_type=="Interaction":
            col1 = st.selectbox("Column 1", num_cols, key="inter_col1")
            col2 = st.selectbox("Column 2", num_cols, key="inter_col2")
            new_col = st.text_input("New column name", value=f"{col1}_x_{col2}", key="inter_newcol")
            if st.button("Create Interaction Feature", key="feat5"):
                work = df.copy()
                work[new_col] = work[col1] * work[col2]
                st.session_state.df_work = work
                st.session_state.history.append(f"Interaction Feature: {new_col}")
                st.success("✅ Done!")

    # -- Advanced Preprocessing --
    with tab6:
        st.markdown("### Automated Preprocessing Pipeline")
        
        pipeline_steps = []
        
        # Missing value imputation
        impute_method = st.selectbox("Missing Value Imputation", 
                                   ["None", "Mean", "Median", "Mode", "Constant", "KNN"])
        if impute_method != "None":
            pipeline_steps.append(f"Imputation: {impute_method}")
        
        # Scaling
        scaling_method = st.selectbox("Feature Scaling", 
                                    ["None", "Standard", "MinMax", "Robust"])
        if scaling_method != "None":
            pipeline_steps.append(f"Scaling: {scaling_method}")
        
        # Encoding
        encoding_method = st.selectbox("Categorical Encoding", 
                                     ["None", "One-Hot", "Label", "Target"])
        if encoding_method != "None":
            pipeline_steps.append(f"Encoding: {encoding_method}")
        
        # Feature selection
        feature_selection = st.selectbox("Feature Selection", 
                                       ["None", "Variance Threshold", "SelectKBest", "PCA"])
        if feature_selection != "None":
            pipeline_steps.append(f"Feature Selection: {feature_selection}")
        
        # Display pipeline
        if pipeline_steps:
            st.markdown("### Pipeline Steps")
            for step in pipeline_steps:
                st.markdown(f'<div class="pipeline-step">{step}</div>', unsafe_allow_html=True)
        
        if st.button("Apply Preprocessing Pipeline", key="pipeline"):
            work = df.copy()
            
            # Apply imputation
            if impute_method != "None":
                numeric_cols = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]
                categorical_cols = [c for c in work.columns if work[c].dtype == 'object' or work[c].dtype.name == 'category']
                
                if impute_method == "Mean":
                    for col in numeric_cols:
                        work[col].fillna(work[col].mean(), inplace=True)
                elif impute_method == "Median":
                    for col in numeric_cols:
                        work[col].fillna(work[col].median(), inplace=True)
                elif impute_method == "Mode":
                    for col in categorical_cols:
                        work[col].fillna(work[col].mode()[0] if not work[col].mode().empty else "Unknown", inplace=True)
                elif impute_method == "Constant":
                    const_val = st.text_input("Constant value", value="0")
                    for col in numeric_cols + categorical_cols:
                        work[col].fillna(const_val, inplace=True)
                elif impute_method == "KNN":
                    from sklearn.impute import KNNImputer
                    imputer = KNNImputer(n_neighbors=5)
                    work[numeric_cols] = imputer.fit_transform(work[numeric_cols])
            
            # Apply scaling
            if scaling_method != "None":
                numeric_cols = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]
                
                if scaling_method == "Standard":
                    scaler = StandardScaler()
                    work[numeric_cols] = scaler.fit_transform(work[numeric_cols])
                elif scaling_method == "MinMax":
                    scaler = MinMaxScaler()
                    work[numeric_cols] = scaler.fit_transform(work[numeric_cols])
                elif scaling_method == "Robust":
                    scaler = RobustScaler()
                    work[numeric_cols] = scaler.fit_transform(work[numeric_cols])
            
            # Apply encoding
            if encoding_method != "None":
                categorical_cols = [c for c in work.columns if work[c].dtype == 'object' or work[c].dtype.name == 'category']
                
                if encoding_method == "One-Hot":
                    work = pd.get_dummies(work, columns=categorical_cols, drop_first=False)
                elif encoding_method == "Label":
                    for col in categorical_cols:
                        le = LabelEncoder()
                        work[col] = le.fit_transform(work[col].astype(str))
                elif encoding_method == "Target":
                    if "target" in st.session_state:
                        target = st.session_state.target
                        for col in categorical_cols:
                            encoding = work.groupby(col)[target].mean().to_dict()
                            work[col] = work[col].map(encoding)
                    else:
                        st.error("Please set a target variable first in the ML tab")
            
            st.session_state.df_work = work
            st.session_state.history.append("Applied preprocessing pipeline")
            st.success("✅ Pipeline applied successfully!")

    # ------------------------------
    # EDA & Visualization
    # ------------------------------
    st.subheader("📈 Exploratory Data Analysis & Visuals")
    eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs(["Summary Stats","Correlations","Charts", "Advanced EDA"])

    # Summary Stats
    with eda_tab1:
        st.dataframe(df.describe(include="all").transpose(), height=300, use_container_width=True)
        
        # Skewness and kurtosis
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            skewness = df[numeric_cols].skew().to_frame(name='Skewness')
            kurtosis = df[numeric_cols].kurtosis().to_frame(name='Kurtosis')
            stats_df = pd.concat([skewness, kurtosis], axis=1)
            st.markdown("### Distribution Statistics")
            st.dataframe(stats_df, use_container_width=True)
            
            # Highlight highly skewed features
            highly_skewed = skewness[abs(skewness['Skewness']) > 1].index.tolist()
            if highly_skewed:
                st.markdown(f"<div class='info-box'>💡 Highly skewed features detected: {', '.join(highly_skewed)}. Consider applying log or power transformations.</div>", unsafe_allow_html=True)

    # Correlations
    with eda_tab2:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax, fmt=".2f")
            st.pyplot(fig)
            
            # Show highly correlated pairs
            st.markdown("### Highly Correlated Features")
            corr_pairs = corr.unstack().sort_values(ascending=False)
            corr_pairs = corr_pairs[corr_pairs < 1.0]  # Remove self-correlations
            top_pairs = corr_pairs.head(10)
            st.dataframe(top_pairs.reset_index().rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: "Correlation"}))
            
            # Correlation threshold
            threshold = st.slider("Correlation Threshold", 0.5, 0.95, 0.8, 0.05)
            high_corr = [(i, j) for i in corr.index for j in corr.columns if i != j and abs(corr.loc[i, j]) > threshold]
            
            if high_corr:
                st.markdown(f"<div class='warning-box'>⚠️ Features with correlation > {threshold}: {', '.join([f'({i}, {j})' for i, j in high_corr[:5]])}{'...' if len(high_corr) > 5 else ''}</div>", unsafe_allow_html=True)
        else:
            st.info("Need at least 2 numeric columns for correlation analysis.")

    # Charts
    with eda_tab3:
        chart_type = st.selectbox("Chart Type", ["Scatter","Line","Bar","Histogram","Box","Pie","Area", "Violin", "Heatmap", "Pairplot"], key="chart_type")
        
        if chart_type == "Pairplot":
            if len(num_cols) >= 2:
                sample_size = st.slider("Sample Size", 50, 1000, 200)
                hue_col = st.selectbox("Color By", [None] + list(df.columns), key="pair_hue")
                
                sample_df = df.sample(min(sample_size, len(df)))
                fig = px.scatter_matrix(sample_df, dimensions=num_cols[:4], color=hue_col)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns for pairplot.")
        elif chart_type == "Heatmap":
            z_metric = st.selectbox("Metric", ["Count", "Mean", "Sum"], key="heat_metric")
            x_col = st.selectbox("X Axis", [None] + list(df.columns), key="heat_x")
            y_col = st.selectbox("Y Axis", [None] + list(df.columns), key="heat_y")
            z_col = st.selectbox("Value", [None] + num_cols, key="heat_z")
            
            if x_col and y_col and z_col:
                pivot_df = df.pivot_table(values=z_col, index=y_col, columns=x_col, aggfunc=z_metric.lower())
                fig = px.imshow(pivot_df, aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
        else:
            x_col = st.selectbox("X", [None] + list(df.columns), key="x_col")
            y_col = st.selectbox("Y", [None] + list(df.columns), key="y_col")
            color_col = st.selectbox("Color", [None] + list(df.columns), key="color_col")
            facet_col = st.selectbox("Facet", [None] + list(df.columns), key="facet_col")
            
            fig = None
            try:
                if chart_type == "Scatter" and x_col and y_col: 
                    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, facet_col=facet_col, 
                                    title=f"Scatter Plot: {x_col} vs {y_col}")
                elif chart_type == "Line" and x_col and y_col: 
                    fig = px.line(df, x=x_col, y=y_col, color=color_col, facet_col=facet_col,
                                 title=f"Line Plot: {y_col} over {x_col}")
                elif chart_type == "Bar" and x_col and y_col: 
                    fig = px.bar(df, x=x_col, y=y_col, color=color_col, facet_col=facet_col,
                                title=f"Bar Chart: {y_col} by {x_col}")
                elif chart_type == "Histogram" and x_col: 
                    fig = px.histogram(df, x=x_col, color=color_col, facet_col=facet_col, marginal="box",
                                      title=f"Distribution of {x_col}")
                elif chart_type == "Box" and x_col and y_col: 
                    fig = px.box(df, x=x_col, y=y_col, color=color_col, facet_col=facet_col,
                                title=f"Box Plot: {y_col} by {x_col}")
                elif chart_type == "Pie" and x_col: 
                    fig = px.pie(df, names=x_col, color=color_col, title=f"Proportion of {x_col}")
                elif chart_type == "Area" and x_col and y_col: 
                    fig = px.area(df, x=x_col, y=y_col, color=color_col, facet_col=facet_col,
                                 title=f"Area Chart: {y_col} over {x_col}")
                elif chart_type == "Violin" and x_col and y_col: 
                    fig = px.violin(df, x=x_col, y=y_col, color=color_col, box=True,
                                   title=f"Violin Plot: {y_col} by {x_col}")
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    fname, fbytes = plotly_download_link(fig, f"{chart_type.lower()}_chart.html")
                    st.download_button("⬇️ Download Chart (HTML)", data=fbytes, file_name=fname, mime="text/html")
            except Exception as e:
                st.error(f"Chart error: {e}")

    # Advanced EDA
    with eda_tab4:
        st.markdown("### Dimensionality Reduction")
        
        reduction_method = st.selectbox("Method", ["PCA", "t-SNE", "UMAP"], key="reduction_method")
        n_components = st.slider("Components", 2, 3, 2, key="n_components")
        color_col = st.selectbox("Color By", [None] + list(df.columns), key="reduction_color")
        
        if st.button("Apply Dimensionality Reduction", key="apply_reduction"):
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(numeric_cols) < 2:
                st.error("Need at least 2 numeric columns for dimensionality reduction")
            else:
                # Handle missing values
                X = df[numeric_cols].dropna()
                
                if reduction_method == "PCA":
                    reducer = PCA(n_components=n_components)
                elif reduction_method == "t-SNE":
                    reducer = TSNE(n_components=n_components, random_state=42)
                else:  # UMAP
                    reducer = umap.UMAP(n_components=n_components, random_state=42)
                
                # Scale the data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Apply dimensionality reduction
                embedding = reducer.fit_transform(X_scaled)
                
                # Create plot
                if n_components == 2:
                    result_df = pd.DataFrame({
                        'Component 1': embedding[:, 0],
                        'Component 2': embedding[:, 1]
                    })
                else:  # 3 components
                    result_df = pd.DataFrame({
                        'Component 1': embedding[:, 0],
                        'Component 2': embedding[:, 1],
                        'Component 3': embedding[:, 2]
                    })
                
                # Add color column if selected
                if color_col:
                    result_df[color_col] = df.loc[X.index, color_col].values
                
                # Plot
                if n_components == 2:
                    fig = px.scatter(result_df, x='Component 1', y='Component 2', color=color_col,
                                    title=f'{reduction_method} Visualization')
                else:
                    fig = px.scatter_3d(result_df, x='Component 1', y='Component 2', z='Component 3', 
                                       color=color_col, title=f'{reduction_method} Visualization')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Explained variance for PCA
                if reduction_method == "PCA":
                    explained_variance = reducer.explained_variance_ratio_
                    st.markdown(f"**Explained Variance:** {explained_variance[0]:.3f}, {explained_variance[1]:.3f}")
                    if n_components == 3:
                        st.markdown(f", {explained_variance[2]:.3f}")

    # ------------------------------
    # Quick Filters
    # ------------------------------
    st.subheader("🔎 Filters")
    with st.expander("Filter Rows"):
        text_col = st.selectbox("Text Column", [None] + [c for c in df.columns if df[c].dtype == "object"], key="filter_text_col")
        query_text = st.text_input("Contains text", key="filter_query")
        num_col = st.selectbox("Numeric Column", [None] + num_cols, key="filter_num_col")
        rng = None
        if num_col:
            min_val, max_val = float(df[num_col].min()), float(df[num_col].max())
            rng = st.slider("Range", min_val, max_val, value=(min_val, max_val), key="filter_range")
        if st.button("Apply Filters", key="filter"):
            work = df.copy()
            if text_col and query_text: 
                work = work[work[text_col].astype(str).str.contains(query_text, case=False, na=False)]
            if num_col and rng: 
                work = work[work[num_col].between(rng[0], rng[1])]
            st.session_state.df_work = work
            st.session_state.history.append("Filters applied")
            st.success(f"✅ Done! {len(work)} rows remaining.")

    # ------------------------------
    # ML Preview
    # ------------------------------
    st.subheader("🤖 ML Preview")
    ml_tab1, ml_tab2, ml_tab3 = st.tabs(["Regression","Classification","Clustering"])

    # Regression
    with ml_tab1:
        target = st.selectbox("Target Column (Numeric)", [c for c in num_cols], key="reg_target")
        features = st.multiselect("Feature Columns", [c for c in df.columns if c != target], key="reg_features")
        
        # Model selection
        model_type = st.selectbox("Model", ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost", 
                                          "ElasticNet", "Huber Regression", "SVM", "K-Neighbors", "MLP"], key="reg_model")
        
        if st.button("Run Regression", key="ml1") and target and features:
            work = df.dropna(subset=[target] + features)
            X, y = work[features], work[target]
            
            # Encode categorical features
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            else:
                X_encoded = X
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
            
            # Train model
            if model_type == "Random Forest":
                model = RandomForestRegressor(random_state=42)
            elif model_type == "Gradient Boosting":
                model = GradientBoostingRegressor(random_state=42)
            elif model_type == "XGBoost" and XGB_INSTALLED:
                model = xgb.XGBRegressor(random_state=42)
            elif model_type == "LightGBM" and LGBM_INSTALLED:
                model = lgb.LGBMRegressor(random_state=42)
            elif model_type == "CatBoost" and CATBOOST_INSTALLED:
                model = cb.CatBoostRegressor(random_state=42, verbose=0)
            elif model_type == "ElasticNet":
                model = ElasticNet(random_state=42)
            elif model_type == "Huber Regression":
                model = HuberRegressor()
            elif model_type == "SVM":
                model = SVR()
            elif model_type == "K-Neighbors":
                model = KNeighborsRegressor()
            elif model_type == "MLP":
                model = MLPRegressor(random_state=42)
            else:
                st.error(f"{model_type} is not available. Please install required library.")
                model = None
                
            if model:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("MSE", f"{mean_squared_error(y_test, y_pred):.4f}")
                with col2:
                    st.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
                    
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'feature': X_encoded.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    fig_imp = px.bar(feature_importance.head(10), x='importance', y='feature', 
                                    title='Top 10 Feature Importance')
                    st.plotly_chart(fig_imp, use_container_width=True)
                
                # Actual vs Predicted
                fig_avp = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'},
                                   title='Actual vs Predicted Values')
                fig_avp.add_shape(type='line', line=dict(dash='dash'), x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max())
                st.plotly_chart(fig_avp, use_container_width=True)
                
                # Store target for feature engineering
                st.session_state.target = target

    # Classification
    with ml_tab2:
        target = st.selectbox("Target Column (Categorical)", [c for c in df.columns if c not in num_cols], key="clf_target")
        features = st.multiselect("Feature Columns", [c for c in df.columns if c != target], key="clf_features")
        
        # Model selection
        model_type = st.selectbox("Model", ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost", 
                                          "Logistic Regression", "SVM", "K-Neighbors", "MLP"], key="clf_model")
        
        if st.button("Run Classification", key="ml2") and target and features:
            work = df.dropna(subset=[target] + features)
            X, y = work[features], work[target]
            
            # Encode categorical features
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            else:
                X_encoded = X
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
            
            # Train model
            if model_type == "Random Forest":
                model = RandomForestClassifier(random_state=42)
            elif model_type == "Gradient Boosting":
                model = GradientBoostingClassifier(random_state=42)
            elif model_type == "XGBoost" and XGB_INSTALLED:
                model = xgb.XGBClassifier(random_state=42)
            elif model_type == "LightGBM" and LGBM_INSTALLED:
                model = lgb.LGBMClassifier(random_state=42)
            elif model_type == "CatBoost" and CATBOOST_INSTALLED:
                model = cb.CatBoostClassifier(random_state=42, verbose=0)
            elif model_type == "Logistic Regression":
                model = LogisticRegression(random_state=42)
            elif model_type == "SVM":
                model = SVC(random_state=42)
            elif model_type == "K-Neighbors":
                model = KNeighborsClassifier()
            elif model_type == "MLP":
                model = MLPClassifier(random_state=42)
            else:
                st.error(f"{model_type} is not available. Please install required library.")
                model = None
                
            if model:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                with col2:
                    st.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.4f}")
                with col3:
                    st.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.4f}")
                with col4:
                    st.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.4f}")
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = px.imshow(cm, text_auto=True, aspect="auto", 
                                  labels=dict(x="Predicted", y="Actual", color="Count"),
                                  x=sorted(y_test.unique()), y=sorted(y_test.unique()))
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # ROC curve for binary classification
                if y_prob is not None and len(np.unique(y_test)) == 2:
                    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                    roc_auc = auc(fpr, tpr)
                    
                    fig_roc = px.area(x=fpr, y=tpr, title=f'ROC Curve (AUC = {roc_auc:.2f})',
                                     labels=dict(x='False Positive Rate', y='True Positive Rate'))
                    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                    st.plotly_chart(fig_roc, use_container_width=True)
                
                # Store target for feature engineering
                st.session_state.target = target

    # Clustering
    with ml_tab3:
        features = st.multiselect("Feature Columns", num_cols, key="cluster_features")
        
        # Model selection
        model_type = st.selectbox("Model", ["K-Means", "DBSCAN", "Agglomerative", "Gaussian Mixture", "Spectral"], key="cluster_model")
        
        # Model parameters
        if model_type == "K-Means":
            n_clusters = st.slider("Number of Clusters", 2, 10, 3, key="kmeans_n")
        elif model_type == "DBSCAN":
            eps = st.slider("EPS", 0.1, 5.0, 0.5, 0.1, key="dbscan_eps")
            min_samples = st.slider("Min Samples", 1, 20, 5, key="dbscan_min")
        elif model_type == "Agglomerative":
            n_clusters = st.slider("Number of Clusters", 2, 10, 3, key="agg_n")
            linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"], key="agg_linkage")
        elif model_type == "Gaussian Mixture":
            n_clusters = st.slider("Number of Clusters", 2, 10, 3, key="gmm_n")
        elif model_type == "Spectral":
            n_clusters = st.slider("Number of Clusters", 2, 10, 3, key="spec_n")
        
        if st.button("Run Clustering", key="ml3") and features:
            work = df[features].dropna()
            
            # Scale the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(work)
            
            # Train model
            if model_type == "K-Means":
                model = KMeans(n_clusters=n_clusters, random_state=42)
            elif model_type == "DBSCAN":
                model = DBSCAN(eps=eps, min_samples=min_samples)
            elif model_type == "Agglomerative":
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            elif model_type == "Gaussian Mixture":
                model = GaussianMixture(n_components=n_clusters, random_state=42)
            elif model_type == "Spectral":
                model = SpectralClustering(n_clusters=n_clusters, random_state=42)
            
            labels = model.fit_predict(X_scaled)
            
            # Add clusters to dataframe
            result_df = df.copy()
            result_df["Cluster"] = np.nan
            result_df.loc[work.index, "Cluster"] = labels
            
            st.dataframe(result_df.head(200))
            
            # Cluster evaluation
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(X_scaled, labels)
                st.metric("Silhouette Score", f"{silhouette:.4f}")
            
            # Visualize clusters
            if len(features) >= 2:
                # Use first two features for 2D visualization
                fig = px.scatter(result_df, x=features[0], y=features[1], color="Cluster",
                               title=f"Cluster Visualization ({model_type})")
                st.plotly_chart(fig, use_container_width=True)
            
            # 3D visualization if we have at least 3 features
            if len(features) >= 3:
                fig3d = px.scatter_3d(result_df, x=features[0], y=features[1], z=features[2], 
                                    color="Cluster", title=f"3D Cluster Visualization ({model_type})")
                st.plotly_chart(fig3d, use_container_width=True)

    # ------------------------------
    # Download Processed Data
    # ------------------------------
    st.subheader("📥 Download Processed Data")
    colD1, colD2, colD3 = st.columns(3)
    with colD1:
        st.download_button("⬇️ CSV", data=to_csv_bytes(df), file_name="processed_data.csv", mime="text/csv")
    with colD2:
        st.download_button("⬇️ Excel", data=to_excel_bytes(df), file_name="processed_data.xlsx", 
                          mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with colD3:
        st.download_button("⬇️ Pickle", data=to_pickle_bytes(df), file_name="processed_data.pkl", 
                          mime="application/octet-stream")

    # ------------------------------
    # Transformation History
    # ------------------------------
    with st.expander("📝 Transformation History"):
        if st.session_state.history:
            for i, item in enumerate(st.session_state.history, 1):
                st.write(f"{i}. {item}")
        else:
            st.write("No transformations applied yet.")

# Close the dark theme div if applied
if theme == "Dark":
    st.markdown('</div>', unsafe_allow_html=True)
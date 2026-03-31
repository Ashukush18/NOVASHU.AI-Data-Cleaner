import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as st_sns # keeping it standard
import seaborn as sns
import io
from scipy import stats
from sklearn.impute import SimpleImputer

import os
import base64

# Set page configuration
st.set_page_config(page_title="NOVASHUAI | Data Cleaner", page_icon="⚙️", layout="wide")

# Custom UI CSS
st.markdown("""
<style>
    /* Main Headers */
    .main-header { font-weight: 900; font-size: 3.5rem; color: #00C4FF; text-align: center; margin-bottom: 0px; letter-spacing: 2px;}
    .sub-header { font-size: 1.2rem; color: #b0b0b0; text-align: center; margin-top: 5px; margin-bottom: 40px; font-weight: 300; letter-spacing: 1px;}
    
    /* Button Hover */
    .stButton>button { border-radius: 8px; transition: 0.3s; width: 100%; border: 1px solid #333;}
    .stButton>button:hover { border-color: #00C4FF; color: #00C4FF; box-shadow: 0 0 10px rgba(0, 196, 255, 0.4); transform: scale(1.02); }
    
    /* Card design for summary */
    div[data-testid="metric-container"] {
        background-color: #1b1d24; border: 1px solid #2e323b; padding: 20px; border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.4); border-left: 4px solid #00C4FF;
    }
</style>
""", unsafe_allow_html=True)

# Process Logo
logo_path = None
for ext in ["png", "jpg", "jpeg"]:
    if os.path.exists(f"logo.{ext}"):
        logo_path = f"logo.{ext}"
        break
    elif os.path.exists(f"img_1.{ext}"): # fallback for chat upload names
        logo_path = f"img_1.{ext}"
        break

if logo_path:
    with open(logo_path, "rb") as f:
        encoded_logo = base64.b64encode(f.read()).decode()
    logo_html = f'''
        <div style="display: flex; justify-content: center; margin-top: 20px;">
            <img src="data:image/png;base64,{encoded_logo}" 
                 style="width: 170px; height: 170px; border-radius: 50%; object-fit: cover; border: 4px solid #00C4FF; box-shadow: 0 0 25px rgba(0, 196, 255, 0.4); margin-bottom: 20px;">
        </div>
    '''
    st.sidebar.markdown(logo_html, unsafe_allow_html=True)
else:
    # Placeholder if logo not found
    st.sidebar.markdown('''
        <div style="display: flex; justify-content: center; margin-top: 20px;">
            <div style="width: 170px; height: 170px; border-radius: 50%; border: 4px solid #00C4FF; display: flex; align-items: center; justify-content: center; background-color: #1b1d24; box-shadow: 0 0 25px rgba(0, 196, 255, 0.4); margin-bottom: 20px;">
                 <h2 style="color: #00C4FF; margin:0; font-size: 1.3rem; letter-spacing: 1px;">NOVASHU.AI</h2>
            </div>
        </div>
    ''', unsafe_allow_html=True)

st.sidebar.markdown('<h2 style="text-align: center; color: #FAFAFA; margin-bottom: 0px; padding-bottom: 0px; letter-spacing: 2px;">NOVASHU.AI</h2>', unsafe_allow_html=True)
st.sidebar.markdown('<p style="text-align: center; color: #00C4FF; font-size: 0.9rem; font-weight: bold; margin-top: -10px; letter-spacing: 1.5px;">PRO DATA ENGINE</p>', unsafe_allow_html=True)
st.sidebar.divider()

st.markdown('<div class="main-header">N O V A S H U . A I</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Data Cleaning & Intelligent Analysis</div>', unsafe_allow_html=True)

# Initialize session state for data
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'clean_data' not in st.session_state:
    st.session_state.clean_data = None
if 'changes_log' not in st.session_state:
    st.session_state.changes_log = []

# --- 1. Dataset Upload ---
st.sidebar.header("1. 📂 Dataset Upload")
uploaded_file = st.sidebar.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        try:
            # Load data based on extension
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            # Store in session state
            st.session_state.raw_data = df.copy()
            st.session_state.clean_data = df.copy()
            st.session_state.changes_log = [f"Loaded dataset: {uploaded_file.name}"]
            st.session_state.current_file = uploaded_file.name
            st.sidebar.success("File successfully uploaded!")
                
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")

if st.session_state.raw_data is not None:
    # Use tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Data Analysis (EDA)", "🤖 AI Suggestions", "🧹 Smart Cleaning Engine", "📊 Before vs After & Download"])
    
    # Getting current state
    raw_df = st.session_state.raw_data
    clean_df = st.session_state.clean_data
    
    # --- Tab 1: Data Analysis ---
    with tab1:
        st.subheader("Data Preview (First 5 Rows)")
        st.dataframe(clean_df.head())
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", clean_df.shape[0])
        col2.metric("Columns", clean_df.shape[1])
        col3.metric("Missing Values", clean_df.isnull().sum().sum())
        col4.metric("Duplicate Rows", clean_df.duplicated().sum())

        st.subheader("Summary Statistics")
        st.dataframe(clean_df.describe(include='all'))
        
        st.subheader("Data Types & Missing Values per Column")
        info_df = pd.DataFrame({
            "Data Type": clean_df.dtypes,
            "Missing Values": clean_df.isnull().sum(),
            "% Missing": (clean_df.isnull().sum() / len(clean_df) * 100).round(2),
            "Unique Values": clean_df.nunique()
        })
        st.dataframe(info_df)
        
        # Visualizations
        st.subheader("Visualizations")
        numeric_cols = clean_df.select_dtypes(include=np.number).columns.tolist()
        
        if numeric_cols:
            vis_col = st.selectbox("Select a numeric column to visualize", numeric_cols)
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            
            # Histogram
            sns.histplot(clean_df[vis_col].dropna(), kde=True, ax=ax[0])
            ax[0].set_title(f"Histogram of {vis_col}")
            
            # Boxplot
            sns.boxplot(x=clean_df[vis_col].dropna(), ax=ax[1])
            ax[1].set_title(f"Boxplot of {vis_col} (Outlier Detection)")
            
            st.pyplot(fig)
        else:
            st.info("No numeric columns found for visualization.")

    # --- Tab 2: AI Suggestions ---
    with tab2:
        st.header("🤖 AI Data Cleaning Suggestions")
        st.markdown("Here is what the **AI brain** recommends for your dataset based on its analysis.")
        
        suggestions = []
        
        # Checking missing values
        missing_counts = clean_df.isnull().sum()
        for col in missing_counts[missing_counts > 0].index:
            percent_missing = (missing_counts[col] / len(clean_df)) * 100
            dtype = clean_df[col].dtype
            
            if percent_missing > 50:
                suggestions.append(f"🔴 **{col}** has {percent_missing:.1f}% missing values → **Recommend:** Drop column")
            elif pd.api.types.is_numeric_dtype(dtype):
                # check skewness
                skewness = clean_df[col].skew()
                if abs(skewness) > 1:
                    suggestions.append(f"🟠 **{col}** is skewed (skew={skewness:.2f}) and has {percent_missing:.1f}% missing values → **Recommend:** Median imputation")
                else:
                    suggestions.append(f"🟢 **{col}** has normal distribution and {percent_missing:.1f}% missing values → **Recommend:** Mean imputation")
            else:
                 suggestions.append(f"🟡 **{col}** is categorical with {percent_missing:.1f}% missing values → **Recommend:** Mode/Most-frequent imputation")
                 
        # Checking duplicates
        dup_count = clean_df.duplicated().sum()
        if dup_count > 0:
            suggestions.append(f"🔵 Found **{dup_count}** duplicate rows → **Recommend:** Remove duplicates")
            
        # Checking outliers & skewness for numeric columns
        for col in numeric_cols:
            skewness = clean_df[col].skew()
            if abs(skewness) > 1.5:
                suggestions.append(f"🟣 **{col}** is highly skewed (skew={skewness:.2f}) → **Suggest:** Log transformation (if values > 0)")
            
            # Outlier detection using Z-score roughly
            z_scores = np.abs(stats.zscore(clean_df[col].dropna()))
            outlier_percent = (len(np.where(z_scores > 3)[0]) / len(clean_df[col].dropna())) * 100
            if outlier_percent > 1:
                suggestions.append(f"🟤 **{col}** has {outlier_percent:.1f}% outliers (Z-score > 3) → **Suggest:** Apply IQR or Z-score outlier removal")
                
        if not suggestions:
            st.success("🎉 Data looks great! No major issues detected by AI.")
        else:
            for s in suggestions:
                st.markdown(f"- {s}")

    # --- Tab 3: Smart Cleaning Engine ---
    with tab3:
        st.header("🧹 Smart Data Cleaning Actions")
        
        st.markdown("### Handled Missing Values")
        # Let user choose columns to fill
        cols_with_missing = clean_df.columns[clean_df.isnull().sum() > 0].tolist()
        if cols_with_missing:
            selected_col = st.selectbox("Select column to impute", cols_with_missing)
            col_dtype = clean_df[selected_col].dtype
            
            impute_strategy = st.radio("Choose Imputation Strategy", 
                                       ["Mean", "Median", "Mode (Most Frequent)", "Fill with 0 (Numeric)", "Fill with 'none' (Text/Categorical)", "Drop Rows", "Drop Column"])
                                       
            if st.button(f"Apply to {selected_col}"):
                if impute_strategy == "Mean":
                    if pd.api.types.is_numeric_dtype(col_dtype):
                        st.session_state.clean_data[selected_col] = st.session_state.clean_data[selected_col].fillna(st.session_state.clean_data[selected_col].mean())
                        st.session_state.changes_log.append(f"Filled missing values in '{selected_col}' with Mean.")
                        st.success("Applied Mean Imputation!")
                        st.rerun()
                    else:
                        st.error("Mean imputation only applies to numeric columns!")
                elif impute_strategy == "Median":
                    if pd.api.types.is_numeric_dtype(col_dtype):
                        st.session_state.clean_data[selected_col] = st.session_state.clean_data[selected_col].fillna(st.session_state.clean_data[selected_col].median())
                        st.session_state.changes_log.append(f"Filled missing values in '{selected_col}' with Median.")
                        st.success("Applied Median Imputation!")
                        st.rerun()
                    else:
                        st.error("Median imputation only applies to numeric columns!")
                elif impute_strategy == "Mode (Most Frequent)":
                    st.session_state.clean_data[selected_col] = st.session_state.clean_data[selected_col].fillna(st.session_state.clean_data[selected_col].mode()[0])
                    st.session_state.changes_log.append(f"Filled missing values in '{selected_col}' with Mode.")
                    st.success("Applied Mode Imputation!")
                    st.rerun()
                elif impute_strategy == "Fill with 0 (Numeric)":
                    if pd.api.types.is_numeric_dtype(col_dtype):
                        st.session_state.clean_data[selected_col] = st.session_state.clean_data[selected_col].fillna(0)
                        st.session_state.changes_log.append(f"Filled missing values in '{selected_col}' with 0.")
                        st.success("Applied Zero Imputation!")
                        st.rerun()
                    else:
                        st.error("Please use 'none' for text columns, or make sure the column is numeric first.")
                elif impute_strategy == "Fill with 'none' (Text/Categorical)":
                    st.session_state.clean_data[selected_col] = st.session_state.clean_data[selected_col].fillna("none")
                    st.session_state.changes_log.append(f"Filled missing values in '{selected_col}' with 'none'.")
                    st.success("Applied 'none' Imputation!")
                    st.rerun()
                elif impute_strategy == "Drop Rows":
                    st.session_state.clean_data = st.session_state.clean_data.dropna(subset=[selected_col])
                    st.session_state.changes_log.append(f"Dropped rows with missing values in '{selected_col}'.")
                    st.success("Dropped affected rows!")
                    st.rerun()
                elif impute_strategy == "Drop Column":
                    st.session_state.clean_data = st.session_state.clean_data.drop(columns=[selected_col])
                    st.session_state.changes_log.append(f"Dropped column '{selected_col}'.")
                    st.success("Dropped column!")
                    st.rerun()
        else:
            st.success("No missing values found in the dataset.")
            
        st.divider()

        st.markdown("### 🧹 Clean Text & Hidden Blank Spaces")
        st.markdown("Sometimes cells look empty but actually contain hidden spaces. This will find them and strip excess spaces from text.")
        if st.button("Fix Blank Spaces & Trim Text"):
            # Trim spaces string columns
            string_cols = st.session_state.clean_data.select_dtypes(include=['object', 'string']).columns
            for col in string_cols:
                st.session_state.clean_data[col] = st.session_state.clean_data[col].str.strip()
            
            # Replace completely empty strings with true NaN so our imputer can catch them
            st.session_state.clean_data.replace(r'^\s*$', np.nan, regex=True, inplace=True)
            
            st.session_state.changes_log.append("Trimmed spaces from text and converted hidden blank cells to standard Missing Values (NaN).")
            st.success("Cleaned all text columns and identified hidden blank spaces!")
            st.rerun()

        st.divider()

        st.markdown("### Fix Data Types")
        col_to_convert = st.selectbox("Select column to convert type", clean_df.columns, key='dtype_col')
        target_dtype = st.selectbox("Select target data type", ["int64", "float64", "str", "datetime64[ns]"])
        if st.button(f"Convert {col_to_convert} to {target_dtype}"):
            try:
                if target_dtype == "datetime64[ns]":
                    st.session_state.clean_data[col_to_convert] = pd.to_datetime(st.session_state.clean_data[col_to_convert])
                else:
                    st.session_state.clean_data[col_to_convert] = st.session_state.clean_data[col_to_convert].astype(target_dtype)
                
                st.session_state.changes_log.append(f"Converted column '{col_to_convert}' to {target_dtype}.")
                st.success(f"Successfully converted {col_to_convert} to {target_dtype}!")
                st.rerun()
            except Exception as e:
                st.error(f"Error converting data type: {e}")

        st.divider()

        st.markdown("### Remove Duplicates")
        if clean_df.duplicated().sum() > 0:
            if st.button("Drop Duplicate Rows"):
                initial_shape = st.session_state.clean_data.shape[0]
                st.session_state.clean_data.drop_duplicates(inplace=True)
                final_shape = st.session_state.clean_data.shape[0]
                st.session_state.changes_log.append(f"Removed {initial_shape - final_shape} duplicate rows.")
                st.success("Duplicates removed!")
                st.rerun()
        else:
            st.info("No duplicates detected.")
            
        st.divider()
        
        st.markdown("### Handle Outliers (Numeric only)")
        if numeric_cols:
            outlier_col = st.selectbox("Select column to handle outliers", numeric_cols)
            outlier_method = st.radio("Choose Method", ["IQR Method (Cap/Floor)", "Z-score (Remove >3 std)"])
            
            if st.button(f"Apply Outlier Treatment to {outlier_col}"):
                if outlier_method == "IQR Method (Cap/Floor)":
                    Q1 = st.session_state.clean_data[outlier_col].quantile(0.25)
                    Q3 = st.session_state.clean_data[outlier_col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Capping
                    st.session_state.clean_data[outlier_col] = np.where(st.session_state.clean_data[outlier_col] < lower_bound, lower_bound, st.session_state.clean_data[outlier_col])
                    st.session_state.clean_data[outlier_col] = np.where(st.session_state.clean_data[outlier_col] > upper_bound, upper_bound, st.session_state.clean_data[outlier_col])
                    
                    st.session_state.changes_log.append(f"Capped outliers in '{outlier_col}' using IQR method.")
                    st.success(f"IQR method applied to {outlier_col}!")
                    st.rerun()
                    
                elif outlier_method == "Z-score (Remove >3 std)":
                    z_scores = np.abs(stats.zscore(st.session_state.clean_data[outlier_col].dropna()))
                    # For Z-score, keep only rows where Z < 3 or it was missing. So if not missing, Z-score < 3. 
                    # Easier way:
                    valid_idx = st.session_state.clean_data.index[st.session_state.clean_data[outlier_col].isnull() | (np.abs(stats.zscore(st.session_state.clean_data[outlier_col].fillna(st.session_state.clean_data[outlier_col].mean()))) < 3)]
                    initial_shape = st.session_state.clean_data.shape[0]
                    st.session_state.clean_data = st.session_state.clean_data.loc[valid_idx]
                    removed = initial_shape - st.session_state.clean_data.shape[0]
                    
                    st.session_state.changes_log.append(f"Removed {removed} outliers from '{outlier_col}' using Z-score > 3.")
                    st.success(f"Z-score method applied to {outlier_col}! ({removed} rows removed)")
                    st.rerun()
                    
        st.divider()
        st.markdown("### Reset Data")
        if st.button("Reset to Original Content"):
            st.session_state.clean_data = st.session_state.raw_data.copy()
            st.session_state.changes_log = ["Reset to original dataset."]
            st.warning("Data has been reset.")
            st.rerun()


    # --- Tab 4: Compare & Download ---
    with tab4:
        st.header("📊 Before vs After")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Data")
            st.dataframe(raw_df.head(10))
            st.write(f"Shape: {raw_df.shape}")
            st.write(f"Total Missing: {raw_df.isnull().sum().sum()}")
            
        with col2:
            st.subheader("Cleaned Data")
            st.dataframe(clean_df.head(10))
            st.write(f"Shape: {clean_df.shape}")
            st.write(f"Total Missing: {clean_df.isnull().sum().sum()}")
            
        st.subheader("📝 Changes Made (AI Explanation)")
        if st.session_state.changes_log:
            for log in st.session_state.changes_log:
                st.markdown(f"- {log}")
        else:
            st.info("No changes made yet.")

        # Download Section
        st.subheader("📥 Download Cleaned Dataset")
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(clean_df)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='cleaned_data.csv',
            mime='text/csv',
        )

else:
    # Landing instructions
    st.info("👈 Please upload a dataset in the sidebar to get started.")
    
    st.markdown("""
    ### Why use this tool?
    - **No Code Required:** Simply point, click, and clean.
    - **Intelligent Insights:** Our AI engine recommends the best cleaning techniques (Mean vs Median vs Drop).
    - **Transparency:** "Explains what it did" - Keeps track of all your changes in the log!
    """)

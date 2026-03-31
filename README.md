# 🚀 NOVASHU.AI - Automated Data Cleaning AI Tool

Welcome to **NOVASHU.AI**, your intelligent data engine built to clean, analyze, and transform messy datasets into perfectly structured, production-ready data! Say goodbye to manual Excel filtering and endless Python scripts—our AI brain detects the problems and suggests the solution instantly.

---

## ✨ Core Features
- 📂 **Multi-format Dataset Upload**: Seamlessly import both `.csv` and `.xlsx` files.
- 🔍 **Automated EDA (Exploratory Data Analysis)**: Instantly generate statistical summaries, detect data types, missing value percentages, and even plot live histograms and boxplots.
- 🤖 **"AI Brain" Suggestions**: Unique built-in intelligence that automatically scans data for skewness, exact outlier percentages, and distribution abnormalities, recommending precise standardizations.
- 🧹 **Smart Data Cleaning Engine**:
  - Implement Missing Value Imputations (Mean, Median, Mode, `0`, or `'none'`).
  - Strip hidden whitespaces and convert empty cells manually into detectable missing values.
  - Automatically drop blank rows/columns or strip pure duplicates.
  - Type-cast columns instantly (`int64`, `float64`, `str`, `datetime`).
  - Cap and Floor extreme Outliers using advanced IQR or Z-score ( > 3 std ) methodologies.
- 📊 **Before vs After Visual Comparison**: Automatically tracks exactly what actions were taken via a live log, and compares original shapes and values against your newly cleaned data.
- 📥 **One-Click Export**: Save your polished data locally in one click to feed into your Machine Learning models.

---

## 💻 Tech Stack
- **Frontend/UI:** [Streamlit](https://streamlit.io/) (for a highly interactive web app)
- **Data Engine Frameworks:** `pandas`, `numpy`, `scipy`
- **Visualization:** `matplotlib`, `seaborn`

---

## 🛠️ Installation & Setup

1. **Clone or Download the Repository**
2. **Setup your environment:**
   Make sure you have Python 3.8+ installed. It is highly recommended to use a virtual environment.
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Server:**
   ```bash
   python -m streamlit run app.py
   ```
5. **Access the App:** Open the `Local URL` provided in the terminal (usually `http://localhost:8501`) directly in your browser.

---

## 🎨 Design & Branding setup
This dashboard is strictly customized with a premium dark-mode aesthetic. 
If you wish to display your custom brand logo, simply name your image file `logo.png` (or `.jpg`) and ensure it's saved in the exact root folder of the project alongside `app.py`. The system will automatically place it perfectly bounded inside the main styling.

---

*Powered by NOVASHU.AI. Intelligent data preparation with zero code.*


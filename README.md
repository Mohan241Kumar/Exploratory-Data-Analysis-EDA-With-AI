## EDA with AI

**EDA with AI** is an interactive multi-page Streamlit application for **exploratory data analysis**. It combines industry-standard profiling libraries with rich 2D/3D visualizations to help you quickly understand and communicate insights from your datasets—without writing code.

### ✨ Key Features

- **Automated Profiling**
  - Pandas Profiling (ydata-profiling)
  - Sweetviz reports
  - AutoViz visualizations with smart fallbacks
  - Quick summaries of dataset size, types, missing data, and basic statistics

- **3D Visualizations**
  - 3D scatter, surface, line, bar, mesh, contour, volume, density, heatmap, and parallel coordinates
  - PCA-based 3D visualization with explained variance
  - 3D clustering using K-Means and PCA
  - Sampling options for large datasets and export to HTML/PNG

- **Data Quality Insights**
  - Missing value analysis and heatmaps
  - Correlation heatmaps for numeric features
  - Categorical and numeric distribution overviews
  - Data type distribution and memory usage

### 🧰 Tech Stack

- **Python**
- **Streamlit** (multi-page app)
- **Plotly** (interactive plots)
- **ydata-profiling**, **Sweetviz**, **AutoViz**
- **scikit-learn** (PCA, K-Means)
- **NumPy**, **Pandas**

### 🚀 Getting Started

1. **Clone the repository**

git clone <YOUR_REPO_URL>
cd "EDA with ai"

2. **Create and activate a virtual environment (optional but recommended)**
python -m venv .venv
.\.venv\Scripts\activate  # Windows
or
source .venv/bin/activate  # macOS/Linux

3. **Install dependencies**
pip install -r requirements.txt
(If you don’t have a requirements.txt yet, export your current environment or list the libraries used in the project.)

3.**Run the app**
streamlit run app.py
If your main entry file has a different name (e.g. Home.py or main.py), replace app.py accordingly.

4. **Open in browser**
Streamlit will print a local URL (usually http://localhost:8501). Open it in your browser, upload a CSV file on the main page, and start exploring.

📂 Project Structure (example)
.
├─ pages/
│  ├─ 1_Automated_Profiling.py
│  ├─ 9_3D_Visualizations.py
│  └─ ...
├─ .streamlit/
│  └─ config.toml
├─ app.py  (or main entry file)
├─ requirements.txt
└─ README.md

💡 Usage Tips
Start from the main page to upload your dataset (CSV).
Use Automated Profiling to get a high-level overview and detailed HTML reports.
Explore 3D Visualizations to uncover patterns, clusters, and relationships in three dimensions.
Export plots and summaries as HTML, PNG, JSON, or CSV for reporting and sharing.

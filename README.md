# DECODE-MIDAS: Multimorbidity in Intellectual Disability Analysis System

A Streamlit application for analysing disease trajectory patterns using population-level healthcare data from CPRD and SAIL datasets.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python -m streamlit run app.py
   ```

## Project Structure

```
personalised_mltc/
├── app.py
├── requirements.txt
├── decode-logo.png
├── README.md
├── data/
│   ├── raw/
│   ├── preprocessed/
│   ├── individual_conditions/
│   └── decode-logo.png
├── modules/
│   ├── __init__.py
│   ├── analysis.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── ui_tabs.py
│   ├── utils.py
│   └── visualizations.py
├── tools/
│   ├── preprocess_data.py
│   └── extract_individual_conditions.py
└── notebooks/
    ├── DECODECohortPaper.ipynb
    ├── Personalisation_Paper.ipynb
    ├── PowerCalc.ipynb
    └── app.ipynb
```

## Directory Structure

### `/data/` - Data Organisation

The data directory is organised in three subdirectories:

- **`data/raw/`** - Original, unmodified CSV files containing condition pair data from CPRD and SAIL datasets
- **`data/preprocessed/`** - Processed versions with enhanced text cleaning and standardisation (includes expanded abbreviations like "Chronic Kidney Disease" for "CKD")  
- **`data/individual_conditions/`** - Extracted individual condition counts and prevalences derived from the pair data

### `/modules/` - Core Application Logic

Contains the modular components that power the Streamlit application:

- **`analysis.py`** - Statistical analysis functions (sensitivity analysis, condition combinations)
- **`config.py`** - System colour mappings and condition categorisations 
- **`data_loader.py`** - Data loading with preprocessing pipeline integration
- **`preprocessing.py`** - Text standardisation and abbreviation expansion
- **`ui_tabs.py`** - Individual tab rendering logic for the Streamlit interface
- **`utils.py`** - Utility functions for data parsing and manipulation
- **`visualizations.py`** - Network graphs, statistical plots, and HTML report generation

### `/tools/` - Data Processing Scripts

**Important: All tools should be run from the parent directory using `python tools/script_name.py`**

- **`preprocess_data.py`** - Batch preprocessing script that:
  - Reads all CSV files from `data/raw/`
  - Applies enhanced text processing (including Precedence column cleaning)  
  - Saves results to `data/preprocessed/` with `_preprocessed` suffix
  
- **`extract_individual_conditions.py`** - Condition extraction script that:
  - Processes files from `data/preprocessed/`
  - Extracts individual condition counts and prevalences from pair data
  - Saves results to `data/individual_conditions/`

**Usage examples:**
```bash
# Process all raw data files
python tools/preprocess_data.py

# Extract individual conditions from processed data  
python tools/extract_individual_conditions.py
```

### `/notebooks/` - Supplementary Analysis

Contains Jupyter notebooks for additional analysis and research:

- **`DECODECohortPaper.ipynb`** - Cohort analysis for research publication
- **`Personalisation_Paper.ipynb`** - Personalisation methodology development
- **`PowerCalc.ipynb`** - Statistical power calculations
- **`app.ipynb`** - Notebook version of the application for development

## Application Features

The Streamlit application provides several analysis modes:

1. **Sensitivity Analysis** - Explore how odds ratio thresholds affect trajectory discovery
2. **Condition Combinations** - Identify prevalent condition patterns in the population  
3. **Personalised Analysis** - Generate patient-specific trajectory reports
4. **Trajectory Filter** - Custom network visualisation with filtering options
5. **Cohort Network** - Population-level condition relationship networks
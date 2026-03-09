# Predicting Respiratory Disease Hospitalization Surges
## Using Machine Learning on U.S. Hospital Data (2020-2024)

This tutorial demonstrates how to apply machine learning to predict respiratory illness hospitalization surges using real-world healthcare data from the U.S. Department of Health and Human Services (HHS).

## 📋 Overview

- **Objective**: Predict whether hospitals will experience a surge in respiratory illness hospitalizations in the upcoming week
- **Dataset**: HHS Weekly Hospital Respiratory Data (2020-2024) - weekly state-level reporting on COVID-19, Influenza, and RSV hospitalizations
- **Approach**: Time-series classification using engineered features from historical hospital data
- **Models**: Logistic Regression, Random Forest, and Gradient Boosting

## 🚀 Quick Start

### Option 1: Run as Python Script
```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Run the complete tutorial
python tutorial_respiratory_ml.py
```

### Option 2: Run as Jupyter Notebook
```bash
# Install Jupyter if not already installed
pip install jupyter

# Start Jupyter and open the notebook
jupyter notebook tutorial_respiratory_ml.ipynb
```

### Option 3: Run with Docker
```bash
# Build the Docker image
docker build -t respiratory-ml-tutorial .

# Run the container
docker run -p 8888:8888 respiratory-ml-tutorial
```

## 📁 Project Structure

```
├── tutorial_respiratory_ml.py          # Complete Python script
├── tutorial_respiratory_ml.ipynb       # Jupyter notebook version
├── raw_weekly_hospital_respiratory_data_2020_2024.csv  # Dataset
├── respiratory_dashboard.jsx           # React dashboard (optional)
├── tutorial_presentation.pptx         # Presentation slides
├── 01_national_trends.png             # Generated visualization
├── 02_correlation_matrix.png          # Generated visualization
├── 03_model_comparison.png            # Generated visualization
├── 04_feature_importance.png          # Generated visualization
├── 05_confusion_matrix.png            # Generated visualization
└── README.md                          # This file
```

## 🛠️ Requirements

### Python Libraries
```bash
pip install pandas>=1.5.0
pip install numpy>=1.24.0
pip install scikit-learn>=1.3.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
```

### Optional (for dashboard)
```bash
pip install jupyter>=1.0.0
pip install ipywidgets>=8.0.0
```

## 📊 What the Script Does

1. **Data Loading & Cleaning**: Loads HHS hospital data, handles missing values, removes territories with sparse data
2. **Exploratory Analysis**: Creates national-level trends and correlation analysis visualizations
3. **Feature Engineering**: Creates time-series features including:
   - Lagged hospitalization counts (1-4 weeks)
   - Rolling averages (2-4 week windows)
   - Week-over-week change rates
   - Bed occupancy ratios
   - Seasonal indicators
4. **Model Training**: Trains three ML models with temporal validation
5. **Evaluation**: Compares models using ROC curves, precision-recall, and confusion matrices
6. **Interpretation**: Analyzes feature importance and provides clinical context

## 🎯 Learning Objectives

After completing this tutorial, you will be able to:
- Load, clean, and explore a real healthcare dataset
- Engineer meaningful time-series features for prediction
- Frame a healthcare question as a supervised classification problem
- Train and evaluate multiple ML models (Logistic Regression, Random Forest, Gradient Boosting)
- Interpret model results in a clinical/public health context
- Understand ethical considerations of predictive models in healthcare

## 📈 Generated Outputs

The script generates five visualization files:
- `01_national_trends.png` - National respiratory illness trends over time
- `02_correlation_matrix.png` - Correlation between different metrics
- `03_model_comparison.png` - ROC curves and model performance comparison
- `04_feature_importance.png` - Most important predictive features
- `05_confusion_matrix.png` - Confusion matrix for the best model

## 🔧 Running the Tutorial Step-by-Step

### For Python Script Users:
The script is divided into clearly marked sections. You can run it entirely or section by section:

```python
# Run specific sections by uncommenting the desired section
# SECTION 1: IMPORTS AND SETUP
# SECTION 2: DATA LOADING AND INITIAL EXPLORATION
# SECTION 3: DATA CLEANING AND PREPROCESSING
# ... etc
```

### For Jupyter Users:
1. Open `tutorial_respiratory_ml.ipynb` in Jupyter
2. Execute cells sequentially using Shift+Enter or the "Run" button
3. Each section has detailed markdown explanations

## 📊 Dataset Information

- **Source**: U.S. Department of Health and Human Services (HHS)
- **URL**: https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/g62h-syeh
- **Time Period**: 2020-2024 (weekly data)
- **Geographic Coverage**: All U.S. states and territories
- **Key Metrics**: Hospitalization counts, bed occupancy, ICU utilization

## 🎯 Key Results

The tutorial demonstrates that:
- **Lagged hospitalization counts** (especially 1-2 week lags) are the strongest predictors
- **Seasonal patterns** capture the well-known winter respiratory season
- **Gradient Boosting** typically performs best for this classification task
- **Feature importance** analysis reveals the most predictive metrics for surge prediction

## ⚠️ Important Notes

- The script expects the CSV file to be in the same directory
- Generated plots are saved as PNG files in the current directory
- The model uses temporal splitting (train on past data, test on future data) to avoid data leakage
- A "surge" is defined as a >10% week-over-week increase in respiratory hospitalizations

## 🔍 Troubleshooting

### Common Issues:
1. **File not found**: Ensure `raw_weekly_hospital_respiratory_data_2020_2024.csv` is in the same directory
2. **Memory issues**: The dataset is ~6MB - ensure sufficient RAM
3. **Library conflicts**: Use a virtual environment: `python -m venv venv && source venv/bin/activate`

### Performance Tips:
- The script uses all available CPU cores for Random Forest training
- For faster development, reduce `n_estimators` in model definitions
- Use the Jupyter notebook version for interactive exploration

## 📚 Additional Resources

- [HHS Health Data](https://healthdata.gov/) - Source dataset
- [Scikit-learn Documentation](https://scikit-learn.org/) - ML library
- [CDC Respiratory Illness Data](https://www.cdc.gov/) - Additional health data

## 🤝 Contributing

This is an educational tutorial. Feel free to:
- Extend the analysis with additional features
- Try different ML algorithms
- Apply the approach to other healthcare datasets
- Improve the visualizations

## 📄 License

This tutorial is provided for educational purposes. The dataset is subject to the terms of use specified by healthdata.gov.

---

**Happy Learning! 🎉** 

For questions or issues, please refer to the inline comments in the code or the detailed markdown explanations in the Jupyter notebook.

# Mental Health Analysis Dashboard

A comprehensive mental health analysis platform designed for students, providing data-driven insights and personalized recommendations using machine learning algorithms.

## Overview

This dashboard analyzes various factors affecting students' mental health, including academic performance, study patterns, and lifestyle factors, to predict potential mental health concerns like Depression, Anxiety, and Panic Attacks.

## Features

### 1. Mental Health Assessment

- Personal information input
- Academic & lifestyle factor analysis
- Real-time risk assessment for:
  - Depression
  - Anxiety
  - Panic Attacks
- Personalized recommendations based on assessment results

### 2. Personal Insights

- Academic Performance Analysis
- Key Influencing Factors visualization
- Feature importance analysis for mental health indicators
- Interactive data visualizations

### 3. Dataset Analysis

- Comprehensive statistical analysis
- Mental Health Distribution Analysis
- Course-wise and Year-wise trends
- Key correlations and insights
- Interactive data exploration

## Technology Stack

- Python 3.x
- Streamlit for web interface
- XGBoost for machine learning
- Plotly for interactive visualizations
- Pandas & NumPy for data processing
- Scikit-learn for model evaluation

## Installation & Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/mental-health-analysis.git
cd mental-health-analysis
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

## Project Structure

mental-health-analysis/
├── app.py # Main Streamlit application
├── data/ # Data directory
│ └── mental_health_data.csv
├── src/ # Source code
│ ├── init.py
│ ├── model.py # Model definitions
│ └── utils.py # Utility functions
├── models/ # Model storage
│ └── saved_models/ # Saved model files
└── requirements.txt # Project dependencies

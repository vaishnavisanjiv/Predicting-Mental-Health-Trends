# Mental Health Analysis Dashboard

This project provides an interactive dashboard for analyzing student mental health data and providing personalized insights and recommendations.

## Features

- Interactive mental health assessment form
- Personalized insights and visualizations
- Dataset analysis and statistics
- Recommendations based on assessment results

## Project Structure

```
mental-health-analysis/
├── data/
│   └── mental_health_data.csv
├── src/
│   ├── __init__.py
│   ├── dashboard.py
│   └── utils.py
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── models/
│   └── saved_models/
├── requirements.txt
├── README.md
└── run_dashboard.py
```

## Setup Instructions

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the dashboard:

```bash
streamlit run run_dashboard.py
```

## Dashboard Features

### 1. Mental Health Assessment

- Interactive form to collect user information
- Demographics, academic factors, and lifestyle information
- Mental health indicators and support system status

### 2. Personal Insights

- Visualizations comparing user data with dataset
- Personalized recommendations
- Progress tracking

### 3. Dataset Analysis

- Overview of the dataset
- Statistical analysis
- Correlation analysis
- Distribution visualizations

## Data Preprocessing

The project includes data preprocessing utilities in `src/utils.py`:

- Categorical variable encoding
- Feature scaling
- Missing value handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

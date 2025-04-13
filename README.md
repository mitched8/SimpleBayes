# Simple Bayesian Probability Calculator

This is a Streamlit application that demonstrates Bayesian probability calculations with interactive examples.

## Setup and Run Instructions

### 1. Clone the repository
```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create and activate a virtual environment

#### For macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application
```bash
streamlit run app.py
```

The app should open automatically in your default web browser. If it doesn't, you can access it at http://localhost:8501.

## Features

The application provides six different Bayesian probability scenarios:

1. **Medical Test Scenario**: Calculate the probability that a patient has a disease given a positive test result.
   - Set disease prevalence, test sensitivity, and test specificity
   - See how these factors affect the posterior probability

2. **Weather Prediction**: Calculate the probability of rain tomorrow based on observed cloud patterns.
   - Set base probability of rain and likelihood of cloud observations
   - See how new evidence changes the probability of rain

3. **Custom Example**: Define your own hypotheses and probabilities.
   - Set custom hypotheses names
   - Adjust prior probabilities and likelihoods
   - Calculate posterior probabilities and Bayes factors

4. **Real World Data Analysis**: Apply Bayesian inference to the Iris dataset.
   - Analyze how petal length can predict flower species
   - See prior and posterior probabilities for each species
   - Explore feature distributions and their impact on classification
   - Understand how Bayesian methods work with real-world data

5. **A/B Testing Calculator**: Analyze A/B test results using Bayesian statistics.
   - Input visitors and conversion counts for control and treatment variants
   - Set prior beliefs about conversion rates
   - Calculate the probability that B is better than A
   - Visualize prior and posterior distributions
   - Get recommendations based on the results
   - Estimate required sample size for conclusive tests

6. **Time Series Analysis**: Apply Bayesian methods to time series data.
   - Detect change points in time series data
   - Generate Bayesian forecasts with credible intervals
   - Decompose time series into trend, seasonal, and remainder components
   - Upload custom data or use synthetic examples
   - Analyze with various Bayesian time series models

Each scenario includes:
- Interactive sliders to adjust parameters
- Clear tables showing prior and posterior probabilities
- Visualization comparing prior vs posterior probabilities
- Interpretation of the results

## About Bayesian Inference

Bayes' theorem calculates the probability of a hypothesis given some evidence:

P(H|E) = [P(E|H) × P(H)] / P(E)

Where:
- P(H|E) is the posterior probability
- P(E|H) is the likelihood
- P(H) is the prior probability
- P(E) is the evidence

## Requirements

- Python 3.8 or higher
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- SciPy 

## Project Structure

```
SimpleBayes/
├── app.py              # Main application entry point
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
├── utils/              # Utility functions and helpers
│   ├── __init__.py
│   └── bayes_utils.py  # Common Bayesian calculation utilities
└── scenarios/          # Individual scenario implementations
    ├── __init__.py
    ├── medical_test.py
    ├── weather_prediction.py
    ├── time_series.py
    └── ... (other scenarios)
``` 
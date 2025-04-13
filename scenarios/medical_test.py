import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from utils.bayes_utils import calculate_posterior

def render_medical_test():
    """Render the Medical Test scenario page"""
    st.header("Medical Test Scenario")
    st.markdown("""
    In this scenario, we're calculating the probability that a patient has a disease given a positive test result.
    - **Prior**: The base rate of the disease in the population
    - **Sensitivity**: Probability of a positive test if the patient has the disease (true positive rate)
    - **Specificity**: Probability of a negative test if the patient doesn't have the disease (true negative rate)
    """)
    
    # User inputs for medical test scenario
    disease_rate = st.slider("Disease prevalence in population (%)", 0.1, 20.0, 1.0, 0.1) / 100
    sensitivity = st.slider("Test sensitivity (%)", 50.0, 100.0, 95.0, 0.1) / 100
    specificity = st.slider("Test specificity (%)", 50.0, 100.0, 90.0, 0.1) / 100
    
    # Calculate false positive rate
    false_positive_rate = 1 - specificity
    
    # Prior probabilities
    prior_probs = {
        "Has Disease": disease_rate,
        "No Disease": 1 - disease_rate
    }
    
    # Likelihoods of positive test result
    likelihoods = {
        "Has Disease": sensitivity,
        "No Disease": false_positive_rate
    }
    
    # Calculate posteriors
    posteriors = calculate_posterior(prior_probs, likelihoods)
    
    # Create a DataFrame for display
    data = {
        "Hypothesis": ["Has Disease", "No Disease"],
        "Prior Probability": [prior_probs["Has Disease"], prior_probs["No Disease"]],
        "Likelihood of Positive Test": [likelihoods["Has Disease"], likelihoods["No Disease"]],
        "Posterior Probability": [posteriors["Has Disease"], posteriors["No Disease"]]
    }
    
    df = pd.DataFrame(data)
    
    # Display the table
    st.subheader("Bayesian Analysis Results")
    st.dataframe(df.style.format({
        "Prior Probability": "{:.4f}",
        "Likelihood of Positive Test": "{:.4f}",
        "Posterior Probability": "{:.4f}"
    }))
    
    # Display the conclusion
    st.markdown(f"""
    ### Interpretation
    
    If a patient tests positive, the probability they actually have the disease is **{posteriors["Has Disease"]:.2%}**.
    
    This demonstrates Bayes' theorem in action:
    
    P(Disease|Positive) = P(Positive|Disease) × P(Disease) / P(Positive)
    """)
    
    # Visualization of prior vs posterior
    st.subheader("Prior vs Posterior Probabilities")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        x=[0, 1, 3, 4], 
        height=[prior_probs["Has Disease"], prior_probs["No Disease"], 
                posteriors["Has Disease"], posteriors["No Disease"]],
        width=0.6,
        color=["#ff9999", "#66b3ff", "#ff9999", "#66b3ff"]
    )
    
    # Add labels and values on the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    ax.set_xticks([0.5, 3.5])
    ax.set_xticklabels(["Prior", "Posterior"])
    ax.set_ylabel("Probability")
    ax.set_title("Prior vs Posterior Probabilities")
    ax.set_ylim(0, 1.1)
    
    # Add a legend
    legend_elements = [
        Patch(facecolor='#ff9999', label='Has Disease'),
        Patch(facecolor='#66b3ff', label='No Disease')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    st.pyplot(fig) 
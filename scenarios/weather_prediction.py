import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from utils.bayes_utils import calculate_posterior

def render_weather_prediction():
    """Render the Weather Prediction scenario page"""
    st.header("Weather Prediction Scenario")
    st.markdown("""
    In this scenario, we're predicting whether it will rain tomorrow based on observed cloud patterns.
    - **Prior**: The base probability of rain on any given day
    - **Likelihood**: The probability of seeing clouds given different weather outcomes
    """)
    
    # User inputs for weather scenario
    rain_rate = st.slider("Base probability of rain on any day (%)", 10.0, 70.0, 30.0, 0.1) / 100
    clouds_if_rain = st.slider("Probability of clouds if it will rain (%)", 50.0, 100.0, 90.0, 0.1) / 100
    clouds_if_no_rain = st.slider("Probability of clouds if it won't rain (%)", 5.0, 80.0, 40.0, 0.1) / 100
    
    # Prior probabilities
    prior_probs = {
        "Rain": rain_rate,
        "No Rain": 1 - rain_rate
    }
    
    # Likelihoods of observing clouds
    likelihoods = {
        "Rain": clouds_if_rain,
        "No Rain": clouds_if_no_rain
    }
    
    # Calculate posteriors
    posteriors = calculate_posterior(prior_probs, likelihoods)
    
    # Create a DataFrame for display
    data = {
        "Hypothesis": ["Rain Tomorrow", "No Rain Tomorrow"],
        "Prior Probability": [prior_probs["Rain"], prior_probs["No Rain"]],
        "Likelihood of Clouds Today": [likelihoods["Rain"], likelihoods["No Rain"]],
        "Posterior Probability": [posteriors["Rain"], posteriors["No Rain"]]
    }
    
    df = pd.DataFrame(data)
    
    # Display the table
    st.subheader("Bayesian Analysis Results")
    st.dataframe(df.style.format({
        "Prior Probability": "{:.4f}",
        "Likelihood of Clouds Today": "{:.4f}",
        "Posterior Probability": "{:.4f}"
    }))
    
    # Display the conclusion
    st.markdown(f"""
    ### Interpretation
    
    If you observe clouds today, the probability it will rain tomorrow is **{posteriors["Rain"]:.2%}**.
    
    This demonstrates how new evidence (cloud observation) updates our belief about the probability of rain.
    """)
    
    # Visualization of prior vs posterior
    st.subheader("Prior vs Posterior Probabilities")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        x=[0, 1, 3, 4], 
        height=[prior_probs["Rain"], prior_probs["No Rain"], 
                posteriors["Rain"], posteriors["No Rain"]],
        width=0.6,
        color=["#9dc6ff", "#ffcf9e", "#9dc6ff", "#ffcf9e"]
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
        Patch(facecolor='#9dc6ff', label='Rain'),
        Patch(facecolor='#ffcf9e', label='No Rain')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    st.pyplot(fig) 
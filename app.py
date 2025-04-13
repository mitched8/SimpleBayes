import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from scipy import stats

# Set page configuration
st.set_page_config(
    page_title="Simple Bayes Calculator",
    page_icon="📊",
    layout="wide"
)

# Main title and description
st.title("Bayesian Probability Calculator")
st.markdown("This application demonstrates simple Bayesian probability calculations with synthetic data.")

# Function to calculate posterior probabilities using Bayes' theorem
def calculate_posterior(prior_probs, likelihoods):
    """
    Calculate posterior probabilities using Bayes' theorem
    
    Parameters:
    - prior_probs: Dictionary of hypotheses and their prior probabilities
    - likelihoods: Dictionary of likelihoods of data given each hypothesis
    
    Returns:
    - Dictionary of posterior probabilities
    """
    # Calculate evidence (denominator in Bayes' theorem)
    evidence = sum(prior_probs[h] * likelihoods[h] for h in prior_probs)
    
    # Calculate posterior for each hypothesis
    posteriors = {}
    for hypothesis in prior_probs:
        posteriors[hypothesis] = (prior_probs[hypothesis] * likelihoods[hypothesis]) / evidence
    
    return posteriors

# Sidebar with parameters
st.sidebar.header("Bayesian Parameters")

# Scenario selection
scenario = st.sidebar.selectbox(
    "Select a scenario",
    ["Medical Test", "Weather Prediction", "Custom Example", "Real World Data Analysis", "A/B Testing Calculator"]
)

# Main content based on the selected scenario
if scenario == "Medical Test":
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
    comparison_df = pd.DataFrame({
        "Probability Type": ["Prior", "Prior", "Posterior", "Posterior"],
        "Hypothesis": ["Has Disease", "No Disease", "Has Disease", "No Disease"],
        "Probability": [
            prior_probs["Has Disease"], 
            prior_probs["No Disease"],
            posteriors["Has Disease"],
            posteriors["No Disease"]
        ]
    })
    
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
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff9999', label='Has Disease'),
        Patch(facecolor='#66b3ff', label='No Disease')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    st.pyplot(fig)

elif scenario == "Weather Prediction":
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
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#9dc6ff', label='Rain'),
        Patch(facecolor='#ffcf9e', label='No Rain')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    st.pyplot(fig)

elif scenario == "Custom Example":
    st.header("Custom Bayesian Example")
    st.markdown("""
    This is a custom example where you can define your own hypotheses and their probabilities.
    - Define two competing hypotheses
    - Set their prior probabilities
    - Set the likelihood of observing the data under each hypothesis
    """)
    
    # User inputs for custom scenario
    col1, col2 = st.columns(2)
    
    with col1:
        hyp1_name = st.text_input("Name for Hypothesis 1", "Hypothesis A")
        hyp1_prior = st.slider(f"Prior probability of {hyp1_name} (%)", 0.0, 100.0, 50.0, 0.1) / 100
        hyp1_likelihood = st.slider(f"Likelihood of data given {hyp1_name} (%)", 0.0, 100.0, 80.0, 0.1) / 100
    
    with col2:
        hyp2_name = st.text_input("Name for Hypothesis 2", "Hypothesis B")
        hyp2_prior = 1 - hyp1_prior  # Must sum to 1
        st.write(f"Prior probability of {hyp2_name}: {hyp2_prior:.1%}")
        hyp2_likelihood = st.slider(f"Likelihood of data given {hyp2_name} (%)", 0.0, 100.0, 20.0, 0.1) / 100
    
    # Prior probabilities
    prior_probs = {
        hyp1_name: hyp1_prior,
        hyp2_name: hyp2_prior
    }
    
    # Likelihoods
    likelihoods = {
        hyp1_name: hyp1_likelihood,
        hyp2_name: hyp2_likelihood
    }
    
    # Calculate posteriors
    posteriors = calculate_posterior(prior_probs, likelihoods)
    
    # Create a DataFrame for display
    data = {
        "Hypothesis": [hyp1_name, hyp2_name],
        "Prior Probability": [prior_probs[hyp1_name], prior_probs[hyp2_name]],
        "Likelihood of Data": [likelihoods[hyp1_name], likelihoods[hyp2_name]],
        "Posterior Probability": [posteriors[hyp1_name], posteriors[hyp2_name]]
    }
    
    df = pd.DataFrame(data)
    
    # Display the table
    st.subheader("Bayesian Analysis Results")
    st.dataframe(df.style.format({
        "Prior Probability": "{:.4f}",
        "Likelihood of Data": "{:.4f}",
        "Posterior Probability": "{:.4f}"
    }))
    
    # Bayes factor
    bayes_factor = likelihoods[hyp1_name] / likelihoods[hyp2_name]
    posterior_odds = posteriors[hyp1_name] / posteriors[hyp2_name]
    
    st.markdown(f"""
    ### Interpretation
    
    - **Bayes Factor** (likelihood ratio): {bayes_factor:.2f}
    - **Posterior Odds**: {posterior_odds:.2f}
    
    After observing the data, the posterior probability of {hyp1_name} is **{posteriors[hyp1_name]:.2%}**.
    """)
    
    # Visualization of prior vs posterior
    st.subheader("Prior vs Posterior Probabilities")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        x=[0, 1, 3, 4], 
        height=[prior_probs[hyp1_name], prior_probs[hyp2_name], 
                posteriors[hyp1_name], posteriors[hyp2_name]],
        width=0.6,
        color=["#b3d1ff", "#ffb3b3", "#b3d1ff", "#ffb3b3"]
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
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#b3d1ff', label=hyp1_name),
        Patch(facecolor='#ffb3b3', label=hyp2_name)
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    st.pyplot(fig)

elif scenario == "Real World Data Analysis":
    st.header("Real World Data Analysis: Iris Dataset")
    st.markdown("""
    This scenario applies Bayesian inference to the well-known Iris flower dataset.
    
    We'll calculate the probability of a flower belonging to a specific species given its petal length.
    
    - **Prior**: The proportion of each species in the dataset
    - **Likelihood**: The probability of observing a certain petal length given the species
    - **Posterior**: The updated probability of the species given the observed petal length
    """)
    
    # Load the Iris dataset
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    # Display dataset information
    with st.expander("View Dataset Information"):
        st.write("Iris Dataset Shape:", iris_df.shape)
        st.write("First 5 rows of the dataset:")
        st.dataframe(iris_df.head())
        st.write("Summary statistics:")
        st.dataframe(iris_df.describe())
    
    # User input for petal length
    col1, col2 = st.columns(2)
    
    with col1:
        # Let user select a petal length to analyze
        petal_length = st.slider("Select a petal length (cm)", 
                                 float(iris_df["petal length (cm)"].min()),
                                 float(iris_df["petal length (cm)"].max()),
                                 float(iris_df["petal length (cm)"].median()),
                                 0.1)
    
    with col2:
        # Display distribution of petal lengths
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=iris_df, x="petal length (cm)", hue="species", bins=20, kde=True, ax=ax)
        ax.axvline(petal_length, color='red', linestyle='--', linewidth=2)
        ax.set_title("Distribution of Petal Lengths by Species")
        ax.text(petal_length + 0.1, ax.get_ylim()[1] * 0.9, f"Selected: {petal_length:.1f} cm", 
                verticalalignment='top', horizontalalignment='left', color='red')
        st.pyplot(fig)
    
    # Calculate priors (proportion of each species in dataset)
    species_counts = iris_df['species'].value_counts()
    prior_probs = {species: count/len(iris_df) for species, count in species_counts.items()}
    
    # Calculate likelihoods using a normal distribution approximation for each species
    likelihoods = {}
    
    for species in iris.target_names:
        # Get petal lengths for this species
        species_petal_lengths = iris_df[iris_df["species"] == species]["petal length (cm)"]
        
        # Fit a normal distribution to the data
        mean = species_petal_lengths.mean()
        std = species_petal_lengths.std()
        
        # Calculate likelihood using normal PDF
        likelihood = stats.norm.pdf(petal_length, mean, std)
        likelihoods[species] = likelihood
    
    # Calculate posteriors
    posteriors = calculate_posterior(prior_probs, likelihoods)
    
    # Create a DataFrame for display
    data = {
        "Species": list(prior_probs.keys()),
        "Prior Probability": list(prior_probs.values()),
        "Likelihood of Petal Length": list(likelihoods.values()),
        "Posterior Probability": list(posteriors.values())
    }
    
    df = pd.DataFrame(data)
    
    # Display the table
    st.subheader("Bayesian Analysis Results")
    st.dataframe(df.style.format({
        "Prior Probability": "{:.4f}",
        "Likelihood of Petal Length": "{:.6f}",
        "Posterior Probability": "{:.4f}"
    }))
    
    # Find the most likely species
    most_likely_species = max(posteriors, key=posteriors.get)
    
    # Display the conclusion
    st.markdown(f"""
    ### Interpretation
    
    Given a petal length of **{petal_length:.1f} cm**, the flower is most likely a **{most_likely_species}** 
    with a probability of **{posteriors[most_likely_species]:.2%}**.
    
    This demonstrates how we can use observed characteristics (petal length) to update our beliefs about 
    the species classification.
    """)
    
    # Visualization of prior vs posterior
    st.subheader("Prior vs Posterior Probabilities")
    
    # Create a DataFrame for easier plotting
    comparison_data = []
    for species in prior_probs:
        comparison_data.append({"Probability Type": "Prior", "Species": species, "Probability": prior_probs[species]})
        comparison_data.append({"Probability Type": "Posterior", "Species": species, "Probability": posteriors[species]})
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"setosa": "#ff9999", "versicolor": "#66b3ff", "virginica": "#99ff99"}
    
    # Get x positions for bars
    species_list = list(prior_probs.keys())
    x = np.arange(len(species_list))
    width = 0.35
    
    # Plot prior and posterior bars
    prior_bars = ax.bar(x - width/2, [prior_probs[s] for s in species_list], width, label='Prior', alpha=0.7)
    posterior_bars = ax.bar(x + width/2, [posteriors[s] for s in species_list], width, label='Posterior', alpha=0.7)
    
    # Add labels and legend
    ax.set_xlabel('Species')
    ax.set_ylabel('Probability')
    ax.set_title('Prior vs Posterior Probabilities by Species')
    ax.set_xticks(x)
    ax.set_xticklabels(species_list)
    ax.legend()
    
    # Add values on bars
    for bars in [prior_bars, posterior_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    st.pyplot(fig)
    
    # Additional visualization: feature distributions by species
    st.subheader("Exploring Feature Distributions by Species")
    
    feature_option = st.selectbox(
        "Select another feature to compare with petal length:",
        ["sepal length (cm)", "sepal width (cm)", "petal width (cm)"]
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create scatter plot
    sns.scatterplot(data=iris_df, x="petal length (cm)", y=feature_option, hue="species", ax=ax)
    
    # Add vertical line for selected petal length
    ax.axvline(x=petal_length, color='red', linestyle='--')
    
    # Highlight the region around the selected petal length
    rect = plt.Rectangle((petal_length-0.2, ax.get_ylim()[0]), 0.4, ax.get_ylim()[1]-ax.get_ylim()[0], 
                        color='red', alpha=0.1)
    ax.add_patch(rect)
    
    ax.set_title(f"Petal Length vs {feature_option}")
    st.pyplot(fig)

elif scenario == "A/B Testing Calculator":
    st.header("Bayesian A/B Testing Calculator")
    st.markdown("""
    This tool uses Bayesian statistics to analyze A/B test results. Unlike traditional (frequentist) methods, 
    the Bayesian approach directly answers the question: "What is the probability that variant B is better than variant A?"
    
    The calculator uses the Beta-Binomial model, which is a conjugate prior for the binomial distribution:
    - Prior distribution: Beta(α, β) represents our beliefs about conversion rates before seeing the data
    - Likelihood: Binomial distribution for observed conversions
    - Posterior distribution: Beta(α + conversions, β + non-conversions)
    """)
    
    # Input section for test results
    st.subheader("Test Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Variant A (Control)")
        visitors_a = st.number_input("Visitors A", min_value=1, value=1000, step=100)
        conversions_a = st.number_input("Conversions A", min_value=0, max_value=visitors_a, value=100, step=10)
        conversion_rate_a = conversions_a / visitors_a
        st.markdown(f"Observed conversion rate: **{conversion_rate_a:.2%}**")
    
    with col2:
        st.markdown("### Variant B (Treatment)")
        visitors_b = st.number_input("Visitors B", min_value=1, value=1000, step=100)
        conversions_b = st.number_input("Conversions B", min_value=0, max_value=visitors_b, value=120, step=10)
        conversion_rate_b = conversions_b / visitors_b
        st.markdown(f"Observed conversion rate: **{conversion_rate_b:.2%}**")
    
    # Prior settings
    st.subheader("Prior Settings")
    
    prior_option = st.radio(
        "Prior knowledge",
        ["Uninformative prior (Beta(1,1))", "Informed prior", "Custom prior"]
    )
    
    if prior_option == "Uninformative prior (Beta(1,1))":
        alpha_prior_a = 1
        beta_prior_a = 1
        alpha_prior_b = 1
        beta_prior_b = 1
        st.markdown("""
        Using an uninformative prior (Beta(1,1)) which is equivalent to a uniform distribution. 
        This means we assume all conversion rates between 0% and 100% are equally likely before seeing the data.
        """)
    
    elif prior_option == "Informed prior":
        prior_mean = st.slider("Prior mean conversion rate (%)", 0.1, 50.0, 10.0, 0.1) / 100
        prior_strength = st.slider("Prior strength (equivalent sample size)", 2, 100, 10, 2)
        
        # Calculate alpha and beta from mean and strength
        alpha_prior_a = prior_mean * prior_strength
        beta_prior_a = (1 - prior_mean) * prior_strength
        alpha_prior_b = alpha_prior_a  # Same prior for both variants
        beta_prior_b = beta_prior_a
        
        st.markdown(f"""
        Using an informed prior with mean={prior_mean:.2%} and strength={prior_strength}.
        This is equivalent to Beta({alpha_prior_a:.1f}, {beta_prior_a:.1f}), or having already observed 
        {prior_strength} visitors with a {prior_mean:.2%} conversion rate.
        """)
    
    elif prior_option == "Custom prior":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Prior for Variant A")
            alpha_prior_a = st.number_input("Alpha A", min_value=0.1, value=1.0, step=0.1)
            beta_prior_a = st.number_input("Beta A", min_value=0.1, value=1.0, step=0.1)
            prior_mean_a = alpha_prior_a / (alpha_prior_a + beta_prior_a)
            st.markdown(f"Prior mean for A: **{prior_mean_a:.2%}**")
        
        with col2:
            st.markdown("### Prior for Variant B")
            alpha_prior_b = st.number_input("Alpha B", min_value=0.1, value=1.0, step=0.1)
            beta_prior_b = st.number_input("Beta B", min_value=0.1, value=1.0, step=0.1)
            prior_mean_b = alpha_prior_b / (alpha_prior_b + beta_prior_b)
            st.markdown(f"Prior mean for B: **{prior_mean_b:.2%}**")
    
    # Calculate posterior parameters
    alpha_posterior_a = alpha_prior_a + conversions_a
    beta_posterior_a = beta_prior_a + (visitors_a - conversions_a)
    
    alpha_posterior_b = alpha_prior_b + conversions_b
    beta_posterior_b = beta_prior_b + (visitors_b - conversions_b)
    
    # Calculate posterior means
    posterior_mean_a = alpha_posterior_a / (alpha_posterior_a + beta_posterior_a)
    posterior_mean_b = alpha_posterior_b / (alpha_posterior_b + beta_posterior_b)
    
    # Generate samples for Monte Carlo estimation
    np.random.seed(42)  # For reproducibility
    samples_a = np.random.beta(alpha_posterior_a, beta_posterior_a, 100000)
    samples_b = np.random.beta(alpha_posterior_b, beta_posterior_b, 100000)
    
    # Calculate probability that B is better than A
    prob_b_better = np.mean(samples_b > samples_a)
    
    # Calculate expected lift
    expected_lift = (posterior_mean_b - posterior_mean_a) / posterior_mean_a
    
    # Calculate 95% credible intervals
    ci_a = (np.percentile(samples_a, 2.5), np.percentile(samples_a, 97.5))
    ci_b = (np.percentile(samples_b, 2.5), np.percentile(samples_b, 97.5))
    
    # Calculate probability of various lift thresholds
    lift_thresholds = [0.01, 0.02, 0.05, 0.1]  # 1%, 2%, 5%, 10%
    prob_b_better_than_a_by = {}
    for threshold in lift_thresholds:
        prob_b_better_than_a_by[threshold] = np.mean(samples_b > samples_a * (1 + threshold))
    
    # Results section
    st.subheader("Bayesian Analysis Results")
    
    # Key metrics
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric(
            label="Probability B > A",
            value=f"{prob_b_better:.2%}",
            delta=f"{prob_b_better - 0.5:.2%}" if prob_b_better != 0.5 else None,
            delta_color="normal"
        )
    
    with metrics_col2:
        st.metric(
            label="Expected Lift",
            value=f"{expected_lift:.2%}",
            delta=f"{posterior_mean_b - posterior_mean_a:.3%} absolute",
            delta_color="normal"
        )
    
    with metrics_col3:
        risk_of_loss = 1 - prob_b_better
        st.metric(
            label="Risk of Loss",
            value=f"{risk_of_loss:.2%}",
            delta=f"{0.5 - risk_of_loss:.2%}" if risk_of_loss != 0.5 else None,
            delta_color="inverse"
        )
    
    # Probability of minimum lift
    st.subheader("Probability of Achieving Minimum Lift")
    lift_cols = st.columns(len(lift_thresholds))
    
    for i, threshold in enumerate(lift_thresholds):
        with lift_cols[i]:
            st.metric(
                label=f"Prob. B > A by {threshold:.0%}",
                value=f"{prob_b_better_than_a_by[threshold]:.2%}"
            )
    
    # Visualizations
    st.subheader("Visualizations")
    
    # Prior and posterior comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot prior distributions
    x = np.linspace(0, 1, 1000)
    
    prior_a = stats.beta.pdf(x, alpha_prior_a, beta_prior_a)
    prior_b = stats.beta.pdf(x, alpha_prior_b, beta_prior_b)
    
    axes[0].plot(x, prior_a, label=f"A: Beta({alpha_prior_a:.1f}, {beta_prior_a:.1f})", color='blue', linestyle='--')
    axes[0].plot(x, prior_b, label=f"B: Beta({alpha_prior_b:.1f}, {beta_prior_b:.1f})", color='red', linestyle='--')
    axes[0].set_title("Prior Distributions")
    axes[0].set_xlabel("Conversion Rate")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot posterior distributions
    posterior_a = stats.beta.pdf(x, alpha_posterior_a, beta_posterior_a)
    posterior_b = stats.beta.pdf(x, alpha_posterior_b, beta_posterior_b)
    
    axes[1].plot(x, posterior_a, 
                label=f"A: Beta({alpha_posterior_a:.1f}, {beta_posterior_a:.1f})", 
                color='blue')
    axes[1].plot(x, posterior_b, 
                label=f"B: Beta({alpha_posterior_b:.1f}, {beta_posterior_b:.1f})", 
                color='red')
    axes[1].axvline(posterior_mean_a, color='blue', linestyle=':', alpha=0.7, 
                    label=f"Mean A: {posterior_mean_a:.2%}")
    axes[1].axvline(posterior_mean_b, color='red', linestyle=':', alpha=0.7, 
                    label=f"Mean B: {posterior_mean_b:.2%}")
    
    # Shade 95% credible intervals
    axes[1].fill_between(x, 0, posterior_a, where=(x >= ci_a[0]) & (x <= ci_a[1]), 
                        color='blue', alpha=0.1, 
                        label=f"95% CI A: [{ci_a[0]:.2%}, {ci_a[1]:.2%}]")
    axes[1].fill_between(x, 0, posterior_b, where=(x >= ci_b[0]) & (x <= ci_b[1]), 
                        color='red', alpha=0.1, 
                        label=f"95% CI B: [{ci_b[0]:.2%}, {ci_b[1]:.2%}]")
    
    axes[1].set_title("Posterior Distributions")
    axes[1].set_xlabel("Conversion Rate")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Distribution of difference
    st.subheader("Distribution of Difference (B - A)")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate differences
    diff_samples = samples_b - samples_a
    
    # Plot histogram of differences
    sns.histplot(diff_samples, kde=True, ax=ax)
    
    # Add mean and percentiles
    mean_diff = np.mean(diff_samples)
    percentile_2_5 = np.percentile(diff_samples, 2.5)
    percentile_97_5 = np.percentile(diff_samples, 97.5)
    
    ax.axvline(mean_diff, color='red', linestyle='--', 
                label=f"Mean: {mean_diff:.2%}")
    ax.axvline(0, color='black', linestyle='-', alpha=0.7, 
                label="No difference")
    ax.axvline(percentile_2_5, color='green', linestyle=':', 
                label=f"2.5%: {percentile_2_5:.2%}")
    ax.axvline(percentile_97_5, color='green', linestyle=':', 
                label=f"97.5%: {percentile_97_5:.2%}")
    
    # Shade area where B > A
    positive_mask = diff_samples > 0
    ax.fill_between(
        x=np.linspace(0, max(diff_samples), 1000),
        y1=0, 
        y2=ax.get_ylim()[1],
        color='green', 
        alpha=0.1,
        label=f"Prob(B > A): {prob_b_better:.2%}"
    )
    
    ax.set_title("Distribution of Difference in Conversion Rates (B - A)")
    ax.set_xlabel("Difference in Conversion Rate")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Interpretation and recommendations
    st.subheader("Interpretation")
    
    if prob_b_better > 0.95:
        st.success(f"""
        **Strong evidence that B is better than A** (Probability: {prob_b_better:.2%})
        
        With {prob_b_better:.2%} probability, variant B has a higher conversion rate than variant A.
        The expected lift is {expected_lift:.2%}.
        
        **Recommendation**: Implement variant B.
        """)
    elif prob_b_better > 0.9:
        st.info(f"""
        **Moderate evidence that B is better than A** (Probability: {prob_b_better:.2%})
        
        With {prob_b_better:.2%} probability, variant B has a higher conversion rate than variant A.
        The expected lift is {expected_lift:.2%}.
        
        **Recommendation**: Consider implementing variant B, or collect more data if the decision has high stakes.
        """)
    elif prob_b_better > 0.8:
        st.warning(f"""
        **Weak evidence that B is better than A** (Probability: {prob_b_better:.2%})
        
        With {prob_b_better:.2%} probability, variant B has a higher conversion rate than variant A.
        The expected lift is {expected_lift:.2%}.
        
        **Recommendation**: Consider collecting more data before making a decision.
        """)
    elif prob_b_better < 0.2:
        st.error(f"""
        **Evidence that A is better than B** (Probability B > A: {prob_b_better:.2%})
        
        With {1-prob_b_better:.2%} probability, variant A has a higher conversion rate than variant B.
        
        **Recommendation**: Stay with variant A.
        """)
    else:
        st.warning(f"""
        **Inconclusive results** (Probability B > A: {prob_b_better:.2%})
        
        The data doesn't provide strong evidence in either direction.
        
        **Recommendation**: Continue the test to collect more data, or consider other factors in making your decision.
        """)
    
    # Sample size calculator
    st.subheader("Sample Size Planning")
    st.markdown("""
    If your test is inconclusive, you may want to collect more data. This tool helps you estimate 
    how many more samples you need to reach a desired level of confidence.
    """)
    
    target_prob = st.slider("Target probability of detecting a true difference", 0.8, 0.99, 0.95, 0.01)
    min_effect = st.slider("Minimum meaningful effect size (%)", 1.0, 20.0, 5.0, 0.5) / 100
    
    # Estimate required sample size (using simple approximation)
    baseline_rate = posterior_mean_a
    p = baseline_rate
    required_n_per_variant = int(16 * p * (1-p) / (min_effect**2))
    
    st.markdown(f"""
    To detect an absolute difference of at least {min_effect:.1%} with {target_prob:.0%} probability,
    you need approximately **{required_n_per_variant} visitors per variant**.
    
    You currently have {visitors_a} visitors for variant A and {visitors_b} visitors for variant B.
    """)
    
    if visitors_a < required_n_per_variant or visitors_b < required_n_per_variant:
        additional_a = max(0, required_n_per_variant - visitors_a)
        additional_b = max(0, required_n_per_variant - visitors_b)
        
        st.markdown(f"""
        To reach the required sample size, you need approximately:
        - {additional_a} more visitors for variant A
        - {additional_b} more visitors for variant B
        """)

# Sidebar explanation
st.sidebar.markdown("""
## About Bayesian Inference

Bayes' theorem is a mathematical formula used to calculate the probability of a hypothesis, given some observed evidence:

$$ P(H|E) = \\frac{P(E|H) \\cdot P(H)}{P(E)} $$

Where:
- $P(H|E)$ is the posterior probability (probability of hypothesis given the evidence)
- $P(E|H)$ is the likelihood (probability of the evidence given the hypothesis)
- $P(H)$ is the prior probability (initial probability of the hypothesis)
- $P(E)$ is the evidence (total probability of observing the evidence)
""")

# Footer
st.markdown("---")
st.markdown("Created with Streamlit for Bayesian inference demonstration") 
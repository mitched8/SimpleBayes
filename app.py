import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from scipy import stats

# Import our modules
from scenarios.medical_test import render_medical_test
from scenarios.weather_prediction import render_weather_prediction
from scenarios.time_series import render_time_series_analysis

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
    ["Medical Test", "Weather Prediction", "Custom Example", "Real World Data Analysis", "A/B Testing Calculator", "Time Series Analysis"]
)

# Main content based on the selected scenario
if scenario == "Medical Test":
    render_medical_test()
elif scenario == "Weather Prediction":
    render_weather_prediction()
elif scenario == "Time Series Analysis":
    render_time_series_analysis()
elif scenario == "Custom Example":
    # TODO: Move this to a separate module
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
    # TODO: Move this to a separate module
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
    # TODO: Move this to a separate module
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
    
    # Add expandable section explaining the impact of priors
    with st.expander("How do priors impact the analysis?"):
        st.markdown("### Understanding the Impact of Priors")
        st.markdown("""
        The choice between uninformative and informed priors can significantly affect your conclusions, 
        especially with smaller sample sizes. Let's examine how different priors would impact the current analysis.
        """)
        
        # Calculate what the results would be with an uninformative prior
        uninform_alpha_post_a = 1 + conversions_a
        uninform_beta_post_a = 1 + (visitors_a - conversions_a)
        uninform_alpha_post_b = 1 + conversions_b
        uninform_beta_post_b = 1 + (visitors_b - conversions_b)
        
        uninform_mean_a = uninform_alpha_post_a / (uninform_alpha_post_a + uninform_beta_post_a)
        uninform_mean_b = uninform_alpha_post_b / (uninform_alpha_post_b + uninform_beta_post_b)
        
        # Generate samples with uninformative prior
        uninform_samples_a = np.random.beta(uninform_alpha_post_a, uninform_beta_post_a, 100000)
        uninform_samples_b = np.random.beta(uninform_alpha_post_b, uninform_beta_post_b, 100000)
        uninform_prob_b_better = np.mean(uninform_samples_b > uninform_samples_a)
        
        # Calculate what the results would be with a stronger informed prior
        # Using 5% as mean and 100 as strength for illustration
        strong_prior_mean = 0.05
        strong_prior_strength = 100
        strong_alpha_prior = strong_prior_mean * strong_prior_strength
        strong_beta_prior = (1 - strong_prior_mean) * strong_prior_strength
        
        strong_alpha_post_a = strong_alpha_prior + conversions_a
        strong_beta_post_a = strong_beta_prior + (visitors_a - conversions_a)
        strong_alpha_post_b = strong_alpha_prior + conversions_b
        strong_beta_post_b = strong_beta_prior + (visitors_b - conversions_b)
        
        strong_mean_a = strong_alpha_post_a / (strong_alpha_post_a + strong_beta_post_a)
        strong_mean_b = strong_alpha_post_b / (strong_alpha_post_b + strong_beta_post_b)
        
        # Generate samples with strong informed prior
        strong_samples_a = np.random.beta(strong_alpha_post_a, strong_beta_post_a, 100000)
        strong_samples_b = np.random.beta(strong_alpha_post_b, strong_beta_post_b, 100000)
        strong_prob_b_better = np.mean(strong_samples_b > strong_samples_a)
        
        # Calculate for current prior
        current_alpha_post_a = alpha_prior_a + conversions_a
        current_beta_post_a = beta_prior_a + (visitors_a - conversions_a)
        current_alpha_post_b = alpha_prior_b + conversions_b
        current_beta_post_b = beta_prior_b + (visitors_b - conversions_b)
        
        current_mean_a = current_alpha_post_a / (current_alpha_post_a + current_beta_post_a)
        current_mean_b = current_alpha_post_b / (current_alpha_post_b + current_beta_post_b)
        
        # Generate samples for current prior
        current_samples_a = np.random.beta(current_alpha_post_a, current_beta_post_a, 100000)
        current_samples_b = np.random.beta(current_alpha_post_b, current_beta_post_b, 100000)
        current_prob_b_better = np.mean(current_samples_b > current_samples_a)
        
        # Create a comparison table
        comparison_data = {
            "Prior Type": ["Uninformative (Beta(1,1))", 
                          f"Current ({prior_option})", 
                          "Strong Informed (5% mean, strength=100)"],
            "Posterior Mean A": [uninform_mean_a, current_mean_a, strong_mean_a],
            "Posterior Mean B": [uninform_mean_b, current_mean_b, strong_mean_b],
            "Probability B > A": [uninform_prob_b_better, current_prob_b_better, strong_prob_b_better]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display the table
        st.dataframe(comparison_df.style.format({
            "Posterior Mean A": "{:.2%}",
            "Posterior Mean B": "{:.2%}",
            "Probability B > A": "{:.2%}"
        }))
        
        st.markdown("""
        ### Key Takeaways
        
        1. **Uninformative Prior**: Relies almost entirely on the observed data. This can lead to higher confidence with small samples, as it doesn't incorporate any skepticism about extreme results.
        
        2. **Informed Prior**: Pulls posterior estimates toward the prior mean. The strength parameter determines how much influence the prior has relative to the observed data.
        
        3. **Mathematical Explanation**:
           
           With an uninformative prior Beta(1,1), the posterior calculation is:
           ```
           alpha_posterior_A = 1 + conversions_A
           beta_posterior_A = 1 + (visitors_A - conversions_A)
           ```
           
           With an informed prior with mean μ and strength s, the calculation is:
           ```
           alpha_prior = μ × s
           beta_prior = (1-μ) × s
           alpha_posterior_A = alpha_prior + conversions_A
           beta_posterior_A = beta_prior + (visitors_A - conversions_A)
           ```
           
           The posterior mean is then:
           ```
           posterior_mean = alpha_posterior / (alpha_posterior + beta_posterior)
           ```
           
           This can also be written as a weighted average:
           ```
           posterior_mean = (prior_strength × prior_mean + visitors × observed_rate) / (prior_strength + visitors)
           ```
           
        4. **When Prior Choice Matters Most**:
           - Small sample sizes
           - Observed rates far from typical values
           - High-stakes decisions
        
        5. **Recommendation**: If you have reliable historical data about typical conversion rates, using an informed prior can protect against overconfidence based on small samples. If you're exploring something new or want to be more data-driven, an uninformative prior might be appropriate.
        
        ### Why Informed Priors Matter Even for "Independent" Tests
        
        A common question is: "Why should prior knowledge influence my analysis of completely new variants A and B?"
        
        The answer lies in understanding that while variants A and B are independent experiments, conversion rates across similar types of tests tend to follow patterns:
        
        1. **Domain Knowledge**: If you've run dozens of e-commerce tests and found that add-to-cart conversion rates typically range from 3-7%, seeing a 25% conversion rate in a small sample is more likely due to random chance than a true breakthrough.
        
        2. **Protection Against Small Sample Noise**: Small samples (less than a few hundred observations) can produce extreme results by random chance alone. Informed priors help guard against overinterpreting this noise.
        
        3. **Pragmatic Bayesian Approach**: We're not saying new variants are connected to previous tests; we're saying they belong to the same class of phenomena that tends to behave in certain ways.
        
        4. **Gradual Updating**: As sample size increases, the influence of the prior diminishes and the data dominates. With enough data, even a strong prior will be overcome by consistent evidence.
        
        Consider this concrete example: If you test a new button color and see 5/50 conversions (10%) for A and 10/50 conversions (20%) for B:
        
        - The **uninformed approach** would fully trust this 10% difference despite the small sample.
        - The **informed approach** would say: "While the data suggests B is better, such large differences are rare in button tests based on our history. Let's collect more data before making a decision."
        
        This is the essence of Bayesian thinking: updating beliefs incrementally as evidence accumulates, rather than treating each experiment as if we have no prior knowledge about typical conversion rates in our domain.
        """)
        
        # Add a visual to show the "pull" effect of priors
        st.subheader("Visual Comparison of Prior Effects")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Observed data reference line
        observed_rate_a = conversions_a / visitors_a
        observed_rate_b = conversions_b / visitors_b
        
        x = ['Variant A', 'Variant B']
        y_observed = [observed_rate_a, observed_rate_b]
        y_uninform = [uninform_mean_a, uninform_mean_b]
        y_current = [current_mean_a, current_mean_b]
        y_strong = [strong_mean_a, strong_mean_b]
        
        ax.plot(x, y_observed, 'o-', label='Observed Rates', color='black', linewidth=2)
        ax.plot(x, y_uninform, 's-', label='With Uninformative Prior', color='blue', linewidth=2)
        ax.plot(x, y_current, 'd-', label=f'With Current Prior', color='green', linewidth=2)
        ax.plot(x, y_strong, '^-', label='With Strong Prior (5%)', color='red', linewidth=2)
        
        # Add horizontal line for the strong prior mean
        ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.3, label='Strong Prior Mean (5%)')
        
        ax.set_ylabel('Conversion Rate')
        ax.set_title('How Different Priors Pull the Posterior Means')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add some explanatory text annotations
        if abs(strong_mean_a - observed_rate_a) > 0.02:  # Only annotate if there's a visible difference
            ax.annotate('Prior pulls\nposterior toward\nprior mean',
                       xy=('Variant A', strong_mean_a),
                       xytext=('Variant A', (observed_rate_a + strong_mean_a)/2 + 0.02),
                       arrowprops=dict(arrowstyle='->'),
                       ha='center')
        
        st.pyplot(fig)
    
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
    
    with st.expander("How does sample size planning work in Bayesian testing?"):
        st.markdown("""
        ### Understanding Sample Size in Bayesian A/B Testing
        
        Unlike frequentist methods that use power calculations based on p-values, Bayesian sample size planning focuses on:
        
        1. **The probability of detecting a true effect** (similar to statistical power)
        2. **The minimum effect size that matters to your business**
        3. **The level of certainty you need to make a decision**
        
        #### How the calculation works
        
        The sample size estimate uses an approximation based on the normal distribution of the difference between two proportions:
        
        ```
        n ≈ 16 × p × (1-p) / (minimum_effect)²
        ```
        
        Where:
        - n is the required sample size per variant
        - p is the baseline conversion rate
        - minimum_effect is the smallest meaningful difference (absolute, not relative)
        - The constant 16 comes from (zα + zβ)² where zα and zβ correspond to approximately 95% probability
        
        #### Interpreting this estimate
        
        This formula gives you an approximate sample size needed to:
        - Detect a true difference of at least the minimum effect size
        - With the target probability (similar to statistical power)
        - Assuming your current posterior mean is close to the true conversion rate
        
        #### Key insights about sample size
        
        1. **Higher baseline conversion rates need smaller samples** - It's easier to detect changes in high conversion rates because there's less variance as a proportion of the mean.
        
        2. **Smaller effect sizes require larger samples** - The sample size increases with the square of the minimum effect, so halving the minimum detectable effect requires 4× the sample size.
        
        3. **More certainty requires larger samples** - Higher target probabilities require more data.
        
        4. **Bayesian analysis can often make decisions with smaller samples** than frequentist methods because:
           - It directly calculates the probability of B being better than A
           - It incorporates prior knowledge when available
           - It doesn't rely on arbitrary significance thresholds
        
        #### Example calculation
        
        If your baseline conversion rate is 10%, and you want to detect a 2% absolute improvement (12% vs 10%) with 95% probability:
        
        ```
        n ≈ 16 × 0.10 × (1-0.10) / (0.02)² = 16 × 0.10 × 0.90 / 0.0004 = 16 × 0.09 / 0.0004 = 3,600
        ```
        
        You would need approximately 3,600 visitors per variant.
        """)
        
        # Add a visualization of how sample size affects uncertainty
        st.subheader("How Sample Size Affects Uncertainty")
        
        # Create a figure showing posterior distributions with different sample sizes
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.linspace(0, 0.2, 1000)  # Range of conversion rates to plot
        
        # Define different sample sizes
        sample_sizes = [100, 500, 2000, 10000]
        colors = ['#FF9999', '#FFCC99', '#99CCFF', '#99FF99']
        
        # Base parameters
        base_rate = 0.1  # 10% conversion rate
        effect = 0.02  # 2% effect
        
        for i, n in enumerate(sample_sizes):
            # Calculate the parameters for the posterior distribution
            # Assuming uninformative prior and observed data matching expectations
            alpha_a = 1 + n * base_rate
            beta_a = 1 + n * (1 - base_rate)
            
            alpha_b = 1 + n * (base_rate + effect)
            beta_b = 1 + n * (1 - (base_rate + effect))
            
            # Plot the posterior distributions
            posterior_a = stats.beta.pdf(x, alpha_a, beta_a)
            posterior_b = stats.beta.pdf(x, alpha_b, beta_b)
            
            # Scale down the density values to fit multiple curves
            scale_factor = 1.0
            if i > 0:
                scale_factor = 0.8 ** i
            
            ax.plot(x, posterior_a * scale_factor, 
                    color=colors[i], linestyle='-', alpha=0.8,
                    label=f'A (n={n}): {base_rate:.1%} rate')
            ax.plot(x, posterior_b * scale_factor, 
                    color=colors[i], linestyle='--', alpha=0.8,
                    label=f'B (n={n}): {base_rate+effect:.1%} rate')
            
            # Calculate the probability that B > A for this sample size
            samples_a = np.random.beta(alpha_a, beta_a, 10000)
            samples_b = np.random.beta(alpha_b, beta_b, 10000)
            prob_b_better = np.mean(samples_b > samples_a)
            
            # Add text annotation about the probability
            ax.text(0.17, ax.get_ylim()[1] * (0.95 - i*0.15), 
                    f"n={n}: P(B>A) = {prob_b_better:.1%}",
                    fontsize=10, color=colors[i], 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor=colors[i]))
        
        ax.set_xlabel('Conversion Rate')
        ax.set_ylabel('Posterior Density (scaled)')
        ax.set_title('Effect of Sample Size on Posterior Distributions')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        st.markdown("""
        This visualization shows how increasing the sample size:
        
        1. Makes the posterior distributions narrower (more certainty)
        2. Increases the separation between distributions when a real effect exists
        3. Increases the probability of correctly detecting that B is better than A
        
        Notice how with small samples (n=100), there's substantial overlap between the distributions, even though B truly has a 2% higher conversion rate. As the sample size increases, the distributions separate, making it easier to detect the difference.
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
        
        # Add a business context explanation
        st.info(f"""
        **Business Context**: If your test is running at a traffic level of 1,000 visitors per day (combined for both variants), 
        it would take approximately **{(additional_a + additional_b) / 1000:.1f} more days** to reach the recommended sample size.
        
        **Recommendation**: When deciding whether to wait for more data, consider:
        1. The cost of waiting (delayed implementation)
        2. The risk of making the wrong decision
        3. The expected value of the improvement if B is truly better
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
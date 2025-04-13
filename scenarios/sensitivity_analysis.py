import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from utils.bayes_utils import calculate_posterior

def render_sensitivity_analysis():
    """Render the Bayesian Sensitivity Analysis page"""
    st.header("Bayesian Sensitivity Analysis")
    st.markdown("""
    This tool demonstrates how to perform sensitivity analysis using Bayesian methods.
    Sensitivity analysis helps you understand:
    
    1. Which parameters most influence your posterior beliefs
    2. How robust your conclusions are to changes in assumptions
    3. Where to focus efforts for gathering more information
    
    Bayesian sensitivity analysis is particularly valuable because it works with full probability 
    distributions rather than just point estimates.
    """)
    
    # Select analysis type
    analysis_type = st.radio(
        "Select analysis approach",
        ["Parameter Sensitivity", "Prior Sensitivity", "Model Comparison"]
    )
    
    if analysis_type == "Parameter Sensitivity":
        render_parameter_sensitivity()
    elif analysis_type == "Prior Sensitivity":
        render_prior_sensitivity()
    else:
        render_model_comparison()

def render_parameter_sensitivity():
    """Render the Parameter Sensitivity section"""
    st.subheader("Parameter Sensitivity Analysis")
    st.markdown("""
    Parameter sensitivity analysis explores how changes in model parameters affect outcomes.
    
    In this example, we'll examine a medical diagnosis scenario and see how sensitive our
    posterior probability is to changes in test accuracy parameters.
    """)
    
    # Base parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Base Parameters")
        base_prevalence = st.slider("Disease prevalence (%)", 0.1, 20.0, 1.0, 0.1) / 100
        base_sensitivity = st.slider("Test sensitivity (%)", 50.0, 100.0, 95.0, 0.1) / 100
        base_specificity = st.slider("Test specificity (%)", 50.0, 100.0, 90.0, 0.1) / 100
        
        # Calculate base result
        base_ppv = calculate_ppv(base_prevalence, base_sensitivity, base_specificity)
        st.markdown(f"### Base Result")
        st.markdown(f"**Posterior probability of disease: {base_ppv:.2%}**")
    
    with col2:
        st.markdown("### Sensitivity Analysis Parameters")
        parameter_to_vary = st.selectbox(
            "Parameter to analyze sensitivity",
            ["Prevalence", "Sensitivity", "Specificity"]
        )
        
        if parameter_to_vary == "Prevalence":
            min_value = max(0.1, base_prevalence * 100 - 5.0)
            max_value = min(20.0, base_prevalence * 100 + 5.0)
            step_value = (max_value - min_value) / 20
            range_values = np.arange(min_value, max_value, step_value) / 100
        elif parameter_to_vary == "Sensitivity":
            min_value = max(50.0, base_sensitivity * 100 - 10.0)
            max_value = min(100.0, base_sensitivity * 100 + 10.0)
            step_value = (max_value - min_value) / 20
            range_values = np.arange(min_value, max_value, step_value) / 100
        else:  # Specificity
            min_value = max(50.0, base_specificity * 100 - 10.0)
            max_value = min(100.0, base_specificity * 100 + 10.0)
            step_value = (max_value - min_value) / 20
            range_values = np.arange(min_value, max_value, step_value) / 100
    
    # Calculate sensitivity results
    results = []
    for value in range_values:
        if parameter_to_vary == "Prevalence":
            ppv = calculate_ppv(value, base_sensitivity, base_specificity)
            results.append((value, ppv))
        elif parameter_to_vary == "Sensitivity":
            ppv = calculate_ppv(base_prevalence, value, base_specificity)
            results.append((value, ppv))
        else:  # Specificity
            ppv = calculate_ppv(base_prevalence, base_sensitivity, value)
            results.append((value, ppv))
    
    # Create results dataframe
    results_df = pd.DataFrame(results, columns=[parameter_to_vary, "Posterior Probability"])
    
    # Plot sensitivity
    st.subheader("Sensitivity Analysis Results")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(results_df[parameter_to_vary], results_df["Posterior Probability"])
    plt.scatter(results_df[parameter_to_vary], results_df["Posterior Probability"], color='red')
    
    # Mark the base value
    if parameter_to_vary == "Prevalence":
        base_x = base_prevalence
    elif parameter_to_vary == "Sensitivity":
        base_x = base_sensitivity
    else:
        base_x = base_specificity
    
    plt.axvline(x=base_x, color='green', linestyle='--', label=f'Base {parameter_to_vary}')
    plt.axhline(y=base_ppv, color='orange', linestyle='--', label='Base Posterior')
    
    plt.xlabel(parameter_to_vary)
    plt.ylabel("Posterior Probability of Disease")
    plt.title(f"Impact of {parameter_to_vary} on Posterior Probability")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    st.pyplot(fig)
    
    # Calculate elasticity
    central_index = len(range_values) // 2
    if central_index > 0 and central_index < len(range_values) - 1:
        x1, x2 = range_values[central_index-1], range_values[central_index+1]
        y1, y2 = results[central_index-1][1], results[central_index+1][1]
        x0, y0 = range_values[central_index], results[central_index][1]
        
        # Calculate elasticity at the central point
        elasticity = ((y2 - y1) / y0) / ((x2 - x1) / x0)
        
        st.markdown(f"### Elasticity Analysis")
        st.markdown(f"""
        The **elasticity** measures how responsive the posterior probability is to changes in parameters.
        
        At the base values, the elasticity of the posterior probability with respect to {parameter_to_vary.lower()} is:
        **{elasticity:.4f}**
        
        This means that a 1% change in {parameter_to_vary.lower()} results in approximately a {abs(elasticity):.2f}% 
        {'increase' if elasticity >= 0 else 'decrease'} in the posterior probability.
        """)
    
    # Threshold analysis
    st.subheader("Threshold Analysis")
    threshold = st.slider("Decision threshold for posterior probability", 0.0, 1.0, 0.5, 0.01)
    
    # Find where posterior crosses threshold
    crossings = []
    for i in range(len(results) - 1):
        if (results[i][1] < threshold and results[i+1][1] >= threshold) or \
           (results[i][1] >= threshold and results[i+1][1] < threshold):
            # Calculate approximate crossing point by linear interpolation
            x1, y1 = results[i]
            x2, y2 = results[i+1]
            if x2 != x1:  # Avoid division by zero
                x_cross = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
                crossings.append(x_cross)
    
    if crossings:
        st.markdown(f"""
        The posterior probability crosses the threshold of {threshold:.2f} when 
        {parameter_to_vary.lower()} is approximately:
        """)
        for crossing in crossings:
            st.markdown(f"- **{crossing:.4f}** ({crossing:.2%})")
    else:
        st.markdown(f"""
        The posterior probability does not cross the threshold of {threshold:.2f} within 
        the analyzed range of {parameter_to_vary.lower()}.
        """)

def render_prior_sensitivity():
    """Render the Prior Sensitivity section"""
    st.subheader("Prior Sensitivity Analysis")
    st.markdown("""
    Prior sensitivity analysis examines how your choice of prior distribution affects
    the posterior results. This is especially important when:
    
    1. Limited data is available
    2. There are disagreements among experts about prior beliefs
    3. You want to ensure robust decisions across a range of assumptions
    """)
    
    # Example: Beta prior for a coin flip probability
    st.markdown("### Beta Prior for a Coin Flip Probability")
    st.markdown("""
    In this example, we'll examine how different Beta priors affect our posterior 
    belief about a coin's probability of landing heads.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Data input
        heads = st.number_input("Number of heads observed", 0, 1000, 8, 1)
        tails = st.number_input("Number of tails observed", 0, 1000, 2, 1)
        
        # Calculate the maximum likelihood estimate
        mle = heads / (heads + tails)
        st.markdown(f"**Maximum likelihood estimate: {mle:.4f}**")
    
    with col2:
        # Prior specifications
        st.markdown("### Prior Specifications")
        
        prior_type = st.radio(
            "Prior type",
            ["Informative", "Weakly informative", "Uninformative", "Custom"]
        )
        
        if prior_type == "Informative":
            alpha = 10
            beta = 10
        elif prior_type == "Weakly informative":
            alpha = 2
            beta = 2
        elif prior_type == "Uninformative":
            alpha = 1
            beta = 1
        else:  # Custom
            alpha = st.slider("Alpha parameter", 0.1, 20.0, 5.0, 0.1)
            beta = st.slider("Beta parameter", 0.1, 20.0, 5.0, 0.1)
    
    # Calculate posterior for the selected prior
    posterior_alpha = alpha + heads
    posterior_beta = beta + tails
    
    # Calculate the mean of the posterior
    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
    
    # Plot prior and posterior
    x = np.linspace(0, 1, 1000)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot prior
    prior_pdf = stats.beta.pdf(x, alpha, beta)
    ax.plot(x, prior_pdf, 'b--', label=f'Prior: Beta({alpha}, {beta})')
    
    # Plot posterior
    posterior_pdf = stats.beta.pdf(x, posterior_alpha, posterior_beta)
    ax.plot(x, posterior_pdf, 'r-', label=f'Posterior: Beta({posterior_alpha}, {posterior_beta})')
    
    # Mark MLE and posterior mean
    ax.axvline(x=mle, color='green', linestyle=':', label=f'MLE: {mle:.4f}')
    ax.axvline(x=posterior_mean, color='red', linestyle=':', label=f'Posterior Mean: {posterior_mean:.4f}')
    
    ax.set_xlabel('Probability of Heads')
    ax.set_ylabel('Density')
    ax.set_title('Prior and Posterior Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Sensitivity to different priors
    st.subheader("Sensitivity to Different Priors")
    
    # Create a range of alphas and betas
    alphas = [0.5, 1, 2, 5, 10, 20]
    betas = [0.5, 1, 2, 5, 10, 20]
    
    # Calculate posteriors for different combinations
    results = []
    for a in alphas:
        for b in betas:
            post_alpha = a + heads
            post_beta = b + tails
            post_mean = post_alpha / (post_alpha + post_beta)
            results.append({
                'Alpha': a,
                'Beta': b,
                'Prior Mean': a / (a + b),
                'Posterior Mean': post_mean,
                'Prior Strength': a + b,
                'Posterior 95% CI Lower': stats.beta.ppf(0.025, post_alpha, post_beta),
                'Posterior 95% CI Upper': stats.beta.ppf(0.975, post_alpha, post_beta)
            })
    
    results_df = pd.DataFrame(results)
    
    # Plot heatmap of posterior means
    st.markdown("### Heatmap of Posterior Means")
    
    # Reshape data for heatmap
    heatmap_data = results_df.pivot(index='Beta', columns='Alpha', values='Posterior Mean')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis", ax=ax)
    ax.set_title("Posterior Mean by Prior Parameters")
    
    st.pyplot(fig)
    
    # Plot posterior means vs. prior means
    st.markdown("### Posterior Mean vs. Prior Mean")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(
        results_df['Prior Mean'], 
        results_df['Posterior Mean'],
        c=results_df['Prior Strength'],
        cmap='viridis',
        s=100,
        alpha=0.7
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Prior Strength (α + β)')
    
    # Add reference line (y=x)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    
    # Mark MLE
    ax.axhline(y=mle, color='green', linestyle=':', label=f'MLE: {mle:.4f}')
    
    ax.set_xlabel('Prior Mean')
    ax.set_ylabel('Posterior Mean')
    ax.set_title('How Prior Mean Affects Posterior Mean')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    st.pyplot(fig)
    
    # Robustness conclusion
    st.subheader("Robustness Analysis")
    
    # Calculate range of posterior means
    min_post = results_df['Posterior Mean'].min()
    max_post = results_df['Posterior Mean'].max()
    range_post = max_post - min_post
    
    st.markdown(f"""
    Across all the prior configurations tested:
    
    - The **minimum posterior mean** is {min_post:.4f}
    - The **maximum posterior mean** is {max_post:.4f}
    - The **range** is {range_post:.4f}
    
    {'Your conclusions are **robust** to prior specifications.' if range_post < 0.1 else 
     'Your conclusions are **moderately sensitive** to prior specifications.' if range_post < 0.2 else
     'Your conclusions are **highly sensitive** to prior specifications. Consider collecting more data.'}
    """)

def render_model_comparison():
    """Render the Model Comparison section"""
    st.subheader("Bayesian Model Comparison")
    st.markdown("""
    Model comparison allows you to:
    
    1. Compare different model structures or hypotheses
    2. Assess how parameter uncertainty affects model selection
    3. Make more robust predictions by model averaging
    
    This approach goes beyond simple sensitivity analysis by formally 
    accounting for model uncertainty.
    """)
    
    st.markdown("### Simple A/B Test Example")
    st.markdown("""
    Let's compare two models for an A/B test:
    
    1. **Equal model**: Conversion rates are equal for A and B
    2. **Different model**: Conversion rates can be different
    
    We'll examine how the data affects our model selection and how sensitive
    our conclusions are to prior assumptions.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Data input
        visitors_a = st.number_input("Visitors to variant A", 10, 10000, 1000, 10)
        conversions_a = st.number_input("Conversions from variant A", 0, visitors_a, 100, 1)
        
        visitors_b = st.number_input("Visitors to variant B", 10, 10000, 1000, 10)
        conversions_b = st.number_input("Conversions from variant B", 0, visitors_b, 120, 1)
        
        # Calculate conversion rates
        conv_rate_a = conversions_a / visitors_a
        conv_rate_b = conversions_b / visitors_b
        
        st.markdown(f"""
        **Observed conversion rates:**
        - Variant A: {conv_rate_a:.2%} ({conversions_a}/{visitors_a})
        - Variant B: {conv_rate_b:.2%} ({conversions_b}/{visitors_b})
        - Difference: {(conv_rate_b - conv_rate_a):.2%}
        """)
    
    with col2:
        # Prior parameters
        st.markdown("### Prior Specifications")
        
        prior_strength = st.slider(
            "Prior strength (pseudo-observations)", 
            0.1, 100.0, 10.0, 0.1
        )
        
        prior_mean = st.slider(
            "Prior mean conversion rate", 
            0.01, 0.5, 0.1, 0.01
        )
        
        # Calculate prior parameters for Beta distribution
        prior_alpha = prior_mean * prior_strength
        prior_beta = prior_strength - prior_alpha
    
    # Calculate posteriors for each model
    
    # Model 1: Equal conversion rates
    total_visitors = visitors_a + visitors_b
    total_conversions = conversions_a + conversions_b
    
    post_alpha_equal = prior_alpha + total_conversions
    post_beta_equal = prior_beta + (total_visitors - total_conversions)
    
    # Model 2: Different conversion rates
    post_alpha_a = prior_alpha + conversions_a
    post_beta_a = prior_beta + (visitors_a - conversions_a)
    
    post_alpha_b = prior_alpha + conversions_b
    post_beta_b = prior_beta + (visitors_b - conversions_b)
    
    # Calculate log evidence for each model (using Beta-Binomial marginal likelihood)
    from scipy.special import betaln
    
    # Log evidence for equal model
    log_evidence_equal = (
        betaln(post_alpha_equal, post_beta_equal) - 
        betaln(prior_alpha, prior_beta)
    )
    
    # Log evidence for different model
    log_evidence_diff = (
        betaln(post_alpha_a, post_beta_a) - betaln(prior_alpha, prior_beta) +
        betaln(post_alpha_b, post_beta_b) - betaln(prior_alpha, prior_beta)
    )
    
    # Calculate Bayes factor
    log_bayes_factor = log_evidence_diff - log_evidence_equal
    bayes_factor = np.exp(log_bayes_factor)
    
    # Calculate posterior model probabilities (assuming equal prior model probabilities)
    model_diff_prob = bayes_factor / (1 + bayes_factor)
    model_equal_prob = 1 - model_diff_prob
    
    # Display results
    st.subheader("Model Comparison Results")
    
    st.markdown(f"""
    **Bayes factor** (Different vs. Equal): {bayes_factor:.4f}
    
    **Posterior model probabilities:**
    - Equal conversion rates: {model_equal_prob:.2%}
    - Different conversion rates: {model_diff_prob:.2%}
    
    **Interpretation:**
    {interpret_bayes_factor(bayes_factor)}
    """)
    
    # Sensitivity to prior strength
    st.subheader("Sensitivity to Prior Strength")
    
    # Calculate Bayes factors for different prior strengths
    prior_strengths = np.linspace(0.1, 100, 50)
    bayes_factors = []
    
    for strength in prior_strengths:
        p_alpha = prior_mean * strength
        p_beta = strength - p_alpha
        
        # Equal model
        pe_alpha = p_alpha + total_conversions
        pe_beta = p_beta + (total_visitors - total_conversions)
        
        # Different model
        pa_alpha = p_alpha + conversions_a
        pa_beta = p_beta + (visitors_a - conversions_a)
        
        pb_alpha = p_alpha + conversions_b
        pb_beta = p_beta + (visitors_b - conversions_b)
        
        # Log evidence
        le_equal = betaln(pe_alpha, pe_beta) - betaln(p_alpha, p_beta)
        
        le_diff = (
            betaln(pa_alpha, pa_beta) - betaln(p_alpha, p_beta) +
            betaln(pb_alpha, pb_beta) - betaln(p_alpha, p_beta)
        )
        
        # Bayes factor
        lbf = le_diff - le_equal
        bf = np.exp(lbf)
        
        bayes_factors.append(bf)
    
    # Plot sensitivity to prior strength
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogx(prior_strengths, bayes_factors)
    ax.axhline(y=1, color='red', linestyle='--', label='Equal evidence')
    ax.axvline(x=prior_strength, color='green', linestyle=':', 
              label=f'Current prior strength: {prior_strength}')
    
    ax.set_xlabel('Prior Strength (pseudo-observations)')
    ax.set_ylabel('Bayes Factor (Different vs. Equal)')
    ax.set_title('Sensitivity of Bayes Factor to Prior Strength')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    st.pyplot(fig)
    
    # Inference robustness conclusion
    robustness_threshold = 3.0  # Threshold for "strong evidence"
    
    if all(bf > robustness_threshold for bf in bayes_factors):
        robustness_msg = (
            "Your conclusion of **different conversion rates** is robust across "
            "all tested prior strengths."
        )
    elif all(bf < 1/robustness_threshold for bf in bayes_factors):
        robustness_msg = (
            "Your conclusion of **equal conversion rates** is robust across "
            "all tested prior strengths."
        )
    else:
        robustness_msg = (
            "Your conclusion is **sensitive to prior strength**. "
            "Consider collecting more data before making a firm decision."
        )
    
    st.markdown(f"### Robustness Conclusion")
    st.markdown(robustness_msg)
    
    # Prediction using model averaging
    st.subheader("Bayesian Model Averaging")
    
    # Calculate posterior predictive for a new visitor using model averaging
    # Probability of conversion under equal model
    p_conv_equal = post_alpha_equal / (post_alpha_equal + post_beta_equal)
    
    # Probability of conversion under different model
    p_conv_a = post_alpha_a / (post_alpha_a + post_beta_a)
    p_conv_b = post_alpha_b / (post_alpha_b + post_beta_b)
    
    # Model averaged predictions
    p_conv_a_avg = model_equal_prob * p_conv_equal + model_diff_prob * p_conv_a
    p_conv_b_avg = model_equal_prob * p_conv_equal + model_diff_prob * p_conv_b
    
    st.markdown(f"""
    **Model-averaged conversion probability predictions:**
    - Variant A: {p_conv_a_avg:.4f} ({p_conv_a_avg:.2%})
    - Variant B: {p_conv_b_avg:.4f} ({p_conv_b_avg:.2%})
    - Difference: {(p_conv_b_avg - p_conv_a_avg):.4f} ({(p_conv_b_avg - p_conv_a_avg):.2%})
    
    **Probability that B is better than A:** {model_diff_prob * (p_conv_b > p_conv_a):.2%}
    """)

# Helper functions
def calculate_ppv(prevalence, sensitivity, specificity):
    """Calculate positive predictive value (posterior probability of disease)"""
    # Prior probabilities
    prior_probs = {
        "Has Disease": prevalence,
        "No Disease": 1 - prevalence
    }
    
    # Likelihoods of positive test result
    likelihoods = {
        "Has Disease": sensitivity,
        "No Disease": 1 - specificity
    }
    
    # Calculate posteriors
    posteriors = calculate_posterior(prior_probs, likelihoods)
    
    return posteriors["Has Disease"]

def interpret_bayes_factor(bf):
    """Interpret the strength of evidence based on Bayes factor"""
    if bf < 1:
        # Reverse interpretation for BF < 1
        return interpret_bayes_factor(1/bf).replace("different", "equal").replace("Different", "Equal")
    
    if bf > 100:
        return "Extreme evidence for **different conversion rates**."
    elif bf > 30:
        return "Very strong evidence for **different conversion rates**."
    elif bf > 10:
        return "Strong evidence for **different conversion rates**."
    elif bf > 3:
        return "Moderate evidence for **different conversion rates**."
    elif bf > 1:
        return "Weak evidence for **different conversion rates**."
    else:
        return "No evidence for either model." 
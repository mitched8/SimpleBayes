import numpy as np
import pandas as pd
from scipy import stats

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

def generate_beta_samples(alpha, beta, size=100000):
    """
    Generate samples from a Beta distribution
    
    Parameters:
    - alpha: Alpha parameter for Beta distribution
    - beta: Beta parameter for Beta distribution
    - size: Number of samples to generate
    
    Returns:
    - Array of samples
    """
    np.random.seed(42)  # For reproducibility
    return np.random.beta(alpha, beta, size)

def calculate_credible_interval(samples, interval=95):
    """
    Calculate credible interval from samples
    
    Parameters:
    - samples: Array of samples
    - interval: Credible interval percentage (default 95%)
    
    Returns:
    - Tuple of (lower_bound, upper_bound)
    """
    lower_percentile = (100 - interval) / 2
    upper_percentile = 100 - lower_percentile
    return (
        np.percentile(samples, lower_percentile),
        np.percentile(samples, upper_percentile)
    ) 
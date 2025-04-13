import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def render_time_series_analysis():
    """Render the Bayesian Time Series Analysis page"""
    st.header("Bayesian Time Series Analysis")
    st.markdown("""
    This tool demonstrates Bayesian time series analysis techniques. You can use this to:
    
    1. **Detect changes in time series data** (change point detection)
    2. **Forecast future values** with uncertainty estimates
    3. **Identify seasonal patterns** and separate them from trends
    
    In this scenario, we use Bayesian methods to update our beliefs about time series parameters 
    as we observe more data points.
    """)
    
    # Sidebar for time series options
    analysis_type = st.radio(
        "Select analysis type",
        ["Change Point Detection", "Bayesian Forecasting", "Custom Time Series"]
    )
    
    if analysis_type == "Change Point Detection":
        render_change_point_detection()
    elif analysis_type == "Bayesian Forecasting":
        render_bayesian_forecasting()
    else:
        render_custom_time_series()

def render_change_point_detection():
    """Render the Change Point Detection section"""
    st.subheader("Bayesian Change Point Detection")
    st.markdown("""
    Change point detection identifies when the underlying parameters of a time series have changed.
    Bayesian methods are particularly powerful here because they:
    
    1. Quantify uncertainty in the change point location
    2. Allow for incorporating prior knowledge about change frequency
    3. Can detect multiple change points simultaneously
    """)
    
    # Options for synthetic data generation
    st.markdown("### Data Generation")
    
    data_type = st.radio(
        "Data source",
        ["Synthetic data with known change points", "Upload your own data"]
    )
    
    if data_type == "Synthetic data with known change points":
        # Parameters for synthetic data
        n_points = st.slider("Number of data points", 50, 500, 200)
        change_points = st.multiselect(
            "Select change points (positions)",
            options=list(range(10, n_points-10, 10)),
            default=[50, 120]
        )
        
        noise_level = st.slider("Noise level", 0.1, 2.0, 0.5)
        
        # Generate synthetic time series with change points
        data = generate_synthetic_data_with_change_points(n_points, change_points, noise_level)
    else:
        st.markdown("Upload your time series data (CSV with columns 'time' and 'value')")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
        else:
            st.warning("Please upload a CSV file or select synthetic data.")
            return
    
    # Display the data
    st.subheader("Time Series Data")
    st.line_chart(data.set_index('time')['value'])
    
    # Bayesian change point detection parameters
    st.subheader("Change Point Detection Parameters")
    
    prior_change_prob = st.slider(
        "Prior probability of change at each point (%)", 
        0.1, 10.0, 1.0, 0.1
    ) / 100
    
    n_iterations = st.slider("Number of MCMC iterations", 1000, 10000, 5000, 1000)
    
    if st.button("Run Bayesian Change Point Detection"):
        with st.spinner("Running Bayesian change point detection..."):
            change_point_probs = detect_change_points(data, prior_change_prob, n_iterations)
        
        # Display results
        st.subheader("Change Point Detection Results")
        
        # Plot time series with change point probabilities
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot original time series
        ax1.plot(data['time'], data['value'])
        ax1.set_title("Original Time Series")
        ax1.set_ylabel("Value")
        
        # Plot posterior probability of change points
        ax2.bar(data['time'], change_point_probs, alpha=0.7)
        ax2.set_title("Posterior Probability of Change Points")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Probability")
        
        # If using synthetic data, mark the true change points
        if data_type == "Synthetic data with known change points":
            for cp in change_points:
                ax1.axvline(x=cp, color='red', linestyle='--', alpha=0.7)
                ax2.axvline(x=cp, color='red', linestyle='--', alpha=0.7)
        
        st.pyplot(fig)
        
        # Identify the most likely change points
        threshold = 0.2  # Probability threshold for reporting change points
        # Find indices where change_point_probs exceed threshold
        likely_indices = np.where(change_point_probs > threshold)[0]
        likely_change_points = data['time'].iloc[likely_indices].tolist()
        
        if likely_change_points:
            st.markdown(f"### Detected Change Points (Probability > {threshold})")
            for i, cp in enumerate(likely_change_points):
                cp_index = likely_indices[i]
                st.markdown(f"- Time = {cp}: Probability = {change_point_probs[cp_index]:.2f}")
        else:
            st.markdown("No change points with probability above the threshold were detected.")

def render_bayesian_forecasting():
    """Render the Bayesian Forecasting section"""
    st.subheader("Bayesian Time Series Forecasting")
    st.markdown("""
    Bayesian forecasting provides predictions with credible intervals, representing our uncertainty about future values.
    This is especially useful for decision-making because it quantifies risk.
    
    In this example, we use a simple Bayesian structural time series model that can handle:
    1. Trends
    2. Seasonality
    3. Exogenous variables (optional)
    """)
    
    # Options for synthetic data generation
    st.markdown("### Data Generation")
    
    data_type = st.radio(
        "Data source",
        ["Synthetic data with trend and seasonality", "Upload your own data"]
    )
    
    if data_type == "Synthetic data with trend and seasonality":
        # Parameters for synthetic data
        n_points = st.slider("Number of data points", 50, 200, 100)
        
        trend_type = st.selectbox(
            "Trend type",
            ["Linear increasing", "Linear decreasing", "No trend"]
        )
        
        seasonal_period = st.slider("Seasonal period (number of time points)", 0, 50, 12)
        seasonal_amplitude = st.slider("Seasonal amplitude", 0.0, 5.0, 2.0, 0.1)
        
        noise_level = st.slider("Noise level", 0.1, 2.0, 0.5)
        
        # Generate synthetic time series with trend and seasonality
        data = generate_synthetic_data_with_trend_seasonality(
            n_points, trend_type, seasonal_period, seasonal_amplitude, noise_level
        )
    else:
        st.markdown("Upload your time series data (CSV with columns 'time' and 'value')")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
        else:
            st.warning("Please upload a CSV file or select synthetic data.")
            return
    
    # Display the data
    st.subheader("Time Series Data")
    st.line_chart(data.set_index('time')['value'])
    
    # Bayesian forecasting parameters
    st.subheader("Forecasting Parameters")
    
    forecast_horizon = st.slider("Forecast horizon (number of time points)", 1, 50, 20)
    
    include_trend = st.checkbox("Include trend component", True)
    if include_trend:
        trend_flexibility = st.slider(
            "Trend flexibility (higher = more flexible)", 
            0.01, 1.0, 0.1, 0.01
        )
    
    include_seasonality = st.checkbox("Include seasonal component", True)
    if include_seasonality:
        season_length = st.slider("Season length", 2, 52, 12)
    
    if st.button("Run Bayesian Forecasting"):
        with st.spinner("Running Bayesian forecasting..."):
            forecast_result = bayesian_forecast(
                data, 
                forecast_horizon,
                include_trend=include_trend,
                trend_flexibility=trend_flexibility if include_trend else 0.1,
                include_seasonality=include_seasonality,
                season_length=season_length if include_seasonality else 12
            )
        
        # Display results
        st.subheader("Forecasting Results")
        
        # Create DataFrame for forecast results
        forecast_df = pd.DataFrame({
            'time': range(data['time'].iloc[-1] + 1, data['time'].iloc[-1] + 1 + forecast_horizon),
            'forecast_mean': forecast_result['mean'],
            'forecast_lower': forecast_result['lower'],
            'forecast_upper': forecast_result['upper']
        })
        
        # Plot original series and forecast
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot original time series
        ax.plot(data['time'], data['value'], label='Historical Data')
        
        # Plot forecast
        ax.plot(forecast_df['time'], forecast_df['forecast_mean'], 
                label='Forecast', color='red')
        
        # Plot prediction intervals
        ax.fill_between(
            forecast_df['time'],
            forecast_df['forecast_lower'],
            forecast_df['forecast_upper'],
            color='red', alpha=0.2,
            label='95% Credible Interval'
        )
        
        ax.set_title("Bayesian Time Series Forecast")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Display forecast statistics
        st.subheader("Forecast Statistics")
        
        # Add forecasted values to the data
        forecast_stats = forecast_df[['time', 'forecast_mean', 'forecast_lower', 'forecast_upper']]
        forecast_stats.columns = ['Time', 'Mean Forecast', 'Lower 95% CI', 'Upper 95% CI']
        
        st.dataframe(forecast_stats.style.format({
            'Mean Forecast': "{:.2f}",
            'Lower 95% CI': "{:.2f}",
            'Upper 95% CI': "{:.2f}"
        }))

def render_custom_time_series():
    """Render the Custom Time Series section"""
    st.subheader("Custom Bayesian Time Series Analysis")
    st.markdown("""
    In this section, you can upload your own time series data and experiment with
    different Bayesian time series models and parameters.
    """)
    
    st.markdown("Upload your time series data (CSV with columns 'time' and 'value')")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Display the data
        st.subheader("Time Series Data")
        st.line_chart(data.set_index('time')['value'])
        
        # Analysis options
        st.subheader("Analysis Options")
        
        analysis_option = st.selectbox(
            "Select analysis option",
            ["Basic Statistics", "Decomposition", "Change Point Detection", "Forecasting"]
        )
        
        if analysis_option == "Basic Statistics":
            # Calculate basic Bayesian statistics for the time series
            st.markdown("### Basic Bayesian Statistics")
            
            # Mean and credible interval for the whole series
            mean_val = data['value'].mean()
            std_val = data['value'].std()
            
            # Create a simple Bayesian model for the mean
            # Using normal-normal conjugate prior for simplicity
            prior_mean = st.slider("Prior mean", float(mean_val - 3*std_val), float(mean_val + 3*std_val), float(mean_val))
            prior_std = st.slider("Prior standard deviation", 0.1, float(5*std_val), float(2*std_val))
            
            # Calculate posterior distribution for the mean
            n = len(data)
            posterior_mean = (prior_mean / (prior_std**2) + data['value'].sum() / (std_val**2)) / \
                             (1 / (prior_std**2) + n / (std_val**2))
            posterior_std = np.sqrt(1 / (1 / (prior_std**2) + n / (std_val**2)))
            
            # Display results
            st.markdown(f"**Sample mean:** {mean_val:.4f}")
            st.markdown(f"**Sample standard deviation:** {std_val:.4f}")
            st.markdown(f"**Posterior mean:** {posterior_mean:.4f}")
            st.markdown(f"**Posterior standard deviation:** {posterior_std:.4f}")
            
            # Plot the posterior distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.linspace(posterior_mean - 4*posterior_std, posterior_mean + 4*posterior_std, 1000)
            prior = stats.norm.pdf(x, prior_mean, prior_std)
            posterior = stats.norm.pdf(x, posterior_mean, posterior_std)
            
            ax.plot(x, prior, label=f'Prior: N({prior_mean:.2f}, {prior_std:.2f})', linestyle='--')
            ax.plot(x, posterior, label=f'Posterior: N({posterior_mean:.2f}, {posterior_std:.2f})')
            ax.axvline(x=mean_val, color='green', linestyle=':', label=f'Sample Mean: {mean_val:.2f}')
            
            ax.set_title("Posterior Distribution of Mean")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
        elif analysis_option == "Decomposition":
            st.markdown("### Time Series Decomposition")
            st.markdown("""
            Bayesian time series decomposition separates a time series into:
            - Trend component
            - Seasonal component
            - Remainder (unexplained variation)
            """)
            
            # Parameters for decomposition
            season_length = st.slider("Season length", 2, 52, 12)
            
            if st.button("Run Bayesian Decomposition"):
                with st.spinner("Running Bayesian decomposition..."):
                    decomposition = bayesian_decomposition(data, season_length)
                
                # Display results
                fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
                
                # Original series
                axes[0].plot(data['time'], data['value'])
                axes[0].set_title("Original Time Series")
                
                # Trend
                axes[1].plot(data['time'], decomposition['trend'])
                axes[1].set_title("Trend Component")
                
                # Seasonality
                axes[2].plot(data['time'], decomposition['seasonal'])
                axes[2].set_title("Seasonal Component")
                
                # Remainder
                axes[3].plot(data['time'], decomposition['remainder'])
                axes[3].set_title("Remainder Component")
                axes[3].set_xlabel("Time")
                
                plt.tight_layout()
                st.pyplot(fig)
        
        elif analysis_option == "Change Point Detection":
            # Reuse the change point detection functionality
            st.markdown("### Change Point Detection")
            
            prior_change_prob = st.slider(
                "Prior probability of change at each point (%)", 
                0.1, 10.0, 1.0, 0.1
            ) / 100
            
            n_iterations = st.slider("Number of MCMC iterations", 1000, 10000, 5000, 1000)
            
            if st.button("Run Change Point Detection"):
                with st.spinner("Running Bayesian change point detection..."):
                    change_point_probs = detect_change_points(data, prior_change_prob, n_iterations)
                
                # Display results
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                
                # Plot original time series
                ax1.plot(data['time'], data['value'])
                ax1.set_title("Original Time Series")
                ax1.set_ylabel("Value")
                
                # Plot posterior probability of change points
                ax2.bar(data['time'], change_point_probs, alpha=0.7)
                ax2.set_title("Posterior Probability of Change Points")
                ax2.set_xlabel("Time")
                ax2.set_ylabel("Probability")
                
                st.pyplot(fig)
                
        elif analysis_option == "Forecasting":
            # Reuse the forecasting functionality
            st.markdown("### Bayesian Forecasting")
            
            forecast_horizon = st.slider("Forecast horizon (number of time points)", 1, 50, 20)
            
            include_trend = st.checkbox("Include trend component", True)
            if include_trend:
                trend_flexibility = st.slider(
                    "Trend flexibility (higher = more flexible)", 
                    0.01, 1.0, 0.1, 0.01
                )
            
            include_seasonality = st.checkbox("Include seasonal component", True)
            if include_seasonality:
                season_length = st.slider("Season length", 2, 52, 12)
            
            if st.button("Run Forecasting"):
                with st.spinner("Running Bayesian forecasting..."):
                    forecast_result = bayesian_forecast(
                        data, 
                        forecast_horizon,
                        include_trend=include_trend,
                        trend_flexibility=trend_flexibility if include_trend else 0.1,
                        include_seasonality=include_seasonality,
                        season_length=season_length if include_seasonality else 12
                    )
                
                # Display results
                forecast_df = pd.DataFrame({
                    'time': range(data['time'].iloc[-1] + 1, data['time'].iloc[-1] + 1 + forecast_horizon),
                    'forecast_mean': forecast_result['mean'],
                    'forecast_lower': forecast_result['lower'],
                    'forecast_upper': forecast_result['upper']
                })
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot original time series
                ax.plot(data['time'], data['value'], label='Historical Data')
                
                # Plot forecast
                ax.plot(forecast_df['time'], forecast_df['forecast_mean'], 
                        label='Forecast', color='red')
                
                # Plot prediction intervals
                ax.fill_between(
                    forecast_df['time'],
                    forecast_df['forecast_lower'],
                    forecast_df['forecast_upper'],
                    color='red', alpha=0.2,
                    label='95% Credible Interval'
                )
                
                ax.set_title("Bayesian Time Series Forecast")
                ax.set_xlabel("Time")
                ax.set_ylabel("Value")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
    
    else:
        st.warning("Please upload a CSV file to proceed with custom analysis.")

# Helper functions for time series analysis

def generate_synthetic_data_with_change_points(n_points, change_points, noise_level=0.5):
    """Generate synthetic time series data with change points"""
    time = list(range(n_points))
    value = []
    
    # Initialize with a random value
    current_level = np.random.normal(0, 1)
    
    for t in time:
        # Check if this is a change point
        if t in change_points:
            # Generate a new level (change in mean)
            current_level += np.random.normal(0, 2)
        
        # Generate value with noise
        value.append(current_level + np.random.normal(0, noise_level))
    
    return pd.DataFrame({'time': time, 'value': value})

def generate_synthetic_data_with_trend_seasonality(
    n_points, trend_type, seasonal_period=12, seasonal_amplitude=2.0, noise_level=0.5
):
    """Generate synthetic time series data with trend and seasonality"""
    time = list(range(n_points))
    value = []
    
    # Generate trend component
    if trend_type == "Linear increasing":
        trend = [0.05 * t for t in time]
    elif trend_type == "Linear decreasing":
        trend = [-0.05 * t for t in time]
    else:  # No trend
        trend = [0 for _ in time]
    
    # Generate seasonal component
    if seasonal_period > 0:
        seasonal = [seasonal_amplitude * np.sin(2 * np.pi * t / seasonal_period) for t in time]
    else:
        seasonal = [0 for _ in time]
    
    # Combine components with noise
    for t in range(n_points):
        value.append(trend[t] + seasonal[t] + np.random.normal(0, noise_level))
    
    return pd.DataFrame({'time': time, 'value': value})

def detect_change_points(data, prior_change_prob=0.01, n_iterations=5000):
    """
    Detect change points in time series data using a Bayesian approach
    
    This is a simplified implementation that uses a Bayesian online changepoint detection algorithm.
    In a real implementation, you would likely use a more sophisticated method.
    """
    n_points = len(data)
    
    # For this example, we'll use a simple heuristic based on moving windows to approximate
    # the Bayesian changepoint detection results
    window_size = max(10, n_points // 20)
    change_point_probs = np.zeros(n_points)
    
    for i in range(window_size, n_points - window_size):
        # Calculate means and variances of the two windows
        window1 = data['value'].iloc[i-window_size:i]
        window2 = data['value'].iloc[i:i+window_size]
        
        mean1, mean2 = window1.mean(), window2.mean()
        var1, var2 = window1.var(), window2.var()
        
        # Calculate the Bayes factor (approximation)
        # Higher values indicate higher probability of a change point
        if var1 > 0 and var2 > 0:
            mean_diff = abs(mean1 - mean2)
            pooled_std = np.sqrt((var1 + var2) / 2)
            
            # Calculate a rough approximation of the posterior probability
            # This is based on a normal-normal model with uninformative prior
            z_score = mean_diff / (pooled_std * np.sqrt(2/window_size))
            
            # Convert to a probability using a heuristic
            change_point_probs[i] = min(0.99, 1 - np.exp(-0.5 * z_score**2))
            
            # Apply the prior probability
            change_point_probs[i] = (change_point_probs[i] * prior_change_prob) / \
                                   (change_point_probs[i] * prior_change_prob + (1 - change_point_probs[i]) * (1 - prior_change_prob))
    
    return change_point_probs

def bayesian_forecast(data, horizon, include_trend=True, trend_flexibility=0.1, 
                     include_seasonality=True, season_length=12):
    """
    Generate Bayesian forecast for time series data
    
    This is a simplified implementation that approximates a Bayesian structural time series model.
    In a real implementation, you would likely use a more sophisticated method.
    """
    # Extract the time series values
    values = data['value'].values
    n_points = len(values)
    
    # Fit a simple model using statsmodels (as an approximation to a Bayesian model)
    # In a real implementation, you would use proper Bayesian inference
    
    # Simple forecasting based on historical mean and trend
    if include_trend:
        # Calculate trend using simple linear regression
        t = np.arange(n_points)
        slope, intercept = np.polyfit(t, values, 1)
        
        # Adjust the slope based on trend_flexibility
        # Lower flexibility = more constrained to historical trend
        adjusted_slope = slope * (1 - np.exp(-10 * trend_flexibility))
        
        # Generate the mean forecast
        forecast_mean = intercept + adjusted_slope * (t[-1] + 1 + np.arange(horizon))
    else:
        # Use the historical mean
        forecast_mean = np.ones(horizon) * np.mean(values)
    
    # Add seasonal component if requested
    if include_seasonality and n_points >= season_length:
        # Extract the last full season from the data
        last_season = values[-season_length:]
        
        # Repeat the seasonal pattern for the forecast horizon
        seasonal_component = np.tile(last_season, int(np.ceil(horizon / season_length)))[:horizon]
        
        # Normalize the seasonal component to have zero mean
        seasonal_component = seasonal_component - np.mean(seasonal_component)
        
        # Add the seasonal component to the forecast
        forecast_mean = forecast_mean + seasonal_component
    
    # Calculate the prediction intervals
    # This is a simple approximation based on historical variance
    forecast_std = np.std(values) * np.sqrt(1 + np.arange(horizon) / 10)
    
    forecast_lower = forecast_mean - 1.96 * forecast_std
    forecast_upper = forecast_mean + 1.96 * forecast_std
    
    return {
        'mean': forecast_mean,
        'lower': forecast_lower,
        'upper': forecast_upper
    }

def bayesian_decomposition(data, season_length=12):
    """
    Perform Bayesian decomposition of time series data
    
    This is a simplified implementation that approximates a Bayesian decomposition.
    In a real implementation, you would use proper Bayesian inference.
    """
    # Extract the time series values
    values = data['value'].values
    n_points = len(values)
    
    # Simple decomposition (approximation to a Bayesian decomposition)
    
    # Trend component using a moving average
    window_size = min(n_points // 3, season_length * 2)
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size
    
    # Calculate trend using LOESS smoothing (approximation)
    t = np.arange(n_points)
    trend = np.zeros(n_points)
    
    # Simple weighted moving average as an approximation
    for i in range(n_points):
        weights = np.exp(-0.1 * np.abs(t - i))
        trend[i] = np.sum(weights * values) / np.sum(weights)
    
    # Calculate seasonal component
    # Detrend the data
    detrended = values - trend
    
    # Calculate seasonal component by averaging across seasons
    seasonal = np.zeros(n_points)
    if n_points >= season_length:
        # Reshape the detrended data to organize by season
        n_seasons = n_points // season_length
        seasonality_matrix = detrended[:n_seasons * season_length].reshape((n_seasons, season_length))
        
        # Calculate the average seasonal pattern
        avg_seasonal_pattern = np.mean(seasonality_matrix, axis=0)
        
        # Normalize to have zero mean
        avg_seasonal_pattern = avg_seasonal_pattern - np.mean(avg_seasonal_pattern)
        
        # Tile the seasonal pattern to cover the entire time series
        seasonal_full = np.tile(avg_seasonal_pattern, n_seasons + 1)
        seasonal = seasonal_full[:n_points]
    
    # Calculate remainder
    remainder = values - trend - seasonal
    
    return {
        'trend': trend,
        'seasonal': seasonal,
        'remainder': remainder
    } 
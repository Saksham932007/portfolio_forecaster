"""Portfolio Forecasting System - Streamlit Web Interface."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys
import warnings

# Suppress warnings for cleaner interface
warnings.filterwarnings('ignore')

# Add project paths for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'utils'))
sys.path.append(os.path.join(project_root, 'models'))

# Import project modules
try:
    from data_loader import load_data, get_ticker_data, get_multiple_tickers_data, get_data_summary
    from evaluation import calculate_all_metrics, compare_forecasts
    from baseline import naive_forecast, arima_forecast, moving_average_forecast
    from var_model import VARModel
    from deep_learning_model import create_deep_learning_forecaster
    from run_backtest import SimpleBacktester
except ImportError as e:
    st.error(f"Error importing project modules: {e}")
    st.error("Please ensure all required files are in the correct directories")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="Portfolio Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #2E86AB;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin: 1rem 0;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
    
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_app_data():
    """Load and cache the stock data."""
    data_path = os.path.join(project_root, 'data', 'mock_stock_data.csv')
    if not os.path.exists(data_path):
        st.error(f"Data file not found: {data_path}")
        return None
    
    data = load_data(data_path)
    return data


def create_time_series_plot(data, title="Time Series Data"):
    """Create an interactive time series plot."""
    fig = go.Figure()
    
    if isinstance(data, pd.Series):
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.values,
            mode='lines',
            name=title,
            line=dict(color='#2E86AB', width=2)
        ))
    elif isinstance(data, pd.DataFrame):
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7']
        for i, column in enumerate(data.columns):
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[column],
                mode='lines',
                name=column,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        showlegend=True,
        height=500
    )
    
    return fig


def create_forecast_plot(actual, forecasts_dict, title="Forecast Comparison"):
    """Create a plot comparing actual vs forecasted values."""
    fig = go.Figure()
    
    # Plot actual data
    fig.add_trace(go.Scatter(
        x=actual.index,
        y=actual.values,
        mode='lines',
        name='Actual',
        line=dict(color='black', width=3)
    ))
    
    # Plot forecasts
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7']
    for i, (model_name, forecast) in enumerate(forecasts_dict.items()):
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode='lines',
            name=f'{model_name} Forecast',
            line=dict(color=colors[i % len(colors)], width=2, dash='dash')
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        showlegend=True,
        height=500
    )
    
    return fig


def sidebar_navigation():
    """Create sidebar navigation."""
    st.sidebar.markdown("## 📈 Portfolio Forecaster")
    st.sidebar.markdown("---")
    
    pages = {
        "🏠 Home": "home",
        "📊 Data Exploration": "data",
        "🔍 Individual Forecasting": "forecast",
        "📈 Model Comparison": "comparison",
        "🔗 Multivariate Analysis": "multivariate",
        "💼 Portfolio Optimization": "portfolio",
        "🎯 Backtesting": "backtest",
        "❓ Help": "help"
    }
    
    selected_page = st.sidebar.selectbox("Navigate to:", list(pages.keys()))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Stats")
    
    # Quick stats section (will be populated when data is loaded)
    if 'data' in st.session_state and st.session_state.data is not None:
        data_summary = get_data_summary(st.session_state.data)
        st.sidebar.metric("Available Tickers", len(data_summary['tickers']))
        st.sidebar.metric("Date Range", f"{data_summary['date_range'][0].strftime('%Y-%m-%d')} to {data_summary['date_range'][1].strftime('%Y-%m-%d')}")
        st.sidebar.metric("Total Records", data_summary['total_records'])
    
    return pages[selected_page]


def page_home():
    """Home page content."""
    st.markdown('<div class="main-header">Portfolio Forecasting System</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the Portfolio Forecasting System! This application provides comprehensive tools 
    for analyzing and forecasting stock prices using various time series models.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔍 **Forecasting Models**
        - Baseline models (Naive, Moving Average, ARIMA)
        - Vector Autoregression (VAR)
        - Deep Learning (DeepAR, TFT)
        """)
    
    with col2:
        st.markdown("""
        ### 📊 **Analysis Tools**
        - Interactive data visualization
        - Model performance comparison
        - Backtesting framework
        """)
    
    with col3:
        st.markdown("""
        ### 💼 **Portfolio Features**
        - Multi-asset forecasting
        - Portfolio optimization
        - Risk analysis
        """)
    
    st.markdown("---")
    
    # Data loading section
    st.markdown('<div class="sub-header">Getting Started</div>', unsafe_allow_html=True)
    
    if st.button("Load Sample Data", type="primary"):
        with st.spinner("Loading stock data..."):
            data = load_app_data()
            if data is not None:
                st.session_state.data = data
                st.success("✅ Data loaded successfully!")
                
                # Show data summary
                summary = get_data_summary(data)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Tickers", len(summary['tickers']))
                with col2:
                    st.metric("Records", summary['total_records'])
                with col3:
                    st.metric("Start Date", summary['date_range'][0].strftime('%Y-%m-%d'))
                with col4:
                    st.metric("End Date", summary['date_range'][1].strftime('%Y-%m-%d'))
                
                st.info("🎯 Use the sidebar to navigate to different analysis tools!")
            else:
                st.error("❌ Failed to load data. Please check the data file.")


def page_data_exploration():
    """Data exploration page."""
    st.markdown('<div class="main-header">Data Exploration</div>', unsafe_allow_html=True)
    
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please load data first from the Home page.")
        return
    
    data = st.session_state.data
    summary = get_data_summary(data)
    
    # Create tabs for different types of analysis
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Single Ticker", "📈 Multi-Ticker", "🔍 Technical Analysis", "📋 Data Summary"])
    
    with tab1:
        st.markdown("### Single Ticker Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_ticker = st.selectbox("Choose a ticker:", summary['tickers'], key="single_ticker")
        with col2:
            selected_column = st.selectbox("Choose a column:", ['open', 'high', 'low', 'close', 'volume'], 
                                         index=3, key="single_column")  # Default to 'close'
        
        # Get ticker data
        ticker_data = get_ticker_data(data, selected_ticker, selected_column)
        
        # Display basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"${ticker_data.mean():.2f}")
        with col2:
            st.metric("Std Dev", f"${ticker_data.std():.2f}")
        with col3:
            st.metric("Min", f"${ticker_data.min():.2f}")
        with col4:
            st.metric("Max", f"${ticker_data.max():.2f}")
        
        # Interactive plot
        fig = create_time_series_plot(ticker_data, f"{selected_ticker} - {selected_column.title()}")
        st.plotly_chart(fig, width='stretch')
        
        # Additional analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Recent Performance")
            recent_data = ticker_data.tail(30)
            pct_change = ((recent_data.iloc[-1] - recent_data.iloc[0]) / recent_data.iloc[0]) * 100
            st.metric("30-Day Change", f"{pct_change:.2f}%", 
                     delta=f"{ticker_data.iloc[-1] - ticker_data.iloc[-2]:.2f}")
        
        with col2:
            st.markdown("#### Volatility Analysis")
            returns = ticker_data.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            st.metric("Annualized Volatility", f"{volatility:.2%}")
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            st.metric("Sharpe Ratio (approx)", f"{sharpe_ratio:.2f}")
        
        # Historical statistics
        if st.checkbox("Show historical statistics", key="hist_stats"):
            st.markdown("#### Historical Statistics")
            stats_df = pd.DataFrame({
                'Statistic': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
                'Value': [
                    len(ticker_data),
                    ticker_data.mean(),
                    ticker_data.std(),
                    ticker_data.min(),
                    ticker_data.quantile(0.25),
                    ticker_data.median(),
                    ticker_data.quantile(0.75),
                    ticker_data.max()
                ]
            })
            st.dataframe(stats_df, hide_index=True)
    
    with tab2:
        st.markdown("### Multi-Ticker Comparison")
        
        # Ticker selection for comparison
        selected_tickers = st.multiselect(
            "Select tickers to compare:", 
            summary['tickers'], 
            default=summary['tickers'][:3],  # Default to first 3
            key="multi_tickers"
        )
        
        if selected_tickers:
            selected_column_multi = st.selectbox("Choose a column:", ['open', 'high', 'low', 'close', 'volume'], 
                                                index=3, key="multi_column")
            
            # Get multi-ticker data
            multi_data = get_multiple_tickers_data(data, selected_tickers, selected_column_multi)
            
            if not multi_data.empty:
                # Normalized comparison (base 100)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Absolute Prices")
                    fig_abs = create_time_series_plot(multi_data, f"Multi-Ticker {selected_column_multi.title()} Comparison")
                    st.plotly_chart(fig_abs, width='stretch')
                
                with col2:
                    st.markdown("#### Normalized Comparison (Base 100)")
                    normalized_data = multi_data.div(multi_data.iloc[0]) * 100
                    fig_norm = create_time_series_plot(normalized_data, "Normalized Performance (Base 100)")
                    st.plotly_chart(fig_norm, width='stretch')
                
                # Correlation matrix
                st.markdown("#### Correlation Matrix")
                corr_matrix = multi_data.pct_change().corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    title="Correlation Matrix of Returns",
                    color_continuous_scale="RdBu_r",
                    aspect="auto"
                )
                st.plotly_chart(fig_corr, width='stretch')
                
                # Performance summary
                st.markdown("#### Performance Summary")
                performance_data = []
                for ticker in selected_tickers:
                    ticker_series = multi_data[ticker]
                    total_return = ((ticker_series.iloc[-1] - ticker_series.iloc[0]) / ticker_series.iloc[0]) * 100
                    volatility = ticker_series.pct_change().std() * np.sqrt(252) * 100
                    
                    performance_data.append({
                        'Ticker': ticker,
                        'Total Return (%)': f"{total_return:.2f}",
                        'Volatility (%)': f"{volatility:.2f}",
                        'Current Price': f"${ticker_series.iloc[-1]:.2f}",
                        'Max Price': f"${ticker_series.max():.2f}",
                        'Min Price': f"${ticker_series.min():.2f}"
                    })
                
                performance_df = pd.DataFrame(performance_data)
                st.dataframe(performance_df, hide_index=True)
            else:
                st.error("No data available for selected tickers.")
        else:
            st.info("Please select at least one ticker for comparison.")
    
    with tab3:
        st.markdown("### Technical Analysis")
        
        selected_ticker_tech = st.selectbox("Choose a ticker for technical analysis:", 
                                          summary['tickers'], key="tech_ticker")
        
        # Get OHLC data for technical analysis
        ohlc_data = {}
        for col in ['open', 'high', 'low', 'close', 'volume']:
            ohlc_data[col] = get_ticker_data(data, selected_ticker_tech, col)
        
        ohlc_df = pd.DataFrame(ohlc_data)
        
        # Moving averages
        ma_windows = st.multiselect("Select moving average windows:", [5, 10, 20, 50], 
                                   default=[10, 20], key="ma_windows")
        
        # Create technical analysis plot
        fig_tech = go.Figure()
        
        # Add candlestick chart
        fig_tech.add_trace(go.Candlestick(
            x=ohlc_df.index,
            open=ohlc_df['open'],
            high=ohlc_df['high'],
            low=ohlc_df['low'],
            close=ohlc_df['close'],
            name='Price'
        ))
        
        # Add moving averages
        colors = ['blue', 'red', 'green', 'orange']
        for i, window in enumerate(ma_windows):
            ma = ohlc_df['close'].rolling(window=window).mean()
            fig_tech.add_trace(go.Scatter(
                x=ma.index,
                y=ma.values,
                mode='lines',
                name=f'MA{window}',
                line=dict(color=colors[i % len(colors)])
            ))
        
        fig_tech.update_layout(
            title=f"{selected_ticker_tech} - Technical Analysis",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=600
        )
        
        st.plotly_chart(fig_tech, width='stretch')
        
        # Technical indicators summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Price Levels")
            current_price = ohlc_df['close'].iloc[-1]
            high_52w = ohlc_df['high'].tail(252).max() if len(ohlc_df) >= 252 else ohlc_df['high'].max()
            low_52w = ohlc_df['low'].tail(252).min() if len(ohlc_df) >= 252 else ohlc_df['low'].min()
            
            st.metric("Current Price", f"${current_price:.2f}")
            st.metric("52W High", f"${high_52w:.2f}")
            st.metric("52W Low", f"${low_52w:.2f}")
        
        with col2:
            st.markdown("#### Volume Analysis")
            avg_volume = ohlc_df['volume'].mean()
            recent_volume = ohlc_df['volume'].tail(5).mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            st.metric("Avg Volume", f"{avg_volume:,.0f}")
            st.metric("Recent Volume", f"{recent_volume:,.0f}")
            st.metric("Volume Ratio", f"{volume_ratio:.2f}x")
        
        with col3:
            st.markdown("#### Trend Analysis")
            returns_5d = ((current_price - ohlc_df['close'].iloc[-6]) / ohlc_df['close'].iloc[-6]) * 100 if len(ohlc_df) > 5 else 0
            returns_20d = ((current_price - ohlc_df['close'].iloc[-21]) / ohlc_df['close'].iloc[-21]) * 100 if len(ohlc_df) > 20 else 0
            
            st.metric("5-Day Return", f"{returns_5d:.2f}%")
            st.metric("20-Day Return", f"{returns_20d:.2f}%")
            
            # Simple trend indicator
            if ma_windows and len(ma_windows) >= 2:
                ma_short = ohlc_df['close'].rolling(window=min(ma_windows)).mean().iloc[-1]
                ma_long = ohlc_df['close'].rolling(window=max(ma_windows)).mean().iloc[-1]
                trend = "Bullish" if ma_short > ma_long else "Bearish"
                st.metric("Trend", trend)
    
    with tab4:
        st.markdown("### Data Summary")
        
        # Overall dataset summary
        st.markdown("#### Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tickers", len(summary['tickers']))
        with col2:
            st.metric("Total Records", summary['total_records'])
        with col3:
            st.metric("Date Range", f"{summary['num_days']} days")
        with col4:
            st.metric("Columns", len(summary['columns']))
        
        # Ticker-by-ticker summary
        st.markdown("#### Ticker Summary")
        ticker_summary_data = []
        
        for ticker in summary['tickers']:
            ticker_close = get_ticker_data(data, ticker, 'close')
            ticker_volume = get_ticker_data(data, ticker, 'volume')
            
            total_return = ((ticker_close.iloc[-1] - ticker_close.iloc[0]) / ticker_close.iloc[0]) * 100
            volatility = ticker_close.pct_change().std() * np.sqrt(252) * 100
            avg_volume = ticker_volume.mean()
            
            ticker_summary_data.append({
                'Ticker': ticker,
                'Start Price': f"${ticker_close.iloc[0]:.2f}",
                'End Price': f"${ticker_close.iloc[-1]:.2f}",
                'Total Return (%)': f"{total_return:.2f}",
                'Volatility (%)': f"{volatility:.2f}",
                'Avg Volume': f"{avg_volume:,.0f}",
                'Data Points': len(ticker_close)
            })
        
        ticker_summary_df = pd.DataFrame(ticker_summary_data)
        st.dataframe(ticker_summary_df, hide_index=True)
        
        # Data quality check
        st.markdown("#### Data Quality")
        missing_data = data.isnull().sum().sum()
        completeness = ((len(data) - missing_data) / len(data)) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Missing Values", missing_data)
        with col2:
            st.metric("Completeness", f"{completeness:.1f}%")
        
        # Raw data preview
        if st.checkbox("Show raw data preview"):
            st.markdown("#### Raw Data Preview (Last 20 records)")
            st.dataframe(data.tail(20))


def page_individual_forecasting():
    """Individual forecasting page."""
    st.markdown('<div class="main-header">Individual Stock Forecasting</div>', unsafe_allow_html=True)
    
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please load data first from the Home page.")
        return
    
    st.markdown("### Configure Forecasting")
    
    data = st.session_state.data
    summary = get_data_summary(data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_ticker = st.selectbox("Select Ticker:", summary['tickers'])
        forecast_steps = st.slider("Forecast Steps:", 5, 50, 20)
    
    with col2:
        selected_models = st.multiselect(
            "Select Models:", 
            ['Naive', 'Moving Average', 'ARIMA', 'Deep Learning'],
            default=['Naive', 'Moving Average']
        )
        test_size = st.slider("Test Size (%):", 10, 50, 20)
    
    if st.button("Generate Forecasts", type="primary"):
        with st.spinner("Generating forecasts..."):
            # Get data
            ticker_data = get_ticker_data(data, selected_ticker, 'close')
            
            # Split data
            split_point = int(len(ticker_data) * (1 - test_size/100))
            train_data = ticker_data.iloc[:split_point]
            test_data = ticker_data.iloc[split_point:]
            
            forecasts = {}
            metrics = {}
            
            # Generate forecasts based on selected models
            if 'Naive' in selected_models:
                try:
                    naive_pred = naive_forecast(train_data, forecast_steps=len(test_data))
                    forecasts['Naive'] = naive_pred
                    metrics['Naive'] = calculate_all_metrics(test_data.values, naive_pred.values)
                except Exception as e:
                    st.error(f"Naive forecast error: {e}")
            
            if 'Moving Average' in selected_models:
                try:
                    ma_pred = moving_average_forecast(train_data, window=10, forecast_steps=len(test_data))
                    forecasts['Moving Average'] = ma_pred
                    metrics['Moving Average'] = calculate_all_metrics(test_data.values, ma_pred.values)
                except Exception as e:
                    st.error(f"Moving Average forecast error: {e}")
            
            if 'ARIMA' in selected_models:
                try:
                    arima_pred = arima_forecast(train_data, order=(1,1,1), forecast_steps=len(test_data))
                    forecasts['ARIMA'] = arima_pred
                    metrics['ARIMA'] = calculate_all_metrics(test_data.values, arima_pred.values)
                except Exception as e:
                    st.error(f"ARIMA forecast error: {e}")
            
            if 'Deep Learning' in selected_models:
                try:
                    # Simple deep learning forecast (using mock implementation)
                    dl_forecaster = create_deep_learning_forecaster()
                    dl_data = pd.DataFrame({'close': ticker_data}).reset_index()
                    dl_data['time_idx'] = range(len(dl_data))
                    dl_data['ticker'] = selected_ticker
                    
                    dl_pred, _ = dl_forecaster.fit_and_predict_deepar(dl_data, 'close', max_epochs=1)
                    
                    # Align with test data
                    if len(dl_pred) >= len(test_data):
                        dl_pred_aligned = pd.Series(dl_pred[-len(test_data):], index=test_data.index)
                        forecasts['Deep Learning'] = dl_pred_aligned
                        metrics['Deep Learning'] = calculate_all_metrics(test_data.values, dl_pred_aligned.values)
                except Exception as e:
                    st.error(f"Deep Learning forecast error: {e}")
            
            if forecasts:
                # Display results
                st.markdown("### Forecast Results")
                
                # Plot
                fig = create_forecast_plot(test_data, forecasts, f"{selected_ticker} Forecast Comparison")
                st.plotly_chart(fig, width='stretch')
                
                # Metrics table
                if metrics:
                    st.markdown("### Performance Metrics")
                    metrics_df = pd.DataFrame(metrics).T
                    st.dataframe(metrics_df.round(4))
                    
                    # Best model
                    best_model = metrics_df['RMSE'].idxmin()
                    st.success(f"🏆 Best Model: **{best_model}** (RMSE: {metrics_df.loc[best_model, 'RMSE']:.4f})")
            else:
                st.error("No forecasts were generated successfully.")


def page_model_comparison():
    """Model comparison page."""
    st.markdown('<div class="main-header">Model Comparison</div>', unsafe_allow_html=True)
    st.info("🚧 This feature will be implemented in the next phase.")


def page_multivariate_analysis():
    """Multivariate analysis page."""
    st.markdown('<div class="main-header">Multivariate Analysis</div>', unsafe_allow_html=True)
    st.info("🚧 This feature will be implemented in the next phase.")


def page_portfolio_optimization():
    """Portfolio optimization page."""
    st.markdown('<div class="main-header">Portfolio Optimization</div>', unsafe_allow_html=True)
    st.info("🚧 This feature will be implemented in the next phase.")


def page_backtesting():
    """Backtesting page."""
    st.markdown('<div class="main-header">Backtesting Framework</div>', unsafe_allow_html=True)
    st.info("🚧 This feature will be implemented in the next phase.")


def page_help():
    """Help and documentation page."""
    st.markdown('<div class="main-header">Help & Documentation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 📖 How to Use This Application
    
    ### 1. **Getting Started**
    - Start at the Home page and click "Load Sample Data"
    - Navigate using the sidebar menu
    
    ### 2. **Data Exploration**
    - View time series data for different tickers
    - Examine basic statistics and trends
    
    ### 3. **Individual Forecasting**
    - Select a ticker and forecasting models
    - Configure forecast horizon and test size
    - Compare model performance
    
    ### 4. **Available Models**
    
    #### Baseline Models
    - **Naive**: Uses the last observed value
    - **Moving Average**: Average of recent observations
    - **ARIMA**: Auto-regressive integrated moving average
    
    #### Advanced Models
    - **VAR**: Vector Autoregression for multivariate data
    - **Deep Learning**: DeepAR and TFT models
    
    ### 5. **Performance Metrics**
    - **RMSE**: Root Mean Square Error
    - **MAE**: Mean Absolute Error
    - **MAPE**: Mean Absolute Percentage Error
    
    ### 6. **Tips for Best Results**
    - Use at least 100 data points for reliable forecasts
    - Compare multiple models for robust predictions
    - Consider the forecast horizon when selecting models
    
    ### 7. **Technical Details**
    - Built with Python, Streamlit, and scientific libraries
    - Uses mock data for demonstration purposes
    - Implements graceful degradation for optional dependencies
    """)


def main():
    """Main application function."""
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    # Sidebar navigation
    current_page = sidebar_navigation()
    
    # Route to appropriate page
    if current_page == "home":
        page_home()
    elif current_page == "data":
        page_data_exploration()
    elif current_page == "forecast":
        page_individual_forecasting()
    elif current_page == "comparison":
        page_model_comparison()
    elif current_page == "multivariate":
        page_multivariate_analysis()
    elif current_page == "portfolio":
        page_portfolio_optimization()
    elif current_page == "backtest":
        page_backtesting()
    elif current_page == "help":
        page_help()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Portfolio Forecasting System | Built with ❤️ using Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
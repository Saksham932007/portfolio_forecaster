"""Portfolio Forecasting System - Streamlit Web Interface."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
    from performance_reporting import PerformanceReporter
    from api_integration import MarketDataAPI, AlternativeDataAPI
    from config_manager import config_manager
    from model_ensemble import AdvancedEnsemble
    from api_integration import MarketDataAPI, AlternativeDataAPI, get_market_overview
    from interactive_plotting import InteractivePlotter, quick_line_plot, create_dashboard_plot
    
    # Import scientific computing for optimization
    from scipy.optimize import minimize
    from scipy.stats import jarque_bera, skew, kurtosis
    from scipy import stats
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
        "🤖 Deep Learning": "deep_learning",
        "💼 Portfolio Optimization": "portfolio",
        "🎯 Backtesting": "backtest",
        "🛡️ Risk Management": "risk",
        "📋 Performance Reports": "reports",
        "🌐 Market Dashboard": "market",
        "🤖 Ensemble Models": "ensemble",
        "⚙️ Advanced Settings": "settings",
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
    
    data = st.session_state.data
    summary = get_data_summary(data)
    
    # Create tabs for different forecasting approaches
    tab1, tab2 = st.tabs(["🎯 Quick Forecast", "⚙️ Advanced Configuration"])
    
    with tab1:
        st.markdown("### Quick Forecasting Setup")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_ticker = st.selectbox("Select Ticker:", summary['tickers'], key="quick_ticker")
            forecast_steps = st.slider("Forecast Steps:", 5, 50, 20, key="quick_steps")
        
        with col2:
            selected_models = st.multiselect(
                "Select Models:", 
                ['Naive', 'Moving Average', 'ARIMA', 'Deep Learning'],
                default=['Naive', 'Moving Average'],
                key="quick_models"
            )
            test_size = st.slider("Test Size (%):", 10, 50, 20, key="quick_test")
        
        if st.button("Generate Quick Forecasts", type="primary", key="quick_forecast"):
            with st.spinner("Generating forecasts..."):
                # Get data
                ticker_data = get_ticker_data(data, selected_ticker, 'close')
                
                # Split data
                split_point = int(len(ticker_data) * (1 - test_size/100))
                train_data = ticker_data.iloc[:split_point]
                test_data = ticker_data.iloc[split_point:]
                
                forecasts = {}
                metrics = {}
                errors = []
                
                # Generate forecasts based on selected models
                if 'Naive' in selected_models:
                    try:
                        naive_pred = naive_forecast(train_data, forecast_steps=len(test_data))
                        forecasts['Naive'] = naive_pred
                        metrics['Naive'] = calculate_all_metrics(test_data.values, naive_pred.values)
                    except Exception as e:
                        errors.append(f"Naive forecast error: {str(e)}")
                
                if 'Moving Average' in selected_models:
                    try:
                        ma_pred = moving_average_forecast(train_data, window=10, forecast_steps=len(test_data))
                        forecasts['Moving Average'] = ma_pred
                        metrics['Moving Average'] = calculate_all_metrics(test_data.values, ma_pred.values)
                    except Exception as e:
                        errors.append(f"Moving Average forecast error: {str(e)}")
                
                if 'ARIMA' in selected_models:
                    try:
                        arima_pred = arima_forecast(train_data, order=(1,1,1), forecast_steps=len(test_data))
                        forecasts['ARIMA'] = arima_pred
                        metrics['ARIMA'] = calculate_all_metrics(test_data.values, arima_pred.values)
                    except Exception as e:
                        errors.append(f"ARIMA forecast error: {str(e)}")
                
                if 'Deep Learning' in selected_models:
                    try:
                        dl_forecaster = create_deep_learning_forecaster()
                        dl_data = pd.DataFrame({'close': ticker_data}).reset_index()
                        dl_data['time_idx'] = range(len(dl_data))
                        dl_data['ticker'] = selected_ticker
                        
                        dl_pred, _ = dl_forecaster.fit_and_predict_deepar(dl_data, 'close', max_epochs=1)
                        
                        if len(dl_pred) >= len(test_data):
                            dl_pred_aligned = pd.Series(dl_pred[-len(test_data):], index=test_data.index)
                            forecasts['Deep Learning'] = dl_pred_aligned
                            metrics['Deep Learning'] = calculate_all_metrics(test_data.values, dl_pred_aligned.values)
                    except Exception as e:
                        errors.append(f"Deep Learning forecast error: {str(e)}")
                
                # Display errors if any
                for error in errors:
                    st.error(error)
                
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
                        
                        # Model insights
                        st.markdown("### Model Insights")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Accuracy Ranking")
                            ranking = metrics_df.sort_values('RMSE')[['RMSE', 'MAPE']].round(4)
                            for i, (model, row) in enumerate(ranking.iterrows(), 1):
                                st.write(f"{i}. **{model}**: RMSE {row['RMSE']:.4f}, MAPE {row['MAPE']:.2f}%")
                        
                        with col2:
                            st.markdown("#### Performance Summary")
                            avg_rmse = metrics_df['RMSE'].mean()
                            best_rmse = metrics_df['RMSE'].min()
                            worst_rmse = metrics_df['RMSE'].max()
                            
                            st.metric("Average RMSE", f"{avg_rmse:.4f}")
                            st.metric("Best RMSE", f"{best_rmse:.4f}")
                            st.metric("RMSE Range", f"{worst_rmse - best_rmse:.4f}")
                else:
                    st.error("No forecasts were generated successfully. Please check your model selections and try again.")
    
    with tab2:
        st.markdown("### Advanced Forecasting Configuration")
        
        # Model selection with parameters
        st.markdown("#### Model Selection & Parameters")
        
        # Baseline Models Section
        st.markdown("##### Baseline Models")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_naive = st.checkbox("Naive Forecast", value=True, key="adv_naive")
            
            use_ma = st.checkbox("Moving Average", value=True, key="adv_ma")
            if use_ma:
                ma_window = st.slider("MA Window:", 3, 50, 10, key="adv_ma_window")
                ma_multiple = st.checkbox("Multiple MA Windows", key="adv_ma_multi")
                if ma_multiple:
                    ma_windows = st.multiselect("MA Windows:", [5, 10, 15, 20, 30], 
                                               default=[5, 10, 20], key="adv_ma_windows")
        
        with col2:
            use_arima = st.checkbox("ARIMA", value=True, key="adv_arima")
            if use_arima:
                arima_p = st.slider("ARIMA p (AR order):", 0, 5, 1, key="adv_arima_p")
                arima_d = st.slider("ARIMA d (Differencing):", 0, 2, 1, key="adv_arima_d")
                arima_q = st.slider("ARIMA q (MA order):", 0, 5, 1, key="adv_arima_q")
                auto_arima = st.checkbox("Auto-select ARIMA parameters", key="adv_auto_arima")
        
        # Advanced Models Section
        st.markdown("##### Advanced Models")
        
        col3, col4 = st.columns(2)
        
        with col3:
            use_seasonal = st.checkbox("Seasonal Naive", key="adv_seasonal")
            if use_seasonal:
                seasonal_period = st.slider("Seasonal Period:", 5, 50, 12, key="adv_seasonal_period")
        
        with col4:
            use_dl = st.checkbox("Deep Learning Models", key="adv_dl")
            if use_dl:
                dl_epochs = st.slider("Training Epochs:", 1, 10, 3, key="adv_dl_epochs")
                dl_models = st.multiselect("DL Models:", ['DeepAR', 'TFT'], 
                                         default=['DeepAR'], key="adv_dl_models")
        
        # Data Configuration
        st.markdown("#### Data Configuration")
        
        col5, col6 = st.columns(2)
        
        with col5:
            adv_ticker = st.selectbox("Select Ticker:", summary['tickers'], key="adv_ticker")
            adv_column = st.selectbox("Data Column:", ['close', 'open', 'high', 'low'], 
                                    index=0, key="adv_column")
        
        with col6:
            adv_test_size = st.slider("Test Size (%):", 5, 50, 20, key="adv_test_size")
            adv_forecast_steps = st.slider("Forecast Horizon:", 5, 100, 30, key="adv_forecast_steps")
        
        # Additional Options
        st.markdown("#### Additional Options")
        
        col7, col8 = st.columns(2)
        
        with col7:
            confidence_intervals = st.checkbox("Show Confidence Intervals", key="adv_ci")
            if confidence_intervals:
                confidence_level = st.slider("Confidence Level (%):", 80, 99, 95, key="adv_ci_level")
        
        with col8:
            cross_validation = st.checkbox("Time Series Cross-Validation", key="adv_cv")
            if cross_validation:
                cv_folds = st.slider("CV Folds:", 3, 10, 5, key="adv_cv_folds")
        
        # Advanced Forecast Button
        if st.button("Generate Advanced Forecasts", type="primary", key="adv_forecast"):
            with st.spinner("Running advanced forecasting analysis..."):
                # Get data
                ticker_data = get_ticker_data(data, adv_ticker, adv_column)
                
                # Split data
                split_point = int(len(ticker_data) * (1 - adv_test_size/100))
                train_data = ticker_data.iloc[:split_point]
                test_data = ticker_data.iloc[split_point:]
                
                forecasts = {}
                metrics = {}
                model_details = {}
                errors = []
                
                st.markdown("### Advanced Forecast Results")
                
                # Progress tracking
                models_to_run = []
                if use_naive: models_to_run.append("Naive")
                if use_ma: 
                    if ma_multiple and ma_windows:
                        models_to_run.extend([f"MA_{w}" for w in ma_windows])
                    else:
                        models_to_run.append(f"MA_{ma_window}")
                if use_arima: models_to_run.append("ARIMA")
                if use_seasonal: models_to_run.append("Seasonal")
                if use_dl and dl_models: models_to_run.extend(dl_models)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, model in enumerate(models_to_run):
                    status_text.text(f"Running {model}...")
                    
                    try:
                        if model == "Naive":
                            pred = naive_forecast(train_data, forecast_steps=len(test_data))
                            forecasts[model] = pred
                            metrics[model] = calculate_all_metrics(test_data.values, pred.values)
                            model_details[model] = {"type": "baseline", "params": {}}
                        
                        elif model.startswith("MA_"):
                            window = int(model.split("_")[1])
                            pred = moving_average_forecast(train_data, window=window, forecast_steps=len(test_data))
                            forecasts[model] = pred
                            metrics[model] = calculate_all_metrics(test_data.values, pred.values)
                            model_details[model] = {"type": "baseline", "params": {"window": window}}
                        
                        elif model == "ARIMA":
                            if auto_arima:
                                # Simple auto-selection (try a few common orders)
                                best_aic = float('inf')
                                best_order = (1, 1, 1)
                                for p in [0, 1, 2]:
                                    for d in [0, 1]:
                                        for q in [0, 1, 2]:
                                            try:
                                                temp_pred = arima_forecast(train_data, order=(p,d,q), forecast_steps=len(test_data))
                                                # Simple AIC approximation
                                                temp_metrics = calculate_all_metrics(test_data.values, temp_pred.values)
                                                if temp_metrics['RMSE'] < best_aic:
                                                    best_aic = temp_metrics['RMSE']
                                                    best_order = (p, d, q)
                                            except:
                                                continue
                                order = best_order
                            else:
                                order = (arima_p, arima_d, arima_q)
                            
                            pred = arima_forecast(train_data, order=order, forecast_steps=len(test_data))
                            forecasts[model] = pred
                            metrics[model] = calculate_all_metrics(test_data.values, pred.values)
                            model_details[model] = {"type": "baseline", "params": {"order": order}}
                        
                        elif model == "Seasonal":
                            # Use seasonal naive (repeat pattern from seasonal_period ago)
                            if len(train_data) >= seasonal_period:
                                seasonal_pattern = train_data.iloc[-seasonal_period:].values
                                pred_values = np.tile(seasonal_pattern, int(np.ceil(len(test_data) / seasonal_period)))[:len(test_data)]
                                pred = pd.Series(pred_values, index=test_data.index)
                                forecasts[model] = pred
                                metrics[model] = calculate_all_metrics(test_data.values, pred.values)
                                model_details[model] = {"type": "seasonal", "params": {"period": seasonal_period}}
                        
                        elif model in ['DeepAR', 'TFT']:
                            dl_forecaster = create_deep_learning_forecaster()
                            dl_data = pd.DataFrame({adv_column: ticker_data}).reset_index()
                            dl_data['time_idx'] = range(len(dl_data))
                            dl_data['ticker'] = adv_ticker
                            
                            if model == 'DeepAR':
                                pred_values, metadata = dl_forecaster.fit_and_predict_deepar(
                                    dl_data, adv_column, max_epochs=dl_epochs
                                )
                            else:  # TFT
                                pred_values, metadata = dl_forecaster.fit_and_predict_tft(
                                    dl_data, adv_column, max_epochs=dl_epochs
                                )
                            
                            if len(pred_values) >= len(test_data):
                                pred = pd.Series(pred_values[-len(test_data):], index=test_data.index)
                                forecasts[model] = pred
                                metrics[model] = calculate_all_metrics(test_data.values, pred.values)
                                model_details[model] = {"type": "deep_learning", "params": {"epochs": dl_epochs}, "metadata": metadata}
                    
                    except Exception as e:
                        errors.append(f"{model} forecast error: {str(e)}")
                    
                    progress_bar.progress((i + 1) / len(models_to_run))
                
                status_text.text("Analysis complete!")
                
                # Display errors if any
                if errors:
                    st.markdown("#### Warnings")
                    for error in errors:
                        st.warning(error)
                
                if forecasts:
                    # Plot results
                    fig = create_forecast_plot(test_data, forecasts, f"{adv_ticker} Advanced Forecast Comparison")
                    st.plotly_chart(fig, width='stretch')
                    
                    # Detailed metrics
                    st.markdown("#### Detailed Performance Analysis")
                    
                    metrics_df = pd.DataFrame(metrics).T
                    
                    # Add model type information
                    metrics_df['Model Type'] = [model_details[model]['type'] for model in metrics_df.index]
                    
                    # Reorder columns
                    cols = ['Model Type'] + [col for col in metrics_df.columns if col != 'Model Type']
                    metrics_df = metrics_df[cols]
                    
                    st.dataframe(metrics_df.round(4))
                    
                    # Best models by category
                    st.markdown("#### Best Models by Category")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Overall Best**")
                        overall_best = metrics_df['RMSE'].idxmin()
                        st.success(f"{overall_best}")
                        st.write(f"RMSE: {metrics_df.loc[overall_best, 'RMSE']:.4f}")
                    
                    with col2:
                        baseline_models = metrics_df[metrics_df['Model Type'] == 'baseline']
                        if not baseline_models.empty:
                            st.markdown("**Best Baseline**")
                            baseline_best = baseline_models['RMSE'].idxmin()
                            st.info(f"{baseline_best}")
                            st.write(f"RMSE: {baseline_models.loc[baseline_best, 'RMSE']:.4f}")
                    
                    with col3:
                        dl_models_df = metrics_df[metrics_df['Model Type'] == 'deep_learning']
                        if not dl_models_df.empty:
                            st.markdown("**Best Deep Learning**")
                            dl_best = dl_models_df['RMSE'].idxmin()
                            st.info(f"{dl_best}")
                            st.write(f"RMSE: {dl_models_df.loc[dl_best, 'RMSE']:.4f}")
                    
                    # Model parameters summary
                    st.markdown("#### Model Configuration Summary")
                    config_data = []
                    for model, details in model_details.items():
                        if model in metrics_df.index:
                            config_data.append({
                                'Model': model,
                                'Type': details['type'],
                                'Parameters': str(details['params']),
                                'RMSE': f"{metrics_df.loc[model, 'RMSE']:.4f}",
                                'MAPE': f"{metrics_df.loc[model, 'MAPE']:.2f}%"
                            })
                    
                    config_df = pd.DataFrame(config_data)
                    st.dataframe(config_df, hide_index=True)
                    
                else:
                    st.error("No forecasts were generated successfully. Please check your configuration and try again.")


def page_model_comparison():
    """Model comparison page."""
    st.markdown('<div class="main-header">Model Comparison</div>', unsafe_allow_html=True)
    
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please load data first from the Home page.")
        return
    
    data = st.session_state.data
    summary = get_data_summary(data)
    
    st.markdown("""
    Compare multiple forecasting models across different tickers and time horizons 
    to identify the best performing approaches for your portfolio.
    """)
    
    # Configuration section
    st.markdown("### Comparison Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        comparison_tickers = st.multiselect(
            "Select Tickers for Comparison:", 
            summary['tickers'], 
            default=summary['tickers'][:3],
            key="comp_tickers"
        )
        
        comparison_horizons = st.multiselect(
            "Forecast Horizons (days):",
            [5, 10, 15, 20, 30, 50],
            default=[10, 20, 30],
            key="comp_horizons"
        )
    
    with col2:
        comparison_models = st.multiselect(
            "Models to Compare:",
            ['Naive', 'MA_5', 'MA_10', 'MA_20', 'ARIMA_111', 'ARIMA_212', 'DeepAR'],
            default=['Naive', 'MA_10', 'ARIMA_111'],
            key="comp_models"
        )
        
        test_split = st.slider("Test Size (%):", 10, 40, 20, key="comp_test")
    
    if st.button("Run Model Comparison", type="primary", key="run_comparison"):
        if not comparison_tickers or not comparison_models or not comparison_horizons:
            st.error("Please select at least one ticker, model, and forecast horizon.")
            return
        
        # Initialize results storage
        comparison_results = []
        progress_total = len(comparison_tickers) * len(comparison_models) * len(comparison_horizons)
        progress_current = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run comparisons
        for ticker in comparison_tickers:
            # Get ticker data
            ticker_data = get_ticker_data(data, ticker, 'close')
            
            for horizon in comparison_horizons:
                # Adjust split based on horizon
                min_train_size = max(100, horizon * 3)  # Ensure enough training data
                if len(ticker_data) < min_train_size + horizon:
                    continue
                
                split_point = int(len(ticker_data) * (1 - test_split/100))
                train_data = ticker_data.iloc[:split_point]
                test_data = ticker_data.iloc[split_point:split_point + horizon]
                
                if len(test_data) < horizon:
                    test_data = ticker_data.iloc[-horizon:]  # Use last N points
                    train_data = ticker_data.iloc[:-horizon]
                
                for model in comparison_models:
                    progress_current += 1
                    status_text.text(f"Testing {model} on {ticker} (horizon: {horizon})")
                    
                    try:
                        # Generate forecast based on model type
                        if model == 'Naive':
                            forecast = naive_forecast(train_data, forecast_steps=len(test_data))
                        elif model.startswith('MA_'):
                            window = int(model.split('_')[1])
                            forecast = moving_average_forecast(train_data, window=window, forecast_steps=len(test_data))
                        elif model.startswith('ARIMA_'):
                            order_str = model.split('_')[1]
                            order = tuple(int(x) for x in order_str)
                            forecast = arima_forecast(train_data, order=order, forecast_steps=len(test_data))
                        elif model == 'DeepAR':
                            dl_forecaster = create_deep_learning_forecaster()
                            dl_data = pd.DataFrame({'close': ticker_data}).reset_index()
                            dl_data['time_idx'] = range(len(dl_data))
                            dl_data['ticker'] = ticker
                            
                            dl_pred, _ = dl_forecaster.fit_and_predict_deepar(dl_data, 'close', max_epochs=2)
                            forecast = pd.Series(dl_pred[-len(test_data):], index=test_data.index)
                        else:
                            continue
                        
                        # Calculate metrics
                        if len(forecast) == len(test_data):
                            metrics = calculate_all_metrics(test_data.values, forecast.values)
                            
                            comparison_results.append({
                                'Ticker': ticker,
                                'Model': model,
                                'Horizon': horizon,
                                'RMSE': metrics['RMSE'],
                                'MAE': metrics['MAE'], 
                                'MAPE': metrics['MAPE'],
                                'Data_Points': len(train_data)
                            })
                    
                    except Exception as e:
                        st.warning(f"Error with {model} on {ticker} (horizon {horizon}): {str(e)}")
                    
                    progress_bar.progress(progress_current / progress_total)
        
        status_text.text("Analysis complete!")
        
        if comparison_results:
            results_df = pd.DataFrame(comparison_results)
            
            # Display results
            st.markdown("### Comparison Results")
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary Table", "🏆 Best Models", "📈 Performance Charts", "🔍 Detailed Analysis"])
            
            with tab1:
                st.markdown("#### Complete Results Table")
                
                # Format the results table
                display_df = results_df.copy()
                display_df['RMSE'] = display_df['RMSE'].round(4)
                display_df['MAE'] = display_df['MAE'].round(4)
                display_df['MAPE'] = display_df['MAPE'].round(2)
                
                st.dataframe(display_df, hide_index=True)
                
                # Download option
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="model_comparison_results.csv",
                    mime="text/csv"
                )
            
            with tab2:
                st.markdown("#### Best Performing Models")
                
                # Best model overall
                best_overall = results_df.loc[results_df['RMSE'].idxmin()]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**🥇 Overall Best**")
                    st.success(f"**{best_overall['Model']}**")
                    st.write(f"Ticker: {best_overall['Ticker']}")
                    st.write(f"Horizon: {best_overall['Horizon']} days")
                    st.write(f"RMSE: {best_overall['RMSE']:.4f}")
                
                # Best by ticker
                with col2:
                    st.markdown("**🎯 Best by Ticker**")
                    for ticker in comparison_tickers:
                        ticker_results = results_df[results_df['Ticker'] == ticker]
                        if not ticker_results.empty:
                            best_ticker = ticker_results.loc[ticker_results['RMSE'].idxmin()]
                            st.write(f"**{ticker}**: {best_ticker['Model']} (RMSE: {best_ticker['RMSE']:.4f})")
                
                # Best by model
                with col3:
                    st.markdown("**� Best by Model**")
                    for model in comparison_models:
                        model_results = results_df[results_df['Model'] == model]
                        if not model_results.empty:
                            avg_rmse = model_results['RMSE'].mean()
                            st.write(f"**{model}**: Avg RMSE {avg_rmse:.4f}")
                
                # Performance ranking
                st.markdown("#### Model Performance Ranking")
                
                model_ranking = results_df.groupby('Model').agg({
                    'RMSE': ['mean', 'std', 'min'],
                    'MAPE': ['mean', 'std', 'min'],
                    'Ticker': 'count'
                }).round(4)
                
                model_ranking.columns = ['RMSE_Mean', 'RMSE_Std', 'RMSE_Min', 'MAPE_Mean', 'MAPE_Std', 'MAPE_Min', 'Tests']
                model_ranking = model_ranking.sort_values('RMSE_Mean')
                
                st.dataframe(model_ranking)
            
            with tab3:
                st.markdown("#### Performance Visualization")
                
                # RMSE comparison by model
                fig1 = px.box(
                    results_df, 
                    x='Model', 
                    y='RMSE',
                    title="RMSE Distribution by Model",
                    color='Model'
                )
                fig1.update_layout(height=400)
                st.plotly_chart(fig1, width='stretch')
                
                # Performance by horizon
                fig2 = px.line(
                    results_df.groupby(['Horizon', 'Model'])['RMSE'].mean().reset_index(),
                    x='Horizon',
                    y='RMSE', 
                    color='Model',
                    title="Average RMSE by Forecast Horizon",
                    markers=True
                )
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, width='stretch')
                
                # Heatmap of performance
                pivot_data = results_df.pivot_table(
                    values='RMSE', 
                    index='Model', 
                    columns='Ticker', 
                    aggfunc='mean'
                )
                
                if not pivot_data.empty:
                    fig3 = px.imshow(
                        pivot_data.values,
                        labels=dict(x="Ticker", y="Model", color="RMSE"),
                        x=pivot_data.columns,
                        y=pivot_data.index,
                        title="RMSE Heatmap (Model vs Ticker)"
                    )
                    fig3.update_layout(height=400)
                    st.plotly_chart(fig3, width='stretch')
            
            with tab4:
                st.markdown("#### Detailed Statistical Analysis")
                
                # Statistical significance tests
                st.markdown("##### Model Performance Statistics")
                
                for model in comparison_models:
                    model_data = results_df[results_df['Model'] == model]
                    if len(model_data) > 1:
                        st.markdown(f"**{model}:**")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Mean RMSE", f"{model_data['RMSE'].mean():.4f}")
                            st.metric("Std RMSE", f"{model_data['RMSE'].std():.4f}")
                        
                        with col2:
                            st.metric("Mean MAPE", f"{model_data['MAPE'].mean():.2f}%")
                            st.metric("Std MAPE", f"{model_data['MAPE'].std():.2f}%")
                        
                        with col3:
                            st.metric("Success Rate", f"{len(model_data)}/{progress_total//len(comparison_models)} tests")
                            st.metric("Best RMSE", f"{model_data['RMSE'].min():.4f}")
                
                # Correlation analysis
                st.markdown("##### Performance Correlations")
                
                correlation_df = results_df[['RMSE', 'MAE', 'MAPE', 'Horizon', 'Data_Points']].corr()
                
                fig_corr = px.imshow(
                    correlation_df.values,
                    labels=dict(x="Metric", y="Metric", color="Correlation"),
                    x=correlation_df.columns,
                    y=correlation_df.index,
                    title="Metric Correlations",
                    color_continuous_scale="RdBu_r"
                )
                st.plotly_chart(fig_corr, width='stretch')
                
                # Raw correlation matrix
                st.dataframe(correlation_df.round(3))
        
        else:
            st.error("No comparison results were generated. Please check your configuration.")
    
    # Quick comparison section
    st.markdown("---")
    st.markdown("### Quick Model Comparison")
    st.markdown("Select a single ticker for rapid model comparison:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        quick_ticker = st.selectbox("Ticker:", summary['tickers'], key="quick_comp_ticker")
        quick_horizon = st.slider("Forecast Days:", 5, 30, 15, key="quick_comp_horizon")
    
    with col2:
        quick_models = ['Naive', 'MA_10', 'ARIMA_111']
        st.write("**Models:** Naive, MA_10, ARIMA_111")
        quick_test_size = st.slider("Test %:", 10, 30, 15, key="quick_comp_test")
    
    if st.button("Quick Compare", key="quick_compare"):
        ticker_data = get_ticker_data(data, quick_ticker, 'close')
        
        split_point = int(len(ticker_data) * (1 - quick_test_size/100))
        train_data = ticker_data.iloc[:split_point]
        test_data = ticker_data.iloc[split_point:split_point + quick_horizon]
        
        if len(test_data) < quick_horizon:
            test_data = ticker_data.iloc[-quick_horizon:]
            train_data = ticker_data.iloc[:-quick_horizon]
        
        quick_forecasts = {}
        quick_metrics = {}
        
        with st.spinner("Running quick comparison..."):
            # Naive
            try:
                naive_pred = naive_forecast(train_data, forecast_steps=len(test_data))
                quick_forecasts['Naive'] = naive_pred
                quick_metrics['Naive'] = calculate_all_metrics(test_data.values, naive_pred.values)
            except Exception as e:
                st.warning(f"Naive error: {e}")
            
            # MA_10
            try:
                ma_pred = moving_average_forecast(train_data, window=10, forecast_steps=len(test_data))
                quick_forecasts['MA_10'] = ma_pred
                quick_metrics['MA_10'] = calculate_all_metrics(test_data.values, ma_pred.values)
            except Exception as e:
                st.warning(f"MA_10 error: {e}")
            
            # ARIMA
            try:
                arima_pred = arima_forecast(train_data, order=(1,1,1), forecast_steps=len(test_data))
                quick_forecasts['ARIMA_111'] = arima_pred
                quick_metrics['ARIMA_111'] = calculate_all_metrics(test_data.values, arima_pred.values)
            except Exception as e:
                st.warning(f"ARIMA error: {e}")
        
        if quick_forecasts:
            # Plot comparison
            fig_quick = create_forecast_plot(test_data, quick_forecasts, f"{quick_ticker} Quick Comparison")
            st.plotly_chart(fig_quick, width='stretch')
            
            # Quick metrics
            quick_df = pd.DataFrame(quick_metrics).T.round(4)
            st.dataframe(quick_df)
            
            # Winner
            if not quick_df.empty:
                winner = quick_df['RMSE'].idxmin()
                st.success(f"🏆 Winner: **{winner}** (RMSE: {quick_df.loc[winner, 'RMSE']:.4f})")
        else:
            st.error("Quick comparison failed.")


def page_multivariate_analysis():
    """Multivariate analysis page."""
    st.markdown('<div class="main-header">Multivariate Analysis</div>', unsafe_allow_html=True)
    
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please load data first from the Home page.")
        return
    
    data = st.session_state.data
    summary = get_data_summary(data)
    
    st.markdown("""
    Analyze relationships between multiple stocks and create multivariate forecasts 
    using Vector Autoregression (VAR) models that capture cross-asset dependencies.
    """)
    
    # Create tabs for different multivariate analyses
    tab1, tab2, tab3 = st.tabs(["🔗 VAR Forecasting", "📊 Correlation Analysis", "🌐 Portfolio Dynamics"])
    
    with tab1:
        st.markdown("### Vector Autoregression (VAR) Forecasting")
        st.markdown("VAR models capture the linear interdependencies between multiple time series.")
        
        # Configuration
        col1, col2 = st.columns(2)
        
        with col1:
            var_tickers = st.multiselect(
                "Select Tickers for VAR Model:",
                summary['tickers'],
                default=summary['tickers'][:3],
                key="var_tickers",
                help="Select 2-5 tickers for optimal VAR performance"
            )
            
            var_column = st.selectbox(
                "Data Column:",
                ['close', 'open', 'high', 'low'],
                index=0,
                key="var_column"
            )
        
        with col2:
            var_maxlags = st.slider(
                "Maximum Lags:",
                1, 15, 5,
                key="var_maxlags",
                help="Maximum number of lags to consider for VAR model"
            )
            
            var_test_size = st.slider(
                "Test Size (%):",
                10, 40, 20,
                key="var_test_size"
            )
            
            var_forecast_steps = st.slider(
                "Forecast Steps:",
                5, 50, 20,
                key="var_forecast_steps"
            )
        
        # Advanced VAR options
        with st.expander("Advanced VAR Configuration"):
            var_ic = st.selectbox(
                "Information Criterion for Lag Selection:",
                ['aic', 'bic', 'hqic', 'fpe'],
                index=0,
                key="var_ic",
                help="Criterion used to select optimal number of lags"
            )
            
            var_trend = st.selectbox(
                "Trend Component:",
                ['const', 'none', 'linear', 'quadratic'],
                index=0,
                key="var_trend",
                help="Type of deterministic trend to include"
            )
            
            var_diff = st.checkbox(
                "Apply Differencing",
                value=True,
                key="var_diff",
                help="Apply first differencing to ensure stationarity"
            )
        
        if len(var_tickers) >= 2:
            if st.button("Run VAR Analysis", type="primary", key="run_var"):
                with st.spinner("Running VAR analysis..."):
                    try:
                        # Get multivariate data
                        multi_data = get_multiple_tickers_data(data, var_tickers, var_column)
                        
                        if multi_data.empty:
                            st.error("No data available for selected tickers.")
                            return
                        
                        # Split data
                        split_point = int(len(multi_data) * (1 - var_test_size/100))
                        train_data = multi_data.iloc[:split_point]
                        test_data = multi_data.iloc[split_point:]
                        
                        st.markdown("### VAR Model Results")
                        
                        # Initialize VAR model
                        var_model = VARModel(maxlags=var_maxlags)
                        
                        # Fit the model
                        var_model.fit(train_data)
                        
                        # Generate forecasts
                        forecasts, conf_intervals = var_model.predict(steps=min(var_forecast_steps, len(test_data)))
                        
                        # Display model summary
                        st.markdown("#### Model Summary")
                        model_summary = var_model.get_model_summary()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Selected Lags", model_summary.get('lags', 'N/A'))
                        with col2:
                            st.metric("AIC", f"{model_summary.get('aic', 0):.2f}")
                        with col3:
                            st.metric("BIC", f"{model_summary.get('bic', 0):.2f}")
                        
                        # Forecast visualization
                        st.markdown("#### Forecast Results")
                        
                        for ticker in var_tickers:
                            if ticker in forecasts.columns and ticker in test_data.columns:
                                fig = go.Figure()
                                
                                # Historical data
                                historical_data = multi_data[ticker].tail(100)  # Show last 100 points for context
                                fig.add_trace(go.Scatter(
                                    x=historical_data.index,
                                    y=historical_data.values,
                                    mode='lines',
                                    name='Historical',
                                    line=dict(color='blue', width=2)
                                ))
                                
                                # Actual test data
                                actual_test = test_data[ticker].iloc[:len(forecasts)]
                                fig.add_trace(go.Scatter(
                                    x=actual_test.index,
                                    y=actual_test.values,
                                    mode='lines',
                                    name='Actual',
                                    line=dict(color='black', width=2)
                                ))
                                
                                # Forecasts
                                fig.add_trace(go.Scatter(
                                    x=forecasts.index,
                                    y=forecasts[ticker].values,
                                    mode='lines',
                                    name='VAR Forecast',
                                    line=dict(color='red', width=2, dash='dash')
                                ))
                                
                                # Confidence intervals if available
                                if conf_intervals is not None and ticker in conf_intervals:
                                    lower_bound = conf_intervals[ticker].iloc[:, 0]
                                    upper_bound = conf_intervals[ticker].iloc[:, 1]
                                    
                                    fig.add_trace(go.Scatter(
                                        x=forecasts.index,
                                        y=upper_bound.values,
                                        mode='lines',
                                        line=dict(width=0),
                                        showlegend=False,
                                        hoverinfo='skip'
                                    ))
                                    
                                    fig.add_trace(go.Scatter(
                                        x=forecasts.index,
                                        y=lower_bound.values,
                                        mode='lines',
                                        line=dict(width=0),
                                        fill='tonexty',
                                        fillcolor='rgba(255,0,0,0.2)',
                                        name='Confidence Interval',
                                        hoverinfo='skip'
                                    ))
                                
                                fig.update_layout(
                                    title=f"{ticker} - VAR Forecast",
                                    xaxis_title="Date",
                                    yaxis_title="Price ($)",
                                    height=400
                                )
                                
                                st.plotly_chart(fig, width='stretch')
                        
                        # Performance metrics
                        st.markdown("#### Performance Metrics")
                        
                        var_metrics = {}
                        for ticker in var_tickers:
                            if ticker in forecasts.columns and ticker in test_data.columns:
                                actual_values = test_data[ticker].iloc[:len(forecasts)].values
                                forecast_values = forecasts[ticker].values
                                
                                if len(actual_values) == len(forecast_values):
                                    metrics = calculate_all_metrics(actual_values, forecast_values)
                                    var_metrics[ticker] = metrics
                        
                        if var_metrics:
                            metrics_df = pd.DataFrame(var_metrics).T
                            st.dataframe(metrics_df.round(4))
                            
                            # Best performing asset
                            best_ticker = metrics_df['RMSE'].idxmin()
                            st.success(f"🎯 Best VAR Performance: **{best_ticker}** (RMSE: {metrics_df.loc[best_ticker, 'RMSE']:.4f})")
                        
                        # Granger Causality Tests
                        if hasattr(var_model, 'model') and var_model.model is not None:
                            st.markdown("#### Granger Causality Analysis")
                            st.markdown("Tests whether one time series can predict another:")
                            
                            try:
                                from statsmodels.tsa.stattools import grangercausalitytests
                                
                                causality_results = {}
                                for i, ticker1 in enumerate(var_tickers):
                                    for j, ticker2 in enumerate(var_tickers):
                                        if i != j and ticker1 in train_data.columns and ticker2 in train_data.columns:
                                            try:
                                                # Prepare data for Granger test
                                                test_data_granger = train_data[[ticker2, ticker1]].dropna()
                                                
                                                if len(test_data_granger) > 20:  # Minimum data requirement
                                                    gc_result = grangercausalitytests(test_data_granger, maxlag=3, verbose=False)
                                                    
                                                    # Extract p-value for lag 1
                                                    if 1 in gc_result:
                                                        p_value = gc_result[1][0]['ssr_ftest'][1]
                                                        causality_results[f"{ticker1} → {ticker2}"] = {
                                                            'p_value': p_value,
                                                            'significant': p_value < 0.05
                                                        }
                                            except:
                                                continue
                                
                                if causality_results:
                                    causality_df = pd.DataFrame(causality_results).T
                                    causality_df['Significant'] = causality_df['significant'].map({True: '✓', False: '✗'})
                                    causality_df['P-Value'] = causality_df['p_value'].round(4)
                                    
                                    display_causality = causality_df[['P-Value', 'Significant']].sort_values('P-Value')
                                    st.dataframe(display_causality)
                                    
                                    significant_relationships = display_causality[display_causality['Significant'] == '✓']
                                    if not significant_relationships.empty:
                                        st.info(f"Found {len(significant_relationships)} significant Granger causality relationships (p < 0.05)")
                                    else:
                                        st.info("No significant Granger causality relationships found at 5% level")
                            
                            except ImportError:
                                st.warning("Granger causality tests require additional dependencies")
                            except Exception as e:
                                st.warning(f"Granger causality analysis failed: {str(e)}")
                    
                    except Exception as e:
                        st.error(f"VAR analysis failed: {str(e)}")
                        st.error("This might be due to insufficient data or numerical issues. Try reducing the number of lags or using different tickers.")
        else:
            st.info("Please select at least 2 tickers for VAR analysis.")
    
    with tab2:
        st.markdown("### Correlation Analysis")
        st.markdown("Analyze relationships and dependencies between different assets.")
        
        # Correlation configuration
        col1, col2 = st.columns(2)
        
        with col1:
            corr_tickers = st.multiselect(
                "Select Tickers:",
                summary['tickers'],
                default=summary['tickers'],
                key="corr_tickers"
            )
            
            corr_method = st.selectbox(
                "Correlation Method:",
                ['pearson', 'spearman', 'kendall'],
                key="corr_method"
            )
        
        with col2:
            corr_window = st.slider(
                "Rolling Window (days):",
                10, 100, 30,
                key="corr_window",
                help="Window size for rolling correlation"
            )
            
            corr_type = st.selectbox(
                "Analysis Type:",
                ['Price Correlation', 'Return Correlation', 'Rolling Correlation'],
                key="corr_type"
            )
        
        if len(corr_tickers) >= 2:
            if st.button("Run Correlation Analysis", key="run_corr"):
                with st.spinner("Calculating correlations..."):
                    # Get data
                    corr_data = get_multiple_tickers_data(data, corr_tickers, 'close')
                    
                    if corr_type == 'Price Correlation':
                        analysis_data = corr_data
                        title_suffix = "Price Levels"
                    elif corr_type == 'Return Correlation':
                        analysis_data = corr_data.pct_change().dropna()
                        title_suffix = "Returns"
                    else:  # Rolling Correlation
                        analysis_data = corr_data.pct_change().dropna()
                        title_suffix = f"Rolling {corr_window}-day Returns"
                    
                    # Calculate correlation matrix
                    if corr_type != 'Rolling Correlation':
                        corr_matrix = analysis_data.corr(method=corr_method)
                        
                        # Display correlation heatmap
                        fig_heatmap = px.imshow(
                            corr_matrix.values,
                            labels=dict(x="Ticker", y="Ticker", color="Correlation"),
                            x=corr_matrix.columns,
                            y=corr_matrix.index,
                            title=f"Correlation Matrix - {title_suffix}",
                            color_continuous_scale="RdBu_r",
                            aspect="auto"
                        )
                        fig_heatmap.update_layout(height=500)
                        st.plotly_chart(fig_heatmap, width='stretch')
                        
                        # Correlation statistics
                        st.markdown("#### Correlation Statistics")
                        
                        # Extract upper triangle (excluding diagonal)
                        mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
                        correlations = corr_matrix.where(mask).stack().dropna()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Average Correlation", f"{correlations.mean():.3f}")
                        with col2:
                            st.metric("Max Correlation", f"{correlations.max():.3f}")
                        with col3:
                            st.metric("Min Correlation", f"{correlations.min():.3f}")
                        
                        # Top correlations
                        st.markdown("#### Strongest Correlations")
                        correlations_sorted = correlations.abs().sort_values(ascending=False)
                        top_correlations = correlations_sorted.head(10)
                        
                        for pair, corr_val in top_correlations.items():
                            actual_corr = correlations[pair]
                            st.write(f"**{pair[0]} ↔ {pair[1]}**: {actual_corr:.3f}")
                    
                    else:
                        # Rolling correlation analysis
                        st.markdown("#### Rolling Correlation Analysis")
                        
                        if len(corr_tickers) == 2:
                            # Two-asset rolling correlation
                            ticker1, ticker2 = corr_tickers[0], corr_tickers[1]
                            rolling_corr = analysis_data[ticker1].rolling(corr_window).corr(analysis_data[ticker2])
                            
                            fig_rolling = go.Figure()
                            fig_rolling.add_trace(go.Scatter(
                                x=rolling_corr.index,
                                y=rolling_corr.values,
                                mode='lines',
                                name=f'{ticker1} vs {ticker2}',
                                line=dict(width=2)
                            ))
                            
                            fig_rolling.update_layout(
                                title=f"Rolling {corr_window}-day Correlation: {ticker1} vs {ticker2}",
                                xaxis_title="Date",
                                yaxis_title="Correlation",
                                height=400
                            )
                            
                            st.plotly_chart(fig_rolling, width='stretch')
                            
                            # Rolling correlation statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Mean Rolling Corr", f"{rolling_corr.mean():.3f}")
                            with col2:
                                st.metric("Std Rolling Corr", f"{rolling_corr.std():.3f}")
                            with col3:
                                st.metric("Current Corr", f"{rolling_corr.iloc[-1]:.3f}")
                        
                        else:
                            # Multiple assets - show average rolling correlation
                            rolling_corrs = {}
                            for i in range(len(corr_tickers)):
                                for j in range(i+1, len(corr_tickers)):
                                    ticker1, ticker2 = corr_tickers[i], corr_tickers[j]
                                    rolling_corr = analysis_data[ticker1].rolling(corr_window).corr(analysis_data[ticker2])
                                    rolling_corrs[f"{ticker1}-{ticker2}"] = rolling_corr
                            
                            # Plot multiple rolling correlations
                            fig_multi = go.Figure()
                            colors = px.colors.qualitative.Set3
                            
                            for i, (pair, corr_series) in enumerate(rolling_corrs.items()):
                                fig_multi.add_trace(go.Scatter(
                                    x=corr_series.index,
                                    y=corr_series.values,
                                    mode='lines',
                                    name=pair,
                                    line=dict(color=colors[i % len(colors)], width=2)
                                ))
                            
                            fig_multi.update_layout(
                                title=f"Rolling {corr_window}-day Correlations",
                                xaxis_title="Date",
                                yaxis_title="Correlation",
                                height=500
                            )
                            
                            st.plotly_chart(fig_multi, width='stretch')
                    
                    # Correlation matrix table
                    st.markdown("#### Correlation Matrix")
                    if corr_type != 'Rolling Correlation':
                        st.dataframe(corr_matrix.round(3))
    
    with tab3:
        st.markdown("### Portfolio Dynamics")
        st.markdown("Analyze portfolio-level relationships and risk metrics.")
        
        # Portfolio configuration
        portfolio_tickers = st.multiselect(
            "Select Portfolio Assets:",
            summary['tickers'],
            default=summary['tickers'][:4],
            key="portfolio_tickers"
        )
        
        if len(portfolio_tickers) >= 2:
            # Get portfolio data
            portfolio_data = get_multiple_tickers_data(data, portfolio_tickers, 'close')
            portfolio_returns = portfolio_data.pct_change().dropna()
            
            # Portfolio analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Risk Metrics")
                
                # Individual asset risk
                risk_metrics = {}
                for ticker in portfolio_tickers:
                    returns = portfolio_returns[ticker]
                    risk_metrics[ticker] = {
                        'Volatility': returns.std() * np.sqrt(252),
                        'Sharpe_Approx': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                        'Max_Drawdown': (returns.cumsum().cummax() - returns.cumsum()).max()
                    }
                
                risk_df = pd.DataFrame(risk_metrics).T
                st.dataframe(risk_df.round(4))
            
            with col2:
                st.markdown("#### Portfolio Composition")
                
                # Equal weight portfolio
                equal_weights = np.ones(len(portfolio_tickers)) / len(portfolio_tickers)
                portfolio_return = (portfolio_returns * equal_weights).sum(axis=1)
                
                portfolio_vol = portfolio_return.std() * np.sqrt(252)
                portfolio_sharpe = portfolio_return.mean() / portfolio_return.std() * np.sqrt(252) if portfolio_return.std() > 0 else 0
                
                st.metric("Portfolio Volatility", f"{portfolio_vol:.2%}")
                st.metric("Portfolio Sharpe", f"{portfolio_sharpe:.3f}")
                
                # Weights display
                weights_df = pd.DataFrame({
                    'Ticker': portfolio_tickers,
                    'Weight': [f"{w:.1%}" for w in equal_weights]
                })
                st.dataframe(weights_df, hide_index=True)
            
            # Portfolio correlation structure
            st.markdown("#### Portfolio Correlation Structure")
            
            # Eigenvalue analysis
            corr_matrix = portfolio_returns.corr()
            eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
            
            # Sort eigenvalues in descending order
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sorted_indices]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Eigenvalue plot
                fig_eigen = px.bar(
                    x=range(1, len(eigenvalues) + 1),
                    y=eigenvalues,
                    title="Correlation Matrix Eigenvalues",
                    labels={'x': 'Component', 'y': 'Eigenvalue'}
                )
                st.plotly_chart(fig_eigen, width='stretch')
            
            with col2:
                # Diversification metrics
                st.markdown("**Diversification Metrics:**")
                
                # Average correlation
                mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
                avg_corr = corr_matrix.where(mask).stack().mean()
                
                # Diversification ratio
                individual_vol = portfolio_returns.std()
                weighted_avg_vol = (individual_vol * equal_weights).sum()
                diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1
                
                st.metric("Average Correlation", f"{avg_corr:.3f}")
                st.metric("Diversification Ratio", f"{diversification_ratio:.3f}")
                st.metric("Effective Assets", f"{1/np.sum(equal_weights**2):.1f}")
        
        else:
            st.info("Please select at least 2 assets for portfolio analysis.")


def page_deep_learning():
    """Deep learning models page."""
    st.markdown('<div class="main-header">Deep Learning Models</div>', unsafe_allow_html=True)
    
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please load data first from the Home page.")
        return
    
    data = st.session_state.data
    summary = get_data_summary(data)
    
    st.markdown("""
    Advanced deep learning models for time series forecasting using modern architectures 
    like DeepAR and Temporal Fusion Transformers (TFT) with probabilistic predictions.
    """)
    
    # Create tabs for different deep learning models
    tab1, tab2, tab3 = st.tabs(["🤖 DeepAR", "🔮 Temporal Fusion Transformer", "⚡ Quick Deep Learning"])
    
    with tab1:
        st.markdown("### DeepAR: Deep Autoregressive Model")
        st.markdown("""
        DeepAR is a probabilistic forecasting model that learns seasonal patterns and handles 
        multiple related time series to improve accuracy.
        """)
        
        # DeepAR Configuration
        col1, col2 = st.columns(2)
        
        with col1:
            deepar_ticker = st.selectbox(
                "Select Ticker:",
                summary['tickers'],
                key="deepar_ticker"
            )
            
            deepar_epochs = st.slider(
                "Training Epochs:",
                1, 20, 5,
                key="deepar_epochs",
                help="More epochs = better training but slower"
            )
            
            deepar_prediction_length = st.slider(
                "Prediction Length:",
                5, 50, 20,
                key="deepar_pred_length"
            )
        
        with col2:
            deepar_context_length = st.slider(
                "Context Length:",
                10, 100, 30,
                key="deepar_context",
                help="Length of historical context to use"
            )
            
            deepar_lr = st.selectbox(
                "Learning Rate:",
                [0.001, 0.01, 0.1],
                index=1,
                key="deepar_lr"
            )
            
            deepar_batch_size = st.selectbox(
                "Batch Size:",
                [8, 16, 32],
                index=1,
                key="deepar_batch"
            )
        
        # Advanced DeepAR options
        with st.expander("Advanced DeepAR Configuration"):
            deepar_hidden_size = st.slider(
                "Hidden Layer Size:",
                16, 128, 64,
                key="deepar_hidden"
            )
            
            deepar_num_layers = st.slider(
                "Number of LSTM Layers:",
                1, 4, 2,
                key="deepar_layers"
            )
            
            deepar_dropout = st.slider(
                "Dropout Rate:",
                0.0, 0.5, 0.1,
                key="deepar_dropout"
            )
        
        if st.button("Train DeepAR Model", type="primary", key="train_deepar"):
            with st.spinner("Training DeepAR model... This may take a moment."):
                try:
                    # Get data
                    ticker_data = get_ticker_data(data, deepar_ticker, 'close')
                    
                    # Prepare data for deep learning
                    dl_data = pd.DataFrame({'close': ticker_data}).reset_index()
                    dl_data['time_idx'] = range(len(dl_data))
                    dl_data['ticker'] = deepar_ticker
                    
                    # Create forecaster with custom parameters
                    forecaster = create_deep_learning_forecaster(
                        max_prediction_length=deepar_prediction_length,
                        max_encoder_length=deepar_context_length
                    )
                    
                    # Training progress
                    st.markdown("### Training Progress")
                    training_progress = st.progress(0)
                    epoch_text = st.empty()
                    
                    # Simulate training progress (since we're using mock implementation)
                    for epoch in range(deepar_epochs):
                        epoch_text.text(f"Epoch {epoch + 1}/{deepar_epochs}")
                        training_progress.progress((epoch + 1) / deepar_epochs)
                        
                        # Add small delay to simulate training
                        import time
                        time.sleep(0.5)
                    
                    epoch_text.text("Training completed!")
                    
                    # Generate predictions
                    predictions, metadata = forecaster.fit_and_predict_deepar(
                        dl_data, 
                        'close', 
                        max_epochs=deepar_epochs
                    )
                    
                    # Display results
                    st.markdown("### DeepAR Results")
                    
                    # Model metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Model Type", metadata.get('model_type', 'DeepAR'))
                    with col2:
                        st.metric("Prediction Points", len(predictions))
                    with col3:
                        st.metric("Training Status", "✅ Completed")
                    
                    # Predictions visualization
                    historical_data = ticker_data.tail(100)  # Last 100 points for context
                    
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=historical_data.index,
                        y=historical_data.values,
                        mode='lines',
                        name='Historical',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Predictions
                    pred_dates = pd.date_range(
                        start=ticker_data.index[-1] + pd.Timedelta(days=1),
                        periods=len(predictions),
                        freq='D'
                    )
                    
                    fig.add_trace(go.Scatter(
                        x=pred_dates,
                        y=predictions,
                        mode='lines+markers',
                        name='DeepAR Predictions',
                        line=dict(color='red', width=2),
                        marker=dict(size=6)
                    ))
                    
                    fig.update_layout(
                        title=f"{deepar_ticker} - DeepAR Forecast",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Model analysis
                    st.markdown("#### Model Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Prediction Statistics:**")
                        st.write(f"Mean Prediction: ${np.mean(predictions):.2f}")
                        st.write(f"Std Prediction: ${np.std(predictions):.2f}")
                        st.write(f"Min Prediction: ${np.min(predictions):.2f}")
                        st.write(f"Max Prediction: ${np.max(predictions):.2f}")
                    
                    with col2:
                        st.markdown("**Model Configuration:**")
                        st.write(f"Epochs: {deepar_epochs}")
                        st.write(f"Context Length: {deepar_context_length}")
                        st.write(f"Prediction Length: {deepar_prediction_length}")
                        st.write(f"Implementation: {metadata.get('implementation', 'Mock')}")
                    
                    # Export predictions
                    pred_df = pd.DataFrame({
                        'Date': pred_dates,
                        'DeepAR_Prediction': predictions
                    })
                    
                    csv_data = pred_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions",
                        data=csv_data,
                        file_name=f"deepar_predictions_{deepar_ticker}.csv",
                        mime="text/csv"
                    )
                
                except Exception as e:
                    st.error(f"DeepAR training failed: {str(e)}")
                    st.info("This might be due to insufficient data or configuration issues. Try adjusting the parameters.")
    
    with tab2:
        st.markdown("### Temporal Fusion Transformer (TFT)")
        st.markdown("""
        TFT combines the benefits of LSTMs and attention mechanisms to capture both 
        short-term and long-term dependencies with interpretable attention weights.
        """)
        
        # TFT Configuration
        col1, col2 = st.columns(2)
        
        with col1:
            tft_ticker = st.selectbox(
                "Select Ticker:",
                summary['tickers'],
                key="tft_ticker"
            )
            
            tft_epochs = st.slider(
                "Training Epochs:",
                1, 15, 3,
                key="tft_epochs"
            )
            
            tft_prediction_length = st.slider(
                "Prediction Length:",
                5, 40, 15,
                key="tft_pred_length"
            )
        
        with col2:
            tft_context_length = st.slider(
                "Context Length:",
                15, 120, 60,
                key="tft_context"
            )
            
            tft_attention_heads = st.slider(
                "Attention Heads:",
                1, 8, 4,
                key="tft_heads"
            )
            
            tft_hidden_size = st.slider(
                "Hidden Size:",
                32, 256, 128,
                key="tft_hidden"
            )
        
        # TFT Feature Engineering
        with st.expander("Feature Engineering for TFT"):
            use_technical_indicators = st.checkbox(
                "Add Technical Indicators",
                value=True,
                key="tft_tech"
            )
            
            if use_technical_indicators:
                tech_indicators = st.multiselect(
                    "Select Indicators:",
                    ['SMA_5', 'SMA_20', 'RSI', 'MACD'],
                    default=['SMA_5', 'SMA_20'],
                    key="tft_indicators"
                )
            
            use_calendar_features = st.checkbox(
                "Add Calendar Features",
                value=True,
                key="tft_calendar"
            )
        
        if st.button("Train TFT Model", type="primary", key="train_tft"):
            with st.spinner("Training Temporal Fusion Transformer..."):
                try:
                    # Get data
                    ticker_data = get_ticker_data(data, tft_ticker, 'close')
                    
                    # Feature engineering
                    tft_data = pd.DataFrame({'close': ticker_data}).reset_index()
                    tft_data['time_idx'] = range(len(tft_data))
                    tft_data['ticker'] = tft_ticker
                    
                    # Add technical indicators if requested
                    if use_technical_indicators and 'tech_indicators' in locals():
                        if 'SMA_5' in tech_indicators:
                            tft_data['sma_5'] = ticker_data.rolling(5).mean().values
                        if 'SMA_20' in tech_indicators:
                            tft_data['sma_20'] = ticker_data.rolling(20).mean().values
                        if 'RSI' in tech_indicators:
                            # Simple RSI approximation
                            delta = ticker_data.diff()
                            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                            rs = gain / loss
                            tft_data['rsi'] = 100 - (100 / (1 + rs)).values
                    
                    # Add calendar features if requested
                    if use_calendar_features:
                        tft_data['day_of_week'] = pd.to_datetime(tft_data['date']).dt.dayofweek
                        tft_data['month'] = pd.to_datetime(tft_data['date']).dt.month
                    
                    # Create forecaster
                    forecaster = create_deep_learning_forecaster(
                        max_prediction_length=tft_prediction_length,
                        max_encoder_length=tft_context_length
                    )
                    
                    # Training progress
                    st.markdown("### TFT Training Progress")
                    training_progress = st.progress(0)
                    epoch_text = st.empty()
                    
                    for epoch in range(tft_epochs):
                        epoch_text.text(f"Epoch {epoch + 1}/{tft_epochs} - Learning attention patterns...")
                        training_progress.progress((epoch + 1) / tft_epochs)
                        
                        import time
                        time.sleep(0.7)  # Slightly longer for TFT
                    
                    epoch_text.text("TFT training completed!")
                    
                    # Generate predictions
                    predictions, metadata = forecaster.fit_and_predict_tft(
                        tft_data,
                        'close',
                        max_epochs=tft_epochs
                    )
                    
                    # Display results
                    st.markdown("### TFT Results")
                    
                    # Model metadata
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Model", "TFT")
                    with col2:
                        st.metric("Attention Heads", tft_attention_heads)
                    with col3:
                        st.metric("Features", len(tft_data.columns) - 3)  # Exclude date, time_idx, ticker
                    with col4:
                        st.metric("Predictions", len(predictions))
                    
                    # Visualization
                    historical_data = ticker_data.tail(100)
                    
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=historical_data.index,
                        y=historical_data.values,
                        mode='lines',
                        name='Historical',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # TFT Predictions
                    pred_dates = pd.date_range(
                        start=ticker_data.index[-1] + pd.Timedelta(days=1),
                        periods=len(predictions),
                        freq='D'
                    )
                    
                    fig.add_trace(go.Scatter(
                        x=pred_dates,
                        y=predictions,
                        mode='lines+markers',
                        name='TFT Predictions',
                        line=dict(color='green', width=2),
                        marker=dict(size=6, symbol='diamond')
                    ))
                    
                    fig.update_layout(
                        title=f"{tft_ticker} - Temporal Fusion Transformer Forecast",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Feature importance (simulated for mock implementation)
                    st.markdown("#### Feature Importance Analysis")
                    
                    # Mock feature importance
                    features = ['close_lag_1', 'close_lag_2', 'trend', 'seasonal']
                    if use_technical_indicators:
                        features.extend(['sma_5', 'sma_20'])
                    if use_calendar_features:
                        features.extend(['day_of_week', 'month'])
                    
                    importance_scores = np.random.beta(2, 5, len(features))  # Mock importance
                    importance_scores = importance_scores / importance_scores.sum()  # Normalize
                    
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': importance_scores
                    }).sort_values('Importance', ascending=False)
                    
                    fig_importance = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="TFT Feature Importance"
                    )
                    fig_importance.update_layout(height=400)
                    st.plotly_chart(fig_importance, width='stretch')
                    
                    # Model insights
                    st.markdown("#### Model Insights")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Top 3 Features:**")
                        for i, (_, row) in enumerate(importance_df.head(3).iterrows()):
                            st.write(f"{i+1}. {row['Feature']}: {row['Importance']:.3f}")
                    
                    with col2:
                        st.markdown("**Prediction Characteristics:**")
                        pred_trend = "Upward" if predictions[-1] > predictions[0] else "Downward"
                        pred_volatility = np.std(predictions) / np.mean(predictions)
                        
                        st.write(f"Trend: {pred_trend}")
                        st.write(f"Volatility: {pred_volatility:.3f}")
                        st.write(f"Range: ${np.ptp(predictions):.2f}")
                
                except Exception as e:
                    st.error(f"TFT training failed: {str(e)}")
    
    with tab3:
        st.markdown("### Quick Deep Learning Comparison")
        st.markdown("Compare DeepAR and TFT models side-by-side with simplified configuration.")
        
        # Quick configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            quick_ticker = st.selectbox(
                "Ticker:",
                summary['tickers'],
                key="quick_dl_ticker"
            )
        
        with col2:
            quick_epochs = st.slider(
                "Epochs:",
                1, 5, 2,
                key="quick_dl_epochs"
            )
        
        with col3:
            quick_pred_length = st.slider(
                "Forecast Days:",
                5, 25, 10,
                key="quick_dl_pred"
            )
        
        if st.button("Compare DL Models", type="primary", key="quick_dl_compare"):
            with st.spinner("Training both models..."):
                try:
                    # Get data
                    ticker_data = get_ticker_data(data, quick_ticker, 'close')
                    
                    # Prepare data
                    dl_data = pd.DataFrame({'close': ticker_data}).reset_index()
                    dl_data['time_idx'] = range(len(dl_data))
                    dl_data['ticker'] = quick_ticker
                    
                    # Create forecaster
                    forecaster = create_deep_learning_forecaster(
                        max_prediction_length=quick_pred_length,
                        max_encoder_length=min(30, len(ticker_data) // 3)
                    )
                    
                    # Train both models
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Training DeepAR...**")
                        deepar_progress = st.progress(0)
                        for i in range(quick_epochs):
                            deepar_progress.progress((i + 1) / quick_epochs)
                            import time
                            time.sleep(0.3)
                        
                        deepar_pred, deepar_meta = forecaster.fit_and_predict_deepar(
                            dl_data, 'close', max_epochs=quick_epochs
                        )
                        st.success("✅ DeepAR Complete")
                    
                    with col2:
                        st.markdown("**Training TFT...**")
                        tft_progress = st.progress(0)
                        for i in range(quick_epochs):
                            tft_progress.progress((i + 1) / quick_epochs)
                            import time
                            time.sleep(0.3)
                        
                        tft_pred, tft_meta = forecaster.fit_and_predict_tft(
                            dl_data, 'close', max_epochs=quick_epochs
                        )
                        st.success("✅ TFT Complete")
                    
                    # Comparison visualization
                    st.markdown("### Model Comparison Results")
                    
                    historical_data = ticker_data.tail(50)
                    pred_dates = pd.date_range(
                        start=ticker_data.index[-1] + pd.Timedelta(days=1),
                        periods=quick_pred_length,
                        freq='D'
                    )
                    
                    fig = go.Figure()
                    
                    # Historical
                    fig.add_trace(go.Scatter(
                        x=historical_data.index,
                        y=historical_data.values,
                        mode='lines',
                        name='Historical',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # DeepAR
                    fig.add_trace(go.Scatter(
                        x=pred_dates,
                        y=deepar_pred,
                        mode='lines+markers',
                        name='DeepAR',
                        line=dict(color='red', width=2),
                        marker=dict(size=5)
                    ))
                    
                    # TFT
                    fig.add_trace(go.Scatter(
                        x=pred_dates,
                        y=tft_pred,
                        mode='lines+markers',
                        name='TFT',
                        line=dict(color='green', width=2),
                        marker=dict(size=5, symbol='diamond')
                    ))
                    
                    fig.update_layout(
                        title=f"{quick_ticker} - Deep Learning Model Comparison",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Comparison metrics
                    st.markdown("### Comparison Metrics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**DeepAR Results:**")
                        st.write(f"Mean Prediction: ${np.mean(deepar_pred):.2f}")
                        st.write(f"Prediction Range: ${np.ptp(deepar_pred):.2f}")
                        st.write(f"Final Value: ${deepar_pred[-1]:.2f}")
                        st.write(f"Implementation: {deepar_meta.get('implementation', 'Mock')}")
                    
                    with col2:
                        st.markdown("**TFT Results:**")
                        st.write(f"Mean Prediction: ${np.mean(tft_pred):.2f}")
                        st.write(f"Prediction Range: ${np.ptp(tft_pred):.2f}")
                        st.write(f"Final Value: ${tft_pred[-1]:.2f}")
                        st.write(f"Implementation: {tft_meta.get('implementation', 'Mock')}")
                    
                    # Model comparison summary
                    st.markdown("### Summary")
                    
                    deepar_volatility = np.std(deepar_pred) / np.mean(deepar_pred)
                    tft_volatility = np.std(tft_pred) / np.mean(tft_pred)
                    
                    if deepar_volatility < tft_volatility:
                        st.info("📊 **DeepAR** shows more stable predictions with lower volatility")
                    else:
                        st.info("📊 **TFT** shows more stable predictions with lower volatility")
                    
                    # Export comparison
                    comparison_df = pd.DataFrame({
                        'Date': pred_dates,
                        'DeepAR': deepar_pred,
                        'TFT': tft_pred
                    })
                    
                    csv_comparison = comparison_df.to_csv(index=False)
                    st.download_button(
                        label="Download Comparison Results",
                        data=csv_comparison,
                        file_name=f"dl_comparison_{quick_ticker}.csv",
                        mime="text/csv"
                    )
                
                except Exception as e:
                    st.error(f"Deep learning comparison failed: {str(e)}")
    
    # Model selection guidance
    st.markdown("---")
    st.markdown("### 🎯 Model Selection Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **When to use DeepAR:**
        - Multiple related time series
        - Probabilistic forecasts needed
        - Seasonal patterns present
        - Sparse or intermittent data
        """)
    
    with col2:
        st.markdown("""
        **When to use TFT:**
        - Complex feature interactions
        - Need interpretability
        - Multiple external variables
        - Long-term dependencies important
        """)
    
    st.info("""
    💡 **Note:** This implementation uses mock deep learning models for demonstration. 
    In production, you would need PyTorch Forecasting library and proper GPU resources 
    for optimal performance.
    """)


def page_portfolio_optimization():
    """Portfolio optimization page."""
    st.markdown('<div class="main-header">Portfolio Optimization</div>', unsafe_allow_html=True)
    
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please load data first from the Home page.")
        return
    
    data = st.session_state.data
    summary = get_data_summary(data)
    
    st.markdown("""
    Optimize portfolio weights using modern portfolio theory, risk parity, and forecast-based allocation strategies 
    to maximize returns while managing risk.
    """)
    
    # Create tabs for different optimization approaches
    tab1, tab2, tab3, tab4 = st.tabs(["⚖️ Modern Portfolio Theory", "🎯 Risk Parity", "🔮 Forecast-Based", "📊 Portfolio Analysis"])
    
    with tab1:
        st.markdown("### Modern Portfolio Theory (MPT)")
        st.markdown("Optimize portfolio weights to maximize Sharpe ratio or minimize volatility.")
        
        # Asset selection
        col1, col2 = st.columns(2)
        
        with col1:
            mpt_assets = st.multiselect(
                "Select Portfolio Assets:",
                summary['tickers'],
                default=summary['tickers'][:4],
                key="mpt_assets"
            )
            
            mpt_objective = st.selectbox(
                "Optimization Objective:",
                ['Max Sharpe Ratio', 'Min Volatility', 'Max Return'],
                key="mpt_objective"
            )
        
        with col2:
            mpt_lookback = st.slider(
                "Lookback Period (days):",
                30, 250, 120,
                key="mpt_lookback",
                help="Historical period for calculating returns and covariance"
            )
            
            mpt_target_return = st.slider(
                "Target Return (annual %):",
                5.0, 25.0, 12.0,
                key="mpt_target",
                help="Used for minimum volatility with target return"
            )
        
        # Risk constraints
        with st.expander("Risk Constraints"):
            max_weight = st.slider(
                "Maximum Asset Weight:",
                0.1, 1.0, 0.4,
                key="mpt_max_weight"
            )
            
            min_weight = st.slider(
                "Minimum Asset Weight:",
                0.0, 0.2, 0.05,
                key="mpt_min_weight"
            )
            
            risk_free_rate = st.slider(
                "Risk-Free Rate (%):",
                0.0, 5.0, 2.0,
                key="mpt_rf_rate"
            ) / 100
        
        if len(mpt_assets) >= 2:
            if st.button("Optimize Portfolio", type="primary", key="run_mpt"):
                with st.spinner("Running portfolio optimization..."):
                    try:
                        # Get returns data
                        returns_data = {}
                        for asset in mpt_assets:
                            asset_prices = get_ticker_data(data, asset, 'close')
                            asset_returns = asset_prices.pct_change().dropna()
                            returns_data[asset] = asset_returns.tail(mpt_lookback)
                        
                        returns_df = pd.DataFrame(returns_data).dropna()
                        
                        if len(returns_df) < 30:
                            st.error("Insufficient data for optimization. Please select more assets or increase lookback period.")
                            return
                        
                        # Calculate expected returns and covariance matrix
                        expected_returns = returns_df.mean() * 252  # Annualized
                        cov_matrix = returns_df.cov() * 252  # Annualized
                        
                        # Simple portfolio optimization using scipy
                        from scipy.optimize import minimize
                        
                        n_assets = len(mpt_assets)
                        
                        def portfolio_stats(weights, returns, cov_matrix, risk_free_rate):
                            """Calculate portfolio statistics."""
                            portfolio_return = np.dot(weights, returns)
                            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
                            return portfolio_return, portfolio_vol, sharpe_ratio
                        
                        def negative_sharpe(weights):
                            """Objective function for maximizing Sharpe ratio."""
                            _, _, sharpe = portfolio_stats(weights, expected_returns, cov_matrix, risk_free_rate)
                            return -sharpe
                        
                        def portfolio_volatility(weights):
                            """Objective function for minimizing volatility."""
                            _, vol, _ = portfolio_stats(weights, expected_returns, cov_matrix, risk_free_rate)
                            return vol
                        
                        def negative_return(weights):
                            """Objective function for maximizing return."""
                            ret, _, _ = portfolio_stats(weights, expected_returns, cov_matrix, risk_free_rate)
                            return -ret
                        
                        # Constraints and bounds
                        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
                        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
                        
                        # Initial guess (equal weights)
                        initial_guess = np.array([1/n_assets] * n_assets)
                        
                        # Optimize based on selected objective
                        if mpt_objective == 'Max Sharpe Ratio':
                            result = minimize(negative_sharpe, initial_guess, method='SLSQP', 
                                            bounds=bounds, constraints=constraints)
                        elif mpt_objective == 'Min Volatility':
                            result = minimize(portfolio_volatility, initial_guess, method='SLSQP',
                                            bounds=bounds, constraints=constraints)
                        else:  # Max Return
                            result = minimize(negative_return, initial_guess, method='SLSQP',
                                            bounds=bounds, constraints=constraints)
                        
                        if result.success:
                            optimal_weights = result.x
                            
                            # Calculate portfolio metrics
                            opt_return, opt_vol, opt_sharpe = portfolio_stats(
                                optimal_weights, expected_returns, cov_matrix, risk_free_rate
                            )
                            
                            # Display results
                            st.markdown("#### Optimization Results")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Expected Return", f"{opt_return:.2%}")
                            with col2:
                                st.metric("Volatility", f"{opt_vol:.2%}")
                            with col3:
                                st.metric("Sharpe Ratio", f"{opt_sharpe:.3f}")
                            
                            # Optimal weights
                            st.markdown("#### Optimal Weights")
                            weights_df = pd.DataFrame({
                                'Asset': mpt_assets,
                                'Weight': optimal_weights,
                                'Weight (%)': [f"{w:.1%}" for w in optimal_weights]
                            }).sort_values('Weight', ascending=False)
                            
                            st.dataframe(weights_df, hide_index=True)
                            
                            # Pie chart of weights
                            fig_pie = px.pie(
                                weights_df,
                                values='Weight',
                                names='Asset',
                                title="Optimal Portfolio Allocation"
                            )
                            st.plotly_chart(fig_pie, width='stretch')
                            
                            # Compare with equal-weight portfolio
                            st.markdown("#### Comparison with Equal-Weight Portfolio")
                            
                            equal_weights = np.array([1/n_assets] * n_assets)
                            eq_return, eq_vol, eq_sharpe = portfolio_stats(
                                equal_weights, expected_returns, cov_matrix, risk_free_rate
                            )
                            
                            comparison_df = pd.DataFrame({
                                'Portfolio': ['Optimized', 'Equal-Weight'],
                                'Expected Return': [f"{opt_return:.2%}", f"{eq_return:.2%}"],
                                'Volatility': [f"{opt_vol:.2%}", f"{eq_vol:.2%}"],
                                'Sharpe Ratio': [f"{opt_sharpe:.3f}", f"{eq_sharpe:.3f}"]
                            })
                            
                            st.dataframe(comparison_df, hide_index=True)
                            
                            if opt_sharpe > eq_sharpe:
                                st.success(f"🎯 Optimized portfolio improves Sharpe ratio by {((opt_sharpe/eq_sharpe - 1) * 100):.1f}%")
                            else:
                                st.info("📊 Equal-weight portfolio shows competitive performance")
                        
                        else:
                            st.error("Optimization failed. Try adjusting constraints or selecting different assets.")
                    
                    except Exception as e:
                        st.error(f"Portfolio optimization failed: {str(e)}")
                        st.info("This might be due to insufficient data or numerical issues.")
        else:
            st.info("Please select at least 2 assets for portfolio optimization.")
    
    with tab2:
        st.markdown("### Risk Parity Portfolio")
        st.markdown("Allocate capital based on risk contribution rather than dollar amounts.")
        
        # Risk parity configuration
        col1, col2 = st.columns(2)
        
        with col1:
            rp_assets = st.multiselect(
                "Select Assets:",
                summary['tickers'],
                default=summary['tickers'][:3],
                key="rp_assets"
            )
        
        with col2:
            rp_lookback = st.slider(
                "Lookback Period:",
                60, 250, 120,
                key="rp_lookback"
            )
        
        if len(rp_assets) >= 2:
            if st.button("Calculate Risk Parity", key="calc_risk_parity"):
                with st.spinner("Calculating risk parity weights..."):
                    try:
                        # Get returns data
                        returns_data = {}
                        for asset in rp_assets:
                            asset_prices = get_ticker_data(data, asset, 'close')
                            asset_returns = asset_prices.pct_change().dropna()
                            returns_data[asset] = asset_returns.tail(rp_lookback)
                        
                        returns_df = pd.DataFrame(returns_data).dropna()
                        
                        # Calculate volatilities (simple risk parity)
                        volatilities = returns_df.std() * np.sqrt(252)
                        
                        # Inverse volatility weights
                        inv_vol_weights = 1 / volatilities
                        risk_parity_weights = inv_vol_weights / inv_vol_weights.sum()
                        
                        # Display results
                        st.markdown("#### Risk Parity Results")
                        
                        rp_df = pd.DataFrame({
                            'Asset': rp_assets,
                            'Volatility': [f"{vol:.2%}" for vol in volatilities],
                            'Weight': risk_parity_weights.values,
                            'Weight (%)': [f"{w:.1%}" for w in risk_parity_weights.values]
                        }).sort_values('Weight', ascending=False)
                        
                        st.dataframe(rp_df, hide_index=True)
                        
                        # Visualization
                        fig_rp = px.bar(
                            rp_df,
                            x='Asset',
                            y='Weight',
                            title="Risk Parity Weights",
                            color='Weight',
                            color_continuous_scale='viridis'
                        )
                        st.plotly_chart(fig_rp, width='stretch')
                        
                        # Risk contribution analysis
                        st.markdown("#### Risk Contribution Analysis")
                        
                        # Calculate portfolio volatility and risk contributions
                        cov_matrix = returns_df.cov() * 252
                        portfolio_vol = np.sqrt(np.dot(risk_parity_weights.values.T, 
                                                     np.dot(cov_matrix, risk_parity_weights.values)))
                        
                        risk_contributions = (risk_parity_weights.values * 
                                            np.dot(cov_matrix, risk_parity_weights.values)) / portfolio_vol
                        
                        risk_contrib_df = pd.DataFrame({
                            'Asset': rp_assets,
                            'Risk Contribution': risk_contributions,
                            'Risk Contrib (%)': [f"{rc:.1%}" for rc in risk_contributions]
                        })
                        
                        st.dataframe(risk_contrib_df, hide_index=True)
                        
                        st.info(f"📊 Portfolio Volatility: {portfolio_vol:.2%}")
                    
                    except Exception as e:
                        st.error(f"Risk parity calculation failed: {str(e)}")
        else:
            st.info("Please select at least 2 assets for risk parity analysis.")
    
    with tab3:
        st.markdown("### Forecast-Based Allocation")
        st.markdown("Allocate capital based on forecasting model predictions.")
        
        # Forecast-based configuration
        col1, col2 = st.columns(2)
        
        with col1:
            fb_assets = st.multiselect(
                "Select Assets:",
                summary['tickers'],
                default=summary['tickers'][:3],
                key="fb_assets"
            )
            
            fb_model = st.selectbox(
                "Forecasting Model:",
                ['Moving Average', 'ARIMA', 'VAR'],
                key="fb_model"
            )
        
        with col2:
            fb_horizon = st.slider(
                "Forecast Horizon:",
                5, 30, 10,
                key="fb_horizon"
            )
            
            fb_allocation_method = st.selectbox(
                "Allocation Method:",
                ['Proportional to Expected Return', 'Rank-Based', 'Long-Short'],
                key="fb_allocation"
            )
        
        if len(fb_assets) >= 2:
            if st.button("Generate Forecast-Based Portfolio", key="calc_forecast_portfolio"):
                with st.spinner("Generating forecasts and calculating allocations..."):
                    try:
                        forecasts = {}
                        current_prices = {}
                        
                        # Generate forecasts for each asset
                        for asset in fb_assets:
                            asset_prices = get_ticker_data(data, asset, 'close')
                            current_price = asset_prices.iloc[-1]
                            current_prices[asset] = current_price
                            
                            # Generate forecast based on selected model
                            if fb_model == 'Moving Average':
                                forecast = moving_average_forecast(asset_prices, window=10, forecast_steps=fb_horizon)
                            elif fb_model == 'ARIMA':
                                forecast = arima_forecast(asset_prices, order=(1,1,1), forecast_steps=fb_horizon)
                            else:  # VAR
                                if len(fb_assets) >= 2:
                                    # Use VAR for multivariate forecast
                                    multi_data = get_multiple_tickers_data(data, fb_assets, 'close')
                                    var_model = VARModel(maxlags=3)
                                    var_model.fit(multi_data)
                                    var_forecasts, _ = var_model.predict(steps=fb_horizon)
                                    forecast = var_forecasts[asset]
                                else:
                                    # Fallback to MA if only one asset
                                    forecast = moving_average_forecast(asset_prices, window=10, forecast_steps=fb_horizon)
                            
                            forecasts[asset] = forecast
                        
                        # Calculate expected returns
                        expected_returns = {}
                        for asset in fb_assets:
                            forecast_price = forecasts[asset].iloc[-1] if hasattr(forecasts[asset], 'iloc') else forecasts[asset][-1]
                            expected_return = (forecast_price - current_prices[asset]) / current_prices[asset]
                            expected_returns[asset] = expected_return
                        
                        # Calculate allocations based on method
                        if fb_allocation_method == 'Proportional to Expected Return':
                            # Normalize positive returns and set negative to zero
                            pos_returns = {k: max(0, v) for k, v in expected_returns.items()}
                            total_pos_return = sum(pos_returns.values())
                            
                            if total_pos_return > 0:
                                weights = {k: v / total_pos_return for k, v in pos_returns.items()}
                            else:
                                # Equal weights if all returns are negative
                                weights = {k: 1/len(fb_assets) for k in fb_assets}
                        
                        elif fb_allocation_method == 'Rank-Based':
                            # Rank assets by expected return
                            sorted_assets = sorted(expected_returns.items(), key=lambda x: x[1], reverse=True)
                            
                            # Assign weights: highest rank gets highest weight
                            total_ranks = sum(range(1, len(fb_assets) + 1))
                            weights = {}
                            for i, (asset, _) in enumerate(sorted_assets):
                                rank_weight = (len(fb_assets) - i) / total_ranks
                                weights[asset] = rank_weight
                        
                        else:  # Long-Short
                            # Long positive expected returns, short negative
                            total_abs_return = sum(abs(v) for v in expected_returns.values())
                            
                            if total_abs_return > 0:
                                weights = {k: v / total_abs_return for k, v in expected_returns.items()}
                            else:
                                weights = {k: 1/len(fb_assets) for k in fb_assets}
                        
                        # Display results
                        st.markdown("#### Forecast-Based Portfolio Results")
                        
                        fb_df = pd.DataFrame({
                            'Asset': fb_assets,
                            'Current Price': [f"${current_prices[asset]:.2f}" for asset in fb_assets],
                            'Forecast Price': [f"${forecasts[asset].iloc[-1] if hasattr(forecasts[asset], 'iloc') else forecasts[asset][-1]:.2f}" for asset in fb_assets],
                            'Expected Return': [f"{expected_returns[asset]:.2%}" for asset in fb_assets],
                            'Weight': [weights[asset] for asset in fb_assets],
                            'Weight (%)': [f"{weights[asset]:.1%}" for asset in fb_assets]
                        }).sort_values('Weight', ascending=False)
                        
                        st.dataframe(fb_df, hide_index=True)
                        
                        # Visualization
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_weights = px.pie(
                                fb_df,
                                values='Weight',
                                names='Asset',
                                title="Forecast-Based Allocation"
                            )
                            st.plotly_chart(fig_weights, width='stretch')
                        
                        with col2:
                            fig_returns = px.bar(
                                fb_df,
                                x='Asset',
                                y=[float(er.strip('%'))/100 for er in fb_df['Expected Return']],
                                title="Expected Returns by Asset",
                                color=[float(er.strip('%'))/100 for er in fb_df['Expected Return']],
                                color_continuous_scale='RdYlGn'
                            )
                            fig_returns.update_yaxis(title="Expected Return")
                            st.plotly_chart(fig_returns, width='stretch')
                        
                        # Portfolio summary
                        portfolio_expected_return = sum(weights[asset] * expected_returns[asset] for asset in fb_assets)
                        
                        st.markdown("#### Portfolio Summary")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Portfolio Expected Return", f"{portfolio_expected_return:.2%}")
                        with col2:
                            st.metric("Best Asset", fb_df.iloc[0]['Asset'])
                        with col3:
                            st.metric("Forecast Model", fb_model)
                    
                    except Exception as e:
                        st.error(f"Forecast-based allocation failed: {str(e)}")
        else:
            st.info("Please select at least 2 assets for forecast-based allocation.")
    
    with tab4:
        st.markdown("### Portfolio Analysis")
        st.markdown("Analyze and compare different portfolio strategies.")
        
        # Portfolio comparison
        analysis_assets = st.multiselect(
            "Select Assets for Analysis:",
            summary['tickers'],
            default=summary['tickers'][:4],
            key="analysis_assets"
        )
        
        if len(analysis_assets) >= 2:
            if st.button("Run Portfolio Analysis", key="run_portfolio_analysis"):
                with st.spinner("Analyzing portfolio strategies..."):
                    try:
                        # Get data
                        returns_data = {}
                        prices_data = {}
                        for asset in analysis_assets:
                            asset_prices = get_ticker_data(data, asset, 'close')
                            asset_returns = asset_prices.pct_change().dropna()
                            returns_data[asset] = asset_returns.tail(120)
                            prices_data[asset] = asset_prices.tail(120)
                        
                        returns_df = pd.DataFrame(returns_data).dropna()
                        prices_df = pd.DataFrame(prices_data).dropna()
                        
                        # Calculate different portfolio strategies
                        portfolios = {}
                        
                        # Equal Weight
                        n_assets = len(analysis_assets)
                        equal_weights = np.array([1/n_assets] * n_assets)
                        portfolios['Equal Weight'] = {
                            'weights': equal_weights,
                            'returns': (returns_df * equal_weights).sum(axis=1)
                        }
                        
                        # Market Cap Weight (approximated by price)
                        market_weights = prices_df.iloc[-1] / prices_df.iloc[-1].sum()
                        portfolios['Market Weight'] = {
                            'weights': market_weights.values,
                            'returns': (returns_df * market_weights.values).sum(axis=1)
                        }
                        
                        # Inverse Volatility (Risk Parity approximation)
                        volatilities = returns_df.std()
                        inv_vol_weights = (1 / volatilities) / (1 / volatilities).sum()
                        portfolios['Risk Parity'] = {
                            'weights': inv_vol_weights.values,
                            'returns': (returns_df * inv_vol_weights.values).sum(axis=1)
                        }
                        
                        # Calculate portfolio metrics
                        portfolio_metrics = {}
                        for name, portfolio in portfolios.items():
                            port_returns = portfolio['returns']
                            
                            annual_return = port_returns.mean() * 252
                            annual_vol = port_returns.std() * np.sqrt(252)
                            sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
                            max_drawdown = (port_returns.cumsum().cummax() - port_returns.cumsum()).max()
                            
                            portfolio_metrics[name] = {
                                'Annual Return': annual_return,
                                'Volatility': annual_vol,
                                'Sharpe Ratio': sharpe_ratio,
                                'Max Drawdown': max_drawdown
                            }
                        
                        # Display results
                        st.markdown("#### Portfolio Performance Comparison")
                        
                        metrics_df = pd.DataFrame(portfolio_metrics).T
                        metrics_df['Annual Return'] = metrics_df['Annual Return'].map(lambda x: f"{x:.2%}")
                        metrics_df['Volatility'] = metrics_df['Volatility'].map(lambda x: f"{x:.2%}")
                        metrics_df['Sharpe Ratio'] = metrics_df['Sharpe Ratio'].map(lambda x: f"{x:.3f}")
                        metrics_df['Max Drawdown'] = metrics_df['Max Drawdown'].map(lambda x: f"{x:.2%}")
                        
                        st.dataframe(metrics_df)
                        
                        # Cumulative returns chart
                        st.markdown("#### Cumulative Returns Comparison")
                        
                        fig_cumret = go.Figure()
                        
                        colors = ['blue', 'red', 'green', 'orange']
                        for i, (name, portfolio) in enumerate(portfolios.items()):
                            cumulative_returns = (1 + portfolio['returns']).cumprod()
                            
                            fig_cumret.add_trace(go.Scatter(
                                x=cumulative_returns.index,
                                y=cumulative_returns.values,
                                mode='lines',
                                name=name,
                                line=dict(color=colors[i], width=2)
                            ))
                        
                        fig_cumret.update_layout(
                            title="Cumulative Returns Comparison",
                            xaxis_title="Date",
                            yaxis_title="Cumulative Return",
                            height=500
                        )
                        
                        st.plotly_chart(fig_cumret, width='stretch')
                        
                        # Portfolio weights comparison
                        st.markdown("#### Portfolio Weights Comparison")
                        
                        weights_comparison = pd.DataFrame({
                            'Asset': analysis_assets,
                            'Equal Weight': [f"{w:.1%}" for w in portfolios['Equal Weight']['weights']],
                            'Market Weight': [f"{w:.1%}" for w in portfolios['Market Weight']['weights']],
                            'Risk Parity': [f"{w:.1%}" for w in portfolios['Risk Parity']['weights']]
                        })
                        
                        st.dataframe(weights_comparison, hide_index=True)
                        
                        # Best performing portfolio
                        numeric_metrics = {name: metrics['Sharpe Ratio'] for name, metrics in portfolio_metrics.items()}
                        best_portfolio = max(numeric_metrics, key=numeric_metrics.get)
                        
                        st.success(f"🏆 Best Performing Strategy: **{best_portfolio}** (Sharpe: {numeric_metrics[best_portfolio]:.3f})")
                    
                    except Exception as e:
                        st.error(f"Portfolio analysis failed: {str(e)}")
        else:
            st.info("Please select at least 2 assets for portfolio analysis.")
    
    # Educational content
    st.markdown("---")
    st.markdown("### 📚 Portfolio Optimization Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Modern Portfolio Theory:**
        - Balances risk and return
        - Uses historical correlations
        - Assumes normal distributions
        - Best for stable markets
        """)
    
    with col2:
        st.markdown("""
        **Risk Parity:**
        - Equal risk contribution
        - Diversifies risk sources
        - Less sensitive to correlations
        - Good for volatile markets
        """)
    
    st.info("""
    💡 **Tip:** Combine multiple approaches and rebalance regularly. 
    No single strategy works in all market conditions.
    """)


def page_backtesting():
    """Portfolio backtesting page."""
    st.markdown('<div class="main-header">Portfolio Backtesting</div>', unsafe_allow_html=True)
    
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please load data first from the Home page.")
        return
    
    data = st.session_state.data
    summary = get_data_summary(data)
    
    st.markdown("""
    Test portfolio strategies against historical data to evaluate performance, risk metrics, 
    and trading costs before deploying real capital.
    """)
    
    # Create tabs for different backtesting approaches
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Strategy Backtest", "🔄 Rebalancing Analysis", "💰 Transaction Costs", "📈 Performance Analytics"])
    
    with tab1:
        st.markdown("### Strategy Backtesting")
        st.markdown("Test different portfolio strategies against historical data.")
        
        # Strategy configuration
        col1, col2 = st.columns(2)
        
        with col1:
            bt_assets = st.multiselect(
                "Select Portfolio Assets:",
                summary['tickers'],
                default=summary['tickers'][:3],
                key="bt_assets"
            )
            
            bt_strategy = st.selectbox(
                "Portfolio Strategy:",
                ['Equal Weight', 'Market Cap Weight', 'Risk Parity', 'Momentum', 'Mean Reversion'],
                key="bt_strategy"
            )
            
            bt_start_date = st.date_input(
                "Backtest Start Date:",
                value=datetime.now() - timedelta(days=365),
                key="bt_start"
            )
        
        with col2:
            bt_end_date = st.date_input(
                "Backtest End Date:",
                value=datetime.now() - timedelta(days=30),
                key="bt_end"
            )
            
            bt_rebalance_freq = st.selectbox(
                "Rebalancing Frequency:",
                ['Monthly', 'Quarterly', 'Semi-Annually', 'Annually'],
                key="bt_rebalance"
            )
            
            bt_initial_capital = st.number_input(
                "Initial Capital ($):",
                min_value=1000,
                value=100000,
                step=1000,
                key="bt_capital"
            )
        
        # Strategy parameters
        with st.expander("Strategy Parameters"):
            if bt_strategy == 'Momentum':
                momentum_lookback = st.slider(
                    "Momentum Lookback (days):",
                    10, 60, 20,
                    key="momentum_lookback"
                )
                momentum_threshold = st.slider(
                    "Momentum Threshold:",
                    0.0, 0.2, 0.05,
                    key="momentum_threshold"
                )
            
            elif bt_strategy == 'Mean Reversion':
                mr_lookback = st.slider(
                    "Mean Reversion Lookback (days):",
                    20, 100, 50,
                    key="mr_lookback"
                )
                mr_threshold = st.slider(
                    "Deviation Threshold:",
                    0.5, 3.0, 2.0,
                    key="mr_threshold"
                )
        
        if len(bt_assets) >= 2:
            if st.button("Run Backtest", type="primary", key="run_backtest"):
                with st.spinner("Running portfolio backtest..."):
                    try:
                        # Get historical data for backtest period
                        bt_data = {}
                        for asset in bt_assets:
                            asset_prices = get_ticker_data(data, asset, 'close')
                            # Filter data by date range
                            mask = (asset_prices.index >= pd.Timestamp(bt_start_date)) & (asset_prices.index <= pd.Timestamp(bt_end_date))
                            bt_data[asset] = asset_prices[mask]
                        
                        bt_df = pd.DataFrame(bt_data).dropna()
                        
                        if len(bt_df) < 30:
                            st.error("Insufficient data for the selected date range. Please adjust the dates.")
                            return
                        
                        # Calculate returns
                        returns_df = bt_df.pct_change().dropna()
                        
                        # Set rebalancing frequency
                        freq_map = {
                            'Monthly': '1M',
                            'Quarterly': '3M',
                            'Semi-Annually': '6M',
                            'Annually': '12M'
                        }
                        rebalance_freq = freq_map[bt_rebalance_freq]
                        
                        # Generate rebalancing dates
                        rebalance_dates = pd.date_range(
                            start=bt_df.index[0],
                            end=bt_df.index[-1],
                            freq=rebalance_freq
                        )
                        
                        # Initialize portfolio
                        portfolio_value = pd.Series(index=bt_df.index, dtype=float)
                        portfolio_weights = pd.DataFrame(index=rebalance_dates, columns=bt_assets)
                        
                        current_capital = bt_initial_capital
                        
                        # Backtesting loop
                        for i, rebal_date in enumerate(rebalance_dates):
                            # Find closest date in data
                            closest_date = bt_df.index[bt_df.index >= rebal_date][0] if any(bt_df.index >= rebal_date) else bt_df.index[-1]
                            
                            # Calculate weights based on strategy
                            if bt_strategy == 'Equal Weight':
                                weights = np.array([1/len(bt_assets)] * len(bt_assets))
                            
                            elif bt_strategy == 'Market Cap Weight':
                                # Approximate market cap by price
                                prices = bt_df.loc[closest_date]
                                weights = prices / prices.sum()
                                weights = weights.values
                            
                            elif bt_strategy == 'Risk Parity':
                                # Use inverse volatility
                                if i == 0:
                                    lookback_data = returns_df.loc[:closest_date].tail(60)
                                else:
                                    prev_rebal = rebalance_dates[i-1]
                                    lookback_data = returns_df.loc[prev_rebal:closest_date]
                                
                                if len(lookback_data) > 5:
                                    volatilities = lookback_data.std()
                                    inv_vol = 1 / volatilities
                                    weights = (inv_vol / inv_vol.sum()).values
                                else:
                                    weights = np.array([1/len(bt_assets)] * len(bt_assets))
                            
                            elif bt_strategy == 'Momentum':
                                # Calculate momentum scores
                                if i == 0:
                                    lookback_data = returns_df.loc[:closest_date].tail(momentum_lookback)
                                else:
                                    prev_rebal = rebalance_dates[i-1]
                                    lookback_data = returns_df.loc[prev_rebal:closest_date].tail(momentum_lookback)
                                
                                if len(lookback_data) > 5:
                                    momentum_scores = lookback_data.mean()
                                    # Only invest in assets above threshold
                                    positive_momentum = momentum_scores > momentum_threshold
                                    if positive_momentum.any():
                                        weights = np.zeros(len(bt_assets))
                                        weights[positive_momentum] = 1 / positive_momentum.sum()
                                    else:
                                        weights = np.array([1/len(bt_assets)] * len(bt_assets))
                                else:
                                    weights = np.array([1/len(bt_assets)] * len(bt_assets))
                            
                            elif bt_strategy == 'Mean Reversion':
                                # Calculate mean reversion signals
                                if i == 0:
                                    lookback_data = returns_df.loc[:closest_date].tail(mr_lookback)
                                else:
                                    prev_rebal = rebalance_dates[i-1]
                                    lookback_data = returns_df.loc[prev_rebal:closest_date].tail(mr_lookback)
                                
                                if len(lookback_data) > 5:
                                    mean_returns = lookback_data.mean()
                                    std_returns = lookback_data.std()
                                    recent_returns = lookback_data.tail(5).mean()
                                    
                                    # Z-scores for mean reversion
                                    z_scores = (recent_returns - mean_returns) / std_returns
                                    
                                    # Invest more in assets that have deviated negatively
                                    mr_signals = -z_scores  # Negative z-score means buy signal
                                    mr_signals = np.clip(mr_signals, -mr_threshold, mr_threshold)
                                    
                                    # Normalize to weights
                                    weights = (mr_signals - mr_signals.min()) / (mr_signals.max() - mr_signals.min())
                                    weights = weights / weights.sum()
                                    weights = weights.values
                                else:
                                    weights = np.array([1/len(bt_assets)] * len(bt_assets))
                            
                            # Store weights
                            portfolio_weights.loc[closest_date] = weights
                            
                            # Calculate portfolio value from this rebalancing date
                            if i < len(rebalance_dates) - 1:
                                next_rebal = rebalance_dates[i + 1]
                                period_data = bt_df.loc[closest_date:next_rebal]
                                period_returns = period_data.pct_change().dropna()
                            else:
                                period_data = bt_df.loc[closest_date:]
                                period_returns = period_data.pct_change().dropna()
                            
                            # Calculate portfolio returns for this period
                            for date in period_data.index:
                                if date == closest_date:
                                    portfolio_value[date] = current_capital
                                else:
                                    if date in period_returns.index:
                                        period_return = np.dot(weights, period_returns.loc[date])
                                        current_capital *= (1 + period_return)
                                        portfolio_value[date] = current_capital
                        
                        # Calculate benchmark (equal weight buy and hold)
                        equal_weights = np.array([1/len(bt_assets)] * len(bt_assets))
                        benchmark_returns = (returns_df * equal_weights).sum(axis=1)
                        benchmark_value = (1 + benchmark_returns).cumprod() * bt_initial_capital
                        
                        # Calculate performance metrics
                        portfolio_returns = portfolio_value.pct_change().dropna()
                        
                        # Annualized metrics
                        trading_days = 252
                        total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
                        annualized_return = (1 + total_return) ** (trading_days / len(portfolio_value)) - 1
                        annualized_vol = portfolio_returns.std() * np.sqrt(trading_days)
                        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
                        
                        # Maximum drawdown
                        cumulative = portfolio_value / portfolio_value.iloc[0]
                        running_max = cumulative.cummax()
                        drawdown = (cumulative - running_max) / running_max
                        max_drawdown = drawdown.min()
                        
                        # Benchmark metrics
                        bench_total_return = (benchmark_value.iloc[-1] / benchmark_value.iloc[0]) - 1
                        bench_annualized_return = (1 + bench_total_return) ** (trading_days / len(benchmark_value)) - 1
                        bench_returns = benchmark_value.pct_change().dropna()
                        bench_vol = bench_returns.std() * np.sqrt(trading_days)
                        bench_sharpe = bench_annualized_return / bench_vol if bench_vol > 0 else 0
                        
                        # Display results
                        st.markdown("#### Backtest Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Return", f"{total_return:.2%}")
                        with col2:
                            st.metric("Annualized Return", f"{annualized_return:.2%}")
                        with col3:
                            st.metric("Volatility", f"{annualized_vol:.2%}")
                        with col4:
                            st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Max Drawdown", f"{max_drawdown:.2%}")
                        with col2:
                            st.metric("Final Value", f"${portfolio_value.iloc[-1]:,.0f}")
                        with col3:
                            st.metric("Total Trades", f"{len(rebalance_dates)}")
                        with col4:
                            alpha = annualized_return - bench_annualized_return
                            st.metric("Alpha vs Benchmark", f"{alpha:.2%}")
                        
                        # Performance chart
                        st.markdown("#### Portfolio Performance")
                        
                        fig_perf = go.Figure()
                        
                        fig_perf.add_trace(go.Scatter(
                            x=portfolio_value.index,
                            y=portfolio_value.values,
                            mode='lines',
                            name=f'{bt_strategy} Portfolio',
                            line=dict(color='blue', width=2)
                        ))
                        
                        fig_perf.add_trace(go.Scatter(
                            x=benchmark_value.index,
                            y=benchmark_value.values,
                            mode='lines',
                            name='Equal Weight Benchmark',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                        
                        # Add rebalancing points
                        rebal_values = [portfolio_value[portfolio_value.index >= date].iloc[0] if any(portfolio_value.index >= date) else portfolio_value.iloc[-1] for date in rebalance_dates]
                        
                        fig_perf.add_trace(go.Scatter(
                            x=rebalance_dates,
                            y=rebal_values,
                            mode='markers',
                            name='Rebalancing Points',
                            marker=dict(color='green', size=8, symbol='diamond')
                        ))
                        
                        fig_perf.update_layout(
                            title="Portfolio Value Over Time",
                            xaxis_title="Date",
                            yaxis_title="Portfolio Value ($)",
                            height=500,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_perf, width='stretch')
                        
                        # Portfolio weights over time
                        st.markdown("#### Portfolio Weights Over Time")
                        
                        # Interpolate weights for visualization
                        weights_filled = portfolio_weights.reindex(bt_df.index).fillna(method='ffill')
                        
                        fig_weights = go.Figure()
                        
                        colors = px.colors.qualitative.Set3
                        for i, asset in enumerate(bt_assets):
                            fig_weights.add_trace(go.Scatter(
                                x=weights_filled.index,
                                y=weights_filled[asset] * 100,
                                mode='lines',
                                name=asset,
                                fill='tonexty' if i > 0 else 'tozeroy',
                                line=dict(color=colors[i % len(colors)])
                            ))
                        
                        fig_weights.update_layout(
                            title="Portfolio Allocation Over Time",
                            xaxis_title="Date",
                            yaxis_title="Weight (%)",
                            height=400,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_weights, width='stretch')
                        
                        # Performance comparison table
                        st.markdown("#### Strategy vs Benchmark Comparison")
                        
                        comparison_df = pd.DataFrame({
                            'Metric': ['Total Return', 'Annualized Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown'],
                            'Strategy': [
                                f"{total_return:.2%}",
                                f"{annualized_return:.2%}",
                                f"{annualized_vol:.2%}",
                                f"{sharpe_ratio:.3f}",
                                f"{max_drawdown:.2%}"
                            ],
                            'Benchmark': [
                                f"{bench_total_return:.2%}",
                                f"{bench_annualized_return:.2%}",
                                f"{bench_vol:.2%}",
                                f"{bench_sharpe:.3f}",
                                f"{(benchmark_value / benchmark_value.cummax() - 1).min():.2%}"
                            ]
                        })
                        
                        st.dataframe(comparison_df, hide_index=True)
                        
                        if sharpe_ratio > bench_sharpe:
                            st.success(f"🎯 Strategy outperformed benchmark by {((sharpe_ratio/bench_sharpe - 1) * 100):.1f}% in risk-adjusted returns!")
                        else:
                            st.info("📊 Benchmark showed better risk-adjusted performance")
                    
                    except Exception as e:
                        st.error(f"Backtesting failed: {str(e)}")
                        st.info("This might be due to insufficient data or calculation errors.")
        else:
            st.info("Please select at least 2 assets for backtesting.")
    
    with tab2:
        st.markdown("### Rebalancing Analysis")
        st.markdown("Analyze the impact of different rebalancing frequencies on portfolio performance.")
        
        # Rebalancing comparison configuration
        col1, col2 = st.columns(2)
        
        with col1:
            rebal_assets = st.multiselect(
                "Select Assets:",
                summary['tickers'],
                default=summary['tickers'][:3],
                key="rebal_assets"
            )
        
        with col2:
            rebal_period = st.selectbox(
                "Analysis Period:",
                ['1 Year', '2 Years', '3 Years'],
                key="rebal_period"
            )
        
        if len(rebal_assets) >= 2:
            if st.button("Analyze Rebalancing Impact", key="analyze_rebalancing"):
                with st.spinner("Analyzing rebalancing frequencies..."):
                    try:
                        # Set analysis period
                        period_map = {'1 Year': 365, '2 Years': 730, '3 Years': 1095}
                        days_back = period_map[rebal_period]
                        
                        start_date = datetime.now() - timedelta(days=days_back)
                        
                        # Get data
                        rebal_data = {}
                        for asset in rebal_assets:
                            asset_prices = get_ticker_data(data, asset, 'close')
                            mask = asset_prices.index >= pd.Timestamp(start_date)
                            rebal_data[asset] = asset_prices[mask]
                        
                        rebal_df = pd.DataFrame(rebal_data).dropna()
                        returns_df = rebal_df.pct_change().dropna()
                        
                        # Test different rebalancing frequencies
                        frequencies = ['1M', '3M', '6M', '12M', 'None']  # None = buy and hold
                        rebal_results = {}
                        
                        for freq in frequencies:
                            if freq == 'None':
                                # Buy and hold strategy
                                equal_weights = np.array([1/len(rebal_assets)] * len(rebal_assets))
                                portfolio_returns = (returns_df * equal_weights).sum(axis=1)
                            else:
                                # Rebalancing strategy
                                rebal_dates = pd.date_range(
                                    start=rebal_df.index[0],
                                    end=rebal_df.index[-1],
                                    freq=freq
                                )
                                
                                portfolio_returns = pd.Series(index=returns_df.index, dtype=float)
                                
                                for i, rebal_date in enumerate(rebal_dates):
                                    closest_date = rebal_df.index[rebal_df.index >= rebal_date][0] if any(rebal_df.index >= rebal_date) else rebal_df.index[-1]
                                    
                                    if i < len(rebal_dates) - 1:
                                        next_rebal = rebal_dates[i + 1]
                                        period_returns = returns_df.loc[closest_date:next_rebal]
                                    else:
                                        period_returns = returns_df.loc[closest_date:]
                                    
                                    # Equal weight rebalancing
                                    equal_weights = np.array([1/len(rebal_assets)] * len(rebal_assets))
                                    period_portfolio_returns = (period_returns * equal_weights).sum(axis=1)
                                    portfolio_returns.loc[period_portfolio_returns.index] = period_portfolio_returns
                                
                                portfolio_returns = portfolio_returns.dropna()
                            
                            # Calculate metrics
                            total_return = (1 + portfolio_returns).cumprod().iloc[-1] - 1
                            annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
                            volatility = portfolio_returns.std() * np.sqrt(252)
                            sharpe = annualized_return / volatility if volatility > 0 else 0
                            
                            # Transaction costs (approximate)
                            num_rebalances = len(pd.date_range(start=rebal_df.index[0], end=rebal_df.index[-1], freq=freq)) if freq != 'None' else 0
                            transaction_cost = num_rebalances * 0.001 * len(rebal_assets)  # 0.1% per asset per rebalance
                            
                            rebal_results[freq] = {
                                'Total Return': total_return,
                                'Annualized Return': annualized_return,
                                'Volatility': volatility,
                                'Sharpe Ratio': sharpe,
                                'Rebalances': num_rebalances,
                                'Transaction Cost': transaction_cost,
                                'Net Return': total_return - transaction_cost
                            }
                        
                        # Display results
                        st.markdown("#### Rebalancing Frequency Comparison")
                        
                        freq_labels = {'1M': 'Monthly', '3M': 'Quarterly', '6M': 'Semi-Annual', '12M': 'Annual', 'None': 'Buy & Hold'}
                        
                        results_df = pd.DataFrame(rebal_results).T
                        results_df.index = [freq_labels[freq] for freq in results_df.index]
                        
                        # Format the dataframe for display
                        display_df = pd.DataFrame({
                            'Frequency': results_df.index,
                            'Total Return': [f"{r:.2%}" for r in results_df['Total Return']],
                            'Annualized Return': [f"{r:.2%}" for r in results_df['Annualized Return']],
                            'Volatility': [f"{r:.2%}" for r in results_df['Volatility']],
                            'Sharpe Ratio': [f"{r:.3f}" for r in results_df['Sharpe Ratio']],
                            'Rebalances': [int(r) for r in results_df['Rebalances']],
                            'Transaction Cost': [f"{r:.2%}" for r in results_df['Transaction Cost']],
                            'Net Return': [f"{r:.2%}" for r in results_df['Net Return']]
                        })
                        
                        st.dataframe(display_df, hide_index=True)
                        
                        # Best frequency
                        best_freq = results_df['Sharpe Ratio'].idxmax()
                        best_sharpe = results_df.loc[best_freq, 'Sharpe Ratio']
                        
                        st.success(f"🏆 Best Rebalancing Frequency: **{best_freq}** (Sharpe: {best_sharpe:.3f})")
                        
                        # Visualization
                        fig_rebal = go.Figure()
                        
                        for freq in results_df.index:
                            fig_rebal.add_trace(go.Bar(
                                name=freq,
                                x=[freq],
                                y=[results_df.loc[freq, 'Sharpe Ratio']],
                                text=f"{results_df.loc[freq, 'Sharpe Ratio']:.3f}",
                                textposition='auto'
                            ))
                        
                        fig_rebal.update_layout(
                            title="Sharpe Ratio by Rebalancing Frequency",
                            xaxis_title="Rebalancing Frequency",
                            yaxis_title="Sharpe Ratio",
                            height=400
                        )
                        
                        st.plotly_chart(fig_rebal, width='stretch')
                    
                    except Exception as e:
                        st.error(f"Rebalancing analysis failed: {str(e)}")
        else:
            st.info("Please select at least 2 assets for rebalancing analysis.")
    
    with tab3:
        st.markdown("### Transaction Cost Analysis")
        st.markdown("Evaluate the impact of trading costs on portfolio performance.")
        
        # Transaction cost configuration
        col1, col2 = st.columns(2)
        
        with col1:
            tc_spread = st.slider(
                "Bid-Ask Spread (%):",
                0.0, 1.0, 0.1,
                step=0.05,
                key="tc_spread",
                help="Typical bid-ask spread cost"
            )
            
            tc_commission = st.number_input(
                "Commission per Trade ($):",
                min_value=0.0,
                value=1.0,
                step=0.5,
                key="tc_commission"
            )
        
        with col2:
            tc_impact = st.slider(
                "Market Impact (%):",
                0.0, 0.5, 0.05,
                step=0.01,
                key="tc_impact",
                help="Price impact from trading"
            )
            
            tc_slippage = st.slider(
                "Slippage (%):",
                0.0, 0.2, 0.02,
                step=0.01,
                key="tc_slippage",
                help="Execution slippage"
            )
        
        # Portfolio for cost analysis
        tc_assets = st.multiselect(
            "Select Assets for Cost Analysis:",
            summary['tickers'],
            default=summary['tickers'][:3],
            key="tc_assets"
        )
        
        if len(tc_assets) >= 2:
            if st.button("Analyze Transaction Costs", key="analyze_costs"):
                with st.spinner("Calculating transaction cost impact..."):
                    try:
                        # Get portfolio data
                        tc_data = {}
                        for asset in tc_assets:
                            asset_prices = get_ticker_data(data, asset, 'close')
                            tc_data[asset] = asset_prices.tail(252)  # 1 year of data
                        
                        tc_df = pd.DataFrame(tc_data).dropna()
                        returns_df = tc_df.pct_change().dropna()
                        
                        # Calculate total transaction cost
                        total_cost_pct = tc_spread + tc_impact + tc_slippage
                        
                        # Simulate different rebalancing frequencies with costs
                        frequencies = ['1M', '3M', '6M', '12M']
                        cost_results = {}
                        
                        for freq in frequencies:
                            # Calculate rebalancing dates
                            rebal_dates = pd.date_range(
                                start=tc_df.index[0],
                                end=tc_df.index[-1],
                                freq=freq
                            )
                            
                            # Calculate costs
                            num_rebalances = len(rebal_dates)
                            trades_per_rebalance = len(tc_assets) * 2  # Buy and sell
                            total_trades = num_rebalances * trades_per_rebalance
                            
                            # Cost breakdown
                            spread_cost = num_rebalances * tc_spread / 100 * len(tc_assets)
                            commission_cost = total_trades * tc_commission / 100000  # As percentage of $100k portfolio
                            impact_cost = num_rebalances * tc_impact / 100 * len(tc_assets)
                            slippage_cost = num_rebalances * tc_slippage / 100 * len(tc_assets)
                            
                            total_cost = spread_cost + commission_cost + impact_cost + slippage_cost
                            
                            # Portfolio performance without costs
                            equal_weights = np.array([1/len(tc_assets)] * len(tc_assets))
                            portfolio_returns = (returns_df * equal_weights).sum(axis=1)
                            gross_return = (1 + portfolio_returns).cumprod().iloc[-1] - 1
                            
                            # Net return after costs
                            net_return = gross_return - total_cost
                            
                            cost_results[freq] = {
                                'Gross Return': gross_return,
                                'Total Cost': total_cost,
                                'Net Return': net_return,
                                'Spread Cost': spread_cost,
                                'Commission Cost': commission_cost,
                                'Impact Cost': impact_cost,
                                'Slippage Cost': slippage_cost,
                                'Rebalances': num_rebalances,
                                'Total Trades': total_trades
                            }
                        
                        # Display results
                        st.markdown("#### Transaction Cost Impact by Frequency")
                        
                        freq_labels = {'1M': 'Monthly', '3M': 'Quarterly', '6M': 'Semi-Annual', '12M': 'Annual'}
                        
                        cost_df = pd.DataFrame(cost_results).T
                        cost_df.index = [freq_labels[freq] for freq in cost_df.index]
                        
                        display_cost_df = pd.DataFrame({
                            'Frequency': cost_df.index,
                            'Gross Return': [f"{r:.2%}" for r in cost_df['Gross Return']],
                            'Total Cost': [f"{r:.2%}" for r in cost_df['Total Cost']],
                            'Net Return': [f"{r:.2%}" for r in cost_df['Net Return']],
                            'Rebalances': [int(r) for r in cost_df['Rebalances']],
                            'Total Trades': [int(r) for r in cost_df['Total Trades']]
                        })
                        
                        st.dataframe(display_cost_df, hide_index=True)
                        
                        # Cost breakdown chart
                        st.markdown("#### Cost Breakdown by Component")
                        
                        cost_components = ['Spread Cost', 'Commission Cost', 'Impact Cost', 'Slippage Cost']
                        
                        fig_cost = go.Figure()
                        
                        for freq in cost_df.index:
                            costs = [cost_df.loc[freq, comp] * 100 for comp in cost_components]  # Convert to percentage
                            
                            fig_cost.add_trace(go.Bar(
                                name=freq,
                                x=cost_components,
                                y=costs,
                                text=[f"{c:.2f}%" for c in costs],
                                textposition='auto'
                            ))
                        
                        fig_cost.update_layout(
                            title="Transaction Cost Breakdown by Frequency",
                            xaxis_title="Cost Component",
                            yaxis_title="Cost (%)",
                            barmode='group',
                            height=400
                        )
                        
                        st.plotly_chart(fig_cost, width='stretch')
                        
                        # Net return comparison
                        st.markdown("#### Net Return After Costs")
                        
                        fig_net = go.Figure()
                        
                        frequencies_list = list(cost_df.index)
                        net_returns = [cost_df.loc[freq, 'Net Return'] * 100 for freq in frequencies_list]
                        
                        fig_net.add_trace(go.Bar(
                            x=frequencies_list,
                            y=net_returns,
                            text=[f"{nr:.2f}%" for nr in net_returns],
                            textposition='auto',
                            marker_color='lightblue'
                        ))
                        
                        fig_net.update_layout(
                            title="Net Returns by Rebalancing Frequency",
                            xaxis_title="Rebalancing Frequency",
                            yaxis_title="Net Return (%)",
                            height=400
                        )
                        
                        st.plotly_chart(fig_net, width='stretch')
                        
                        # Best frequency after costs
                        best_net_freq = cost_df['Net Return'].idxmax()
                        best_net_return = cost_df.loc[best_net_freq, 'Net Return']
                        
                        st.success(f"💰 Best Frequency After Costs: **{best_net_freq}** (Net Return: {best_net_return:.2%})")
                        
                        # Cost insights
                        total_cost_range = cost_df['Total Cost'].max() - cost_df['Total Cost'].min()
                        if total_cost_range > 0.05:  # 5%
                            st.warning("⚠️ High variation in costs across frequencies. Consider less frequent rebalancing.")
                        else:
                            st.info("ℹ️ Transaction costs are relatively stable across frequencies.")
                    
                    except Exception as e:
                        st.error(f"Transaction cost analysis failed: {str(e)}")
        else:
            st.info("Please select at least 2 assets for transaction cost analysis.")
    
    with tab4:
        st.markdown("### Performance Analytics")
        st.markdown("Advanced performance metrics and risk analysis for backtested portfolios.")
        
        # Performance analytics configuration
        pa_strategy = st.selectbox(
            "Select Strategy for Detailed Analysis:",
            ['Equal Weight', 'Risk Parity', 'Momentum', 'Mean Reversion'],
            key="pa_strategy"
        )
        
        pa_assets = st.multiselect(
            "Select Assets:",
            summary['tickers'],
            default=summary['tickers'][:4],
            key="pa_assets"
        )
        
        if len(pa_assets) >= 2:
            if st.button("Generate Performance Analytics", key="generate_analytics"):
                with st.spinner("Generating detailed performance analytics..."):
                    try:
                        # Get data for analysis (2 years)
                        start_date = datetime.now() - timedelta(days=730)
                        
                        pa_data = {}
                        for asset in pa_assets:
                            asset_prices = get_ticker_data(data, asset, 'close')
                            mask = asset_prices.index >= pd.Timestamp(start_date)
                            pa_data[asset] = asset_prices[mask]
                        
                        pa_df = pd.DataFrame(pa_data).dropna()
                        returns_df = pa_df.pct_change().dropna()
                        
                        # Simple strategy implementation for analytics
                        if pa_strategy == 'Equal Weight':
                            weights = np.array([1/len(pa_assets)] * len(pa_assets))
                        elif pa_strategy == 'Risk Parity':
                            volatilities = returns_df.std()
                            inv_vol = 1 / volatilities
                            weights = (inv_vol / inv_vol.sum()).values
                        else:
                            weights = np.array([1/len(pa_assets)] * len(pa_assets))  # Simplified for demo
                        
                        portfolio_returns = (returns_df * weights).sum(axis=1)
                        
                        # Advanced performance metrics
                        st.markdown("#### Advanced Performance Metrics")
                        
                        # Calculate comprehensive metrics
                        annual_return = portfolio_returns.mean() * 252
                        annual_vol = portfolio_returns.std() * np.sqrt(252)
                        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
                        
                        # Sortino ratio (downside deviation)
                        downside_returns = portfolio_returns[portfolio_returns < 0]
                        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
                        sortino_ratio = annual_return / downside_std if downside_std > 0 else 0
                        
                        # Calmar ratio (return / max drawdown)
                        cumulative = (1 + portfolio_returns).cumprod()
                        running_max = cumulative.cummax()
                        drawdown = (cumulative - running_max) / running_max
                        max_drawdown = abs(drawdown.min())
                        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
                        
                        # Value at Risk (VaR)
                        var_95 = np.percentile(portfolio_returns, 5)
                        var_99 = np.percentile(portfolio_returns, 1)
                        
                        # Expected Shortfall (Conditional VaR)
                        es_95 = portfolio_returns[portfolio_returns <= var_95].mean()
                        es_99 = portfolio_returns[portfolio_returns <= var_99].mean()
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")
                            st.metric("Calmar Ratio", f"{calmar_ratio:.3f}")
                        
                        with col2:
                            st.metric("Sortino Ratio", f"{sortino_ratio:.3f}")
                            st.metric("Max Drawdown", f"{max_drawdown:.2%}")
                        
                        with col3:
                            st.metric("VaR (95%)", f"{var_95:.2%}")
                            st.metric("VaR (99%)", f"{var_99:.2%}")
                        
                        with col4:
                            st.metric("ES (95%)", f"{es_95:.2%}")
                            st.metric("ES (99%)", f"{es_99:.2%}")
                        
                        # Rolling metrics analysis
                        st.markdown("#### Rolling Performance Metrics")
                        
                        # Calculate rolling Sharpe ratio
                        rolling_window = 60  # 60 days
                        rolling_returns = portfolio_returns.rolling(window=rolling_window).mean() * 252
                        rolling_vol = portfolio_returns.rolling(window=rolling_window).std() * np.sqrt(252)
                        rolling_sharpe = rolling_returns / rolling_vol
                        
                        fig_rolling = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=('Rolling Returns', 'Rolling Volatility', 'Rolling Sharpe Ratio', 'Drawdown'),
                            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                   [{"secondary_y": False}, {"secondary_y": False}]]
                        )
                        
                        # Rolling returns
                        fig_rolling.add_trace(
                            go.Scatter(x=rolling_returns.index, y=rolling_returns * 100, name="Rolling Return (%)", line=dict(color='blue')),
                            row=1, col=1
                        )
                        
                        # Rolling volatility
                        fig_rolling.add_trace(
                            go.Scatter(x=rolling_vol.index, y=rolling_vol * 100, name="Rolling Volatility (%)", line=dict(color='red')),
                            row=1, col=2
                        )
                        
                        # Rolling Sharpe
                        fig_rolling.add_trace(
                            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, name="Rolling Sharpe", line=dict(color='green')),
                            row=2, col=1
                        )
                        
                        # Drawdown
                        fig_rolling.add_trace(
                            go.Scatter(x=drawdown.index, y=drawdown * 100, name="Drawdown (%)", fill='tozeroy', line=dict(color='orange')),
                            row=2, col=2
                        )
                        
                        fig_rolling.update_layout(height=600, showlegend=False, title_text="Rolling Performance Analysis")
                        st.plotly_chart(fig_rolling, width='stretch')
                        
                        # Returns distribution analysis
                        st.markdown("#### Returns Distribution Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Histogram of returns
                            fig_hist = px.histogram(
                                x=portfolio_returns * 100,
                                nbins=50,
                                title="Distribution of Daily Returns",
                                labels={'x': 'Daily Return (%)', 'y': 'Frequency'}
                            )
                            
                            # Add normal distribution overlay
                            x_norm = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 100) * 100
                            y_norm = ((1 / (np.sqrt(2 * np.pi) * portfolio_returns.std() * 100)) * 
                                     np.exp(-0.5 * ((x_norm - portfolio_returns.mean() * 100) / (portfolio_returns.std() * 100)) ** 2))
                            
                            # Scale to match histogram
                            y_norm = y_norm * len(portfolio_returns) * (portfolio_returns.max() - portfolio_returns.min()) * 100 / 50
                            
                            fig_hist.add_trace(
                                go.Scatter(x=x_norm, y=y_norm, mode='lines', name='Normal Distribution', line=dict(color='red', dash='dash'))
                            )
                            
                            st.plotly_chart(fig_hist, width='stretch')
                        
                        with col2:
                            # Q-Q plot approximation (scatter plot)
                            from scipy import stats
                            
                            sorted_returns = np.sort(portfolio_returns)
                            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_returns)))
                            
                            fig_qq = go.Figure()
                            fig_qq.add_trace(go.Scatter(
                                x=theoretical_quantiles,
                                y=sorted_returns,
                                mode='markers',
                                name='Sample Quantiles',
                                marker=dict(color='blue', size=4)
                            ))
                            
                            # Add 45-degree line
                            line_min = min(theoretical_quantiles.min(), sorted_returns.min())
                            line_max = max(theoretical_quantiles.max(), sorted_returns.max())
                            fig_qq.add_trace(go.Scatter(
                                x=[line_min, line_max],
                                y=[line_min, line_max],
                                mode='lines',
                                name='Normal Line',
                                line=dict(color='red', dash='dash')
                            ))
                            
                            fig_qq.update_layout(
                                title="Q-Q Plot vs Normal Distribution",
                                xaxis_title="Theoretical Quantiles",
                                yaxis_title="Sample Quantiles"
                            )
                            
                            st.plotly_chart(fig_qq, width='stretch')
                        
                        # Statistical tests
                        st.markdown("#### Statistical Tests")
                        
                        # Normality test (Jarque-Bera)
                        from scipy.stats import jarque_bera, skew, kurtosis
                        
                        jb_stat, jb_pvalue = jarque_bera(portfolio_returns)
                        returns_skew = skew(portfolio_returns)
                        returns_kurtosis = kurtosis(portfolio_returns)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Skewness", f"{returns_skew:.3f}")
                        with col2:
                            st.metric("Kurtosis", f"{returns_kurtosis:.3f}")
                        with col3:
                            st.metric("JB Statistic", f"{jb_stat:.2f}")
                        with col4:
                            normality_status = "Normal" if jb_pvalue > 0.05 else "Non-Normal"
                            st.metric("Normality", normality_status)
                        
                        if jb_pvalue < 0.05:
                            st.warning("⚠️ Returns are not normally distributed. Consider using robust risk measures.")
                        else:
                            st.success("✅ Returns appear to be normally distributed.")
                        
                        # Performance summary
                        st.markdown("#### Performance Summary")
                        
                        if sharpe_ratio > 1.0:
                            performance_grade = "Excellent"
                            grade_color = "🟢"
                        elif sharpe_ratio > 0.5:
                            performance_grade = "Good"
                            grade_color = "🟡"
                        else:
                            performance_grade = "Needs Improvement"
                            grade_color = "🔴"
                        
                        st.info(f"""
                        **Strategy Performance Grade:** {grade_color} {performance_grade}
                        
                        **Key Insights:**
                        - Risk-adjusted return (Sharpe): {sharpe_ratio:.3f}
                        - Downside protection (Sortino): {sortino_ratio:.3f}
                        - Drawdown management (Calmar): {calmar_ratio:.3f}
                        - Tail risk (VaR 95%): {var_95:.2%}
                        """)
                    
                    except Exception as e:
                        st.error(f"Performance analytics failed: {str(e)}")
        else:
            st.info("Please select at least 2 assets for performance analytics.")
    
    # Educational content
    st.markdown("---")
    st.markdown("### 📚 Backtesting Best Practices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Key Considerations:**
        - **Survivorship Bias:** Use complete historical data
        - **Look-ahead Bias:** Only use past information
        - **Transaction Costs:** Include realistic trading costs
        - **Market Regime Changes:** Test across different periods
        """)
    
    with col2:
        st.markdown("""
        **Performance Metrics:**
        - **Sharpe Ratio:** Risk-adjusted returns
        - **Sortino Ratio:** Downside risk focus
        - **Calmar Ratio:** Return vs max drawdown
        - **VaR/ES:** Tail risk measures
        """)
    
    st.info("""
    💡 **Tip:** Past performance doesn't guarantee future results. 
    Use backtesting to understand strategy behavior, not to predict future returns.
    """)


def page_risk_management():
    """Advanced risk management page."""
    st.markdown('<div class="main-header">Advanced Risk Management</div>', unsafe_allow_html=True)
    
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please load data first from the Home page.")
        return
    
    data = st.session_state.data
    summary = get_data_summary(data)
    
    st.markdown("""
    Comprehensive risk analysis including Value at Risk, Expected Shortfall, stress testing, 
    and scenario analysis for portfolio risk management.
    """)
    
    # Create tabs for different risk management approaches
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Risk Metrics", "⚠️ VaR Analysis", "🎯 Stress Testing", "📈 Scenario Analysis", "🛡️ Risk Controls"])
    
    with tab1:
        st.markdown("### Portfolio Risk Metrics")
        st.markdown("Calculate comprehensive risk metrics for individual assets and portfolios.")
        
        # Asset selection for risk analysis
        col1, col2 = st.columns(2)
        
        with col1:
            risk_assets = st.multiselect(
                "Select Assets for Risk Analysis:",
                summary['tickers'],
                default=summary['tickers'][:4],
                key="risk_assets"
            )
            
            risk_timeframe = st.selectbox(
                "Analysis Timeframe:",
                ['1 Month', '3 Months', '6 Months', '1 Year', '2 Years'],
                index=3,
                key="risk_timeframe"
            )
        
        with col2:
            risk_confidence = st.selectbox(
                "Confidence Level:",
                ['90%', '95%', '99%'],
                index=1,
                key="risk_confidence"
            )
            
            portfolio_value = st.number_input(
                "Portfolio Value ($):",
                min_value=1000,
                value=1000000,
                step=10000,
                key="portfolio_value"
            )
        
        if len(risk_assets) >= 1:
            if st.button("Calculate Risk Metrics", type="primary", key="calc_risk_metrics"):
                with st.spinner("Calculating comprehensive risk metrics..."):
                    try:
                        # Get timeframe in days
                        timeframe_map = {'1 Month': 30, '3 Months': 90, '6 Months': 180, '1 Year': 365, '2 Years': 730}
                        days_back = timeframe_map[risk_timeframe]
                        
                        start_date = datetime.now() - timedelta(days=days_back)
                        confidence_level = float(risk_confidence.strip('%')) / 100
                        
                        # Get data for all selected assets
                        risk_data = {}
                        for asset in risk_assets:
                            asset_prices = get_ticker_data(data, asset, 'close')
                            mask = asset_prices.index >= pd.Timestamp(start_date)
                            risk_data[asset] = asset_prices[mask]
                        
                        risk_df = pd.DataFrame(risk_data).dropna()
                        returns_df = risk_df.pct_change().dropna()
                        
                        # Calculate individual asset risk metrics
                        st.markdown("#### Individual Asset Risk Metrics")
                        
                        asset_metrics = {}
                        for asset in risk_assets:
                            asset_returns = returns_df[asset]
                            
                            # Basic statistics
                            mean_return = asset_returns.mean() * 252
                            volatility = asset_returns.std() * np.sqrt(252)
                            
                            # Value at Risk
                            var_1d = np.percentile(asset_returns, (1 - confidence_level) * 100)
                            var_annual = var_1d * np.sqrt(252)
                            
                            # Expected Shortfall (Conditional VaR)
                            es_1d = asset_returns[asset_returns <= var_1d].mean()
                            es_annual = es_1d * np.sqrt(252)
                            
                            # Maximum drawdown
                            cumulative = (1 + asset_returns).cumprod()
                            running_max = cumulative.cummax()
                            drawdowns = (cumulative - running_max) / running_max
                            max_drawdown = drawdowns.min()
                            
                            # Skewness and kurtosis
                            returns_skew = skew(asset_returns)
                            returns_kurtosis = kurtosis(asset_returns)
                            
                            # Beta (using first asset as market proxy)
                            if asset != risk_assets[0]:
                                market_returns = returns_df[risk_assets[0]]
                                covariance = np.cov(asset_returns, market_returns)[0, 1]
                                market_variance = np.var(market_returns)
                                beta = covariance / market_variance if market_variance > 0 else 0
                            else:
                                beta = 1.0
                            
                            asset_metrics[asset] = {
                                'Mean Return': mean_return,
                                'Volatility': volatility,
                                'VaR (1d)': var_1d,
                                'VaR (Annual)': var_annual,
                                'ES (1d)': es_1d,
                                'ES (Annual)': es_annual,
                                'Max Drawdown': max_drawdown,
                                'Skewness': returns_skew,
                                'Kurtosis': returns_kurtosis,
                                'Beta': beta
                            }
                        
                        # Display individual metrics
                        metrics_df = pd.DataFrame(asset_metrics).T
                        
                        display_metrics = pd.DataFrame({
                            'Asset': metrics_df.index,
                            'Ann. Return': [f"{r:.2%}" for r in metrics_df['Mean Return']],
                            'Volatility': [f"{r:.2%}" for r in metrics_df['Volatility']],
                            f'VaR {risk_confidence}': [f"{r:.2%}" for r in metrics_df['VaR (Annual)']],
                            f'ES {risk_confidence}': [f"{r:.2%}" for r in metrics_df['ES (Annual)']],
                            'Max DD': [f"{r:.2%}" for r in metrics_df['Max Drawdown']],
                            'Skewness': [f"{r:.3f}" for r in metrics_df['Skewness']],
                            'Kurtosis': [f"{r:.3f}" for r in metrics_df['Kurtosis']],
                            'Beta': [f"{r:.3f}" for r in metrics_df['Beta']]
                        })
                        
                        st.dataframe(display_metrics, hide_index=True)
                        
                        # Risk ranking
                        st.markdown("#### Risk Ranking")
                        
                        # Calculate risk score (lower is better)
                        risk_scores = {}
                        for asset in risk_assets:
                            # Combine volatility, VaR, and max drawdown for risk score
                            vol_rank = metrics_df.loc[asset, 'Volatility']
                            var_rank = abs(metrics_df.loc[asset, 'VaR (Annual)'])
                            dd_rank = abs(metrics_df.loc[asset, 'Max Drawdown'])
                            
                            risk_score = (vol_rank + var_rank + dd_rank) / 3
                            risk_scores[asset] = risk_score
                        
                        # Sort by risk score
                        sorted_risk = sorted(risk_scores.items(), key=lambda x: x[1])
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Lowest Risk Assets:**")
                            for i, (asset, score) in enumerate(sorted_risk[:3]):
                                st.write(f"{i+1}. {asset} (Risk Score: {score:.3f})")
                        
                        with col2:
                            st.markdown("**Highest Risk Assets:**")
                            for i, (asset, score) in enumerate(sorted_risk[-3:]):
                                st.write(f"{i+1}. {asset} (Risk Score: {score:.3f})")
                        
                        # Risk visualization
                        st.markdown("#### Risk-Return Scatter Plot")
                        
                        fig_risk_return = go.Figure()
                        
                        fig_risk_return.add_trace(go.Scatter(
                            x=[metrics_df.loc[asset, 'Volatility'] * 100 for asset in risk_assets],
                            y=[metrics_df.loc[asset, 'Mean Return'] * 100 for asset in risk_assets],
                            mode='markers+text',
                            text=risk_assets,
                            textposition='top center',
                            marker=dict(
                                size=[abs(metrics_df.loc[asset, 'Max Drawdown']) * 1000 for asset in risk_assets],
                                color=[metrics_df.loc[asset, 'Beta'] for asset in risk_assets],
                                colorscale='RdYlBu',
                                showscale=True,
                                colorbar=dict(title="Beta")
                            ),
                            name="Assets"
                        ))
                        
                        fig_risk_return.update_layout(
                            title="Risk-Return Analysis (Bubble size = Max Drawdown)",
                            xaxis_title="Volatility (%)",
                            yaxis_title="Expected Return (%)",
                            height=500
                        )
                        
                        st.plotly_chart(fig_risk_return, width='stretch')
                        
                        # Portfolio-level risk (if multiple assets)
                        if len(risk_assets) > 1:
                            st.markdown("#### Portfolio Risk Analysis")
                            
                            # Equal weight portfolio
                            equal_weights = np.array([1/len(risk_assets)] * len(risk_assets))
                            portfolio_returns = (returns_df * equal_weights).sum(axis=1)
                            
                            # Portfolio metrics
                            port_mean_return = portfolio_returns.mean() * 252
                            port_volatility = portfolio_returns.std() * np.sqrt(252)
                            port_var_1d = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
                            port_var_annual = port_var_1d * np.sqrt(252)
                            port_es_1d = portfolio_returns[portfolio_returns <= port_var_1d].mean()
                            port_es_annual = port_es_1d * np.sqrt(252)
                            
                            # Portfolio maximum drawdown
                            port_cumulative = (1 + portfolio_returns).cumprod()
                            port_running_max = port_cumulative.cummax()
                            port_drawdowns = (port_cumulative - port_running_max) / port_running_max
                            port_max_drawdown = port_drawdowns.min()
                            
                            # Display portfolio metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Portfolio Return", f"{port_mean_return:.2%}")
                            with col2:
                                st.metric("Portfolio Volatility", f"{port_volatility:.2%}")
                            with col3:
                                st.metric(f"Portfolio VaR {risk_confidence}", f"{port_var_annual:.2%}")
                            with col4:
                                st.metric("Portfolio Max DD", f"{port_max_drawdown:.2%}")
                            
                            # Diversification benefit
                            weighted_vol = np.sqrt(np.dot(equal_weights**2, [metrics_df.loc[asset, 'Volatility']**2 for asset in risk_assets]))
                            diversification_ratio = weighted_vol / port_volatility
                            
                            st.info(f"🎯 **Diversification Benefit:** {diversification_ratio:.2f}x reduction in risk through diversification")
                    
                    except Exception as e:
                        st.error(f"Risk metrics calculation failed: {str(e)}")
        else:
            st.info("Please select at least 1 asset for risk analysis.")
    
    with tab2:
        st.markdown("### Value at Risk (VaR) Analysis")
        st.markdown("Detailed VaR analysis using different methodologies.")
        
        # VaR configuration
        col1, col2 = st.columns(2)
        
        with col1:
            var_assets = st.multiselect(
                "Select Assets:",
                summary['tickers'],
                default=summary['tickers'][:3],
                key="var_assets"
            )
            
            var_method = st.selectbox(
                "VaR Methodology:",
                ['Historical Simulation', 'Parametric (Normal)', 'Monte Carlo'],
                key="var_method"
            )
        
        with col2:
            var_confidence_levels = st.multiselect(
                "Confidence Levels:",
                ['90%', '95%', '99%', '99.9%'],
                default=['95%', '99%'],
                key="var_confidence_levels"
            )
            
            var_horizon = st.selectbox(
                "Time Horizon:",
                ['1 Day', '1 Week', '1 Month'],
                key="var_horizon"
            )
        
        if len(var_assets) >= 1 and len(var_confidence_levels) >= 1:
            if st.button("Calculate VaR Analysis", key="calc_var"):
                with st.spinner("Performing VaR analysis..."):
                    try:
                        # Get data
                        var_data = {}
                        for asset in var_assets:
                            asset_prices = get_ticker_data(data, asset, 'close')
                            var_data[asset] = asset_prices.tail(500)  # Use last 500 days
                        
                        var_df = pd.DataFrame(var_data).dropna()
                        returns_df = var_df.pct_change().dropna()
                        
                        # Time horizon adjustment
                        horizon_map = {'1 Day': 1, '1 Week': 5, '1 Month': 21}
                        horizon_days = horizon_map[var_horizon]
                        
                        # Portfolio returns (equal weight)
                        equal_weights = np.array([1/len(var_assets)] * len(var_assets))
                        portfolio_returns = (returns_df * equal_weights).sum(axis=1)
                        
                        # Calculate VaR using different methods
                        var_results = {}
                        
                        for conf_str in var_confidence_levels:
                            confidence = float(conf_str.strip('%')) / 100
                            alpha = 1 - confidence
                            
                            if var_method == 'Historical Simulation':
                                # Historical simulation VaR
                                var_1d = np.percentile(portfolio_returns, alpha * 100)
                                var_horizon_adj = var_1d * np.sqrt(horizon_days)
                                
                            elif var_method == 'Parametric (Normal)':
                                # Parametric VaR assuming normal distribution
                                mean_return = portfolio_returns.mean()
                                vol = portfolio_returns.std()
                                var_1d = mean_return + vol * stats.norm.ppf(alpha)
                                var_horizon_adj = mean_return * horizon_days + vol * np.sqrt(horizon_days) * stats.norm.ppf(alpha)
                                
                            else:  # Monte Carlo
                                # Monte Carlo simulation
                                np.random.seed(42)
                                mean_return = portfolio_returns.mean()
                                vol = portfolio_returns.std()
                                
                                # Generate random scenarios
                                n_simulations = 10000
                                random_returns = np.random.normal(mean_return, vol, n_simulations)
                                
                                # Adjust for horizon
                                if horizon_days > 1:
                                    simulated_returns = []
                                    for _ in range(n_simulations):
                                        path_returns = np.random.normal(mean_return, vol, horizon_days)
                                        total_return = np.prod(1 + path_returns) - 1
                                        simulated_returns.append(total_return)
                                    random_returns = np.array(simulated_returns)
                                else:
                                    random_returns = random_returns
                                
                                var_horizon_adj = np.percentile(random_returns, alpha * 100)
                                var_1d = var_horizon_adj if horizon_days == 1 else var_horizon_adj / np.sqrt(horizon_days)
                            
                            # Expected Shortfall
                            if var_method == 'Historical Simulation':
                                es_1d = portfolio_returns[portfolio_returns <= var_1d].mean()
                                es_horizon_adj = es_1d * np.sqrt(horizon_days)
                            elif var_method == 'Monte Carlo':
                                es_horizon_adj = random_returns[random_returns <= var_horizon_adj].mean()
                                es_1d = es_horizon_adj if horizon_days == 1 else es_horizon_adj / np.sqrt(horizon_days)
                            else:  # Parametric
                                mean_return = portfolio_returns.mean()
                                vol = portfolio_returns.std()
                                es_1d = mean_return - vol * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
                                es_horizon_adj = mean_return * horizon_days - vol * np.sqrt(horizon_days) * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
                            
                            var_results[conf_str] = {
                                'VaR (1 Day)': var_1d,
                                f'VaR ({var_horizon})': var_horizon_adj,
                                'ES (1 Day)': es_1d,
                                f'ES ({var_horizon})': es_horizon_adj
                            }
                        
                        # Display VaR results
                        st.markdown(f"#### VaR Results - {var_method}")
                        
                        var_display_df = pd.DataFrame(var_results).T
                        var_display_df = var_display_df.applymap(lambda x: f"{x:.2%}")
                        
                        st.dataframe(var_display_df)
                        
                        # VaR at different confidence levels chart
                        st.markdown("#### VaR by Confidence Level")
                        
                        confidence_values = [float(conf.strip('%')) for conf in var_confidence_levels]
                        var_values = [var_results[conf][f'VaR ({var_horizon})'] * 100 for conf in var_confidence_levels]
                        es_values = [var_results[conf][f'ES ({var_horizon})'] * 100 for conf in var_confidence_levels]
                        
                        fig_var = go.Figure()
                        
                        fig_var.add_trace(go.Scatter(
                            x=confidence_values,
                            y=var_values,
                            mode='lines+markers',
                            name='VaR',
                            line=dict(color='red', width=3)
                        ))
                        
                        fig_var.add_trace(go.Scatter(
                            x=confidence_values,
                            y=es_values,
                            mode='lines+markers',
                            name='Expected Shortfall',
                            line=dict(color='darkred', width=3, dash='dash')
                        ))
                        
                        fig_var.update_layout(
                            title=f"VaR and ES by Confidence Level ({var_horizon})",
                            xaxis_title="Confidence Level (%)",
                            yaxis_title="Loss (%)",
                            height=400
                        )
                        
                        st.plotly_chart(fig_var, width='stretch')
                        
                        # Backtesting (if using historical simulation)
                        if var_method == 'Historical Simulation':
                            st.markdown("#### VaR Backtesting")
                            
                            # Use 95% VaR for backtesting
                            var_95 = var_results['95%']['VaR (1 Day)']
                            
                            # Count violations
                            violations = (portfolio_returns < var_95).sum()
                            total_observations = len(portfolio_returns)
                            violation_rate = violations / total_observations
                            expected_violations = 0.05  # 5% for 95% VaR
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Actual Violations", f"{violations}")
                            with col2:
                                st.metric("Violation Rate", f"{violation_rate:.2%}")
                            with col3:
                                expected_count = int(total_observations * expected_violations)
                                st.metric("Expected Violations", f"{expected_count}")
                            
                            # Traffic light system for VaR performance
                            if abs(violation_rate - expected_violations) < 0.01:  # Within 1%
                                st.success("✅ VaR model performance: Excellent")
                            elif abs(violation_rate - expected_violations) < 0.02:  # Within 2%
                                st.warning("⚠️ VaR model performance: Acceptable")
                            else:
                                st.error("❌ VaR model performance: Poor - Consider model recalibration")
                    
                    except Exception as e:
                        st.error(f"VaR analysis failed: {str(e)}")
        else:
            st.info("Please select assets and confidence levels for VaR analysis.")
    
    with tab3:
        st.markdown("### Stress Testing")
        st.markdown("Test portfolio performance under extreme market conditions.")
        
        # Stress testing configuration
        stress_assets = st.multiselect(
            "Select Assets for Stress Testing:",
            summary['tickers'],
            default=summary['tickers'][:3],
            key="stress_assets"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            stress_type = st.selectbox(
                "Stress Test Type:",
                ['Historical Scenarios', 'Custom Shocks', 'Correlation Breakdown'],
                key="stress_type"
            )
        
        with col2:
            stress_portfolio_value = st.number_input(
                "Portfolio Value ($):",
                min_value=1000,
                value=1000000,
                step=10000,
                key="stress_portfolio_value"
            )
        
        if len(stress_assets) >= 1:
            if stress_type == 'Historical Scenarios':
                st.markdown("#### Historical Crisis Scenarios")
                
                # Define historical crisis periods
                crisis_periods = {
                    'COVID-19 Crash (Feb-Mar 2020)': ('2020-02-01', '2020-03-31'),
                    'Global Financial Crisis (2008)': ('2008-09-01', '2008-12-31'),
                    'Dot-com Crash (2000-2002)': ('2000-03-01', '2002-10-31'),
                    'Black Monday (1987)': ('1987-10-01', '1987-11-30')
                }
                
                selected_crisis = st.selectbox(
                    "Select Historical Crisis:",
                    list(crisis_periods.keys()),
                    key="selected_crisis"
                )
                
                if st.button("Run Historical Stress Test", key="run_historical_stress"):
                    with st.spinner("Running historical stress test..."):
                        try:
                            start_crisis, end_crisis = crisis_periods[selected_crisis]
                            
                            # Get data for crisis period
                            crisis_data = {}
                            for asset in stress_assets:
                                asset_prices = get_ticker_data(data, asset, 'close')
                                mask = (asset_prices.index >= pd.Timestamp(start_crisis)) & (asset_prices.index <= pd.Timestamp(end_crisis))
                                crisis_data[asset] = asset_prices[mask]
                            
                            crisis_df = pd.DataFrame(crisis_data).dropna()
                            
                            if len(crisis_df) > 0:
                                crisis_returns = crisis_df.pct_change().dropna()
                                
                                # Calculate total returns during crisis
                                total_returns = (crisis_df.iloc[-1] / crisis_df.iloc[0] - 1)
                                
                                # Portfolio impact (equal weights)
                                equal_weights = np.array([1/len(stress_assets)] * len(stress_assets))
                                portfolio_impact = np.dot(equal_weights, total_returns)
                                portfolio_loss = stress_portfolio_value * portfolio_impact
                                
                                st.markdown(f"#### {selected_crisis} Impact")
                                
                                # Individual asset impacts
                                impact_df = pd.DataFrame({
                                    'Asset': stress_assets,
                                    'Total Return': [f"{r:.2%}" for r in total_returns],
                                    'Dollar Impact': [f"${stress_portfolio_value * w * r:,.0f}" for w, r in zip(equal_weights, total_returns)]
                                })
                                
                                st.dataframe(impact_df, hide_index=True)
                                
                                # Portfolio summary
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Portfolio Impact", f"{portfolio_impact:.2%}")
                                with col2:
                                    st.metric("Dollar Loss", f"${portfolio_loss:,.0f}")
                                
                                # Visualization
                                fig_stress = go.Figure()
                                
                                for asset in stress_assets:
                                    normalized_prices = crisis_df[asset] / crisis_df[asset].iloc[0]
                                    fig_stress.add_trace(go.Scatter(
                                        x=crisis_df.index,
                                        y=normalized_prices,
                                        mode='lines',
                                        name=asset,
                                        line=dict(width=2)
                                    ))
                                
                                fig_stress.update_layout(
                                    title=f"Price Performance During {selected_crisis}",
                                    xaxis_title="Date",
                                    yaxis_title="Normalized Price",
                                    height=400
                                )
                                
                                st.plotly_chart(fig_stress, width='stretch')
                                
                                if portfolio_impact < -0.20:  # More than 20% loss
                                    st.error("❌ Severe portfolio vulnerability to this type of crisis")
                                elif portfolio_impact < -0.10:  # More than 10% loss
                                    st.warning("⚠️ Moderate portfolio vulnerability")
                                else:
                                    st.success("✅ Portfolio shows resilience to this crisis type")
                            else:
                                st.warning("Insufficient data for the selected crisis period.")
                        
                        except Exception as e:
                            st.error(f"Historical stress test failed: {str(e)}")
            
            elif stress_type == 'Custom Shocks':
                st.markdown("#### Custom Shock Scenarios")
                
                # Custom shock inputs
                col1, col2 = st.columns(2)
                
                with col1:
                    market_shock = st.slider(
                        "Market Shock (%):",
                        -50, 0, -20,
                        key="market_shock",
                        help="Apply uniform shock to all assets"
                    )
                
                with col2:
                    correlation_shock = st.slider(
                        "Correlation Increase:",
                        0.0, 1.0, 0.5,
                        key="correlation_shock",
                        help="Increase in asset correlations during stress"
                    )
                
                # Individual asset shocks
                st.markdown("**Individual Asset Shocks:**")
                asset_shocks = {}
                for asset in stress_assets:
                    asset_shocks[asset] = st.slider(
                        f"{asset} Shock (%):",
                        -70, 30, market_shock,
                        key=f"shock_{asset}"
                    )
                
                if st.button("Run Custom Stress Test", key="run_custom_stress"):
                    with st.spinner("Running custom stress test..."):
                        try:
                            # Calculate portfolio impact
                            equal_weights = np.array([1/len(stress_assets)] * len(stress_assets))
                            individual_impacts = np.array([asset_shocks[asset]/100 for asset in stress_assets])
                            
                            # Portfolio impact without correlation adjustment
                            base_portfolio_impact = np.dot(equal_weights, individual_impacts)
                            
                            # Adjust for correlation increase (simplified)
                            correlation_adjustment = correlation_shock * 0.1  # Approximate correlation impact
                            adjusted_portfolio_impact = base_portfolio_impact * (1 + correlation_adjustment)
                            
                            portfolio_loss = stress_portfolio_value * adjusted_portfolio_impact
                            
                            st.markdown("#### Custom Stress Test Results")
                            
                            # Results table
                            stress_results_df = pd.DataFrame({
                                'Asset': stress_assets,
                                'Applied Shock': [f"{asset_shocks[asset]:.1f}%" for asset in stress_assets],
                                'Dollar Impact': [f"${stress_portfolio_value * w * (asset_shocks[asset]/100):,.0f}" 
                                                for w, asset in zip(equal_weights, stress_assets)]
                            })
                            
                            st.dataframe(stress_results_df, hide_index=True)
                            
                            # Portfolio summary
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Base Portfolio Impact", f"{base_portfolio_impact:.2%}")
                            with col2:
                                st.metric("Correlation-Adjusted Impact", f"{adjusted_portfolio_impact:.2%}")
                            with col3:
                                st.metric("Total Dollar Loss", f"${portfolio_loss:,.0f}")
                            
                            # Stress test visualization
                            fig_custom_stress = go.Figure()
                            
                            fig_custom_stress.add_trace(go.Bar(
                                x=stress_assets,
                                y=[asset_shocks[asset] for asset in stress_assets],
                                text=[f"{asset_shocks[asset]:.1f}%" for asset in stress_assets],
                                textposition='auto',
                                marker_color=['red' if shock < 0 else 'green' for shock in [asset_shocks[asset] for asset in stress_assets]]
                            ))
                            
                            fig_custom_stress.update_layout(
                                title="Applied Stress Shocks by Asset",
                                xaxis_title="Asset",
                                yaxis_title="Shock (%)",
                                height=400
                            )
                            
                            st.plotly_chart(fig_custom_stress, width='stretch')
                        
                        except Exception as e:
                            st.error(f"Custom stress test failed: {str(e)}")
            
            else:  # Correlation Breakdown
                st.markdown("#### Correlation Breakdown Analysis")
                st.markdown("Analyze what happens when asset correlations break down during market stress.")
                
                if len(stress_assets) >= 2:
                    if st.button("Analyze Correlation Breakdown", key="run_correlation_breakdown"):
                        with st.spinner("Analyzing correlation breakdown scenarios..."):
                            try:
                                # Get recent data
                                corr_data = {}
                                for asset in stress_assets:
                                    asset_prices = get_ticker_data(data, asset, 'close')
                                    corr_data[asset] = asset_prices.tail(252)  # 1 year
                                
                                corr_df = pd.DataFrame(corr_data).dropna()
                                returns_df = corr_df.pct_change().dropna()
                                
                                # Normal correlation matrix
                                normal_corr = returns_df.corr()
                                
                                # Stress correlation scenarios
                                scenarios = {
                                    'Normal Market': normal_corr,
                                    'Moderate Stress (Corr +20%)': normal_corr * 1.2,
                                    'High Stress (Corr +50%)': normal_corr * 1.5,
                                    'Crisis (All Corr = 0.9)': pd.DataFrame(0.9, index=normal_corr.index, columns=normal_corr.columns)
                                }
                                
                                # Set diagonal to 1 for all scenarios
                                for scenario_name, corr_matrix in scenarios.items():
                                    np.fill_diagonal(corr_matrix.values, 1.0)
                                    # Clip correlations to [-1, 1]
                                    scenarios[scenario_name] = corr_matrix.clip(-1, 1)
                                
                                # Calculate portfolio variance under different scenarios
                                equal_weights = np.array([1/len(stress_assets)] * len(stress_assets))
                                asset_vols = returns_df.std().values * np.sqrt(252)  # Annualized volatilities
                                
                                scenario_results = {}
                                for scenario_name, corr_matrix in scenarios.items():
                                    # Portfolio variance = w'Σw where Σ is covariance matrix
                                    cov_matrix = np.outer(asset_vols, asset_vols) * corr_matrix.values
                                    portfolio_variance = np.dot(equal_weights, np.dot(cov_matrix, equal_weights))
                                    portfolio_vol = np.sqrt(portfolio_variance)
                                    
                                    scenario_results[scenario_name] = {
                                        'Portfolio Volatility': portfolio_vol,
                                        'Average Correlation': (corr_matrix.values.sum() - len(stress_assets)) / (len(stress_assets) * (len(stress_assets) - 1))
                                    }
                                
                                # Display results
                                st.markdown("#### Correlation Breakdown Results")
                                
                                scenario_df = pd.DataFrame(scenario_results).T
                                scenario_display = pd.DataFrame({
                                    'Scenario': scenario_df.index,
                                    'Avg Correlation': [f"{r:.3f}" for r in scenario_df['Average Correlation']],
                                    'Portfolio Volatility': [f"{r:.2%}" for r in scenario_df['Portfolio Volatility']],
                                    'Vol Increase': [f"{((r / scenario_df.iloc[0]['Portfolio Volatility']) - 1):.1%}" 
                                                   for r in scenario_df['Portfolio Volatility']]
                                })
                                
                                st.dataframe(scenario_display, hide_index=True)
                                
                                # Correlation heatmaps
                                st.markdown("#### Correlation Matrix Comparison")
                                
                                fig_corr = make_subplots(
                                    rows=2, cols=2,
                                    subplot_titles=list(scenarios.keys()),
                                    specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
                                           [{"type": "heatmap"}, {"type": "heatmap"}]]
                                )
                                
                                positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
                                for i, (scenario_name, corr_matrix) in enumerate(scenarios.items()):
                                    row, col = positions[i]
                                    
                                    fig_corr.add_trace(
                                        go.Heatmap(
                                            z=corr_matrix.values,
                                            x=corr_matrix.columns,
                                            y=corr_matrix.index,
                                            colorscale='RdBu',
                                            zmid=0,
                                            showscale=(i == 0)
                                        ),
                                        row=row, col=col
                                    )
                                
                                fig_corr.update_layout(height=600, title_text="Correlation Matrix Under Different Stress Scenarios")
                                st.plotly_chart(fig_corr, width='stretch')
                                
                                # Risk assessment
                                vol_increase = (scenario_df.iloc[-1]['Portfolio Volatility'] / scenario_df.iloc[0]['Portfolio Volatility'] - 1) * 100
                                
                                if vol_increase > 50:
                                    st.error(f"❌ High correlation breakdown risk: {vol_increase:.1f}% volatility increase in crisis")
                                elif vol_increase > 25:
                                    st.warning(f"⚠️ Moderate correlation breakdown risk: {vol_increase:.1f}% volatility increase")
                                else:
                                    st.success(f"✅ Low correlation breakdown risk: {vol_increase:.1f}% volatility increase")
                            
                            except Exception as e:
                                st.error(f"Correlation breakdown analysis failed: {str(e)}")
                else:
                    st.info("Please select at least 2 assets for correlation breakdown analysis.")
        else:
            st.info("Please select at least 1 asset for stress testing.")
    
    with tab4:
        st.markdown("### Scenario Analysis")
        st.markdown("Analyze portfolio performance under different economic scenarios.")
        
        # Economic scenario configuration
        scenario_assets = st.multiselect(
            "Select Assets for Scenario Analysis:",
            summary['tickers'],
            default=summary['tickers'][:4],
            key="scenario_assets"
        )
        
        if len(scenario_assets) >= 1:
            # Predefined economic scenarios
            economic_scenarios = {
                'Economic Expansion': {
                    'description': 'Strong GDP growth, low unemployment, rising interest rates',
                    'equity_impact': 0.15,
                    'bond_impact': -0.05,
                    'commodity_impact': 0.10,
                    'currency_impact': 0.05
                },
                'Economic Recession': {
                    'description': 'Negative GDP growth, high unemployment, falling interest rates',
                    'equity_impact': -0.25,
                    'bond_impact': 0.08,
                    'commodity_impact': -0.15,
                    'currency_impact': -0.10
                },
                'High Inflation': {
                    'description': 'Rising prices, potential stagflation, aggressive monetary policy',
                    'equity_impact': -0.10,
                    'bond_impact': -0.12,
                    'commodity_impact': 0.20,
                    'currency_impact': -0.08
                },
                'Deflation': {
                    'description': 'Falling prices, economic stagnation, zero interest rates',
                    'equity_impact': -0.20,
                    'bond_impact': 0.15,
                    'commodity_impact': -0.25,
                    'currency_impact': 0.05
                },
                'Geopolitical Crisis': {
                    'description': 'International tensions, trade disruptions, flight to safety',
                    'equity_impact': -0.18,
                    'bond_impact': 0.06,
                    'commodity_impact': 0.15,
                    'currency_impact': -0.12
                }
            }
            
            selected_scenarios = st.multiselect(
                "Select Economic Scenarios:",
                list(economic_scenarios.keys()),
                default=list(economic_scenarios.keys())[:3],
                key="selected_scenarios"
            )
            
            # Asset classification for scenario impacts
            st.markdown("#### Asset Classification")
            st.markdown("Classify your assets to apply appropriate scenario impacts:")
            
            asset_types = {}
            for asset in scenario_assets:
                asset_types[asset] = st.selectbox(
                    f"{asset} Asset Type:",
                    ['Equity', 'Bond', 'Commodity', 'Currency'],
                    key=f"type_{asset}"
                )
            
            if st.button("Run Scenario Analysis", key="run_scenario_analysis"):
                with st.spinner("Running scenario analysis..."):
                    try:
                        # Calculate current portfolio value and weights
                        equal_weights = np.array([1/len(scenario_assets)] * len(scenario_assets))
                        base_portfolio_value = 1000000  # $1M base portfolio
                        
                        scenario_results = {}
                        
                        for scenario_name in selected_scenarios:
                            scenario = economic_scenarios[scenario_name]
                            
                            # Apply impacts based on asset classification
                            asset_impacts = {}
                            for asset in scenario_assets:
                                asset_type = asset_types[asset].lower()
                                if asset_type == 'equity':
                                    impact = scenario['equity_impact']
                                elif asset_type == 'bond':
                                    impact = scenario['bond_impact']
                                elif asset_type == 'commodity':
                                    impact = scenario['commodity_impact']
                                else:  # currency
                                    impact = scenario['currency_impact']
                                
                                asset_impacts[asset] = impact
                            
                            # Calculate portfolio impact
                            portfolio_impact = np.dot(equal_weights, list(asset_impacts.values()))
                            portfolio_value_change = base_portfolio_value * portfolio_impact
                            new_portfolio_value = base_portfolio_value + portfolio_value_change
                            
                            scenario_results[scenario_name] = {
                                'description': scenario['description'],
                                'portfolio_impact': portfolio_impact,
                                'value_change': portfolio_value_change,
                                'new_value': new_portfolio_value,
                                'asset_impacts': asset_impacts
                            }
                        
                        # Display scenario results
                        st.markdown("#### Scenario Analysis Results")
                        
                        scenario_summary = pd.DataFrame({
                            'Scenario': list(scenario_results.keys()),
                            'Portfolio Impact': [f"{r['portfolio_impact']:.2%}" for r in scenario_results.values()],
                            'Value Change': [f"${r['value_change']:,.0f}" for r in scenario_results.values()],
                            'New Portfolio Value': [f"${r['new_value']:,.0f}" for r in scenario_results.values()]
                        })
                        
                        st.dataframe(scenario_summary, hide_index=True)
                        
                        # Detailed scenario breakdown
                        st.markdown("#### Detailed Impact by Asset")
                        
                        for scenario_name, results in scenario_results.items():
                            with st.expander(f"{scenario_name} - {results['description']}"):
                                asset_detail = pd.DataFrame({
                                    'Asset': scenario_assets,
                                    'Asset Type': [asset_types[asset] for asset in scenario_assets],
                                    'Impact': [f"{results['asset_impacts'][asset]:.2%}" for asset in scenario_assets],
                                    'Dollar Impact': [f"${base_portfolio_value * w * results['asset_impacts'][asset]:,.0f}" 
                                                    for w, asset in zip(equal_weights, scenario_assets)]
                                })
                                
                                st.dataframe(asset_detail, hide_index=True)
                        
                        # Scenario comparison chart
                        st.markdown("#### Scenario Impact Comparison")
                        
                        fig_scenario = go.Figure()
                        
                        scenario_names = list(scenario_results.keys())
                        portfolio_impacts = [scenario_results[s]['portfolio_impact'] * 100 for s in scenario_names]
                        
                        colors = ['green' if impact > 0 else 'red' for impact in portfolio_impacts]
                        
                        fig_scenario.add_trace(go.Bar(
                            x=scenario_names,
                            y=portfolio_impacts,
                            text=[f"{impact:.1f}%" for impact in portfolio_impacts],
                            textposition='auto',
                            marker_color=colors
                        ))
                        
                        fig_scenario.update_layout(
                            title="Portfolio Impact by Economic Scenario",
                            xaxis_title="Economic Scenario",
                            yaxis_title="Portfolio Impact (%)",
                            height=400
                        )
                        
                        st.plotly_chart(fig_scenario, width='stretch')
                        
                        # Risk assessment
                        worst_scenario = min(scenario_results.values(), key=lambda x: x['portfolio_impact'])
                        best_scenario = max(scenario_results.values(), key=lambda x: x['portfolio_impact'])
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Best Case Scenario:**")
                            best_name = [k for k, v in scenario_results.items() if v == best_scenario][0]
                            st.success(f"{best_name}: +{best_scenario['portfolio_impact']:.2%}")
                        
                        with col2:
                            st.markdown("**Worst Case Scenario:**")
                            worst_name = [k for k, v in scenario_results.items() if v == worst_scenario][0]
                            st.error(f"{worst_name}: {worst_scenario['portfolio_impact']:.2%}")
                        
                        # Scenario risk score
                        scenario_range = best_scenario['portfolio_impact'] - worst_scenario['portfolio_impact']
                        if scenario_range > 0.4:  # More than 40% range
                            st.warning("⚠️ High scenario sensitivity - Consider diversification")
                        else:
                            st.info("ℹ️ Moderate scenario sensitivity - Reasonable diversification")
                    
                    except Exception as e:
                        st.error(f"Scenario analysis failed: {str(e)}")
        else:
            st.info("Please select at least 1 asset for scenario analysis.")
    
    with tab5:
        st.markdown("### Risk Controls and Limits")
        st.markdown("Set up risk monitoring and control mechanisms for portfolio management.")
        
        # Risk control configuration
        control_assets = st.multiselect(
            "Select Assets for Risk Controls:",
            summary['tickers'],
            default=summary['tickers'][:3],
            key="control_assets"
        )
        
        if len(control_assets) >= 1:
            st.markdown("#### Position Limits")
            
            # Position size limits
            col1, col2 = st.columns(2)
            
            with col1:
                max_single_position = st.slider(
                    "Maximum Single Position (%):",
                    5, 50, 25,
                    key="max_single_position",
                    help="Maximum allocation to any single asset"
                )
                
                max_sector_exposure = st.slider(
                    "Maximum Sector Exposure (%):",
                    10, 80, 40,
                    key="max_sector_exposure",
                    help="Maximum allocation to any single sector"
                )
            
            with col2:
                concentration_limit = st.slider(
                    "Concentration Limit - Top 3 Holdings (%):",
                    30, 90, 60,
                    key="concentration_limit",
                    help="Maximum combined weight of top 3 holdings"
                )
                
                min_diversification = st.slider(
                    "Minimum Number of Holdings:",
                    3, 20, 8,
                    key="min_diversification",
                    help="Minimum number of different holdings"
                )
            
            st.markdown("#### Risk Limits")
            
            col1, col2 = st.columns(2)
            
            with col1:
                max_portfolio_var = st.slider(
                    "Maximum Portfolio VaR (95%, 1-day) (%):",
                    1.0, 10.0, 3.0,
                    step=0.5,
                    key="max_portfolio_var",
                    help="Maximum daily Value at Risk"
                )
                
                max_drawdown_limit = st.slider(
                    "Maximum Drawdown Alert (%):",
                    5, 30, 15,
                    key="max_drawdown_limit",
                    help="Alert threshold for maximum drawdown"
                )
            
            with col2:
                max_correlation = st.slider(
                    "Maximum Average Correlation:",
                    0.3, 0.9, 0.7,
                    step=0.05,
                    key="max_correlation",
                    help="Alert when average correlation exceeds this level"
                )
                
                min_liquidity = st.selectbox(
                    "Minimum Liquidity Requirement:",
                    ['Daily', 'Weekly', 'Monthly'],
                    index=1,
                    key="min_liquidity",
                    help="Minimum liquidity for portfolio holdings"
                )
            
            if st.button("Check Risk Controls", key="check_risk_controls"):
                with st.spinner("Checking portfolio against risk controls..."):
                    try:
                        # Get current portfolio data
                        control_data = {}
                        for asset in control_assets:
                            asset_prices = get_ticker_data(data, asset, 'close')
                            control_data[asset] = asset_prices.tail(252)
                        
                        control_df = pd.DataFrame(control_data).dropna()
                        returns_df = control_df.pct_change().dropna()
                        
                        # Current portfolio (equal weights for example)
                        equal_weights = np.array([1/len(control_assets)] * len(control_assets))
                        portfolio_weights = dict(zip(control_assets, equal_weights))
                        
                        # Check position limits
                        violations = []
                        warnings = []
                        
                        # 1. Single position limit
                        max_weight = max(equal_weights)
                        if max_weight > max_single_position / 100:
                            violations.append(f"Single position limit exceeded: {max_weight:.1%} > {max_single_position}%")
                        
                        # 2. Concentration limit (top 3)
                        sorted_weights = sorted(equal_weights, reverse=True)
                        top_3_concentration = sum(sorted_weights[:3])
                        if top_3_concentration > concentration_limit / 100:
                            violations.append(f"Concentration limit exceeded: {top_3_concentration:.1%} > {concentration_limit}%")
                        
                        # 3. Minimum diversification
                        if len(control_assets) < min_diversification:
                            violations.append(f"Insufficient diversification: {len(control_assets)} holdings < {min_diversification} minimum")
                        
                        # 4. Portfolio VaR
                        portfolio_returns = (returns_df * equal_weights).sum(axis=1)
                        portfolio_var_95 = abs(np.percentile(portfolio_returns, 5))
                        if portfolio_var_95 > max_portfolio_var / 100:
                            violations.append(f"Portfolio VaR exceeded: {portfolio_var_95:.2%} > {max_portfolio_var}%")
                        
                        # 5. Maximum drawdown check
                        cumulative = (1 + portfolio_returns).cumprod()
                        running_max = cumulative.cummax()
                        current_drawdown = ((cumulative.iloc[-1] - running_max.iloc[-1]) / running_max.iloc[-1])
                        if abs(current_drawdown) > max_drawdown_limit / 100:
                            violations.append(f"Drawdown alert: {current_drawdown:.2%} > {max_drawdown_limit}% threshold")
                        
                        # 6. Correlation check
                        correlation_matrix = returns_df.corr()
                        avg_correlation = (correlation_matrix.values.sum() - len(control_assets)) / (len(control_assets) * (len(control_assets) - 1))
                        if avg_correlation > max_correlation:
                            warnings.append(f"High correlation warning: {avg_correlation:.3f} > {max_correlation} threshold")
                        
                        # Display results
                        st.markdown("#### Risk Control Results")
                        
                        if violations:
                            st.error("❌ **Risk Control Violations:**")
                            for violation in violations:
                                st.error(f"• {violation}")
                        else:
                            st.success("✅ **All Risk Controls Passed**")
                        
                        if warnings:
                            st.warning("⚠️ **Risk Warnings:**")
                            for warning in warnings:
                                st.warning(f"• {warning}")
                        
                        # Risk dashboard
                        st.markdown("#### Risk Dashboard")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Largest Position", 
                                f"{max_weight:.1%}",
                                delta=f"Limit: {max_single_position}%"
                            )
                        
                        with col2:
                            st.metric(
                                "Top 3 Concentration", 
                                f"{top_3_concentration:.1%}",
                                delta=f"Limit: {concentration_limit}%"
                            )
                        
                        with col3:
                            st.metric(
                                "Portfolio VaR (95%)", 
                                f"{portfolio_var_95:.2%}",
                                delta=f"Limit: {max_portfolio_var}%"
                            )
                        
                        with col4:
                            st.metric(
                                "Avg Correlation", 
                                f"{avg_correlation:.3f}",
                                delta=f"Alert: {max_correlation}"
                            )
                        
                        # Portfolio composition pie chart
                        st.markdown("#### Current Portfolio Composition")
                        
                        fig_composition = px.pie(
                            values=equal_weights,
                            names=control_assets,
                            title="Portfolio Allocation"
                        )
                        
                        st.plotly_chart(fig_composition, width='stretch')
                        
                        # Risk control recommendations
                        st.markdown("#### Risk Management Recommendations")
                        
                        if len(violations) > 0:
                            st.markdown("**Immediate Actions Required:**")
                            if max_weight > max_single_position / 100:
                                st.write("• Reduce position size in largest holding")
                            if top_3_concentration > concentration_limit / 100:
                                st.write("• Diversify away from top holdings")
                            if portfolio_var_95 > max_portfolio_var / 100:
                                st.write("• Reduce portfolio risk through diversification or hedging")
                        else:
                            st.markdown("**Maintenance Recommendations:**")
                            st.write("• Continue regular monitoring of risk metrics")
                            st.write("• Review and update risk limits quarterly")
                            st.write("• Monitor market conditions for changing correlations")
                    
                    except Exception as e:
                        st.error(f"Risk control check failed: {str(e)}")
        else:
            st.info("Please select at least 1 asset for risk control monitoring.")
    
    # Educational content
    st.markdown("---")
    st.markdown("### 📚 Risk Management Best Practices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Risk Measurement:**
        - **VaR:** Maximum expected loss at given confidence
        - **Expected Shortfall:** Average loss beyond VaR
        - **Stress Testing:** Performance under extreme scenarios
        - **Correlation Analysis:** Diversification effectiveness
        """)
    
    with col2:
        st.markdown("""
        **Risk Controls:**
        - **Position Limits:** Prevent over-concentration
        - **Stop Losses:** Limit downside exposure
        - **Diversification Requirements:** Spread risk
        - **Regular Monitoring:** Early warning systems
        """)
    
    st.info("""
    💡 **Key Principle:** Risk management is about preserving capital and ensuring sustainable returns. 
    The goal is not to eliminate risk but to understand, measure, and control it effectively.
    """)
    
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


def page_performance_reporting():
    """Performance reporting page."""
    st.markdown('<div class="main-header">Performance Reporting</div>', unsafe_allow_html=True)
    
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please load data first from the Home page.")
        return
    
    data = st.session_state.data
    summary = get_data_summary(data)
    
    st.markdown("""
    Generate comprehensive performance reports including metrics, attribution analysis, 
    and professional-grade visualizations for portfolio presentation and analysis.
    """)
    
    # Initialize performance reporter
    reporter = PerformanceReporter()
    
    # Create tabs for different reporting features
    tab1, tab2, tab3 = st.tabs(["📈 Portfolio Report", "📊 Detailed Analytics", "📋 Executive Summary"])
    
    with tab1:
        st.markdown("### Portfolio Performance Report")
        st.markdown("Generate a comprehensive performance report for your portfolio strategy.")
        
        # Portfolio configuration
        col1, col2 = st.columns(2)
        
        with col1:
            report_assets = st.multiselect(
                "Select Portfolio Assets:",
                summary['tickers'],
                default=summary['tickers'][:4],
                key="report_assets"
            )
            
            report_strategy = st.selectbox(
                "Portfolio Strategy:",
                ['Equal Weight', 'Risk Parity', 'Market Cap Weight'],
                key="report_strategy"
            )
            
            report_period = st.selectbox(
                "Reporting Period:",
                ['6 Months', '1 Year', '2 Years', '3 Years'],
                index=1,
                key="report_period"
            )
        
        with col2:
            benchmark_asset = st.selectbox(
                "Benchmark Asset (Optional):",
                ['None'] + summary['tickers'],
                key="benchmark_asset"
            )
            
            include_risk_metrics = st.checkbox(
                "Include Advanced Risk Metrics",
                value=True,
                key="include_risk_metrics"
            )
        
        if len(report_assets) >= 1:
            if st.button("Generate Performance Report", type="primary", key="generate_report"):
                with st.spinner("Generating comprehensive performance report..."):
                    try:
                        # Get data for reporting period
                        period_map = {'6 Months': 180, '1 Year': 365, '2 Years': 730, '3 Years': 1095}
                        days_back = period_map[report_period]
                        start_date = datetime.now() - timedelta(days=days_back)
                        
                        # Get portfolio data
                        portfolio_data = {}
                        for asset in report_assets:
                            asset_prices = get_ticker_data(data, asset, 'close')
                            mask = asset_prices.index >= pd.Timestamp(start_date)
                            portfolio_data[asset] = asset_prices[mask]
                        
                        portfolio_df = pd.DataFrame(portfolio_data).dropna()
                        
                        if len(portfolio_df) < 30:
                            st.error("Insufficient data for the selected period. Please adjust the reporting period.")
                            return
                        
                        portfolio_returns_df = portfolio_df.pct_change().dropna()
                        
                        # Calculate portfolio weights
                        if report_strategy == 'Equal Weight':
                            weights = np.array([1/len(report_assets)] * len(report_assets))
                        elif report_strategy == 'Risk Parity':
                            volatilities = portfolio_returns_df.std()
                            inv_vol = 1 / volatilities
                            weights = (inv_vol / inv_vol.sum()).values
                        else:  # Market Cap Weight
                            prices = portfolio_df.iloc[-1]
                            weights = (prices / prices.sum()).values
                        
                        # Calculate portfolio returns
                        portfolio_returns = (portfolio_returns_df * weights).sum(axis=1)
                        
                        # Get benchmark data if selected
                        benchmark_returns = None
                        if benchmark_asset != 'None':
                            benchmark_prices = get_ticker_data(data, benchmark_asset, 'close')
                            mask = benchmark_prices.index >= pd.Timestamp(start_date)
                            benchmark_prices = benchmark_prices[mask]
                            benchmark_returns = benchmark_prices.pct_change().dropna()
                            
                            # Align dates
                            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
                            portfolio_returns = portfolio_returns.loc[common_dates]
                            benchmark_returns = benchmark_returns.loc[common_dates]
                        
                        # Calculate performance metrics
                        metrics = reporter.calculate_performance_metrics(
                            portfolio_returns, 
                            benchmark_returns
                        )
                        
                        # Display performance summary
                        st.markdown("#### Performance Summary")
                        
                        # Key metrics display
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Return", f"{metrics['total_return']:.2%}")
                        with col2:
                            st.metric("Annualized Return", f"{metrics['annualized_return']:.2%}")
                        with col3:
                            st.metric("Volatility", f"{metrics['volatility']:.2%}")
                        with col4:
                            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
                        with col2:
                            st.metric("Win Rate", f"{metrics['win_rate']:.2%}")
                        with col3:
                            st.metric("VaR (95%)", f"{metrics['var_95']:.2%}")
                        with col4:
                            if 'alpha' in metrics:
                                st.metric("Alpha", f"{metrics['alpha']:.2%}")
                            else:
                                st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.3f}")
                        
                        # Generate performance chart
                        st.markdown("#### Portfolio Performance Analysis")
                        
                        performance_chart = reporter.generate_performance_summary_chart(
                            portfolio_returns, 
                            benchmark_returns,
                            f"{report_strategy} Portfolio Performance ({report_period})"
                        )
                        
                        if performance_chart:
                            st.plotly_chart(performance_chart, width='stretch')
                        
                        # Portfolio composition
                        st.markdown("#### Portfolio Composition")
                        
                        composition_df = pd.DataFrame({
                            'Asset': report_assets,
                            'Weight': weights,
                            'Weight (%)': [f"{w:.1%}" for w in weights]
                        }).sort_values('Weight', ascending=False)
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.dataframe(composition_df, hide_index=True)
                        
                        with col2:
                            fig_pie = px.pie(
                                composition_df,
                                values='Weight',
                                names='Asset',
                                title="Portfolio Allocation"
                            )
                            st.plotly_chart(fig_pie, width='stretch')
                        
                        # Monthly performance table
                        st.markdown("#### Monthly Performance Breakdown")
                        
                        monthly_table = reporter.generate_monthly_performance_table(portfolio_returns)
                        if monthly_table is not None:
                            st.dataframe(monthly_table, width='stretch')
                        
                        # Performance vs benchmark comparison
                        if benchmark_returns is not None:
                            st.markdown("#### Portfolio vs Benchmark Comparison")
                            
                            bench_metrics = reporter.calculate_performance_metrics(benchmark_returns)
                            
                            comparison_df = pd.DataFrame({
                                'Metric': ['Total Return', 'Annualized Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown'],
                                'Portfolio': [
                                    f"{metrics['total_return']:.2%}",
                                    f"{metrics['annualized_return']:.2%}",
                                    f"{metrics['volatility']:.2%}",
                                    f"{metrics['sharpe_ratio']:.3f}",
                                    f"{metrics['max_drawdown']:.2%}"
                                ],
                                'Benchmark': [
                                    f"{bench_metrics['total_return']:.2%}",
                                    f"{bench_metrics['annualized_return']:.2%}",
                                    f"{bench_metrics['volatility']:.2%}",
                                    f"{bench_metrics['sharpe_ratio']:.3f}",
                                    f"{bench_metrics['max_drawdown']:.2%}"
                                ]
                            })
                            
                            st.dataframe(comparison_df, hide_index=True)
                            
                            if 'alpha' in metrics and metrics['alpha'] > 0:
                                st.success(f"🎯 Portfolio generated {metrics['alpha']:.2%} alpha vs benchmark")
                            elif 'alpha' in metrics:
                                st.info(f"📊 Portfolio underperformed benchmark by {abs(metrics['alpha']):.2%}")
                    
                    except Exception as e:
                        st.error(f"Performance report generation failed: {str(e)}")
        else:
            st.info("Please select at least 1 asset for performance reporting.")
    
    with tab2:
        st.markdown("### Detailed Performance Analytics")
        st.markdown("Deep dive into risk-return characteristics and statistical analysis.")
        
        # Analytics configuration
        analytics_assets = st.multiselect(
            "Select Assets for Analytics:",
            summary['tickers'],
            default=summary['tickers'][:3],
            key="analytics_assets"
        )
        
        if len(analytics_assets) >= 1:
            if st.button("Generate Detailed Analytics", key="generate_analytics"):
                with st.spinner("Generating detailed analytics..."):
                    try:
                        # Get data (last 2 years for comprehensive analysis)
                        start_date = datetime.now() - timedelta(days=730)
                        
                        analytics_data = {}
                        for asset in analytics_assets:
                            asset_prices = get_ticker_data(data, asset, 'close')
                            mask = asset_prices.index >= pd.Timestamp(start_date)
                            analytics_data[asset] = asset_prices[mask]
                        
                        analytics_df = pd.DataFrame(analytics_data).dropna()
                        analytics_returns = analytics_df.pct_change().dropna()
                        
                        # Portfolio returns (equal weight)
                        equal_weights = np.array([1/len(analytics_assets)] * len(analytics_assets))
                        portfolio_returns = (analytics_returns * equal_weights).sum(axis=1)
                        
                        # Generate detailed analytics
                        metrics = reporter.calculate_performance_metrics(portfolio_returns)
                        
                        # Risk-return analysis chart
                        st.markdown("#### Risk-Return Analysis")
                        
                        risk_return_chart = reporter.generate_risk_return_analysis(portfolio_returns)
                        if risk_return_chart:
                            st.plotly_chart(risk_return_chart, width='stretch')
                        
                        # Statistical analysis
                        st.markdown("#### Statistical Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Distribution Statistics:**")
                            stats_df = pd.DataFrame({
                                'Statistic': ['Mean (Daily)', 'Std Dev (Daily)', 'Skewness', 'Kurtosis', 'Jarque-Bera p-value'],
                                'Value': [
                                    f"{portfolio_returns.mean():.4f}",
                                    f"{portfolio_returns.std():.4f}",
                                    f"{metrics['skewness']:.3f}",
                                    f"{metrics['kurtosis']:.3f}",
                                    f"{stats.jarque_bera(portfolio_returns)[1]:.4f}"
                                ]
                            })
                            st.dataframe(stats_df, hide_index=True)
                        
                        with col2:
                            st.markdown("**Performance Percentiles:**")
                            percentiles_df = pd.DataFrame({
                                'Percentile': ['5th', '25th', '50th (Median)', '75th', '95th'],
                                'Daily Return': [
                                    f"{np.percentile(portfolio_returns, 5):.3%}",
                                    f"{np.percentile(portfolio_returns, 25):.3%}",
                                    f"{np.percentile(portfolio_returns, 50):.3%}",
                                    f"{np.percentile(portfolio_returns, 75):.3%}",
                                    f"{np.percentile(portfolio_returns, 95):.3%}"
                                ]
                            })
                            st.dataframe(percentiles_df, hide_index=True)
                        
                        # Correlation analysis
                        if len(analytics_assets) > 1:
                            st.markdown("#### Asset Correlation Analysis")
                            
                            corr_matrix = analytics_returns.corr()
                            
                            fig_corr = px.imshow(
                                corr_matrix,
                                title="Asset Correlation Matrix",
                                color_continuous_scale='RdBu',
                                aspect='auto'
                            )
                            st.plotly_chart(fig_corr, width='stretch')
                            
                            # Average correlation
                            avg_corr = (corr_matrix.values.sum() - len(analytics_assets)) / (len(analytics_assets) * (len(analytics_assets) - 1))
                            st.info(f"Average correlation: {avg_corr:.3f}")
                    
                    except Exception as e:
                        st.error(f"Detailed analytics generation failed: {str(e)}")
        else:
            st.info("Please select at least 1 asset for detailed analytics.")
    
    with tab3:
        st.markdown("### Executive Summary Report")
        st.markdown("Generate a high-level executive summary for stakeholder presentation.")
        
        # Executive summary configuration
        exec_assets = st.multiselect(
            "Select Portfolio Assets:",
            summary['tickers'],
            default=summary['tickers'][:3],
            key="exec_assets"
        )
        
        exec_period = st.selectbox(
            "Reporting Period:",
            ['Quarter', 'Year-to-Date', '1 Year', '3 Years'],
            index=2,
            key="exec_period"
        )
        
        if len(exec_assets) >= 1:
            if st.button("Generate Executive Summary", type="primary", key="generate_exec_summary"):
                with st.spinner("Generating executive summary..."):
                    try:
                        # Get data based on period
                        period_map = {'Quarter': 90, 'Year-to-Date': 365, '1 Year': 365, '3 Years': 1095}
                        days_back = period_map[exec_period]
                        start_date = datetime.now() - timedelta(days=days_back)
                        
                        exec_data = {}
                        for asset in exec_assets:
                            asset_prices = get_ticker_data(data, asset, 'close')
                            mask = asset_prices.index >= pd.Timestamp(start_date)
                            exec_data[asset] = asset_prices[mask]
                        
                        exec_df = pd.DataFrame(exec_data).dropna()
                        exec_returns = exec_df.pct_change().dropna()
                        
                        # Portfolio returns (equal weight)
                        equal_weights = np.array([1/len(exec_assets)] * len(exec_assets))
                        portfolio_returns = (exec_returns * equal_weights).sum(axis=1)
                        
                        # Calculate key metrics
                        metrics = reporter.calculate_performance_metrics(portfolio_returns)
                        
                        # Executive summary layout
                        st.markdown("#### 📊 Executive Summary")
                        st.markdown(f"**Portfolio Performance Report - {exec_period}**")
                        st.markdown("---")
                        
                        # Key highlights
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "📈 Total Return",
                                f"{metrics['total_return']:.1%}",
                                help="Overall portfolio performance"
                            )
                        
                        with col2:
                            st.metric(
                                "📊 Risk-Adjusted Return",
                                f"{metrics['sharpe_ratio']:.2f}",
                                help="Sharpe ratio - return per unit of risk"
                            )
                        
                        with col3:
                            st.metric(
                                "⚠️ Maximum Drawdown", 
                                f"{metrics['max_drawdown']:.1%}",
                                help="Largest peak-to-trough decline"
                            )
                        
                        # Performance narrative
                        st.markdown("#### 📋 Performance Narrative")
                        
                        # Generate automated narrative
                        performance_grade = "Strong" if metrics['total_return'] > 0.1 else "Moderate" if metrics['total_return'] > 0 else "Weak"
                        risk_assessment = "Low" if metrics['volatility'] < 0.15 else "Moderate" if metrics['volatility'] < 0.25 else "High"
                        
                        narrative = f"""
                        **Portfolio Overview:**
                        The portfolio delivered a **{performance_grade.lower()}** performance during the {exec_period.lower()} period, 
                        generating a total return of **{metrics['total_return']:.2%}**. 
                        
                        **Risk Management:**
                        The portfolio exhibited **{risk_assessment.lower()}** volatility at **{metrics['volatility']:.1%}** annualized, 
                        with a maximum drawdown of **{metrics['max_drawdown']:.1%}**.
                        
                        **Risk-Adjusted Performance:**
                        The Sharpe ratio of **{metrics['sharpe_ratio']:.2f}** indicates {'excellent' if metrics['sharpe_ratio'] > 1 else 'good' if metrics['sharpe_ratio'] > 0.5 else 'poor'} 
                        risk-adjusted returns.
                        """
                        
                        st.markdown(narrative)
                        
                        # Key metrics table
                        st.markdown("#### 📈 Key Performance Indicators")
                        
                        kpi_df = pd.DataFrame({
                            'Metric': [
                                'Total Return',
                                'Annualized Return', 
                                'Volatility',
                                'Sharpe Ratio',
                                'Maximum Drawdown',
                                'Win Rate'
                            ],
                            'Value': [
                                f"{metrics['total_return']:.2%}",
                                f"{metrics['annualized_return']:.2%}",
                                f"{metrics['volatility']:.2%}",
                                f"{metrics['sharpe_ratio']:.3f}",
                                f"{metrics['max_drawdown']:.2%}",
                                f"{metrics['win_rate']:.1%}"
                            ]
                        })
                        
                        st.dataframe(kpi_df, hide_index=True)
                        
                        # Simple performance chart for executive view
                        st.markdown("#### 📊 Portfolio Performance Trend")
                        
                        cumulative_returns = (1 + portfolio_returns).cumprod()
                        
                        fig_exec = go.Figure()
                        fig_exec.add_trace(go.Scatter(
                            x=cumulative_returns.index,
                            y=cumulative_returns.values,
                            mode='lines',
                            name='Portfolio Value',
                            line=dict(color='blue', width=3),
                            fill='tonexty'
                        ))
                        
                        fig_exec.update_layout(
                            title=f"Portfolio Growth - {exec_period}",
                            xaxis_title="Date",
                            yaxis_title="Cumulative Return",
                            height=400,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_exec, width='stretch')
                        
                        # Executive summary footer
                        st.markdown("---")
                        st.markdown(f"*Report generated on {datetime.now().strftime('%B %d, %Y')} | Portfolio Forecasting System*")
                    
                    except Exception as e:
                        st.error(f"Executive summary generation failed: {str(e)}")
        else:
            st.info("Please select at least 1 asset for executive summary.")


def get_market_overview():
    """Get comprehensive market overview data."""
    try:
        market_api = MarketDataAPI()
        
        # Get market indices
        major_indices = {
            'S&P 500': '^GSPC',
            'NASDAQ': '^IXIC', 
            'Dow Jones': '^DJI',
            'Russell 2000': '^RUT',
            'VIX': '^VIX'
        }
        
        indices_data = {}
        for name, symbol in major_indices.items():
            try:
                data = market_api.get_enhanced_stock_data(symbol, period='1y')
                if data and not data['price_data'].empty:
                    price_data = data['price_data']
                    current_price = price_data['Close'].iloc[-1]
                    prev_price = price_data['Close'].iloc[-2] if len(price_data) > 1 else current_price
                    change_1d = ((current_price / prev_price) - 1) * 100
                    
                    indices_data[name] = {
                        'current_price': current_price,
                        'change_1d': change_1d,
                        'data': price_data
                    }
            except:
                continue
        
        # Generate economic indicators (simulated for demo)
        economic_indicators = {
            'GDP Growth Rate': np.random.uniform(1.5, 3.5),
            'Unemployment Rate': np.random.uniform(3.0, 6.0), 
            'Inflation Rate': np.random.uniform(1.0, 4.0),
            'Federal Funds Rate': np.random.uniform(0.0, 5.5),
            'Consumer Confidence': np.random.uniform(90, 130)
        }
        
        # Market calendar information
        now = datetime.now()
        market_calendar = {
            'current_time': now,
            'market_status': 'OPEN' if 9 <= now.hour < 16 and now.weekday() < 5 else 'CLOSED',
            'next_market_open': now.replace(hour=9, minute=30, second=0) + timedelta(days=1) if now.hour >= 16 or now.weekday() >= 5 else now.replace(hour=9, minute=30, second=0),
            'next_market_close': now.replace(hour=16, minute=0, second=0) if now.hour < 16 and now.weekday() < 5 else now.replace(hour=16, minute=0, second=0) + timedelta(days=1)
        }
        
        return {
            'market_indices': indices_data,
            'economic_indicators': economic_indicators,
            'market_calendar': market_calendar
        }
        
    except Exception as e:
        st.error(f"Error fetching market overview: {str(e)}")
        return {
            'market_indices': {},
            'economic_indicators': {},
            'market_calendar': {}
        }


def page_market_dashboard():
    """Market dashboard with real-time data and analysis."""
    st.markdown('<div class="main-header">Market Dashboard</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Real-time market data, analysis, and insights powered by multiple data sources 
    including market indices, sector performance, economic indicators, and alternative data.
    """)
    
    # Initialize APIs
    market_api = MarketDataAPI()
    alt_api = AlternativeDataAPI()
    
    # Create tabs for different market views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏛️ Market Overview", "📊 Sector Analysis", "💱 Global Markets", "🔍 Stock Analysis", "📈 Technical Signals"])
    
    with tab1:
        st.markdown("### Market Overview")
        st.markdown("Real-time view of major market indices and economic indicators.")
        
        if st.button("Refresh Market Data", type="primary", key="refresh_market"):
            with st.spinner("Fetching real-time market data..."):
                try:
                    # Get market overview
                    market_overview = get_market_overview()
                    
                    # Market indices
                    st.markdown("#### Major Market Indices")
                    
                    indices_data = market_overview['market_indices']
                    if indices_data:
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        for i, (name, data) in enumerate(indices_data.items()):
                            with [col1, col2, col3, col4, col5][i % 5]:
                                change_color = "normal" if data['change_1d'] >= 0 else "inverse"
                                st.metric(
                                    name,
                                    f"{data['current_price']:.2f}",
                                    f"{data['change_1d']:.2f}%",
                                    delta_color=change_color
                                )
                        
                        # Indices performance chart
                        st.markdown("#### Indices Performance (YTD)")
                        
                        fig_indices = go.Figure()
                        
                        for name, data in indices_data.items():
                            if 'data' in data:
                                normalized_data = data['data']['Close'] / data['data']['Close'].iloc[0]
                                fig_indices.add_trace(go.Scatter(
                                    x=normalized_data.index,
                                    y=normalized_data.values,
                                    mode='lines',
                                    name=name,
                                    line=dict(width=2)
                                ))
                        
                        fig_indices.update_layout(
                            title="Market Indices Performance (Normalized)",
                            xaxis_title="Date",
                            yaxis_title="Normalized Price",
                            height=400,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_indices, width='stretch')
                    
                    # Economic indicators
                    st.markdown("#### Economic Indicators")
                    
                    econ_indicators = market_overview['economic_indicators']
                    if econ_indicators:
                        econ_df = pd.DataFrame([
                            {'Indicator': k, 'Value': f"{v:.2f}" + ('%' if 'Rate' in k or 'Growth' in k or 'Inflation' in k else ''), 
                             'Status': '✅ Positive' if v > 0 else '⚠️ Negative' if 'Unemployment' not in k else '⚠️ High'}
                            for k, v in econ_indicators.items()
                        ])
                        
                        st.dataframe(econ_df, hide_index=True)
                    
                    # Market calendar
                    st.markdown("#### Market Information")
                    
                    calendar_info = market_overview['market_calendar']
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        status_color = "🟢" if calendar_info['market_status'] == "OPEN" else "🔴"
                        st.info(f"{status_color} **Market Status:** {calendar_info['market_status']}")
                        st.info(f"⏰ **Current Time:** {calendar_info['current_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    with col2:
                        st.info(f"🔔 **Next Open:** {calendar_info['next_market_open'].strftime('%Y-%m-%d %H:%M')}")
                        st.info(f"🔔 **Next Close:** {calendar_info['next_market_close'].strftime('%Y-%m-%d %H:%M')}")
                
                except Exception as e:
                    st.error(f"Error fetching market data: {str(e)}")
        else:
            st.info("Click 'Refresh Market Data' to load the latest market information.")
    
    with tab2:
        st.markdown("### Sector Analysis")
        st.markdown("Performance analysis across different market sectors.")
        
        if st.button("Refresh Sector Data", key="refresh_sectors"):
            with st.spinner("Loading sector performance data..."):
                try:
                    sector_data = market_api.get_sector_performance()
                    
                    if sector_data:
                        # Sector performance metrics
                        st.markdown("#### Sector Performance Summary")
                        
                        sector_df = pd.DataFrame([
                            {
                                'Sector': sector,
                                'ETF': data['etf_symbol'],
                                'Price': f"${data['current_price']:.2f}",
                                '1D Change': f"{data['change_1d']:.2f}%",
                                '1W Change': f"{data['change_1w']:.2f}%",
                                '1M Change': f"{data['change_1m']:.2f}%",
                                'Volatility': f"{data['volatility']:.1f}%"
                            }
                            for sector, data in sector_data.items()
                        ])
                        
                        st.dataframe(sector_df, hide_index=True)
                        
                        # Sector performance visualization
                        st.markdown("#### Sector Performance Comparison")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # 1-day performance
                            fig_1d = go.Figure(data=[
                                go.Bar(
                                    x=list(sector_data.keys()),
                                    y=[data['change_1d'] for data in sector_data.values()],
                                    marker_color=['green' if x >= 0 else 'red' for x in [data['change_1d'] for data in sector_data.values()]],
                                    text=[f"{x:.1f}%" for x in [data['change_1d'] for data in sector_data.values()]],
                                    textposition='auto'
                                )
                            ])
                            fig_1d.update_layout(
                                title="1-Day Sector Performance",
                                xaxis_title="Sector",
                                yaxis_title="Change (%)",
                                height=400
                            )
                            fig_1d.update_xaxes(tickangle=45)
                            st.plotly_chart(fig_1d, width='stretch')
                        
                        with col2:
                            # 1-month performance
                            fig_1m = go.Figure(data=[
                                go.Bar(
                                    x=list(sector_data.keys()),
                                    y=[data['change_1m'] for data in sector_data.values()],
                                    marker_color=['green' if x >= 0 else 'red' for x in [data['change_1m'] for data in sector_data.values()]],
                                    text=[f"{x:.1f}%" for x in [data['change_1m'] for data in sector_data.values()]],
                                    textposition='auto'
                                )
                            ])
                            fig_1m.update_layout(
                                title="1-Month Sector Performance",
                                xaxis_title="Sector",
                                yaxis_title="Change (%)",
                                height=400
                            )
                            fig_1m.update_xaxes(tickangle=45)
                            st.plotly_chart(fig_1m, width='stretch')
                        
                        # Sector insights
                        best_sector = max(sector_data.items(), key=lambda x: x[1]['change_1d'])
                        worst_sector = min(sector_data.items(), key=lambda x: x[1]['change_1d'])
                        
                        st.markdown("#### Sector Insights")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.success(f"🏆 **Best Performing Sector (1D):** {best_sector[0]}")
                            st.write(f"Change: {best_sector[1]['change_1d']:.2f}%")
                            st.write(f"ETF: {best_sector[1]['etf_symbol']}")
                        
                        with col2:
                            st.error(f"📉 **Worst Performing Sector (1D):** {worst_sector[0]}")
                            st.write(f"Change: {worst_sector[1]['change_1d']:.2f}%")
                            st.write(f"ETF: {worst_sector[1]['etf_symbol']}")
                
                except Exception as e:
                    st.error(f"Error fetching sector data: {str(e)}")
    
    with tab3:
        st.markdown("### Global Markets")
        st.markdown("Cryptocurrency and forex market overview.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Cryptocurrency Markets")
            
            if st.button("Refresh Crypto Data", key="refresh_crypto"):
                with st.spinner("Loading cryptocurrency data..."):
                    try:
                        crypto_data = market_api.get_crypto_data()
                        
                        if crypto_data:
                            crypto_df = pd.DataFrame([
                                {
                                    'Crypto': crypto,
                                    'Price': f"${data['current_price']:,.2f}",
                                    '24h Change': f"{data['change_24h']:.2f}%",
                                    'Volatility': f"{data['volatility']:.1f}%",
                                    'Volume': f"${data['volume_24h']:,.0f}" if data['volume_24h'] > 0 else "N/A"
                                }
                                for crypto, data in crypto_data.items()
                            ])
                            
                            st.dataframe(crypto_df, hide_index=True)
                            
                            # Crypto performance chart
                            if len(crypto_data) > 1:
                                crypto_changes = [data['change_24h'] for data in crypto_data.values()]
                                crypto_names = list(crypto_data.keys())
                                
                                fig_crypto = go.Figure(data=[
                                    go.Bar(
                                        x=crypto_names,
                                        y=crypto_changes,
                                        marker_color=['green' if x >= 0 else 'red' for x in crypto_changes],
                                        text=[f"{x:.1f}%" for x in crypto_changes],
                                        textposition='auto'
                                    )
                                ])
                                
                                fig_crypto.update_layout(
                                    title="24h Cryptocurrency Performance",
                                    xaxis_title="Cryptocurrency",
                                    yaxis_title="24h Change (%)",
                                    height=300
                                )
                                
                                st.plotly_chart(fig_crypto, width='stretch')
                    
                    except Exception as e:
                        st.error(f"Error fetching crypto data: {str(e)}")
        
        with col2:
            st.markdown("#### Foreign Exchange")
            
            if st.button("Refresh Forex Data", key="refresh_forex"):
                with st.spinner("Loading forex data..."):
                    try:
                        forex_data = market_api.get_forex_data()
                        
                        if forex_data:
                            forex_df = pd.DataFrame([
                                {
                                    'Currency Pair': pair,
                                    'Rate': f"{data['current_rate']:.4f}",
                                    '1D Change': f"{data['change_1d']:.2f}%",
                                    'Volatility': f"{data['volatility']:.1f}%"
                                }
                                for pair, data in forex_data.items()
                            ])
                            
                            st.dataframe(forex_df, hide_index=True)
                            
                            # Forex performance chart
                            if len(forex_data) > 1:
                                forex_changes = [data['change_1d'] for data in forex_data.values()]
                                forex_names = list(forex_data.keys())
                                
                                fig_forex = go.Figure(data=[
                                    go.Bar(
                                        x=forex_names,
                                        y=forex_changes,
                                        marker_color=['green' if x >= 0 else 'red' for x in forex_changes],
                                        text=[f"{x:.2f}%" for x in forex_changes],
                                        textposition='auto'
                                    )
                                ])
                                
                                fig_forex.update_layout(
                                    title="1-Day Forex Performance",
                                    xaxis_title="Currency Pair",
                                    yaxis_title="1D Change (%)",
                                    height=300
                                )
                                
                                st.plotly_chart(fig_forex, width='stretch')
                    
                    except Exception as e:
                        st.error(f"Error fetching forex data: {str(e)}")
    
    with tab4:
        st.markdown("### Individual Stock Analysis")
        st.markdown("Deep dive analysis for individual stocks with fundamentals and alternative data.")
        
        # Stock symbol input
        analysis_symbol = st.text_input(
            "Enter Stock Symbol:",
            value="AAPL",
            key="analysis_symbol",
            help="Enter a valid stock ticker symbol (e.g., AAPL, GOOGL, TSLA)"
        ).upper()
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_fundamentals = st.checkbox("Include Fundamentals", value=True, key="include_fundamentals")
        
        with col2:
            include_alt_data = st.checkbox("Include Alternative Data", value=True, key="include_alt_data")
        
        if st.button("Analyze Stock", type="primary", key="analyze_stock"):
            if analysis_symbol:
                with st.spinner(f"Analyzing {analysis_symbol}..."):
                    try:
                        # Get enhanced stock data
                        stock_data = market_api.get_enhanced_stock_data(
                            analysis_symbol, 
                            period='6mo', 
                            include_fundamentals=include_fundamentals
                        )
                        
                        if stock_data and not stock_data['price_data'].empty:
                            # Basic price information
                            st.markdown(f"#### {analysis_symbol} Analysis")
                            
                            price_data = stock_data['price_data']
                            current_price = price_data['Close'].iloc[-1]
                            prev_price = price_data['Close'].iloc[-2] if len(price_data) > 1 else current_price
                            change_1d = ((current_price / prev_price) - 1) * 100
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Current Price", f"${current_price:.2f}")
                            with col2:
                                st.metric("1D Change", f"{change_1d:.2f}%", delta_color="normal" if change_1d >= 0 else "inverse")
                            with col3:
                                st.metric("Volume", f"{price_data['Volume'].iloc[-1]:,.0f}" if 'Volume' in price_data else "N/A")
                            with col4:
                                high_52w = price_data['High'].max()
                                st.metric("52W High", f"${high_52w:.2f}")
                            
                            # Price chart
                            st.markdown(f"#### {analysis_symbol} Price Chart")
                            
                            fig_stock = go.Figure()
                            
                            fig_stock.add_trace(go.Candlestick(
                                x=price_data.index,
                                open=price_data['Open'],
                                high=price_data['High'], 
                                low=price_data['Low'],
                                close=price_data['Close'],
                                name=analysis_symbol
                            ))
                            
                            fig_stock.update_layout(
                                title=f"{analysis_symbol} Price Chart (6 Months)",
                                xaxis_title="Date",
                                yaxis_title="Price ($)",
                                height=500,
                                xaxis_rangeslider_visible=False
                            )
                            
                            st.plotly_chart(fig_stock, width='stretch')
                            
                            # Company fundamentals
                            if include_fundamentals and 'company_info' in stock_data and stock_data['company_info']:
                                st.markdown("#### Company Information")
                                
                                info = stock_data['company_info']
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.info(f"**Company:** {info.get('name', 'N/A')}")
                                    st.info(f"**Sector:** {info.get('sector', 'N/A')}")
                                    st.info(f"**Industry:** {info.get('industry', 'N/A')}")
                                
                                with col2:
                                    market_cap = info.get('market_cap', 0)
                                    if market_cap > 0:
                                        st.info(f"**Market Cap:** ${market_cap/1e9:.1f}B")
                                    st.info(f"**P/E Ratio:** {info.get('pe_ratio', 'N/A')}")
                                    st.info(f"**Beta:** {info.get('beta', 'N/A')}")
                            
                            # Technical indicators
                            st.markdown("#### Technical Analysis")
                            
                            tech_indicators = market_api.get_technical_indicators(analysis_symbol)
                            
                            if tech_indicators:
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    if tech_indicators['rsi']:
                                        rsi_color = "🔴" if tech_indicators['rsi'] > 70 else "🟢" if tech_indicators['rsi'] < 30 else "🟡"
                                        st.metric("RSI", f"{tech_indicators['rsi']:.1f}", help="Relative Strength Index")
                                        st.write(f"{rsi_color} RSI Signal")
                                
                                with col2:
                                    if tech_indicators['sma_20']:
                                        st.metric("20-Day SMA", f"${tech_indicators['sma_20']:.2f}")
                                        if tech_indicators['price_vs_sma20']:
                                            st.write(f"Price vs SMA20: {tech_indicators['price_vs_sma20']:.1f}%")
                                
                                with col3:
                                    if tech_indicators['macd'] and tech_indicators['macd_signal']:
                                        macd_signal = "Bullish" if tech_indicators['macd'] > tech_indicators['macd_signal'] else "Bearish"
                                        st.metric("MACD Signal", macd_signal)
                                        st.write(f"MACD: {tech_indicators['macd']:.3f}")
                                
                                # Technical signals summary
                                if tech_indicators['signals']:
                                    st.markdown("**Technical Signals:**")
                                    for signal in tech_indicators['signals']:
                                        st.write(f"• {signal}")
                            
                            # Alternative data
                            if include_alt_data:
                                st.markdown("#### Alternative Data Analysis")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Social sentiment
                                    social_data = alt_api.get_social_sentiment(analysis_symbol)
                                    
                                    st.markdown("**Social Media Sentiment:**")
                                    sentiment_color = "🟢" if social_data['overall_sentiment'] == 'Bullish' else "🔴" if social_data['overall_sentiment'] == 'Bearish' else "🟡"
                                    st.write(f"{sentiment_color} Overall Sentiment: **{social_data['overall_sentiment']}**")
                                    st.write(f"Twitter Mentions: {social_data['twitter_mentions']:,}")
                                    st.write(f"Reddit Posts: {social_data['reddit_posts']:,}")
                                    st.write(f"Trending Score: {social_data['trending_score']}/100")
                                
                                with col2:
                                    # ESG data
                                    esg_data = alt_api.get_esg_scores(analysis_symbol)
                                    
                                    st.markdown("**ESG Scores:**")
                                    st.write(f"Overall ESG: {esg_data['overall_esg_score']}/100")
                                    st.write(f"Environmental: {esg_data['environmental_score']}/100")
                                    st.write(f"Social: {esg_data['social_score']}/100")
                                    st.write(f"Governance: {esg_data['governance_score']}/100")
                                
                                # News sentiment
                                news_data = market_api.get_news_sentiment(analysis_symbol)
                                
                                st.markdown("**News Analysis:**")
                                sentiment_emoji = "📈" if news_data['sentiment_score'] > 0 else "📉" if news_data['sentiment_score'] < 0 else "➡️"
                                st.write(f"{sentiment_emoji} News Sentiment: **{news_data['sentiment_label']}** (Score: {news_data['sentiment_score']:.2f})")
                                st.write(f"News Articles: {news_data['news_count']}")
                                
                                with st.expander("Recent Headlines"):
                                    for headline in news_data['latest_headlines']:
                                        st.write(f"• {headline}")
                        
                        else:
                            st.error(f"Could not fetch data for {analysis_symbol}. Please check the symbol and try again.")
                    
                    except Exception as e:
                        st.error(f"Error analyzing {analysis_symbol}: {str(e)}")
            else:
                st.warning("Please enter a stock symbol.")
    
    with tab5:
        st.markdown("### Technical Signals Scanner")
        st.markdown("Scan multiple stocks for technical trading signals.")
        
        # Default watchlist
        default_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
        
        # Watchlist input
        watchlist_input = st.text_area(
            "Enter Stock Symbols (comma-separated):",
            value=', '.join(default_symbols),
            key="watchlist_input",
            help="Enter multiple stock symbols separated by commas"
        )
        
        signal_types = st.multiselect(
            "Select Signal Types:",
            ['RSI Overbought/Oversold', 'MACD Crossover', 'Price vs Moving Average', 'Volume Analysis'],
            default=['RSI Overbought/Oversold', 'MACD Crossover'],
            key="signal_types"
        )
        
        if st.button("Scan for Signals", type="primary", key="scan_signals"):
            if watchlist_input:
                symbols = [symbol.strip().upper() for symbol in watchlist_input.split(',')]
                
                with st.spinner(f"Scanning {len(symbols)} stocks for technical signals..."):
                    try:
                        signals_results = []
                        
                        for symbol in symbols:
                            try:
                                tech_data = market_api.get_technical_indicators(symbol, period='3mo')
                                
                                if tech_data:
                                    symbol_signals = []
                                    
                                    # RSI signals
                                    if 'RSI Overbought/Oversold' in signal_types and tech_data['rsi']:
                                        if tech_data['rsi'] > 70:
                                            symbol_signals.append("RSI Overbought")
                                        elif tech_data['rsi'] < 30:
                                            symbol_signals.append("RSI Oversold")
                                    
                                    # MACD signals
                                    if 'MACD Crossover' in signal_types and tech_data['macd'] and tech_data['macd_signal']:
                                        if tech_data['macd'] > tech_data['macd_signal']:
                                            symbol_signals.append("MACD Bullish")
                                        else:
                                            symbol_signals.append("MACD Bearish")
                                    
                                    # Moving average signals
                                    if 'Price vs Moving Average' in signal_types and tech_data['price_vs_sma20']:
                                        if tech_data['price_vs_sma20'] > 5:
                                            symbol_signals.append("Above SMA20")
                                        elif tech_data['price_vs_sma20'] < -5:
                                            symbol_signals.append("Below SMA20")
                                    
                                    # Volume analysis
                                    if 'Volume Analysis' in signal_types and tech_data['volume'] and tech_data['avg_volume']:
                                        volume_ratio = tech_data['volume'] / tech_data['avg_volume']
                                        if volume_ratio > 1.5:
                                            symbol_signals.append("High Volume")
                                        elif volume_ratio < 0.5:
                                            symbol_signals.append("Low Volume")
                                    
                                    signals_results.append({
                                        'Symbol': symbol,
                                        'Price': f"${tech_data['current_price']:.2f}",
                                        'RSI': f"{tech_data['rsi']:.1f}" if tech_data['rsi'] else "N/A",
                                        'MACD': "Bull" if tech_data['macd'] and tech_data['macd_signal'] and tech_data['macd'] > tech_data['macd_signal'] else "Bear",
                                        'Signals': ', '.join(symbol_signals) if symbol_signals else 'No Signals'
                                    })
                            
                            except Exception as e:
                                print(f"Error scanning {symbol}: {e}")
                                continue
                        
                        if signals_results:
                            st.markdown("#### Technical Signals Results")
                            
                            signals_df = pd.DataFrame(signals_results)
                            st.dataframe(signals_df, hide_index=True)
                            
                            # Signal summary
                            total_signals = sum(1 for result in signals_results if result['Signals'] != 'No Signals')
                            st.info(f"Found signals in {total_signals} out of {len(signals_results)} stocks scanned.")
                            
                            # Highlight significant signals
                            overbought_stocks = [r['Symbol'] for r in signals_results if 'RSI Overbought' in r['Signals']]
                            oversold_stocks = [r['Symbol'] for r in signals_results if 'RSI Oversold' in r['Signals']]
                            
                            if overbought_stocks:
                                st.warning(f"⚠️ **Overbought Stocks:** {', '.join(overbought_stocks)}")
                            
                            if oversold_stocks:
                                st.success(f"💡 **Oversold Stocks:** {', '.join(oversold_stocks)}")
                        
                        else:
                            st.warning("No valid data found for the provided symbols.")
                    
                    except Exception as e:
                        st.error(f"Error during signal scanning: {str(e)}")
            else:
                st.warning("Please enter at least one stock symbol.")
    
    # Information footer
    st.markdown("---")
    st.info("""
    💡 **Market Dashboard Features:**
    - Real-time market data and indices
    - Sector performance analysis
    - Global markets (crypto & forex)
    - Individual stock fundamental and technical analysis
    - Alternative data integration (social sentiment, ESG scores)
    - Technical signals scanner
    """)


def page_ensemble_models():
    """Advanced ensemble forecasting with multiple models."""
    st.markdown('<div class="main-header">🤖 Ensemble Models</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Advanced ensemble forecasting combining multiple machine learning models 
    for improved accuracy and robustness. Choose from voting, stacking, and 
    weighted ensemble methods.
    """)
    
    # Data input section
    st.markdown("### 📊 Data Selection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ensemble_symbol = st.text_input(
            "Stock Symbol:", 
            value="AAPL", 
            key="ensemble_symbol",
            help="Enter a stock ticker symbol"
        ).upper()
    
    with col2:
        ensemble_period = st.selectbox(
            "Data Period:",
            ["1y", "2y", "5y", "max"],
            index=1,
            key="ensemble_period"
        )
    
    if st.button("Load Data for Ensemble Analysis", type="primary"):
        if ensemble_symbol:
            with st.spinner(f"Loading {ensemble_symbol} data for ensemble analysis..."):
                try:
                    # Load data
                    data = get_ticker_data(ensemble_symbol, period=ensemble_period)
                    
                    if data is not None and len(data) > 100:
                        st.session_state.ensemble_data = data
                        st.session_state.ensemble_symbol = ensemble_symbol
                        st.success(f"Loaded {len(data)} data points for {ensemble_symbol}")
                        
                        # Display basic data info
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Data Points", len(data))
                        with col2:
                            st.metric("Date Range", f"{(data.index[-1] - data.index[0]).days} days")
                        with col3:
                            current_price = data['Close'].iloc[-1]
                            st.metric("Current Price", f"${current_price:.2f}")
                        with col4:
                            volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
                            st.metric("Annual Volatility", f"{volatility:.1f}%")
                    
                    else:
                        st.error(f"Insufficient data for {ensemble_symbol}. Need at least 100 data points.")
                
                except Exception as e:
                    st.error(f"Error loading data for {ensemble_symbol}: {str(e)}")
        else:
            st.warning("Please enter a stock symbol.")
    
    # Ensemble configuration and analysis
    if 'ensemble_data' in st.session_state and st.session_state.ensemble_data is not None:
        st.markdown("---")
        st.markdown("### 🎛️ Ensemble Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Ensemble Methods")
            
            use_voting = st.checkbox(
                "Voting Ensemble", 
                value=True, 
                help="Combine predictions using voting"
            )
            
            use_stacking = st.checkbox(
                "Stacking Ensemble", 
                value=True, 
                help="Use meta-learner for stacking"
            )
            
            use_blending = st.checkbox(
                "Weighted Average", 
                value=True, 
                help="Performance-weighted ensemble"
            )
            
            st.markdown("#### Model Selection")
            
            available_models = [
                'Linear Regression', 'Ridge Regression', 'Lasso Regression',
                'Elastic Net', 'Random Forest', 'Gradient Boosting',
                'Support Vector Regression', 'Neural Network'
            ]
            
            selected_models = st.multiselect(
                "Select Base Models:",
                available_models,
                default=available_models[:6],
                help="Choose models to include in ensemble"
            )
        
        with col2:
            st.markdown("#### Forecast Parameters")
            
            forecast_periods = st.slider(
                "Forecast Periods",
                min_value=7,
                max_value=90,
                value=30,
                help="Number of days to forecast"
            )
            
            confidence_level = st.slider(
                "Confidence Level",
                min_value=0.80,
                max_value=0.99,
                value=0.95,
                step=0.01,
                help="Confidence level for prediction intervals"
            )
            
            lookback_window = st.slider(
                "Lookback Window",
                min_value=10,
                max_value=60,
                value=20,
                help="Number of past observations for features"
            )
            
            weight_method = st.selectbox(
                "Ensemble Weight Method:",
                ["performance_based", "equal", "inverse_variance"],
                help="Method for calculating ensemble weights"
            )
        
        # Run ensemble analysis
        if st.button("🚀 Run Ensemble Analysis", type="primary"):
            if selected_models:
                with st.spinner("Training ensemble models and generating forecasts..."):
                    try:
                        # Initialize ensemble
                        ensemble = AdvancedEnsemble(
                            use_stacking=use_stacking,
                            use_voting=use_voting,
                            use_blending=use_blending
                        )
                        
                        # Use closing prices for training
                        price_data = st.session_state.ensemble_data['Close']
                        
                        # Train ensemble
                        ensemble.fit(price_data, lookback_window=lookback_window)
                        
                        # Generate forecasts
                        forecast_results = ensemble.forecast(
                            price_data, 
                            periods=forecast_periods, 
                            confidence_interval=confidence_level
                        )
                        
                        st.session_state.ensemble_results = forecast_results
                        st.session_state.ensemble_model = ensemble
                        
                        st.success("Ensemble analysis completed successfully!")
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("### 📈 Ensemble Forecast Results")
                        
                        # Create forecast visualization
                        fig = go.Figure()
                        
                        # Historical data
                        recent_data = price_data.tail(60)  # Last 60 days
                        fig.add_trace(go.Scatter(
                            x=recent_data.index,
                            y=recent_data.values,
                            mode='lines',
                            name='Historical',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Forecast
                        fig.add_trace(go.Scatter(
                            x=forecast_results['dates'],
                            y=forecast_results['forecast'],
                            mode='lines',
                            name='Ensemble Forecast',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                        
                        # Confidence intervals
                        fig.add_trace(go.Scatter(
                            x=list(forecast_results['dates']) + list(forecast_results['dates'][::-1]),
                            y=list(forecast_results['upper_bound']) + list(forecast_results['lower_bound'][::-1]),
                            fill='toself',
                            fillcolor='rgba(255, 0, 0, 0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name=f'{confidence_level:.0%} Confidence Interval',
                            showlegend=True
                        ))
                        
                        fig.update_layout(
                            title=f"Ensemble Forecast for {st.session_state.ensemble_symbol}",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            height=500,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        current_price = price_data.iloc[-1]
                        forecast_price = forecast_results['forecast'][-1]
                        price_change = ((forecast_price / current_price) - 1) * 100
                        
                        with col1:
                            st.metric("Current Price", f"${current_price:.2f}")
                        with col2:
                            st.metric(
                                f"Forecast ({forecast_periods}d)", 
                                f"${forecast_price:.2f}",
                                f"{price_change:+.1f}%"
                            )
                        with col3:
                            forecast_range = forecast_results['upper_bound'][-1] - forecast_results['lower_bound'][-1]
                            st.metric("Forecast Range", f"${forecast_range:.2f}")
                        with col4:
                            avg_confidence_width = np.mean(forecast_results['upper_bound'] - forecast_results['lower_bound'])
                            st.metric("Avg Confidence Width", f"${avg_confidence_width:.2f}")
                        
                        # Model performance comparison
                        st.markdown("### 🏆 Model Performance Comparison")
                        
                        model_importance = ensemble.get_model_importance()
                        
                        # Model performance chart
                        fig_models = go.Figure()
                        
                        fig_models.add_trace(go.Bar(
                            x=model_importance['Model'],
                            y=model_importance['Weight'],
                            name='Ensemble Weight',
                            marker_color='lightblue',
                            yaxis='y'
                        ))
                        
                        fig_models.add_trace(go.Scatter(
                            x=model_importance['Model'],
                            y=1.0 / (model_importance['MSE'] + 1e-8),  # Inverse MSE for better visualization
                            mode='markers+lines',
                            name='Performance Score',
                            marker=dict(size=10, color='red'),
                            yaxis='y2'
                        ))
                        
                        fig_models.update_layout(
                            title="Model Weights vs Performance",
                            xaxis_title="Model",
                            yaxis=dict(title="Ensemble Weight", side='left'),
                            yaxis2=dict(title="Performance Score", side='right', overlaying='y'),
                            height=400
                        )
                        
                        st.plotly_chart(fig_models, use_container_width=True)
                        
                        # Model performance table
                        st.markdown("#### Model Performance Details")
                        
                        performance_df = model_importance.copy()
                        performance_df['Weight'] = performance_df['Weight'].apply(lambda x: f"{x:.3f}")
                        performance_df['MSE'] = performance_df['MSE'].apply(lambda x: f"{x:.6f}" if x != np.inf else "N/A")
                        performance_df['MSE_Std'] = performance_df['MSE_Std'].apply(lambda x: f"{x:.6f}" if x != np.inf else "N/A")
                        performance_df['Importance_Score'] = performance_df['Importance_Score'].apply(lambda x: f"{x:.6f}")
                        
                        st.dataframe(performance_df, hide_index=True)
                        
                        # Ensemble summary
                        ensemble_summary = ensemble.get_ensemble_summary()
                        
                        st.markdown("### 📊 Ensemble Summary")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.info(f"**Total Models:** {ensemble_summary['total_models']}")
                            st.info(f"**Fitted Models:** {ensemble_summary['fitted_models']}")
                            st.info(f"**Ensemble Methods:** {', '.join(ensemble_summary['ensemble_methods'])}")
                        
                        with col2:
                            if ensemble_summary['best_model']:
                                st.success(f"**Best Model:** {ensemble_summary['best_model'].upper()}")
                            if ensemble_summary['worst_model']:
                                st.error(f"**Worst Model:** {ensemble_summary['worst_model'].upper()}")
                            if ensemble_summary['average_performance']:
                                st.info(f"**Average MSE:** {ensemble_summary['average_performance']:.6f}")
                    
                    except Exception as e:
                        st.error(f"Error in ensemble analysis: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            else:
                st.warning("Please select at least one base model for the ensemble.")
    
    else:
        st.info("👆 Load data to begin ensemble analysis")
    
    # Feature explanations
    st.markdown("---")
    st.markdown("### 💡 Ensemble Methods Explained")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🗳️ Voting Ensemble**
        - Combines predictions from multiple models
        - Each model gets equal vote by default
        - Can use weighted voting based on performance
        - Reduces variance and improves stability
        """)
    
    with col2:
        st.markdown("""
        **🏗️ Stacking Ensemble**
        - Uses meta-learner to combine base models
        - Base models trained on original data
        - Meta-learner learns optimal combination
        - Often achieves best performance
        """)
    
    with col3:
        st.markdown("""
        **⚖️ Weighted Average**
        - Weights models by their performance
        - Better models get higher weights
        - Automatic weight calculation
        - Balances accuracy and simplicity
        """)
    
    st.markdown("""
    ### 🎯 Model Features
    
    **Base Models Include:**
    - **Linear Models:** Linear, Ridge, Lasso, Elastic Net regression
    - **Tree Models:** Random Forest, Gradient Boosting
    - **Non-linear:** Support Vector Regression, Neural Networks
    
    **Advanced Features:**
    - Automatic feature engineering (technical indicators)
    - Cross-validation for robust evaluation
    - Performance-based weighting
    - Confidence interval estimation
    - Model importance analysis
    """)


def page_advanced_settings():
    """Advanced system configuration and settings page."""
    st.markdown('<div class="main-header">⚙️ Advanced Settings</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Configure advanced system settings including model parameters, data sources, 
    risk management settings, and user interface preferences.
    """)
    
    # Create tabs for different setting categories
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🤖 Model Settings", 
        "📊 Data Settings", 
        "⚖️ Risk Settings", 
        "🎨 UI Settings", 
        "🔌 API Settings", 
        "🔧 System Settings"
    ])
    
    with tab1:
        st.markdown("### Model Configuration")
        st.markdown("Configure forecasting models and ensemble settings.")
        
        model_config = config_manager.get_config('model')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### General Model Settings")
            
            new_periods = st.number_input(
                "Default Forecast Periods", 
                min_value=1, 
                max_value=365, 
                value=model_config.default_periods,
                help="Default number of periods to forecast"
            )
            
            new_confidence = st.slider(
                "Confidence Interval", 
                min_value=0.8, 
                max_value=0.999, 
                value=model_config.confidence_interval,
                step=0.01,
                help="Confidence level for forecast intervals"
            )
            
            new_cv_folds = st.number_input(
                "Cross-Validation Folds", 
                min_value=2, 
                max_value=10, 
                value=model_config.cross_validation_folds,
                help="Number of folds for cross-validation"
            )
            
            st.markdown("#### ARIMA Settings")
            
            new_arima_auto = st.checkbox(
                "Auto ARIMA", 
                value=model_config.arima_auto,
                help="Automatically select ARIMA parameters"
            )
            
            if not new_arima_auto:
                col_p, col_d, col_q = st.columns(3)
                
                with col_p:
                    new_arima_max_p = st.number_input("Max P", min_value=0, max_value=10, value=model_config.arima_max_p)
                with col_d:
                    new_arima_max_d = st.number_input("Max D", min_value=0, max_value=5, value=model_config.arima_max_d)
                with col_q:
                    new_arima_max_q = st.number_input("Max Q", min_value=0, max_value=10, value=model_config.arima_max_q)
            else:
                new_arima_max_p = model_config.arima_max_p
                new_arima_max_d = model_config.arima_max_d
                new_arima_max_q = model_config.arima_max_q
            
            new_arima_seasonal = st.checkbox(
                "Seasonal ARIMA", 
                value=model_config.arima_seasonal,
                help="Include seasonal components"
            )
            
            if new_arima_seasonal:
                new_arima_m = st.number_input(
                    "Seasonal Periods", 
                    min_value=2, 
                    max_value=52, 
                    value=model_config.arima_m,
                    help="Number of periods in a season"
                )
            else:
                new_arima_m = model_config.arima_m
        
        with col2:
            st.markdown("#### Prophet Settings")
            
            new_prophet_growth = st.selectbox(
                "Growth Model", 
                ["linear", "logistic"], 
                index=0 if model_config.prophet_growth == "linear" else 1,
                help="Prophet growth model type"
            )
            
            new_prophet_yearly = st.checkbox(
                "Yearly Seasonality", 
                value=model_config.prophet_yearly_seasonality
            )
            
            new_prophet_weekly = st.checkbox(
                "Weekly Seasonality", 
                value=model_config.prophet_weekly_seasonality
            )
            
            new_prophet_daily = st.checkbox(
                "Daily Seasonality", 
                value=model_config.prophet_daily_seasonality
            )
            
            new_prophet_changepoint = st.slider(
                "Changepoint Prior Scale", 
                min_value=0.001, 
                max_value=0.5, 
                value=model_config.prophet_changepoint_prior_scale,
                step=0.001,
                help="Flexibility of trend changes"
            )
            
            new_prophet_seasonality = st.slider(
                "Seasonality Prior Scale", 
                min_value=0.1, 
                max_value=50.0, 
                value=model_config.prophet_seasonality_prior_scale,
                step=0.1,
                help="Strength of seasonality"
            )
            
            st.markdown("#### LSTM Settings")
            
            new_lstm_units = st.number_input(
                "LSTM Units", 
                min_value=10, 
                max_value=200, 
                value=model_config.lstm_units,
                help="Number of LSTM units per layer"
            )
            
            new_lstm_layers = st.number_input(
                "LSTM Layers", 
                min_value=1, 
                max_value=5, 
                value=model_config.lstm_layers,
                help="Number of LSTM layers"
            )
            
            new_lstm_dropout = st.slider(
                "Dropout Rate", 
                min_value=0.0, 
                max_value=0.5, 
                value=model_config.lstm_dropout,
                step=0.05,
                help="Dropout rate for regularization"
            )
            
            new_lstm_epochs = st.number_input(
                "Training Epochs", 
                min_value=10, 
                max_value=500, 
                value=model_config.lstm_epochs,
                help="Number of training epochs"
            )
            
            new_lstm_batch_size = st.number_input(
                "Batch Size", 
                min_value=8, 
                max_value=128, 
                value=model_config.lstm_batch_size,
                help="Training batch size"
            )
            
            new_lstm_lookback = st.number_input(
                "Lookback Window", 
                min_value=10, 
                max_value=200, 
                value=model_config.lstm_lookback,
                help="Number of past observations to use"
            )
        
        st.markdown("#### Ensemble Settings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            new_ensemble_voting = st.checkbox(
                "Use Voting Ensemble", 
                value=model_config.ensemble_use_voting,
                help="Combine models using voting"
            )
        
        with col2:
            new_ensemble_stacking = st.checkbox(
                "Use Stacking Ensemble", 
                value=model_config.ensemble_use_stacking,
                help="Use meta-learner for stacking"
            )
        
        with col3:
            new_ensemble_blending = st.checkbox(
                "Use Blending Ensemble", 
                value=model_config.ensemble_use_blending,
                help="Blend multiple model predictions"
            )
        
        new_ensemble_weight_method = st.selectbox(
            "Ensemble Weight Method",
            ["performance_based", "equal", "inverse_variance"],
            index=["performance_based", "equal", "inverse_variance"].index(model_config.ensemble_weight_method),
            help="Method for calculating ensemble weights"
        )
        
        # Update model configuration
        if st.button("Save Model Settings", type="primary"):
            config_manager.update_config('model',
                default_periods=new_periods,
                confidence_interval=new_confidence,
                cross_validation_folds=new_cv_folds,
                arima_auto=new_arima_auto,
                arima_max_p=new_arima_max_p,
                arima_max_d=new_arima_max_d,
                arima_max_q=new_arima_max_q,
                arima_seasonal=new_arima_seasonal,
                arima_m=new_arima_m,
                prophet_growth=new_prophet_growth,
                prophet_yearly_seasonality=new_prophet_yearly,
                prophet_weekly_seasonality=new_prophet_weekly,
                prophet_daily_seasonality=new_prophet_daily,
                prophet_changepoint_prior_scale=new_prophet_changepoint,
                prophet_seasonality_prior_scale=new_prophet_seasonality,
                lstm_units=new_lstm_units,
                lstm_layers=new_lstm_layers,
                lstm_dropout=new_lstm_dropout,
                lstm_epochs=new_lstm_epochs,
                lstm_batch_size=new_lstm_batch_size,
                lstm_lookback=new_lstm_lookback,
                ensemble_use_voting=new_ensemble_voting,
                ensemble_use_stacking=new_ensemble_stacking,
                ensemble_use_blending=new_ensemble_blending,
                ensemble_weight_method=new_ensemble_weight_method
            )
            config_manager.save_config()
            st.success("Model settings saved successfully!")
    
    with tab2:
        st.markdown("### Data Configuration")
        st.markdown("Configure data sources and processing parameters.")
        
        data_config = config_manager.get_config('data')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Data Sources")
            
            new_default_source = st.selectbox(
                "Default Data Source",
                ["yfinance", "alpha_vantage", "quandl"],
                index=["yfinance", "alpha_vantage", "quandl"].index(data_config.default_source),
                help="Primary data source for market data"
            )
            
            new_backup_sources = st.multiselect(
                "Backup Sources",
                ["yfinance", "alpha_vantage", "quandl"],
                default=data_config.backup_sources,
                help="Fallback data sources"
            )
            
            st.markdown("#### Data Processing")
            
            new_fill_method = st.selectbox(
                "Missing Data Handling",
                ["forward", "backward", "interpolate", "drop"],
                index=["forward", "backward", "interpolate", "drop"].index(data_config.fill_method),
                help="Method for handling missing data"
            )
            
            new_outlier_detection = st.checkbox(
                "Enable Outlier Detection",
                value=data_config.outlier_detection,
                help="Automatically detect and handle outliers"
            )
            
            if new_outlier_detection:
                new_outlier_method = st.selectbox(
                    "Outlier Detection Method",
                    ["iqr", "zscore", "isolation_forest"],
                    index=["iqr", "zscore", "isolation_forest"].index(data_config.outlier_method),
                    help="Statistical method for outlier detection"
                )
                
                new_outlier_threshold = st.slider(
                    "Outlier Threshold",
                    min_value=1.0,
                    max_value=5.0,
                    value=data_config.outlier_threshold,
                    step=0.1,
                    help="Threshold for outlier detection"
                )
            else:
                new_outlier_method = data_config.outlier_method
                new_outlier_threshold = data_config.outlier_threshold
        
        with col2:
            st.markdown("#### Data Limits")
            
            new_default_frequency = st.selectbox(
                "Default Frequency",
                ["D", "W", "M", "Q", "Y"],
                index=["D", "W", "M", "Q", "Y"].index(data_config.default_frequency),
                help="Default data frequency"
            )
            
            new_min_data_points = st.number_input(
                "Minimum Data Points",
                min_value=50,
                max_value=1000,
                value=data_config.min_data_points,
                help="Minimum required data points for analysis"
            )
            
            new_max_data_points = st.number_input(
                "Maximum Data Points",
                min_value=1000,
                max_value=50000,
                value=data_config.max_data_points,
                help="Maximum data points to load"
            )
            
            st.markdown("#### Market Data Settings")
            
            new_market_hours_only = st.checkbox(
                "Market Hours Only",
                value=data_config.market_hours_only,
                help="Only use data from market hours"
            )
            
            new_adjust_splits = st.checkbox(
                "Adjust for Stock Splits",
                value=data_config.adjust_splits,
                help="Adjust prices for stock splits"
            )
            
            new_adjust_dividends = st.checkbox(
                "Adjust for Dividends",
                value=data_config.adjust_dividends,
                help="Adjust prices for dividends"
            )
        
        # Update data configuration
        if st.button("Save Data Settings", type="primary"):
            config_manager.update_config('data',
                default_source=new_default_source,
                backup_sources=new_backup_sources,
                fill_method=new_fill_method,
                outlier_detection=new_outlier_detection,
                outlier_method=new_outlier_method,
                outlier_threshold=new_outlier_threshold,
                default_frequency=new_default_frequency,
                min_data_points=new_min_data_points,
                max_data_points=new_max_data_points,
                market_hours_only=new_market_hours_only,
                adjust_splits=new_adjust_splits,
                adjust_dividends=new_adjust_dividends
            )
            config_manager.save_config()
            st.success("Data settings saved successfully!")
    
    with tab3:
        st.markdown("### Risk Management Configuration")
        st.markdown("Configure risk analysis and stress testing parameters.")
        
        risk_config = config_manager.get_config('risk')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Value at Risk (VaR)")
            
            current_var_levels = risk_config.var_confidence_levels
            new_var_levels_str = st.text_input(
                "VaR Confidence Levels (comma-separated)",
                value=", ".join([str(x) for x in current_var_levels]),
                help="Confidence levels for VaR calculation (e.g., 0.95, 0.99)"
            )
            
            try:
                new_var_levels = [float(x.strip()) for x in new_var_levels_str.split(",")]
            except:
                new_var_levels = current_var_levels
                st.error("Invalid VaR confidence levels format")
            
            new_var_methods = st.multiselect(
                "VaR Methods",
                ["historical", "parametric", "monte_carlo"],
                default=risk_config.var_methods,
                help="Methods for VaR calculation"
            )
            
            new_var_lookback = st.number_input(
                "VaR Lookback Days",
                min_value=30,
                max_value=1000,
                value=risk_config.var_lookback_days,
                help="Historical data window for VaR"
            )
            
            st.markdown("#### Monte Carlo Settings")
            
            new_mc_simulations = st.number_input(
                "Monte Carlo Simulations",
                min_value=1000,
                max_value=100000,
                value=risk_config.mc_simulations,
                step=1000,
                help="Number of Monte Carlo simulations"
            )
            
            new_mc_horizon = st.number_input(
                "Time Horizon (Days)",
                min_value=1,
                max_value=1000,
                value=risk_config.mc_time_horizon,
                help="Time horizon for Monte Carlo analysis"
            )
        
        with col2:
            st.markdown("#### Portfolio Metrics")
            
            new_benchmark = st.text_input(
                "Benchmark Symbol",
                value=risk_config.benchmark_symbol,
                help="Benchmark index for comparison (e.g., ^GSPC for S&P 500)"
            )
            
            new_risk_free_rate = st.slider(
                "Risk-Free Rate",
                min_value=0.0,
                max_value=0.1,
                value=risk_config.risk_free_rate,
                step=0.001,
                format="%.3f",
                help="Annual risk-free rate for Sharpe ratio calculation"
            )
            
            st.markdown("#### Stress Testing")
            
            # Display current stress scenarios
            st.markdown("**Current Stress Scenarios:**")
            
            for scenario_name, scenario_values in risk_config.stress_scenarios.items():
                with st.expander(f"📊 {scenario_name}"):
                    for asset_class, shock in scenario_values.items():
                        st.write(f"• **{asset_class.title()}:** {shock:+.1%}")
            
            # Stress shock sizes
            current_shock_sizes = risk_config.stress_shock_sizes
            new_shock_sizes_str = st.text_input(
                "Stress Test Shock Sizes (comma-separated)",
                value=", ".join([str(x) for x in current_shock_sizes]),
                help="Shock sizes for stress testing (e.g., 0.01, 0.05, 0.10)"
            )
            
            try:
                new_shock_sizes = [float(x.strip()) for x in new_shock_sizes_str.split(",")]
            except:
                new_shock_sizes = current_shock_sizes
                st.error("Invalid shock sizes format")
        
        # Update risk configuration
        if st.button("Save Risk Settings", type="primary"):
            config_manager.update_config('risk',
                var_confidence_levels=new_var_levels,
                var_methods=new_var_methods,
                var_lookback_days=new_var_lookback,
                mc_simulations=new_mc_simulations,
                mc_time_horizon=new_mc_horizon,
                benchmark_symbol=new_benchmark,
                risk_free_rate=new_risk_free_rate,
                stress_shock_sizes=new_shock_sizes
            )
            config_manager.save_config()
            st.success("Risk settings saved successfully!")
    
    with tab4:
        st.markdown("### User Interface Configuration")
        st.markdown("Customize the appearance and behavior of the application.")
        
        ui_config = config_manager.get_config('ui')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Theme and Styling")
            
            new_theme = st.selectbox(
                "Theme",
                ["light", "dark", "auto"],
                index=["light", "dark", "auto"].index(ui_config.theme),
                help="Application theme"
            )
            
            new_color_scheme = st.selectbox(
                "Color Scheme",
                ["blue", "green", "red", "purple", "orange"],
                index=["blue", "green", "red", "purple", "orange"].index(ui_config.color_scheme),
                help="Primary color scheme"
            )
            
            new_chart_style = st.selectbox(
                "Chart Style",
                ["plotly_white", "plotly", "plotly_dark", "ggplot2"],
                index=["plotly_white", "plotly", "plotly_dark", "ggplot2"].index(ui_config.chart_style),
                help="Default chart styling"
            )
            
            st.markdown("#### Layout Settings")
            
            new_sidebar_width = st.slider(
                "Sidebar Width",
                min_value=200,
                max_value=500,
                value=ui_config.sidebar_width,
                step=10,
                help="Width of the navigation sidebar"
            )
            
            new_main_content = st.selectbox(
                "Main Content Layout",
                ["wide", "centered"],
                index=["wide", "centered"].index(ui_config.main_content_width),
                help="Layout for main content area"
            )
            
            new_show_tooltips = st.checkbox(
                "Show Tooltips",
                value=ui_config.show_tooltips,
                help="Display helpful tooltips"
            )
            
            new_show_help = st.checkbox(
                "Show Help Text",
                value=ui_config.show_help_text,
                help="Display contextual help text"
            )
        
        with col2:
            st.markdown("#### Chart Settings")
            
            new_chart_height = st.slider(
                "Default Chart Height",
                min_value=300,
                max_value=800,
                value=ui_config.default_chart_height,
                step=50,
                help="Default height for charts in pixels"
            )
            
            new_interactive_charts = st.checkbox(
                "Interactive Charts",
                value=ui_config.interactive_charts,
                help="Enable interactive chart features"
            )
            
            new_show_grid = st.checkbox(
                "Show Grid Lines",
                value=ui_config.show_grid,
                help="Display grid lines on charts"
            )
            
            new_show_legend = st.checkbox(
                "Show Chart Legends",
                value=ui_config.show_legend,
                help="Display legends on charts"
            )
            
            st.markdown("#### Data Display")
            
            new_max_table_rows = st.number_input(
                "Maximum Table Rows",
                min_value=100,
                max_value=10000,
                value=ui_config.max_table_rows,
                step=100,
                help="Maximum rows to display in data tables"
            )
            
            new_decimal_places = st.number_input(
                "Decimal Places",
                min_value=0,
                max_value=10,
                value=ui_config.decimal_places,
                help="Number of decimal places for numeric display"
            )
            
            new_currency_symbol = st.text_input(
                "Currency Symbol",
                value=ui_config.currency_symbol,
                help="Symbol to use for currency display"
            )
        
        # Update UI configuration
        if st.button("Save UI Settings", type="primary"):
            config_manager.update_config('ui',
                theme=new_theme,
                color_scheme=new_color_scheme,
                chart_style=new_chart_style,
                sidebar_width=new_sidebar_width,
                main_content_width=new_main_content,
                show_tooltips=new_show_tooltips,
                show_help_text=new_show_help,
                default_chart_height=new_chart_height,
                interactive_charts=new_interactive_charts,
                show_grid=new_show_grid,
                show_legend=new_show_legend,
                max_table_rows=new_max_table_rows,
                decimal_places=new_decimal_places,
                currency_symbol=new_currency_symbol
            )
            config_manager.save_config()
            st.success("UI settings saved successfully!")
            st.info("Some UI changes may require a page refresh to take effect.")
    
    with tab5:
        st.markdown("### API Configuration")
        st.markdown("Configure external API connections and settings.")
        
        api_config = config_manager.get_config('api')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### API Keys")
            st.info("API keys are loaded from environment variables for security.")
            
            # Display API key status
            api_status = [
                ("Alpha Vantage", api_config.alpha_vantage_key, "ALPHA_VANTAGE_API_KEY"),
                ("Quandl", api_config.quandl_key, "QUANDL_API_KEY"),
                ("FRED", api_config.fred_key, "FRED_API_KEY"),
                ("News API", api_config.news_api_key, "NEWS_API_KEY")
            ]
            
            for name, key, env_var in api_status:
                status = "✅ Configured" if key else "❌ Not Set"
                color = "success" if key else "error"
                st.write(f"**{name}:** {status}")
                
                if not key:
                    st.caption(f"Set environment variable: {env_var}")
            
            st.markdown("#### Rate Limiting")
            
            new_requests_per_minute = st.number_input(
                "Requests per Minute",
                min_value=1,
                max_value=300,
                value=api_config.requests_per_minute,
                help="Maximum API requests per minute"
            )
            
            new_retry_attempts = st.number_input(
                "Retry Attempts",
                min_value=1,
                max_value=10,
                value=api_config.retry_attempts,
                help="Number of retry attempts for failed requests"
            )
            
            new_retry_delay = st.slider(
                "Retry Delay (seconds)",
                min_value=0.1,
                max_value=5.0,
                value=api_config.retry_delay,
                step=0.1,
                help="Delay between retry attempts"
            )
        
        with col2:
            st.markdown("#### Timeout Settings")
            
            new_connection_timeout = st.number_input(
                "Connection Timeout (seconds)",
                min_value=1,
                max_value=60,
                value=api_config.connection_timeout,
                help="Timeout for establishing connections"
            )
            
            new_read_timeout = st.number_input(
                "Read Timeout (seconds)",
                min_value=1,
                max_value=120,
                value=api_config.read_timeout,
                help="Timeout for reading responses"
            )
            
            st.markdown("#### Cache Settings")
            
            new_enable_cache = st.checkbox(
                "Enable API Cache",
                value=api_config.enable_cache,
                help="Cache API responses to reduce requests"
            )
            
            if new_enable_cache:
                new_cache_directory = st.text_input(
                    "Cache Directory",
                    value=api_config.cache_directory,
                    help="Directory to store cached responses"
                )
                
                new_cache_expiry = st.number_input(
                    "Cache Expiry (hours)",
                    min_value=1,
                    max_value=168,
                    value=api_config.cache_expiry_hours,
                    help="Time before cached data expires"
                )
            else:
                new_cache_directory = api_config.cache_directory
                new_cache_expiry = api_config.cache_expiry_hours
            
            st.markdown("#### Data Quality")
            
            new_validate_data = st.checkbox(
                "Validate API Data",
                value=api_config.validate_data,
                help="Validate data quality from API responses"
            )
            
            if new_validate_data:
                new_min_quality_score = st.slider(
                    "Minimum Quality Score",
                    min_value=0.0,
                    max_value=1.0,
                    value=api_config.min_data_quality_score,
                    step=0.1,
                    help="Minimum acceptable data quality score"
                )
            else:
                new_min_quality_score = api_config.min_data_quality_score
            
            new_auto_fallback = st.checkbox(
                "Auto Fallback",
                value=api_config.auto_fallback,
                help="Automatically fallback to backup APIs"
            )
        
        # Update API configuration
        if st.button("Save API Settings", type="primary"):
            config_manager.update_config('api',
                requests_per_minute=new_requests_per_minute,
                retry_attempts=new_retry_attempts,
                retry_delay=new_retry_delay,
                connection_timeout=new_connection_timeout,
                read_timeout=new_read_timeout,
                enable_cache=new_enable_cache,
                cache_directory=new_cache_directory,
                cache_expiry_hours=new_cache_expiry,
                validate_data=new_validate_data,
                min_data_quality_score=new_min_quality_score,
                auto_fallback=new_auto_fallback
            )
            config_manager.save_config()
            st.success("API settings saved successfully!")
    
    with tab6:
        st.markdown("### System Configuration")
        st.markdown("Configure system-wide settings and performance parameters.")
        
        system_config = config_manager.get_config('system')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Logging")
            
            new_log_level = st.selectbox(
                "Log Level",
                ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                index=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].index(system_config.log_level),
                help="Minimum log level to record"
            )
            
            new_log_rotation = st.checkbox(
                "Log Rotation",
                value=system_config.log_rotation,
                help="Automatically rotate log files"
            )
            
            if new_log_rotation:
                new_max_log_size = st.text_input(
                    "Maximum Log Size",
                    value=system_config.max_log_size,
                    help="Maximum size before log rotation (e.g., 10MB)"
                )
            else:
                new_max_log_size = system_config.max_log_size
            
            st.markdown("#### Performance")
            
            new_max_workers = st.number_input(
                "Maximum Workers",
                min_value=1,
                max_value=16,
                value=system_config.max_workers,
                help="Maximum number of worker threads"
            )
            
            new_memory_limit = st.text_input(
                "Memory Limit",
                value=system_config.memory_limit,
                help="Maximum memory usage (e.g., 4GB)"
            )
            
            new_enable_gpu = st.checkbox(
                "Enable GPU",
                value=system_config.enable_gpu,
                help="Use GPU acceleration if available"
            )
        
        with col2:
            st.markdown("#### Storage")
            
            new_data_dir = st.text_input(
                "Data Directory",
                value=system_config.data_directory,
                help="Directory for storing data files"
            )
            
            new_models_dir = st.text_input(
                "Models Directory",
                value=system_config.models_directory,
                help="Directory for storing trained models"
            )
            
            new_exports_dir = st.text_input(
                "Exports Directory",
                value=system_config.exports_directory,
                help="Directory for exported files"
            )
            
            st.markdown("#### Security")
            
            new_enable_auth = st.checkbox(
                "Enable Authentication",
                value=system_config.enable_auth,
                help="Require user authentication"
            )
            
            if new_enable_auth:
                new_session_timeout = st.number_input(
                    "Session Timeout (seconds)",
                    min_value=300,
                    max_value=86400,
                    value=system_config.session_timeout,
                    help="User session timeout duration"
                )
            else:
                new_session_timeout = system_config.session_timeout
            
            st.markdown("#### Updates")
            
            new_check_updates = st.checkbox(
                "Check for Updates",
                value=system_config.check_updates,
                help="Check for system updates"
            )
            
            new_auto_update = st.checkbox(
                "Auto Update",
                value=system_config.auto_update,
                help="Automatically install updates"
            )
        
        # Update system configuration
        if st.button("Save System Settings", type="primary"):
            config_manager.update_config('system',
                log_level=new_log_level,
                log_rotation=new_log_rotation,
                max_log_size=new_max_log_size,
                max_workers=new_max_workers,
                memory_limit=new_memory_limit,
                enable_gpu=new_enable_gpu,
                data_directory=new_data_dir,
                models_directory=new_models_dir,
                exports_directory=new_exports_dir,
                enable_auth=new_enable_auth,
                session_timeout=new_session_timeout,
                check_updates=new_check_updates,
                auto_update=new_auto_update
            )
            config_manager.save_config()
            st.success("System settings saved successfully!")
    
    # Configuration validation and summary
    st.markdown("---")
    st.markdown("### Configuration Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Validation results
        validation = config_manager.validate_config()
        
        if validation['valid']:
            st.success("✅ Configuration is valid")
        else:
            st.error("❌ Configuration has errors")
        
        if validation['warnings']:
            st.warning("⚠️ Configuration warnings:")
            for warning in validation['warnings']:
                st.write(f"• {warning}")
        
        if validation['errors']:
            st.error("🚨 Configuration errors:")
            for error in validation['errors']:
                st.write(f"• {error}")
    
    with col2:
        # Configuration summary
        summary = config_manager.get_summary()
        
        st.info("📊 Configuration Summary:")
        st.write(f"• **Config File:** {summary['config_file']}")
        st.write(f"• **Total Sections:** {len(summary['sections'])}")
        st.write(f"• **API Keys Configured:** {summary['api_keys_configured']}/4")
        
        if st.button("Reset to Defaults", key="reset_all_config"):
            if st.confirm("Are you sure you want to reset all configuration to defaults?"):
                config_manager.reset_to_defaults()
                config_manager.save_config()
                st.success("Configuration reset to defaults!")
                st.rerun()
        
        if st.button("Export Configuration", key="export_config"):
            st.download_button(
                label="Download Config File",
                data=open(config_manager.config_path, 'r').read(),
                file_name="portfolio_config.yaml",
                mime="text/yaml"
            )


def page_help():
    """Help and documentation page."""
    st.markdown('<div class="main-header">Help & Documentation</div>', unsafe_allow_html=True)
    st.info("🚧 This feature will be implemented in the next phase.")


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
    elif current_page == "deep_learning":
        page_deep_learning()
    elif current_page == "portfolio":
        page_portfolio_optimization()
    elif current_page == "backtest":
        page_backtesting()
    elif current_page == "risk":
        page_risk_management()
    elif current_page == "reports":
        page_performance_reporting()
    elif current_page == "market":
        page_market_dashboard()
    elif current_page == "ensemble":
        page_ensemble_models()
    elif current_page == "settings":
        page_advanced_settings()
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
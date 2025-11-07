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
    elif current_page == "deep_learning":
        page_deep_learning()
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
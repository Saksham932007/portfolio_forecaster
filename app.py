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
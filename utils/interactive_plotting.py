"""Advanced interactive plotting utilities for the Portfolio Forecasting System."""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class InteractivePlotter:
    """Advanced interactive plotting class with multiple chart types and customization options."""
    
    def __init__(self, theme: str = "plotly"):
        """
        Initialize the plotter with a theme.
        
        Args:
            theme: Plotly theme ('plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn', etc.)
        """
        self.theme = theme
        self.colors = px.colors.qualitative.Set3
        self.default_height = 500
        
    def create_candlestick_chart(self, data: pd.DataFrame, title: str = "Candlestick Chart") -> go.Figure:
        """
        Create an interactive candlestick chart.
        
        Args:
            data: DataFrame with OHLC data (columns: open, high, low, close)
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure(data=go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name="OHLC"
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template=self.theme,
            height=self.default_height,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def create_multi_asset_comparison(self, data: pd.DataFrame, title: str = "Multi-Asset Comparison", 
                                    normalize: bool = False) -> go.Figure:
        """
        Create a multi-asset comparison chart.
        
        Args:
            data: DataFrame with multiple assets as columns
            title: Chart title
            normalize: Whether to normalize to base 100
            
        Returns:
            Plotly Figure object
        """
        plot_data = data.copy()
        
        if normalize:
            plot_data = plot_data.div(plot_data.iloc[0]) * 100
            ylabel = "Normalized Price (Base 100)"
        else:
            ylabel = "Price ($)"
        
        fig = go.Figure()
        
        for i, column in enumerate(plot_data.columns):
            fig.add_trace(go.Scatter(
                x=plot_data.index,
                y=plot_data[column],
                mode='lines',
                name=column,
                line=dict(color=self.colors[i % len(self.colors)], width=2),
                hovertemplate=f"<b>{column}</b><br>" +
                             "Date: %{x}<br>" +
                             f"Value: %{{y:.2f}}<br>" +
                             "<extra></extra>"
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=ylabel,
            template=self.theme,
            height=self.default_height,
            hovermode='x unified'
        )
        
        return fig
    
    def create_forecast_comparison(self, actual: pd.Series, forecasts: Dict[str, pd.Series], 
                                 title: str = "Forecast Comparison", 
                                 confidence_intervals: Optional[Dict[str, pd.DataFrame]] = None) -> go.Figure:
        """
        Create a forecast comparison chart with confidence intervals.
        
        Args:
            actual: Actual values
            forecasts: Dictionary of model names and their forecasts
            title: Chart title
            confidence_intervals: Optional confidence intervals for each model
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        # Plot actual data
        fig.add_trace(go.Scatter(
            x=actual.index,
            y=actual.values,
            mode='lines',
            name='Actual',
            line=dict(color='black', width=3),
            hovertemplate="<b>Actual</b><br>" +
                         "Date: %{x}<br>" +
                         "Value: %{y:.2f}<br>" +
                         "<extra></extra>"
        ))
        
        # Plot forecasts with confidence intervals
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            color = self.colors[i % len(self.colors)]
            
            # Main forecast line
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast.values,
                mode='lines+markers',
                name=f'{model_name} Forecast',
                line=dict(color=color, width=2, dash='dash'),
                marker=dict(size=4),
                hovertemplate=f"<b>{model_name}</b><br>" +
                             "Date: %{x}<br>" +
                             "Forecast: %{y:.2f}<br>" +
                             "<extra></extra>"
            ))
            
            # Add confidence intervals if available
            if confidence_intervals and model_name in confidence_intervals:
                ci_data = confidence_intervals[model_name]
                
                # Upper bound
                fig.add_trace(go.Scatter(
                    x=forecast.index,
                    y=ci_data.iloc[:, 1],  # Upper bound
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Lower bound with fill
                fig.add_trace(go.Scatter(
                    x=forecast.index,
                    y=ci_data.iloc[:, 0],  # Lower bound
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor=f'rgba({px.colors.hex_to_rgb(color)[0]}, {px.colors.hex_to_rgb(color)[1]}, {px.colors.hex_to_rgb(color)[2]}, 0.2)',
                    name=f'{model_name} CI',
                    hovertemplate=f"<b>{model_name} CI</b><br>" +
                                 "Date: %{x}<br>" +
                                 "Lower: %{y:.2f}<br>" +
                                 "<extra></extra>"
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Value",
            template=self.theme,
            height=self.default_height,
            hovermode='x unified'
        )
        
        return fig
    
    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame, 
                                 title: str = "Correlation Heatmap") -> go.Figure:
        """
        Create an interactive correlation heatmap.
        
        Args:
            correlation_matrix: Correlation matrix DataFrame
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu_r',
            zmid=0,
            text=correlation_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="<b>%{x} vs %{y}</b><br>" +
                         "Correlation: %{z:.3f}<br>" +
                         "<extra></extra>"
        ))
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=self.default_height,
            xaxis_title="Assets",
            yaxis_title="Assets"
        )
        
        return fig
    
    def create_returns_distribution(self, returns_data: pd.DataFrame, 
                                  title: str = "Returns Distribution") -> go.Figure:
        """
        Create a returns distribution comparison.
        
        Args:
            returns_data: DataFrame with returns for different assets
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        for i, column in enumerate(returns_data.columns):
            fig.add_trace(go.Histogram(
                x=returns_data[column].dropna(),
                name=column,
                opacity=0.7,
                nbinsx=50,
                marker_color=self.colors[i % len(self.colors)],
                hovertemplate=f"<b>{column}</b><br>" +
                             "Return: %{x:.3f}<br>" +
                             "Count: %{y}<br>" +
                             "<extra></extra>"
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Returns",
            yaxis_title="Frequency",
            template=self.theme,
            height=self.default_height,
            barmode='overlay'
        )
        
        return fig
    
    def create_risk_return_scatter(self, returns_data: pd.DataFrame, 
                                 title: str = "Risk-Return Analysis") -> go.Figure:
        """
        Create a risk-return scatter plot.
        
        Args:
            returns_data: DataFrame with returns for different assets
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        # Calculate risk and return metrics
        metrics = []
        for column in returns_data.columns:
            asset_returns = returns_data[column].dropna()
            annual_return = asset_returns.mean() * 252
            annual_volatility = asset_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            
            metrics.append({
                'Asset': column,
                'Return': annual_return,
                'Volatility': annual_volatility,
                'Sharpe': sharpe_ratio
            })
        
        metrics_df = pd.DataFrame(metrics)
        
        fig = px.scatter(
            metrics_df,
            x='Volatility',
            y='Return',
            color='Sharpe',
            size='Sharpe',
            hover_data=['Asset'],
            title=title,
            labels={'Volatility': 'Annual Volatility', 'Return': 'Annual Return'},
            color_continuous_scale='Viridis'
        )
        
        # Add asset labels
        for _, row in metrics_df.iterrows():
            fig.add_annotation(
                x=row['Volatility'],
                y=row['Return'],
                text=row['Asset'],
                showarrow=False,
                yshift=10
            )
        
        fig.update_layout(
            template=self.theme,
            height=self.default_height
        )
        
        return fig
    
    def create_performance_comparison(self, metrics_df: pd.DataFrame, 
                                    title: str = "Model Performance Comparison") -> go.Figure:
        """
        Create a model performance comparison chart.
        
        Args:
            metrics_df: DataFrame with performance metrics
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        # Create subplots for different metrics
        metrics_to_plot = ['RMSE', 'MAE', 'MAPE']
        available_metrics = [col for col in metrics_to_plot if col in metrics_df.columns]
        
        if not available_metrics:
            # Fallback to all numeric columns
            available_metrics = metrics_df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_metrics = len(available_metrics)
        
        if n_metrics == 0:
            # Create empty figure if no metrics available
            fig = go.Figure()
            fig.add_annotation(
                text="No numeric metrics available for visualization",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False
            )
            return fig
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=n_metrics,
            subplot_titles=available_metrics,
            horizontal_spacing=0.1
        )
        
        for i, metric in enumerate(available_metrics):
            fig.add_trace(
                go.Bar(
                    x=metrics_df.index,
                    y=metrics_df[metric],
                    name=metric,
                    marker_color=self.colors[i % len(self.colors)],
                    showlegend=False,
                    hovertemplate=f"<b>%{{x}}</b><br>" +
                                 f"{metric}: %{{y:.4f}}<br>" +
                                 "<extra></extra>"
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=self.default_height
        )
        
        return fig
    
    def create_rolling_statistics(self, data: pd.Series, window: int = 30, 
                                title: str = "Rolling Statistics") -> go.Figure:
        """
        Create a rolling statistics chart.
        
        Args:
            data: Time series data
            window: Rolling window size
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        # Calculate rolling statistics
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        rolling_min = data.rolling(window=window).min()
        rolling_max = data.rolling(window=window).max()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Price & Rolling Mean', 'Rolling Volatility', 'Rolling Min/Max', 'Price Distribution'],
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Price and rolling mean
        fig.add_trace(
            go.Scatter(x=data.index, y=data.values, name='Price', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=rolling_mean.index, y=rolling_mean.values, name=f'MA({window})', 
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # Rolling volatility
        fig.add_trace(
            go.Scatter(x=rolling_std.index, y=rolling_std.values, name='Rolling Std', 
                      line=dict(color='green')),
            row=1, col=2
        )
        
        # Rolling min/max
        fig.add_trace(
            go.Scatter(x=rolling_min.index, y=rolling_min.values, name='Rolling Min', 
                      line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=rolling_max.index, y=rolling_max.values, name='Rolling Max', 
                      line=dict(color='orange')),
            row=2, col=1
        )
        
        # Price distribution
        fig.add_trace(
            go.Histogram(x=data.dropna().values, name='Distribution', 
                        marker_color='lightblue', nbinsx=50),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=700,
            showlegend=True
        )
        
        return fig
    
    def create_seasonal_decomposition(self, data: pd.Series, 
                                    title: str = "Seasonal Decomposition") -> go.Figure:
        """
        Create a seasonal decomposition visualization.
        
        Args:
            data: Time series data
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(data, model='additive', period=min(30, len(data)//3))
            
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
                vertical_spacing=0.05
            )
            
            # Original
            fig.add_trace(
                go.Scatter(x=data.index, y=data.values, name='Original', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Trend
            fig.add_trace(
                go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values, 
                          name='Trend', line=dict(color='red')),
                row=2, col=1
            )
            
            # Seasonal
            fig.add_trace(
                go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values, 
                          name='Seasonal', line=dict(color='green')),
                row=3, col=1
            )
            
            # Residual
            fig.add_trace(
                go.Scatter(x=decomposition.resid.index, y=decomposition.resid.values, 
                          name='Residual', line=dict(color='purple')),
                row=4, col=1
            )
            
            fig.update_layout(
                title=title,
                template=self.theme,
                height=800,
                showlegend=False
            )
            
            return fig
            
        except ImportError:
            # Fallback if statsmodels is not available
            fig = go.Figure()
            fig.add_annotation(
                text="Seasonal decomposition requires statsmodels library",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False
            )
            fig.update_layout(title=title, template=self.theme, height=400)
            return fig
        
        except Exception as e:
            # Fallback for any other errors
            fig = go.Figure()
            fig.add_annotation(
                text=f"Seasonal decomposition failed: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False
            )
            fig.update_layout(title=title, template=self.theme, height=400)
            return fig


# Utility functions for quick plotting
def quick_line_plot(data: Union[pd.Series, pd.DataFrame], title: str = "Line Plot") -> go.Figure:
    """Quick line plot function."""
    plotter = InteractivePlotter()
    
    if isinstance(data, pd.Series):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.values,
            mode='lines',
            name=data.name or 'Data'
        ))
    else:
        fig = plotter.create_multi_asset_comparison(data, title)
    
    fig.update_layout(title=title, template=plotter.theme)
    return fig


def quick_scatter_plot(x_data: pd.Series, y_data: pd.Series, 
                      title: str = "Scatter Plot") -> go.Figure:
    """Quick scatter plot function."""
    plotter = InteractivePlotter()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='markers',
        marker=dict(size=8, opacity=0.7),
        hovertemplate=f"<b>{x_data.name or 'X'}</b>: %{{x}}<br>" +
                     f"<b>{y_data.name or 'Y'}</b>: %{{y}}<br>" +
                     "<extra></extra>"
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_data.name or 'X',
        yaxis_title=y_data.name or 'Y',
        template=plotter.theme
    )
    
    return fig


def create_dashboard_plot(data_dict: Dict[str, pd.Series], 
                         title: str = "Dashboard") -> go.Figure:
    """Create a dashboard with multiple subplots."""
    n_plots = len(data_dict)
    
    if n_plots == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data provided", x=0.5, y=0.5)
        return fig
    
    # Calculate grid layout
    cols = min(2, n_plots)
    rows = (n_plots + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=list(data_dict.keys()),
        vertical_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set3
    
    for i, (name, series) in enumerate(data_dict.items()):
        row = i // cols + 1
        col = i % cols + 1
        
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode='lines',
                name=name,
                line=dict(color=colors[i % len(colors)]),
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title=title,
        height=300 * rows,
        template="plotly"
    )
    
    return fig


# Export the main plotter class and utility functions
__all__ = [
    'InteractivePlotter',
    'quick_line_plot',
    'quick_scatter_plot', 
    'create_dashboard_plot'
]
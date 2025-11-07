"""Performance Reporting Module - Generate comprehensive portfolio performance reports."""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import base64
from scipy import stats


class PerformanceReporter:
    """Generate comprehensive performance reports for portfolios and strategies."""
    
    def __init__(self):
        """Initialize the performance reporter."""
        self.report_data = {}
        self.charts = {}
        
    def calculate_performance_metrics(self, returns, benchmark_returns=None, risk_free_rate=0.02):
        """Calculate comprehensive performance metrics.
        
        Args:
            returns: Portfolio returns series
            benchmark_returns: Benchmark returns series (optional)
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            # Basic statistics
            total_return = (1 + returns).cumprod().iloc[-1] - 1
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = returns.std() * np.sqrt(252)
            
            # Risk-adjusted metrics
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Downside metrics
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Drawdown analysis
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdowns = (cumulative - running_max) / running_max
            max_drawdown = drawdowns.min()
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Value at Risk
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Expected Shortfall
            es_95 = returns[returns <= var_95].mean() if any(returns <= var_95) else 0
            es_99 = returns[returns <= var_99].mean() if any(returns <= var_99) else 0
            
            # Distribution statistics
            returns_skew = stats.skew(returns)
            returns_kurtosis = stats.kurtosis(returns)
            
            # Win rate
            positive_returns = returns[returns > 0]
            win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
            
            # Average win/loss
            avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
            avg_loss = returns[returns < 0].mean() if any(returns < 0) else 0
            win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            metrics = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'var_99': var_99,
                'es_95': es_95,
                'es_99': es_99,
                'skewness': returns_skew,
                'kurtosis': returns_kurtosis,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'win_loss_ratio': win_loss_ratio
            }
            
            # Benchmark comparison if provided
            if benchmark_returns is not None and len(benchmark_returns) == len(returns):
                bench_total_return = (1 + benchmark_returns).cumprod().iloc[-1] - 1
                bench_annualized_return = (1 + bench_total_return) ** (252 / len(benchmark_returns)) - 1
                bench_volatility = benchmark_returns.std() * np.sqrt(252)
                
                # Alpha and Beta
                covariance = np.cov(returns, benchmark_returns)[0, 1] * 252
                benchmark_variance = np.var(benchmark_returns) * 252
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                alpha = annualized_return - (risk_free_rate + beta * (bench_annualized_return - risk_free_rate))
                
                # Information ratio
                active_returns = returns - benchmark_returns
                tracking_error = active_returns.std() * np.sqrt(252)
                information_ratio = active_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
                
                # Correlation
                correlation = np.corrcoef(returns, benchmark_returns)[0, 1]
                
                metrics.update({
                    'alpha': alpha,
                    'beta': beta,
                    'information_ratio': information_ratio,
                    'tracking_error': tracking_error,
                    'correlation': correlation,
                    'benchmark_return': bench_annualized_return,
                    'benchmark_volatility': bench_volatility
                })
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating performance metrics: {e}")
            return {}
    
    def generate_performance_summary_chart(self, returns, benchmark_returns=None, title="Portfolio Performance"):
        """Generate a comprehensive performance summary chart."""
        try:
            # Calculate cumulative returns
            cumulative_returns = (1 + returns).cumprod()
            
            # Create subplot figure
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Cumulative Returns', 'Rolling Volatility', 'Drawdown', 'Monthly Returns Heatmap'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"type": "heatmap"}]],
                vertical_spacing=0.08,
                horizontal_spacing=0.1
            )
            
            # 1. Cumulative returns
            fig.add_trace(
                go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns.values,
                    mode='lines',
                    name='Portfolio',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            if benchmark_returns is not None:
                bench_cumulative = (1 + benchmark_returns).cumprod()
                fig.add_trace(
                    go.Scatter(
                        x=bench_cumulative.index,
                        y=bench_cumulative.values,
                        mode='lines',
                        name='Benchmark',
                        line=dict(color='red', width=2, dash='dash')
                    ),
                    row=1, col=1
                )
            
            # 2. Rolling volatility (20-day)
            rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
            fig.add_trace(
                go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol.values * 100,
                    mode='lines',
                    name='20D Vol',
                    line=dict(color='orange', width=1),
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # 3. Drawdown
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - running_max) / running_max
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values * 100,
                    mode='lines',
                    fill='tozeroy',
                    name='Drawdown',
                    line=dict(color='red', width=1),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # 4. Monthly returns heatmap
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_returns.index = monthly_returns.index.to_period('M')
            
            # Create monthly heatmap data
            monthly_data = monthly_returns.to_frame('returns')
            monthly_data['year'] = monthly_data.index.year
            monthly_data['month'] = monthly_data.index.month
            
            # Pivot for heatmap
            heatmap_data = monthly_data.pivot(index='year', columns='month', values='returns')
            
            # Ensure we have all 12 months
            for month in range(1, 13):
                if month not in heatmap_data.columns:
                    heatmap_data[month] = np.nan
            
            heatmap_data = heatmap_data.reindex(columns=range(1, 13))
            
            fig.add_trace(
                go.Heatmap(
                    z=heatmap_data.values * 100,
                    x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                    y=heatmap_data.index,
                    colorscale='RdYlGn',
                    zmid=0,
                    showscale=False,
                    text=np.round(heatmap_data.values * 100, 1),
                    texttemplate="%{text}%",
                    textfont={"size": 8}
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title=title,
                height=600,
                showlegend=True,
                legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)')
            )
            
            # Update axes
            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
            
            fig.update_xaxes(title_text="Date", row=1, col=2)
            fig.update_yaxes(title_text="Volatility (%)", row=1, col=2)
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
            
            fig.update_xaxes(title_text="Month", row=2, col=2)
            fig.update_yaxes(title_text="Year", row=2, col=2)
            
            return fig
            
        except Exception as e:
            print(f"Error generating performance chart: {e}")
            return None
    
    def generate_risk_return_analysis(self, returns, benchmark_returns=None):
        """Generate risk-return analysis charts."""
        try:
            # Rolling Sharpe ratio
            rolling_window = 60
            rolling_returns = returns.rolling(window=rolling_window).mean() * 252
            rolling_vol = returns.rolling(window=rolling_window).std() * np.sqrt(252)
            rolling_sharpe = rolling_returns / rolling_vol
            
            # Risk-return scatter (monthly data)
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_vol = returns.resample('M').std() * np.sqrt(12)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Rolling Sharpe Ratio', 'Risk-Return Scatter', 'Return Distribution', 'Q-Q Plot'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 1. Rolling Sharpe ratio
            fig.add_trace(
                go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe.values,
                    mode='lines',
                    name='Rolling Sharpe',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Add horizontal line at 1.0
            fig.add_hline(y=1.0, line_dash="dash", line_color="red", 
                         annotation_text="Sharpe = 1.0", row=1, col=1)
            
            # 2. Risk-return scatter
            fig.add_trace(
                go.Scatter(
                    x=monthly_vol.values * 100,
                    y=monthly_returns.values * 100,
                    mode='markers',
                    name='Monthly Risk-Return',
                    marker=dict(color='blue', size=6, opacity=0.7),
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # 3. Return distribution
            fig.add_trace(
                go.Histogram(
                    x=returns.values * 100,
                    nbinsx=50,
                    name='Return Distribution',
                    marker_color='lightblue',
                    opacity=0.7,
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # 4. Q-Q plot
            sorted_returns = np.sort(returns)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_returns)))
            
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sorted_returns,
                    mode='markers',
                    name='Q-Q Plot',
                    marker=dict(color='red', size=4),
                    showlegend=False
                ),
                row=2, col=2
            )
            
            # Add 45-degree line for Q-Q plot
            line_min = min(theoretical_quantiles.min(), sorted_returns.min())
            line_max = max(theoretical_quantiles.max(), sorted_returns.max())
            fig.add_trace(
                go.Scatter(
                    x=[line_min, line_max],
                    y=[line_min, line_max],
                    mode='lines',
                    line=dict(color='blue', dash='dash'),
                    name='Normal Line',
                    showlegend=False
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title="Risk-Return Analysis",
                height=600,
                showlegend=True
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
            
            fig.update_xaxes(title_text="Volatility (%)", row=1, col=2)
            fig.update_yaxes(title_text="Return (%)", row=1, col=2)
            
            fig.update_xaxes(title_text="Return (%)", row=2, col=1)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)
            
            fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
            fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)
            
            return fig
            
        except Exception as e:
            print(f"Error generating risk-return analysis: {e}")
            return None
    
    def generate_attribution_analysis(self, asset_returns, weights):
        """Generate performance attribution analysis."""
        try:
            # Calculate contribution of each asset
            contributions = asset_returns.multiply(weights, axis=1)
            total_contribution = contributions.sum(axis=1)
            
            # Cumulative contributions
            cumulative_contributions = (1 + contributions).cumprod() - 1
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Asset Contribution Over Time', 'Cumulative Asset Contribution'),
                vertical_spacing=0.1
            )
            
            colors = px.colors.qualitative.Set3
            
            # 1. Daily contributions
            for i, asset in enumerate(asset_returns.columns):
                fig.add_trace(
                    go.Scatter(
                        x=contributions.index,
                        y=contributions[asset].values * 100,
                        mode='lines',
                        name=asset,
                        line=dict(color=colors[i % len(colors)], width=1.5),
                        stackgroup='one'
                    ),
                    row=1, col=1
                )
            
            # 2. Cumulative contributions
            for i, asset in enumerate(asset_returns.columns):
                fig.add_trace(
                    go.Scatter(
                        x=cumulative_contributions.index,
                        y=cumulative_contributions[asset].values * 100,
                        mode='lines',
                        name=f"{asset} (Cumulative)",
                        line=dict(color=colors[i % len(colors)], width=2),
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                title="Performance Attribution Analysis",
                height=600,
                showlegend=True,
                legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)')
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Daily Contribution (%)", row=1, col=1)
            fig.update_yaxes(title_text="Cumulative Contribution (%)", row=2, col=1)
            
            return fig
            
        except Exception as e:
            print(f"Error generating attribution analysis: {e}")
            return None
    
    def create_performance_report_summary(self, metrics):
        """Create a formatted performance summary table."""
        try:
            summary_data = {
                'Performance Metrics': [
                    'Total Return',
                    'Annualized Return',
                    'Volatility',
                    'Sharpe Ratio',
                    'Sortino Ratio',
                    'Calmar Ratio',
                    'Maximum Drawdown',
                    'Win Rate',
                    'Win/Loss Ratio'
                ],
                'Values': [
                    f"{metrics.get('total_return', 0):.2%}",
                    f"{metrics.get('annualized_return', 0):.2%}",
                    f"{metrics.get('volatility', 0):.2%}",
                    f"{metrics.get('sharpe_ratio', 0):.3f}",
                    f"{metrics.get('sortino_ratio', 0):.3f}",
                    f"{metrics.get('calmar_ratio', 0):.3f}",
                    f"{metrics.get('max_drawdown', 0):.2%}",
                    f"{metrics.get('win_rate', 0):.2%}",
                    f"{metrics.get('win_loss_ratio', 0):.2f}"
                ]
            }
            
            # Add benchmark comparison if available
            if 'alpha' in metrics:
                benchmark_data = {
                    'Benchmark Comparison': [
                        'Alpha',
                        'Beta', 
                        'Information Ratio',
                        'Tracking Error',
                        'Correlation'
                    ],
                    'Values': [
                        f"{metrics.get('alpha', 0):.2%}",
                        f"{metrics.get('beta', 0):.3f}",
                        f"{metrics.get('information_ratio', 0):.3f}",
                        f"{metrics.get('tracking_error', 0):.2%}",
                        f"{metrics.get('correlation', 0):.3f}"
                    ]
                }
                
                return pd.DataFrame(summary_data), pd.DataFrame(benchmark_data)
            
            return pd.DataFrame(summary_data), None
            
        except Exception as e:
            print(f"Error creating performance summary: {e}")
            return None, None
    
    def generate_monthly_performance_table(self, returns):
        """Generate a monthly performance table."""
        try:
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            
            # Create monthly table
            monthly_data = monthly_returns.to_frame('returns')
            monthly_data['year'] = monthly_data.index.year
            monthly_data['month'] = monthly_data.index.month
            
            # Pivot table
            monthly_table = monthly_data.pivot(index='year', columns='month', values='returns')
            
            # Ensure all months are present
            for month in range(1, 13):
                if month not in monthly_table.columns:
                    monthly_table[month] = np.nan
            
            monthly_table = monthly_table.reindex(columns=range(1, 13))
            
            # Add annual returns
            annual_returns = monthly_returns.groupby(monthly_returns.index.year).apply(lambda x: (1 + x).prod() - 1)
            monthly_table['Annual'] = annual_returns
            
            # Format as percentages
            monthly_table = monthly_table.applymap(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
            
            # Rename columns
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual']
            monthly_table.columns = month_names
            
            return monthly_table
            
        except Exception as e:
            print(f"Error generating monthly performance table: {e}")
            return None
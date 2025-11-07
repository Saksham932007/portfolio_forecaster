"""API Integration Module - Enhanced data sources and external API integration."""

import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')


class MarketDataAPI:
    """Enhanced market data integration with multiple data sources."""
    
    def __init__(self):
        """Initialize API client."""
        self.session = requests.Session()
        self.cache = {}
        
    def get_enhanced_stock_data(self, symbol, period='1y', include_fundamentals=True):
        """Get comprehensive stock data with fundamentals.
        
        Args:
            symbol: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            include_fundamentals: Include fundamental data
            
        Returns:
            Dictionary with price data and fundamentals
        """
        try:
            # Get price data using yfinance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            result = {
                'symbol': symbol,
                'price_data': hist,
                'period': period,
                'last_updated': datetime.now()
            }
            
            if include_fundamentals and len(hist) > 0:
                try:
                    # Get company info
                    info = ticker.info
                    result['company_info'] = {
                        'name': info.get('longName', 'N/A'),
                        'sector': info.get('sector', 'N/A'), 
                        'industry': info.get('industry', 'N/A'),
                        'market_cap': info.get('marketCap', 0),
                        'pe_ratio': info.get('trailingPE', 0),
                        'dividend_yield': info.get('dividendYield', 0),
                        'beta': info.get('beta', 0),
                        '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                        '52_week_low': info.get('fiftyTwoWeekLow', 0)
                    }
                    
                    # Get financial statements
                    result['financials'] = {
                        'income_statement': ticker.financials,
                        'balance_sheet': ticker.balance_sheet,
                        'cash_flow': ticker.cashflow
                    }
                    
                except Exception as e:
                    print(f"Warning: Could not fetch fundamentals for {symbol}: {e}")
                    result['company_info'] = {}
                    result['financials'] = {}
            
            return result
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_market_indices(self):
        """Get major market indices data."""
        indices = {
            'S&P 500': '^GSPC',
            'Dow Jones': '^DJI', 
            'NASDAQ': '^IXIC',
            'Russell 2000': '^RUT',
            'VIX': '^VIX'
        }
        
        market_data = {}
        for name, symbol in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='1y')
                if len(data) > 0:
                    market_data[name] = {
                        'symbol': symbol,
                        'data': data,
                        'current_price': data['Close'].iloc[-1],
                        'change_1d': ((data['Close'].iloc[-1] / data['Close'].iloc[-2]) - 1) * 100 if len(data) > 1 else 0,
                        'ytd_change': ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
                    }
            except Exception as e:
                print(f"Error fetching {name}: {e}")
                
        return market_data
    
    def get_economic_indicators(self):
        """Get key economic indicators (simulated for demonstration)."""
        # In a real implementation, this would fetch from FRED, Bloomberg, etc.
        indicators = {
            'GDP Growth Rate': np.random.normal(2.5, 1.0),
            'Inflation Rate': np.random.normal(3.0, 0.5),
            'Unemployment Rate': np.random.normal(4.0, 0.5), 
            '10Y Treasury Yield': np.random.normal(4.5, 0.3),
            'Federal Funds Rate': np.random.normal(5.0, 0.2),
            'Consumer Confidence': np.random.normal(100, 10),
            'PMI Manufacturing': np.random.normal(50, 5),
            'Dollar Index (DXY)': np.random.normal(103, 2)
        }
        
        return indicators
    
    def get_sector_performance(self):
        """Get sector ETF performance."""
        sectors = {
            'Technology': 'XLK',
            'Healthcare': 'XLV', 
            'Financials': 'XLF',
            'Consumer Discretionary': 'XLY',
            'Energy': 'XLE',
            'Industrials': 'XLI',
            'Materials': 'XLB',
            'Consumer Staples': 'XLP',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Communication Services': 'XLC'
        }
        
        sector_data = {}
        for sector, etf in sectors.items():
            try:
                ticker = yf.Ticker(etf)
                data = ticker.history(period='3mo')
                if len(data) > 0:
                    sector_data[sector] = {
                        'etf_symbol': etf,
                        'current_price': data['Close'].iloc[-1],
                        'change_1d': ((data['Close'].iloc[-1] / data['Close'].iloc[-2]) - 1) * 100 if len(data) > 1 else 0,
                        'change_1w': ((data['Close'].iloc[-1] / data['Close'].iloc[-5]) - 1) * 100 if len(data) > 5 else 0,
                        'change_1m': ((data['Close'].iloc[-1] / data['Close'].iloc[-21]) - 1) * 100 if len(data) > 21 else 0,
                        'volatility': data['Close'].pct_change().std() * np.sqrt(252) * 100
                    }
            except Exception as e:
                print(f"Error fetching sector data for {sector}: {e}")
        
        return sector_data
    
    def get_crypto_data(self, symbols=['BTC-USD', 'ETH-USD']):
        """Get cryptocurrency data."""
        crypto_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='3mo')
                if len(data) > 0:
                    crypto_name = symbol.split('-')[0]
                    crypto_data[crypto_name] = {
                        'symbol': symbol,
                        'data': data,
                        'current_price': data['Close'].iloc[-1],
                        'change_24h': ((data['Close'].iloc[-1] / data['Close'].iloc[-2]) - 1) * 100 if len(data) > 1 else 0,
                        'volatility': data['Close'].pct_change().std() * np.sqrt(365) * 100,
                        'volume_24h': data['Volume'].iloc[-1] if 'Volume' in data.columns else 0
                    }
            except Exception as e:
                print(f"Error fetching crypto data for {symbol}: {e}")
        
        return crypto_data
    
    def get_forex_data(self):
        """Get major forex pairs data."""
        forex_pairs = {
            'EUR/USD': 'EURUSD=X',
            'GBP/USD': 'GBPUSD=X',
            'USD/JPY': 'USDJPY=X',
            'USD/CHF': 'USDCHF=X',
            'AUD/USD': 'AUDUSD=X',
            'USD/CAD': 'USDCAD=X'
        }
        
        forex_data = {}
        for pair_name, symbol in forex_pairs.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='1mo')
                if len(data) > 0:
                    forex_data[pair_name] = {
                        'symbol': symbol,
                        'current_rate': data['Close'].iloc[-1],
                        'change_1d': ((data['Close'].iloc[-1] / data['Close'].iloc[-2]) - 1) * 100 if len(data) > 1 else 0,
                        'volatility': data['Close'].pct_change().std() * np.sqrt(252) * 100,
                        'data': data
                    }
            except Exception as e:
                print(f"Error fetching forex data for {pair_name}: {e}")
        
        return forex_data
    
    def get_market_calendar(self):
        """Get market calendar and trading sessions info."""
        # Simplified market calendar
        now = datetime.now()
        
        calendar_info = {
            'current_time': now,
            'market_status': self._get_market_status(now),
            'next_market_open': self._get_next_market_open(now),
            'next_market_close': self._get_next_market_close(now),
            'upcoming_holidays': self._get_upcoming_holidays()
        }
        
        return calendar_info
    
    def _get_market_status(self, current_time):
        """Determine if market is currently open."""
        # Simplified - assumes US market hours
        weekday = current_time.weekday()
        hour = current_time.hour
        
        if weekday < 5:  # Monday to Friday
            if 9 <= hour < 16:  # 9 AM to 4 PM EST
                return "OPEN"
            else:
                return "CLOSED"
        else:
            return "CLOSED"
    
    def _get_next_market_open(self, current_time):
        """Get next market opening time."""
        # Simplified logic
        if current_time.weekday() < 5 and current_time.hour < 9:
            # Same day opening
            return current_time.replace(hour=9, minute=30, second=0, microsecond=0)
        else:
            # Next weekday opening
            days_ahead = 1
            while (current_time + timedelta(days=days_ahead)).weekday() >= 5:
                days_ahead += 1
            next_day = current_time + timedelta(days=days_ahead)
            return next_day.replace(hour=9, minute=30, second=0, microsecond=0)
    
    def _get_next_market_close(self, current_time):
        """Get next market closing time."""
        if current_time.weekday() < 5 and current_time.hour < 16:
            # Same day closing
            return current_time.replace(hour=16, minute=0, second=0, microsecond=0)
        else:
            # Next weekday closing
            days_ahead = 1
            while (current_time + timedelta(days=days_ahead)).weekday() >= 5:
                days_ahead += 1
            next_day = current_time + timedelta(days=days_ahead)
            return next_day.replace(hour=16, minute=0, second=0, microsecond=0)
    
    def _get_upcoming_holidays(self):
        """Get upcoming market holidays."""
        # Simplified list of major US holidays
        current_year = datetime.now().year
        holidays = [
            f"{current_year}-01-01",  # New Year's Day
            f"{current_year}-07-04",  # Independence Day
            f"{current_year}-12-25",  # Christmas Day
        ]
        return holidays
    
    def get_earnings_calendar(self, symbols):
        """Get earnings calendar for given symbols (simulated)."""
        earnings_data = {}
        
        for symbol in symbols:
            # Simulate earnings data
            next_earnings = datetime.now() + timedelta(days=np.random.randint(1, 90))
            
            earnings_data[symbol] = {
                'next_earnings_date': next_earnings.strftime('%Y-%m-%d'),
                'estimated_eps': round(np.random.normal(2.0, 0.5), 2),
                'previous_eps': round(np.random.normal(1.8, 0.4), 2),
                'revenue_estimate': round(np.random.normal(1000, 200), 1),
                'analyst_recommendations': {
                    'strong_buy': np.random.randint(0, 5),
                    'buy': np.random.randint(3, 10),
                    'hold': np.random.randint(5, 15),
                    'sell': np.random.randint(0, 3),
                    'strong_sell': np.random.randint(0, 2)
                }
            }
        
        return earnings_data
    
    def get_news_sentiment(self, symbol):
        """Get news sentiment for a symbol (simulated)."""
        # In a real implementation, this would use news APIs like NewsAPI, Alpha Vantage, etc.
        sentiment_score = np.random.normal(0, 1)  # -2 to 2 scale
        
        sentiment_data = {
            'symbol': symbol,
            'sentiment_score': sentiment_score,
            'sentiment_label': self._classify_sentiment(sentiment_score),
            'news_count': np.random.randint(5, 50),
            'latest_headlines': [
                f"Market Update: {symbol} Shows Strong Performance",
                f"{symbol} Analyst Upgrade Boosts Investor Confidence", 
                f"Sector Rotation Impacts {symbol} Trading Volume",
                f"{symbol} Earnings Preview: What to Expect",
                f"Technical Analysis: {symbol} Key Support Levels"
            ][:np.random.randint(3, 6)]
        }
        
        return sentiment_data
    
    def _classify_sentiment(self, score):
        """Classify sentiment score into labels."""
        if score > 1:
            return "Very Positive"
        elif score > 0.5:
            return "Positive"
        elif score > -0.5:
            return "Neutral"
        elif score > -1:
            return "Negative"
        else:
            return "Very Negative"
    
    def get_technical_indicators(self, symbol, period='3mo'):
        """Calculate technical indicators for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if len(data) < 20:
                return None
            
            # Calculate technical indicators
            close_prices = data['Close']
            
            # Moving averages
            sma_20 = close_prices.rolling(window=20).mean()
            sma_50 = close_prices.rolling(window=50).mean() if len(close_prices) >= 50 else None
            ema_12 = close_prices.ewm(span=12).mean()
            
            # RSI
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = close_prices.ewm(span=12).mean()
            ema_26 = close_prices.ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            
            # Bollinger Bands
            bb_middle = close_prices.rolling(window=20).mean()
            bb_std = close_prices.rolling(window=20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            current_price = close_prices.iloc[-1]
            
            indicators = {
                'symbol': symbol,
                'current_price': current_price,
                'sma_20': sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else None,
                'sma_50': sma_50.iloc[-1] if sma_50 is not None and not pd.isna(sma_50.iloc[-1]) else None,
                'rsi': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None,
                'macd': macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else None,
                'macd_signal': signal.iloc[-1] if not pd.isna(signal.iloc[-1]) else None,
                'bb_upper': bb_upper.iloc[-1] if not pd.isna(bb_upper.iloc[-1]) else None,
                'bb_lower': bb_lower.iloc[-1] if not pd.isna(bb_lower.iloc[-1]) else None,
                'price_vs_sma20': ((current_price / sma_20.iloc[-1]) - 1) * 100 if not pd.isna(sma_20.iloc[-1]) else None,
                'volume': data['Volume'].iloc[-1] if 'Volume' in data.columns else None,
                'avg_volume': data['Volume'].rolling(window=20).mean().iloc[-1] if 'Volume' in data.columns else None
            }
            
            # Technical signals
            signals = []
            
            if indicators['rsi'] and indicators['rsi'] > 70:
                signals.append("Overbought (RSI)")
            elif indicators['rsi'] and indicators['rsi'] < 30:
                signals.append("Oversold (RSI)")
            
            if indicators['macd'] and indicators['macd_signal']:
                if indicators['macd'] > indicators['macd_signal']:
                    signals.append("Bullish (MACD)")
                else:
                    signals.append("Bearish (MACD)")
            
            if indicators['price_vs_sma20'] and indicators['price_vs_sma20'] > 5:
                signals.append("Above SMA20")
            elif indicators['price_vs_sma20'] and indicators['price_vs_sma20'] < -5:
                signals.append("Below SMA20")
            
            indicators['signals'] = signals
            
            return indicators
            
        except Exception as e:
            print(f"Error calculating technical indicators for {symbol}: {e}")
            return None


class AlternativeDataAPI:
    """Integration with alternative data sources."""
    
    def __init__(self):
        """Initialize alternative data API."""
        self.social_sentiment_cache = {}
        
    def get_social_sentiment(self, symbol):
        """Get social media sentiment (simulated)."""
        # In real implementation, integrate with Twitter API, Reddit API, etc.
        sentiment_metrics = {
            'symbol': symbol,
            'twitter_mentions': np.random.randint(100, 10000),
            'reddit_posts': np.random.randint(10, 500),
            'overall_sentiment': np.random.choice(['Bullish', 'Bearish', 'Neutral'], p=[0.4, 0.3, 0.3]),
            'sentiment_score': np.random.normal(0, 1),
            'trending_score': np.random.randint(1, 100),
            'top_keywords': ['earnings', 'growth', 'revenue', 'analyst', 'upgrade'],
            'influencer_mentions': np.random.randint(0, 20)
        }
        
        return sentiment_metrics
    
    def get_institutional_flow(self, symbol):
        """Get institutional money flow data (simulated)."""
        flow_data = {
            'symbol': symbol,
            'net_institutional_flow': np.random.normal(0, 1000000),  # in dollars
            'insider_trading': {
                'insider_buys': np.random.randint(0, 10),
                'insider_sells': np.random.randint(0, 15),
                'net_insider_sentiment': np.random.choice(['Positive', 'Negative', 'Neutral'])
            },
            'institutional_ownership': np.random.uniform(0.3, 0.8),  # percentage
            'hedge_fund_positions': np.random.randint(10, 100)
        }
        
        return flow_data
    
    def get_esg_scores(self, symbol):
        """Get ESG (Environmental, Social, Governance) scores."""
        esg_data = {
            'symbol': symbol,
            'environmental_score': np.random.randint(60, 95),
            'social_score': np.random.randint(55, 90),
            'governance_score': np.random.randint(70, 98),
            'overall_esg_score': np.random.randint(65, 92),
            'esg_rank_percentile': np.random.randint(20, 90),
            'carbon_intensity': np.random.uniform(50, 200),
            'diversity_score': np.random.randint(40, 95)
        }
        
        return esg_data


def get_market_overview():
    """Get comprehensive market overview."""
    api = MarketDataAPI()
    
    overview = {
        'timestamp': datetime.now(),
        'market_indices': api.get_market_indices(),
        'sector_performance': api.get_sector_performance(),
        'economic_indicators': api.get_economic_indicators(),
        'market_calendar': api.get_market_calendar(),
        'crypto_overview': api.get_crypto_data(),
        'forex_overview': api.get_forex_data()
    }
    
    return overview
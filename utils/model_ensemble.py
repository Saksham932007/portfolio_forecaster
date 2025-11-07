"""
Advanced Model Ensemble System for Portfolio Forecasting

This module provides sophisticated ensemble methods combining multiple
forecasting models to improve prediction accuracy and robustness.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingRegressor, BaggingRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')


class AdvancedEnsemble:
    """Advanced ensemble system for time series forecasting."""
    
    def __init__(self, use_stacking=True, use_voting=True, use_blending=True):
        """
        Initialize the advanced ensemble system.
        
        Parameters:
        -----------
        use_stacking : bool
            Whether to use stacking ensemble
        use_voting : bool
            Whether to use voting ensemble
        use_blending : bool
            Whether to use blending ensemble
        """
        self.use_stacking = use_stacking
        self.use_voting = use_voting
        self.use_blending = use_blending
        
        # Base models
        self.base_models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'svr': SVR(kernel='rbf', C=1.0),
            'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        # Meta-learner for stacking
        self.meta_learner = LinearRegression()
        
        # Ensemble weights
        self.ensemble_weights = {}
        
        # Performance tracking
        self.model_performances = {}
        self.ensemble_performance = {}
        
        # Scaler for feature normalization
        self.scaler = StandardScaler()
        
        # Fitted models
        self.fitted_models = {}
        self.voting_ensemble = None
        self.meta_model = None
        
    def prepare_features(self, data, lookback_window=20):
        """
        Prepare features for ensemble training.
        
        Parameters:
        -----------
        data : pd.Series or pd.DataFrame
            Time series data
        lookback_window : int
            Number of past observations to use as features
            
        Returns:
        --------
        X, y : np.ndarray
            Features and targets
        """
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]  # Use first column if DataFrame
            
        # Create lagged features
        features = []
        targets = []
        
        for i in range(lookback_window, len(data)):
            # Lagged values
            lag_features = data.iloc[i-lookback_window:i].values
            
            # Technical indicators
            recent_data = data.iloc[i-lookback_window:i]
            
            # Moving averages
            ma_5 = recent_data.rolling(5).mean().iloc[-1] if len(recent_data) >= 5 else recent_data.mean()
            ma_10 = recent_data.rolling(10).mean().iloc[-1] if len(recent_data) >= 10 else recent_data.mean()
            ma_20 = recent_data.rolling(20).mean().iloc[-1] if len(recent_data) >= 20 else recent_data.mean()
            
            # Volatility features
            volatility = recent_data.std()
            
            # Momentum features
            momentum_5 = (recent_data.iloc[-1] / recent_data.iloc[-5] - 1) if len(recent_data) >= 5 else 0
            momentum_10 = (recent_data.iloc[-1] / recent_data.iloc[-10] - 1) if len(recent_data) >= 10 else 0
            
            # RSI-like feature
            returns = recent_data.pct_change().dropna()
            gains = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
            losses = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0
            rsi = 100 - (100 / (1 + gains / (losses + 1e-8)))
            
            # Combine all features
            combined_features = np.concatenate([
                lag_features,
                [ma_5, ma_10, ma_20, volatility, momentum_5, momentum_10, rsi]
            ])
            
            features.append(combined_features)
            targets.append(data.iloc[i])
        
        return np.array(features), np.array(targets)
    
    def evaluate_base_models(self, X, y, cv_folds=5):
        """
        Evaluate base models using cross-validation.
        
        Parameters:
        -----------
        X, y : np.ndarray
            Features and targets
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict : Model performance scores
        """
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        self.model_performances = {}
        
        for name, model in self.base_models.items():
            try:
                # Scale features for models that need it
                if name in ['svr', 'mlp', 'lasso', 'ridge', 'elastic']:
                    X_scaled = self.scaler.fit_transform(X)
                    scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='neg_mean_squared_error')
                else:
                    scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
                
                self.model_performances[name] = {
                    'mean_score': -scores.mean(),
                    'std_score': scores.std(),
                    'scores': -scores
                }
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                self.model_performances[name] = {
                    'mean_score': float('inf'),
                    'std_score': float('inf'),
                    'scores': np.array([float('inf')] * cv_folds)
                }
        
        return self.model_performances
    
    def calculate_ensemble_weights(self, method='performance_based'):
        """
        Calculate ensemble weights based on model performance.
        
        Parameters:
        -----------
        method : str
            Method for weight calculation ('performance_based', 'equal', 'inverse_variance')
            
        Returns:
        --------
        dict : Ensemble weights for each model
        """
        if not self.model_performances:
            # Equal weights if no performance data
            n_models = len(self.base_models)
            return {name: 1.0/n_models for name in self.base_models.keys()}
        
        if method == 'performance_based':
            # Inverse of MSE (better performance = higher weight)
            mse_scores = {name: perf['mean_score'] for name, perf in self.model_performances.items()}
            
            # Avoid division by zero
            min_mse = min(mse_scores.values())
            if min_mse <= 0:
                min_mse = 1e-8
            
            # Calculate inverse weights
            inverse_weights = {name: 1.0 / (mse + min_mse) for name, mse in mse_scores.items()}
            
            # Normalize weights
            total_weight = sum(inverse_weights.values())
            weights = {name: w / total_weight for name, w in inverse_weights.items()}
            
        elif method == 'inverse_variance':
            # Weight by inverse of variance
            var_scores = {name: perf['std_score']**2 for name, perf in self.model_performances.items()}
            
            # Avoid division by zero
            min_var = min(var_scores.values())
            if min_var <= 0:
                min_var = 1e-8
            
            inverse_weights = {name: 1.0 / (var + min_var) for name, var in var_scores.items()}
            
            # Normalize weights
            total_weight = sum(inverse_weights.values())
            weights = {name: w / total_weight for name, w in inverse_weights.items()}
            
        else:  # equal
            n_models = len(self.base_models)
            weights = {name: 1.0/n_models for name in self.base_models.keys()}
        
        self.ensemble_weights = weights
        return weights
    
    def fit_voting_ensemble(self, X, y):
        """
        Fit voting ensemble.
        
        Parameters:
        -----------
        X, y : np.ndarray
            Features and targets
        """
        if not self.use_voting:
            return
        
        # Create voting ensemble with weighted average
        estimators = []
        
        for name, model in self.base_models.items():
            try:
                # Clone the model
                if name in ['svr', 'mlp', 'lasso', 'ridge', 'elastic']:
                    # Models that need scaled features
                    X_scaled = self.scaler.fit_transform(X)
                    model.fit(X_scaled, y)
                else:
                    model.fit(X, y)
                
                estimators.append((name, model))
                self.fitted_models[name] = model
                
            except Exception as e:
                print(f"Error fitting {name}: {e}")
                continue
        
        # Create voting regressor
        if estimators:
            self.voting_ensemble = VotingRegressor(estimators=estimators, weights=None)
            self.voting_ensemble.fit(X, y)
    
    def fit_stacking_ensemble(self, X, y):
        """
        Fit stacking ensemble with meta-learner.
        
        Parameters:
        -----------
        X, y : np.ndarray
            Features and targets
        """
        if not self.use_stacking:
            return
        
        # Generate meta-features using cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        meta_features = np.zeros((len(X), len(self.base_models)))
        
        for name, model in self.base_models.items():
            col_idx = list(self.base_models.keys()).index(name)
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y[train_idx]
                
                try:
                    # Scale features for certain models
                    if name in ['svr', 'mlp', 'lasso', 'ridge', 'elastic']:
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_val_scaled = scaler.transform(X_val)
                        
                        model.fit(X_train_scaled, y_train)
                        predictions = model.predict(X_val_scaled)
                    else:
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_val)
                    
                    meta_features[val_idx, col_idx] = predictions
                    
                except Exception as e:
                    print(f"Error in stacking for {name}: {e}")
                    meta_features[val_idx, col_idx] = np.mean(y_train)
        
        # Fit meta-learner
        self.meta_model = LinearRegression()
        self.meta_model.fit(meta_features, y)
    
    def fit(self, data, lookback_window=20):
        """
        Fit the ensemble system.
        
        Parameters:
        -----------
        data : pd.Series or pd.DataFrame
            Time series data
        lookback_window : int
            Number of past observations to use as features
        """
        print("Preparing features for ensemble training...")
        X, y = self.prepare_features(data, lookback_window)
        
        print("Evaluating base models...")
        self.evaluate_base_models(X, y)
        
        print("Calculating ensemble weights...")
        self.calculate_ensemble_weights()
        
        print("Fitting voting ensemble...")
        self.fit_voting_ensemble(X, y)
        
        print("Fitting stacking ensemble...")
        self.fit_stacking_ensemble(X, y)
        
        print("Ensemble training completed!")
        
        return self
    
    def predict_ensemble(self, X, method='weighted_average'):
        """
        Generate ensemble predictions.
        
        Parameters:
        -----------
        X : np.ndarray
            Features for prediction
        method : str
            Ensemble method ('weighted_average', 'voting', 'stacking')
            
        Returns:
        --------
        np.ndarray : Ensemble predictions
        """
        if method == 'voting' and self.voting_ensemble is not None:
            return self.voting_ensemble.predict(X)
        
        elif method == 'stacking' and self.meta_model is not None:
            # Generate meta-features
            meta_features = np.zeros((len(X), len(self.fitted_models)))
            
            for i, (name, model) in enumerate(self.fitted_models.items()):
                try:
                    if name in ['svr', 'mlp', 'lasso', 'ridge', 'elastic']:
                        X_scaled = self.scaler.transform(X)
                        predictions = model.predict(X_scaled)
                    else:
                        predictions = model.predict(X)
                    
                    meta_features[:, i] = predictions
                    
                except Exception as e:
                    print(f"Error predicting with {name}: {e}")
                    meta_features[:, i] = 0
            
            return self.meta_model.predict(meta_features)
        
        else:  # weighted_average
            predictions = []
            weights = []
            
            for name, model in self.fitted_models.items():
                try:
                    if name in ['svr', 'mlp', 'lasso', 'ridge', 'elastic']:
                        X_scaled = self.scaler.transform(X)
                        pred = model.predict(X_scaled)
                    else:
                        pred = model.predict(X)
                    
                    predictions.append(pred)
                    weights.append(self.ensemble_weights.get(name, 1.0))
                    
                except Exception as e:
                    print(f"Error predicting with {name}: {e}")
                    continue
            
            if predictions:
                # Weighted average
                predictions = np.array(predictions)
                weights = np.array(weights)
                weights = weights / weights.sum()  # Normalize
                
                return np.average(predictions, axis=0, weights=weights)
            else:
                return np.zeros(len(X))
    
    def forecast(self, data, periods=30, confidence_interval=0.95):
        """
        Generate multi-step forecasts using ensemble.
        
        Parameters:
        -----------
        data : pd.Series or pd.DataFrame
            Historical time series data
        periods : int
            Number of periods to forecast
        confidence_interval : float
            Confidence interval for forecast bands
            
        Returns:
        --------
        dict : Forecast results including point estimates and confidence bands
        """
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]
        
        # Prepare last window for forecasting
        lookback_window = 20
        last_values = data.tail(lookback_window).values
        
        forecasts = []
        forecast_features = []
        
        # Generate forecasts iteratively
        current_values = last_values.copy()
        
        for step in range(periods):
            # Prepare features for current step
            # Technical indicators for current window
            recent_data = pd.Series(current_values)
            
            ma_5 = recent_data.rolling(5).mean().iloc[-1] if len(recent_data) >= 5 else recent_data.mean()
            ma_10 = recent_data.rolling(10).mean().iloc[-1] if len(recent_data) >= 10 else recent_data.mean()
            ma_20 = recent_data.rolling(20).mean().iloc[-1] if len(recent_data) >= 20 else recent_data.mean()
            
            volatility = recent_data.std()
            
            momentum_5 = (recent_data.iloc[-1] / recent_data.iloc[-5] - 1) if len(recent_data) >= 5 else 0
            momentum_10 = (recent_data.iloc[-1] / recent_data.iloc[-10] - 1) if len(recent_data) >= 10 else 0
            
            returns = recent_data.pct_change().dropna()
            gains = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
            losses = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0
            rsi = 100 - (100 / (1 + gains / (losses + 1e-8)))
            
            # Combine features
            features = np.concatenate([
                current_values,
                [ma_5, ma_10, ma_20, volatility, momentum_5, momentum_10, rsi]
            ])
            
            forecast_features.append(features)
            
            # Generate ensemble predictions
            X_step = features.reshape(1, -1)
            
            # Get predictions from different ensemble methods
            pred_weighted = self.predict_ensemble(X_step, method='weighted_average')
            pred_voting = self.predict_ensemble(X_step, method='voting') if self.voting_ensemble else pred_weighted
            pred_stacking = self.predict_ensemble(X_step, method='stacking') if self.meta_model else pred_weighted
            
            # Combine predictions
            ensemble_pred = np.mean([pred_weighted[0], pred_voting[0], pred_stacking[0]])
            
            forecasts.append(ensemble_pred)
            
            # Update current values for next iteration
            current_values = np.append(current_values[1:], ensemble_pred)
        
        # Calculate confidence intervals using ensemble disagreement
        forecast_features = np.array(forecast_features)
        
        # Get predictions from all models for uncertainty estimation
        all_predictions = []
        
        for name, model in self.fitted_models.items():
            try:
                if name in ['svr', 'mlp', 'lasso', 'ridge', 'elastic']:
                    X_scaled = self.scaler.transform(forecast_features)
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(forecast_features)
                
                all_predictions.append(pred)
                
            except Exception as e:
                continue
        
        if all_predictions:
            all_predictions = np.array(all_predictions)
            
            # Calculate standard deviation across models
            forecast_std = np.std(all_predictions, axis=0)
            
            # Calculate confidence intervals
            from scipy import stats
            alpha = 1 - confidence_interval
            z_score = stats.norm.ppf(1 - alpha/2)
            
            lower_bound = np.array(forecasts) - z_score * forecast_std
            upper_bound = np.array(forecasts) + z_score * forecast_std
        else:
            # Fallback: use historical volatility
            historical_returns = data.pct_change().dropna()
            historical_vol = historical_returns.std()
            
            forecast_std = historical_vol * np.sqrt(np.arange(1, periods + 1))
            
            from scipy import stats
            alpha = 1 - confidence_interval
            z_score = stats.norm.ppf(1 - alpha/2)
            
            lower_bound = np.array(forecasts) - z_score * forecast_std
            upper_bound = np.array(forecasts) + z_score * forecast_std
        
        # Create forecast dates
        if hasattr(data.index, 'freq') and data.index.freq:
            freq = data.index.freq
        else:
            freq = pd.infer_freq(data.index) or 'D'
        
        last_date = data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(1, freq), periods=periods, freq=freq)
        
        return {
            'dates': forecast_dates,
            'forecast': np.array(forecasts),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_interval': confidence_interval,
            'model_performances': self.model_performances,
            'ensemble_weights': self.ensemble_weights
        }
    
    def get_model_importance(self):
        """
        Get model importance scores based on ensemble weights and performance.
        
        Returns:
        --------
        pd.DataFrame : Model importance scores
        """
        importance_data = []
        
        for name in self.base_models.keys():
            performance = self.model_performances.get(name, {})
            weight = self.ensemble_weights.get(name, 0)
            
            importance_data.append({
                'Model': name.upper(),
                'Weight': weight,
                'MSE': performance.get('mean_score', np.inf),
                'MSE_Std': performance.get('std_score', np.inf),
                'Importance_Score': weight * (1.0 / (performance.get('mean_score', np.inf) + 1e-8))
            })
        
        df = pd.DataFrame(importance_data)
        df = df.sort_values('Importance_Score', ascending=False)
        
        return df
    
    def get_ensemble_summary(self):
        """
        Get summary of ensemble configuration and performance.
        
        Returns:
        --------
        dict : Ensemble summary information
        """
        summary = {
            'total_models': len(self.base_models),
            'fitted_models': len(self.fitted_models),
            'ensemble_methods': [],
            'best_model': None,
            'worst_model': None,
            'average_performance': None
        }
        
        # Ensemble methods used
        if self.use_voting and self.voting_ensemble:
            summary['ensemble_methods'].append('Voting')
        if self.use_stacking and self.meta_model:
            summary['ensemble_methods'].append('Stacking')
        if self.ensemble_weights:
            summary['ensemble_methods'].append('Weighted Average')
        
        # Best and worst models
        if self.model_performances:
            performances = {name: perf['mean_score'] for name, perf in self.model_performances.items()}
            summary['best_model'] = min(performances, key=performances.get)
            summary['worst_model'] = max(performances, key=performances.get)
            summary['average_performance'] = np.mean(list(performances.values()))
        
        return summary
"""
Approach 1: Statistical Methods (ARIMA, SARIMA)

Philosophy: Model time series through autoregressive and moving average components.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from ..core.base_model import BaseApproach


def _fit_time_series_model(model: Any):
    """Fit statsmodels models across API versions.

    Newer statsmodels versions reject `disp`; older ones may still accept it.
    """
    try:
        return model.fit()
    except TypeError:
        return model.fit(disp=False)


class ARIMAForecaster(BaseApproach):
    """ARIMA-based forecaster."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Statistical (ARIMA)", config)
        
        self.order = config.get('order', (2, 1, 2)) if config else (2, 1, 2)
        self.seasonal_order = config.get('seasonal_order', None) if config else None
        
        self.model = None
        self.fitted_model = None
        self.train_series = None
        self.forecast_horizon = 24
        
        self.metrics.interpretability_score = 0.9
        self.metrics.maintenance_complexity = 0.3
    
    def train(self, X_train: np.ndarray = None, y_train: np.ndarray = None,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              train_series: np.ndarray = None, forecast_horizon: int = 24) -> None:
        
        if not HAS_STATSMODELS:
            self.is_trained = True
            return
        
        self.forecast_horizon = forecast_horizon
        
        # Use raw series if provided
        if train_series is not None:
            series = train_series.flatten() if train_series.ndim > 1 else train_series
        elif X_train is not None:
            series = X_train[-1].flatten()  # Use last window
        else:
            raise ValueError("Need either train_series or X_train")
        
        self.train_series = series
        
        try:
            if self.seasonal_order:
                self.model = SARIMAX(
                    series,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                self.model = ARIMA(
                    series,
                    order=self.order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            
            self.fitted_model = _fit_time_series_model(self.model)
        except Exception as e:
            print(f"ARIMA fitting failed: {e}")
            self.fitted_model = None
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict for each sample in X."""
        
        if not HAS_STATSMODELS or self.fitted_model is None:
            # Fallback: naive forecast
            return np.tile(X[:, -1, :], (1, self.forecast_horizon, 1))
        
        predictions = []
        
        for i in range(len(X)):
            try:
                forecast = self.fitted_model.forecast(steps=self.forecast_horizon)
                predictions.append(forecast.reshape(-1, 1))
            except:
                predictions.append(np.tile(X[i, -1], (self.forecast_horizon, 1)))
        
        return np.array(predictions)
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Time series = autoregressive + differencing + moving average',
            'inductive_bias': 'Linear relationships, stationarity after differencing',
            'strengths': 'Interpretable parameters, statistical guarantees, handles trends',
            'weaknesses': 'Linear only, manual order selection, univariate focus',
            'best_for': 'Univariate series with clear AR/MA structure'
        }
    
    def get_model_size(self) -> float:
        return 0.01  # Minimal - just parameters
    
    def collect_failure_cases(self, X_test: np.ndarray, y_test: np.ndarray,
                               y_pred: np.ndarray, n_cases: int = 10) -> List[Dict]:
        failures = []
        errors = np.mean((y_test - y_pred) ** 2, axis=(1, 2))
        worst_indices = np.argsort(-errors)[:n_cases]
        
        for idx in worst_indices:
            failures.append({
                'index': int(idx),
                'mse': float(errors[idx]),
                'reason': 'High prediction error'
            })
        
        return failures
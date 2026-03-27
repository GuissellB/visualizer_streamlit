import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    TimeSeriesSplit,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek

# =============================================================================
# 0) MÉTRICAS (OCP): agregas sin tocar runners
# =============================================================================
MetricFn = Callable[[np.ndarray, np.ndarray], Dict[str, object]]


def get_positive_score(model, X):
    """Obtiene un score continuo de la clase positiva si el modelo lo soporta."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return np.ravel(proba)
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def m_accuracy_error() -> Callable:
    def _m(y, yp, model=None, X=None):
        cm = confusion_matrix(y, yp)
        acc = cm.diagonal().sum() / cm.sum() if cm.sum() else np.nan
        # Nota: ConfusionMatrix es útil, pero es un objeto 2D.
        # En Streamlit/Arrow no se puede mostrar dentro de un dataframe.
        # Déjalo en el dict (para quien lo necesite) y filtra al construir tablas.
        return {"ConfusionMatrix": cm, "Accuracy": float(acc), "Error": float(1 - acc)}
    return _m

def m_clf_basic(pos_label=1) -> Callable:
    def _m(y, yp, model=None, X=None):
        out = {}
        for k, fn in [("Recall_Pos", recall_score), ("Precision_Pos", precision_score), ("F1_Pos", f1_score)]:
            try:
                out[k] = float(fn(y, yp, pos_label=pos_label))
            except Exception:
                out[k] = None

        auc = None
        try:
            score = get_positive_score(model, X)
            if score is not None:
                auc = float(roc_auc_score(y, score))
        except Exception:
            auc = None

        out["ROC_AUC_Pos"] = auc
        return out
    return _m

def m_reg_basic(mape=True) -> Callable:
    def _m(y, yp, model=None, X=None):
        y, yp = np.asarray(y), np.asarray(yp)
        mask = ~(np.isnan(y) | np.isnan(yp))
        y, yp = y[mask], yp[mask]

        mse = float(mean_squared_error(y, yp)) if len(y) else None
        out = {
            "MAE": float(mean_absolute_error(y, yp)) if len(y) else None,
            "MSE": mse,
            "RMSE": float(np.sqrt(mse)) if mse is not None else None,
            "R2": float(r2_score(y, yp)) if len(y) else None,
        }
        if mape:
            denom = np.where(y == 0, np.nan, y)
            val = np.nanmean(np.abs((y - yp) / denom)) * 100
            out["MAPE_%"] = float(val) if val == val else None
        return out
    return _m


# =============================================================================
# 1) PREPARACIÓN (SRP): prepara X/y, scaling, split
# =============================================================================
@dataclass
class DataPreparer:
    train_size: float = 0.75
    random_state: Optional[int] = None
    scaler: Optional[object] = None
    scale_X: bool = True

    def __post_init__(self):
        if self.scaler is None and self.scale_X:
            self.scaler = StandardScaler()

    def build_xy(
        self,
        df: pd.DataFrame,
        target: str,
        features: Optional[List[str]] = None,
        y_transform: Optional[Callable[[pd.Series], np.ndarray]] = None,
    ):
        features = features or []
        X = df.drop(columns=[target]) if not features else df[features]
        cols = list(X.columns)

        y_series = df[target]
        y = y_transform(y_series) if y_transform else y_series.values
        return X, y, cols

    def scale_train_test(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        cols: Optional[List[str]] = None,
        clone_scaler: bool = False,
    ):
        if self.scaler is None:
            return X_train, X_test

        cols = cols or list(X_train.columns)
        s = clone(self.scaler) if clone_scaler else self.scaler

        X_train = pd.DataFrame(s.fit_transform(X_train), columns=cols, index=X_train.index)
        X_test = pd.DataFrame(s.transform(X_test), columns=cols, index=X_test.index)
        return X_train, X_test

    def split(
        self,
        df: pd.DataFrame,
        target: str,
        features: Optional[List[str]] = None,
        y_transform: Optional[Callable[[pd.Series], np.ndarray]] = None,
        stratify: bool = False,
    ):
        X, y, cols = self.build_xy(
            df=df,
            target=target,
            features=features,
            y_transform=y_transform,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=self.train_size,
            random_state=self.random_state,
            stratify=y if stratify else None,
        )

        if self.scaler is not None:
            X_train, X_test = self.scale_train_test(X_train, X_test, cols=cols)

        return X_train, X_test, y_train, y_test, cols

    # --- tiempo / forecasting ---
    def build_lagged_xy(
        self,
        df: pd.DataFrame,
        target: str,
        lags: int = 12,
        features: Optional[List[str]] = None,
    ):
        if target not in df.columns:
            raise ValueError(f"'{target}' no existe en el DataFrame.")
        if not isinstance(lags, int) or lags <= 0:
            raise ValueError("lags debe ser un entero positivo.")

        features = features or []
        for c in features:
            if c not in df.columns:
                raise ValueError(f"Feature '{c}' no existe en el DataFrame.")

        X = pd.DataFrame(index=df.index)
        for i in range(1, lags + 1):
            X[f"{target}_lag_{i}"] = df[target].shift(i)

        for c in features:
            if c == target:
                continue
            X[c] = df[c]

        y = pd.to_numeric(df[target], errors="coerce")
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X.loc[mask]
        y = y.loc[mask]

        return X, y.values, list(X.columns)

    def split_time_xy(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: Optional[int] = None,
    ):
        n = len(X)
        if n < 2:
            raise ValueError("Se requieren al menos 2 observaciones para split temporal.")

        if test_size is None:
            test_size = max(1, int(round(n * (1 - self.train_size))))
        if not isinstance(test_size, int) or test_size <= 0 or test_size >= n:
            raise ValueError("test_size debe ser entero en [1, n-1].")

        cut = n - test_size
        X_train, X_test = X.iloc[:cut].copy(), X.iloc[cut:].copy()
        y_train, y_test = y[:cut], y[cut:]

        if self.scaler is not None:
            X_train, X_test = self.scale_train_test(X_train, X_test, cols=list(X.columns))

        return X_train, X_test, y_train, y_test, list(X.columns)

    def split_time_series(
        self,
        series: pd.Series,
        test_size: Optional[int] = None,
    ):
        # Split temporal directo para modelos que trabajan sobre la serie cruda.
        series = pd.to_numeric(series, errors="coerce").dropna()
        n = len(series)
        if n < 2:
            raise ValueError("Se requieren al menos 2 observaciones para split temporal.")

        if test_size is None:
            test_size = max(1, int(round(n * (1 - self.train_size))))
        if not isinstance(test_size, int) or test_size <= 0 or test_size >= n:
            raise ValueError("test_size debe ser entero en [1, n-1].")

        cut = n - test_size
        train_series = series.iloc[:cut].copy()
        test_series = series.iloc[cut:].copy()
        return train_series, test_series


class HoltWintersForecaster(BaseEstimator):
    """Adaptador simple para ExponentialSmoothing."""

    def __init__(self, trend: str = "add", seasonal: str = "add", seasonal_periods: int = 7):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model_ = None

    def fit(self, series: pd.Series):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        # El adaptador normaliza la entrada para que TimeSeriesRunner pueda tratarlo como un estimador más.
        clean_series = pd.to_numeric(pd.Series(series), errors="coerce").dropna().reset_index(drop=True)
        self.model_ = ExponentialSmoothing(
            clean_series,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
        ).fit()
        return self

    def predict(self, steps: int) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("El modelo Holt-Winters aún no fue entrenado.")
        return np.asarray(self.model_.forecast(steps))


class ARIMAForecaster(BaseEstimator):
    """Adaptador simple para ARIMA."""

    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model_ = None

    def fit(self, series: pd.Series):
        from statsmodels.tsa.arima.model import ARIMA

        # Mantiene una interfaz fit/predict mínima y compatible con clone().
        clean_series = pd.to_numeric(pd.Series(series), errors="coerce").dropna().reset_index(drop=True)
        self.model_ = ARIMA(clean_series, order=self.order).fit()
        return self

    def predict(self, steps: int) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("El modelo ARIMA aún no fue entrenado.")
        return np.asarray(self.model_.forecast(steps=steps))


# =============================================================================
# 2) SUPERVISADO
# =============================================================================
class SupervisedRunner:
    """Entrenamiento + evaluación (clasificación/regresión)."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        model,
        task: str,  # "classification" | "regression"
        features: Optional[List[str]] = None,
        preparer: Optional[DataPreparer] = None,
        metrics: Optional[List[Callable]] = None,
        encode_target: bool = False,
        pos_label: int = 1,
        class_weight: Optional[object] = None,
        sampling_method: Optional[str] = None,
    ):
        self.df = df
        self.target = target
        self.model = model
        self.task = task.lower()
        self.features = features or []
        self.preparer = preparer if preparer is not None else DataPreparer()

        self.encode_target = encode_target
        self.pos_label = pos_label
        self._label_encoder = None
        self.class_weight = class_weight
        self.sampling_method = sampling_method

        if metrics is None:
            if self.task == "classification":
                self.metrics = [m_accuracy_error(), m_clf_basic(pos_label)]
            elif self.task == "regression":
                self.metrics = [m_reg_basic()]
            else:
                raise ValueError("task debe ser 'classification' o 'regression'")
        else:
            self.metrics = metrics

        self._prepared = False

    def _build_sampler(self):
        # Construye el sampler indicado para aplicarlo solo sobre el set de entrenamiento.
        if self.sampling_method is None:
            return None

        method = self.sampling_method.lower()

        if method == "undersample":
            return NearMiss()

        if method == "oversample":
            return RandomOverSampler(random_state=self.preparer.random_state)

        if method == "smote_tomek":
            return SMOTETomek(random_state=self.preparer.random_state)

        raise ValueError(
            "sampling_method debe ser uno de: None, 'undersample', 'oversample', 'smote_tomek'"
        )

    def _compute_scale_pos_weight(self, y: np.ndarray, pos_label: int = 1) -> float:
        # XGBoost usa la razón negativos/positivos calculada sobre el y de entrenamiento.
        y_arr = pd.to_numeric(pd.Series(y), errors="coerce")
        pos_count = int((y_arr == pos_label).sum())
        neg_count = int((y_arr != pos_label).sum())

        if pos_count == 0:
            return 1.0

        return max(neg_count / pos_count, 1.0)

    def _apply_class_balancing(self, model, y_train: np.ndarray, class_weight: object = None):
        # Aplica balanceo a modelos de clasificación según los parámetros que expongan.
        if class_weight is None:
            return model

        try:
            params = model.get_params(deep=True)
        except Exception:
            return model

        updates = {}

        class_weight_keys = []
        if "class_weight" in params:
            class_weight_keys.append("class_weight")
        class_weight_keys.extend([k for k in params.keys() if k.endswith("__class_weight")])

        if class_weight_keys:
            updates.update({k: class_weight for k in class_weight_keys})

        scale_pos_weight_keys = []
        if "scale_pos_weight" in params:
            scale_pos_weight_keys.append("scale_pos_weight")
        scale_pos_weight_keys.extend([k for k in params.keys() if k.endswith("__scale_pos_weight")])

        if scale_pos_weight_keys:
            scale_pos_weight = self._compute_scale_pos_weight(y_train, pos_label=self.pos_label)
            updates.update({k: scale_pos_weight for k in scale_pos_weight_keys})

        if not updates:
            return model

        model.set_params(**updates)
        return model

    def _apply_sampling(self, X_train, y_train):
        # Aplica resampling solo a entrenamiento para evitar fuga de información.
        if self.task != "classification" or self.sampling_method is None:
            return X_train, y_train

        sampler = self._build_sampler()
        if sampler is None:
            return X_train, y_train

        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        return X_resampled, y_resampled

    def _y_transform(self, y: pd.Series) -> np.ndarray:
        if self.task == "classification":
            if self.encode_target and (y.dtype == "object" or str(y.dtype).startswith("category")):
                self._label_encoder = LabelEncoder()
                return self._label_encoder.fit_transform(y.values)
            return y.values
        return pd.to_numeric(y, errors="coerce").values

    def _use_stratify(self) -> bool:
        return self.task == "classification"

    def get_cv_strategy(self, n_splits: int = 10, shuffle: bool = True):
        if self.task == "classification":
            return StratifiedKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=self.preparer.random_state,
            )
        if self.task == "regression":
            return KFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=self.preparer.random_state,
            )
        raise ValueError("task debe ser 'classification' o 'regression'")

    def _prepare(self):
        self.X_train, self.X_test, self.y_train, self.y_test, self.feature_names = self.preparer.split(
            df=self.df,
            target=self.target,
            features=self.features,
            y_transform=self._y_transform,
            stratify=self._use_stratify(),
        )
        self._prepared = True

    def get_model_for_current_split(self):
        # Devuelve el modelo configurado con el balanceo que corresponde al y_train actual.
        if not self._prepared:
            self._prepare()
        _, y_train = self._apply_sampling(self.X_train, self.y_train)
        return self._apply_class_balancing(self.model, y_train, self.class_weight)

    def fit_predict(self) -> np.ndarray:
        if not self._prepared:
            self._prepare()
        X_train_fit, y_train_fit = self._apply_sampling(self.X_train, self.y_train)
        self.model = self._apply_class_balancing(self.model, y_train_fit, self.class_weight)
        self.model.fit(X_train_fit, y_train_fit)
        return self.model.predict(self.X_test)

    def evaluate(self) -> Dict[str, object]:
        yp = self.fit_predict()
        out: Dict[str, object] = {}
        for fn in self.metrics:
            out.update(fn(self.y_test, yp, model=self.model, X=self.X_test))
        return out

    def build_evaluator(self, scoring: Optional[str] = None, cv: int = 5):
        """Construye un ModelEvaluator reutilizando el split actual del runner."""
        return ModelEvaluator.from_runner(self, scoring=scoring, cv=cv)

    def evaluate_cv(self, n_splits: int = 10, shuffle: bool = True) -> Dict[str, object]:
        """KFold/StratifiedKFold con métricas mean/std."""
        X, y, cols = self.preparer.build_xy(
            df=self.df,
            target=self.target,
            features=self.features,
            y_transform=self._y_transform,
        )
        cv = self.get_cv_strategy(n_splits=n_splits, shuffle=shuffle)

        fold_metrics: List[Dict[str, object]] = []
        for tr, te in cv.split(X, y):
            X_train, X_test = X.iloc[tr].copy(), X.iloc[te].copy()
            y_train, y_test = y[tr], y[te]

            X_train, X_test = self.preparer.scale_train_test(X_train, X_test, cols=cols, clone_scaler=True)
            X_train, y_train = self._apply_sampling(X_train, y_train)

            model_fold = clone(self.model)
            model_fold = self._apply_class_balancing(model_fold, y_train, self.class_weight)
            model_fold.fit(X_train, y_train)
            yp = model_fold.predict(X_test)

            out: Dict[str, object] = {}
            for fn in self.metrics:
                out.update(fn(y_test, yp, model=model_fold, X=X_test))
            fold_metrics.append(out)

        df_res = pd.DataFrame(fold_metrics)
        final: Dict[str, object] = {}
        for col in df_res.columns:
            if pd.api.types.is_numeric_dtype(df_res[col]):
                final[col] = float(df_res[col].mean())
                final[col + "_std"] = float(df_res[col].std())
        return final


# =============================================================================
# 3) SERIES TEMPORALES (forecast tabular/autoregresivo)
# =============================================================================
class TimeSeriesRunner:
    """Entrenamiento + evaluación para forecasting (sin plots).

    Enfoque:
    - Convierte serie a problema supervisado por lags.
    - Split temporal (sin shuffle).
    - Backtesting con TimeSeriesSplit.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        model,
        lags: int = 12,
        features: Optional[List[str]] = None,
        preparer: Optional[DataPreparer] = None,
        metrics: Optional[List[Callable]] = None,
        test_size: Optional[int] = None,
    ):
        self.df = df.copy()
        self.target = target
        self.model = model
        self.lags = lags
        self.features = features or []
        self.preparer = preparer if preparer is not None else DataPreparer()
        self.metrics = metrics if metrics is not None else [m_reg_basic()]
        self.test_size = test_size

        self._prepared = False
        self._fitted_full = False
        self._last_observed: Optional[pd.Series] = None
        self.feature_names: List[str] = []
        self.test_index = None

    def _uses_series_forecaster(self) -> bool:
        # Estos modelos no usan lags tabulares sino la serie directamente.
        return isinstance(self.model, (HoltWintersForecaster, ARIMAForecaster))

    def _prepare(self):
        if self._uses_series_forecaster():
            # Rama para modelos statsmodels adaptados: separa train/test sobre la serie cruda.
            series = pd.to_numeric(self.df[self.target], errors="coerce").dropna()
            self.train_series, self.test_series = self.preparer.split_time_series(series, test_size=self.test_size)
            self.y_train = self.train_series.values
            self.y_test = self.test_series.values
            self.feature_names = [self.target]
            self.test_index = self.test_series.index
            self.X_train = None
            self.X_test = None
        else:
            # Rama autoregresiva/tabular: construye lags y luego hace split temporal.
            X, y, cols = self.preparer.build_lagged_xy(
                df=self.df,
                target=self.target,
                lags=self.lags,
                features=self.features,
            )
            self.X_train, self.X_test, self.y_train, self.y_test, self.feature_names = self.preparer.split_time_xy(
                X, y, test_size=self.test_size
            )
            self.test_index = self.X_test.index
        self._prepared = True

    def fit_predict(self) -> np.ndarray:
        if not self._prepared:
            self._prepare()
        if self._uses_series_forecaster():
            # En modelos de serie directa, predict recibe la cantidad de pasos del holdout.
            self.model.fit(self.train_series)
            return self.model.predict(len(self.test_series))
        self.model.fit(self.X_train, self.y_train)
        return self.model.predict(self.X_test)

    def evaluate(self) -> Dict[str, object]:
        yp = self.fit_predict()
        out: Dict[str, object] = {}
        for fn in self.metrics:
            out.update(fn(self.y_test, yp, model=self.model, X=self.X_test))
        return out

    def evaluate_cv(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
    ) -> Dict[str, object]:
        if self._uses_series_forecaster():
            # Backtesting temporal para ARIMA/Holt-Winters sin convertir la serie a lags.
            # Aquí la validación cruzada se hace con TimeSeriesSplit:
            # cada fold entrena con pasado y prueba con futuro.
            series = pd.to_numeric(self.df[self.target], errors="coerce").dropna().reset_index(drop=True)
            cv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

            fold_metrics: List[Dict[str, object]] = []
            for tr, te in cv.split(series):
                train_series = series.iloc[tr].copy()
                test_series = series.iloc[te].copy()

                model_fold = clone(self.model)
                model_fold.fit(train_series)
                yp = model_fold.predict(len(test_series))

                out: Dict[str, object] = {}
                for fn in self.metrics:
                    out.update(fn(test_series.values, yp, model=model_fold, X=None))
                fold_metrics.append(out)

            df_res = pd.DataFrame(fold_metrics)
            final: Dict[str, object] = {}
            for col in df_res.columns:
                if pd.api.types.is_numeric_dtype(df_res[col]):
                    final[col] = float(df_res[col].mean())
                    final[col + "_std"] = float(df_res[col].std())
            return final

        X, y, cols = self.preparer.build_lagged_xy(
            df=self.df,
            target=self.target,
            lags=self.lags,
            features=self.features,
        )
        # Para modelos por lags también se usa TimeSeriesSplit,
        # manteniendo el orden temporal del dataset.
        cv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

        fold_metrics: List[Dict[str, object]] = []
        for tr, te in cv.split(X):
            X_train, X_test = X.iloc[tr].copy(), X.iloc[te].copy()
            y_train, y_test = y[tr], y[te]

            X_train, X_test = self.preparer.scale_train_test(X_train, X_test, cols=cols, clone_scaler=True)

            model_fold = clone(self.model)
            model_fold.fit(X_train, y_train)
            yp = model_fold.predict(X_test)

            out: Dict[str, object] = {}
            for fn in self.metrics:
                out.update(fn(y_test, yp, model=model_fold, X=X_test))
            fold_metrics.append(out)

        df_res = pd.DataFrame(fold_metrics)
        final: Dict[str, object] = {}
        for col in df_res.columns:
            if pd.api.types.is_numeric_dtype(df_res[col]):
                final[col] = float(df_res[col].mean())
                final[col + "_std"] = float(df_res[col].std())
        return final

    def fit_full(self):
        """Entrena con toda la serie laggeada (útil para forecast futuro)."""
        if self._uses_series_forecaster():
            # Para modelos de serie directa se ajusta una vez sobre toda la serie disponible.
            series = pd.to_numeric(self.df[self.target], errors="coerce").dropna().reset_index(drop=True)
            self.model.fit(series)
            self._last_observed = series
            self._fitted_full = True
            return self

        X, y, cols = self.preparer.build_lagged_xy(
            df=self.df,
            target=self.target,
            lags=self.lags,
            features=self.features,
        )

        if self.preparer.scaler is not None:
            X = pd.DataFrame(self.preparer.scaler.fit_transform(X), columns=cols, index=X.index)

        self.model.fit(X, y)
        self._last_observed = pd.to_numeric(self.df[self.target], errors="coerce").dropna()
        self._fitted_full = True
        return self

    def forecast(self, steps: int = 1) -> np.ndarray:
        """Forecast recursivo univariado (solo lags del target)."""
        if self._uses_series_forecaster():
            if not self._fitted_full:
                self.fit_full()
            if not isinstance(steps, int) or steps <= 0:
                raise ValueError("steps debe ser entero positivo.")
            return self.model.predict(steps)

        if self.features:
            raise ValueError("forecast recursivo solo soporta target univariado (features vacías).")
        if not self._fitted_full:
            self.fit_full()
        if self._last_observed is None or len(self._last_observed) < self.lags:
            raise ValueError("No hay suficiente historial para forecast.")
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps debe ser entero positivo.")

        history = list(self._last_observed.values.astype(float))
        preds: List[float] = []

        for _ in range(steps):
            # Cada predicción nueva se reutiliza como entrada para el siguiente paso.
            lags_row = [history[-i] for i in range(1, self.lags + 1)]
            X_next = pd.DataFrame([lags_row], columns=[f"{self.target}_lag_{i}" for i in range(1, self.lags + 1)])

            if self.preparer.scaler is not None:
                X_next = pd.DataFrame(self.preparer.scaler.transform(X_next), columns=X_next.columns, index=X_next.index)

            y_hat = float(self.model.predict(X_next)[0])
            preds.append(y_hat)
            history.append(y_hat)

        return np.asarray(preds)


# =============================================================================
# 4) BÚSQUEDA DE HIPERPARÁMETROS
# =============================================================================
class ModelEvaluator:
    """Búsqueda de hiperparámetros sobre un split ya preparado.

    Salida estándar por modelo:
    {
        "NombreModelo": {
            "estimator": mejor_modelo_entrenado,
            "best_params": {...},
            "best_score": float,
            "searcher": objeto_search_cv,
        }
    }
    """

    def __init__(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        task: str = "regression",
        scoring: Optional[str] = None,
        cv: int = 5,
        random_state: Optional[int] = 42,
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.task = task.lower()
        self.scoring = scoring or self._default_scoring()
        self.cv = cv
        self.random_state = random_state
        self.runner = None

    @classmethod
    def from_runner(
        cls,
        runner: SupervisedRunner,
        scoring: Optional[str] = None,
        cv: int = 5,
    ):
        """Crea un evaluador a partir de un SupervisedRunner ya configurado."""
        if not runner._prepared:
            runner._prepare()
        evaluator = cls(
            X_train=runner.X_train,
            X_test=runner.X_test,
            y_train=runner.y_train,
            y_test=runner.y_test,
            task=runner.task,
            scoring=scoring,
            cv=cv,
            random_state=runner.preparer.random_state,
        )
        evaluator.runner = runner
        return evaluator

    def _default_scoring(self) -> str:
        if self.task == "classification":
            return "f1"
        if self.task == "regression":
            return "neg_mean_squared_error"
        raise ValueError("task debe ser 'classification' o 'regression'")

    def _default_cv(self):
        if self.runner is not None:
            return self.runner.get_cv_strategy(n_splits=self.cv, shuffle=True)

        if self.task == "classification":
            return StratifiedKFold(
                n_splits=self.cv,
                shuffle=True,
                random_state=self.random_state,
            )
        if self.task == "regression":
            return KFold(
                n_splits=self.cv,
                shuffle=True,
                random_state=self.random_state,
            )
        raise ValueError("task debe ser 'classification' o 'regression'")

    def _normalize_search_result(self, searcher) -> Dict[str, Any]:
        return {
            "estimator": searcher.best_estimator_,
            "best_params": dict(searcher.best_params_),
            "best_score": float(searcher.best_score_),
            "searcher": searcher,
        }

    def get_evolved_estimator(self, result: Dict[str, Any]):
        """Devuelve el estimador final de un resultado de búsqueda."""
        if "estimator" not in result:
            raise KeyError("El resultado no contiene la llave 'estimator'.")
        return result["estimator"]

    def exhaustive_search(self, model_spaces: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Prueba todas las combinaciones del grid con GridSearchCV.

        Formato esperado:
        {
            "RandomForest": {
                "estimator": RandomForestRegressor(...),
                "param_grid": {"n_estimators": [100, 200], "max_depth": [5, 10]},
            }
        }
        """
        results: Dict[str, Dict[str, Any]] = {}
        cv_strategy = self._default_cv()

        for name, config in model_spaces.items():
            estimator = config["estimator"]
            param_grid = config["param_grid"]

            searcher = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring=self.scoring,
                cv=cv_strategy,
                n_jobs=-1,
                refit=True,
            )
            searcher.fit(self.X_train, self.y_train)
            results[name] = self._normalize_search_result(searcher)

        return results

    def genetic_search(
        self,
        model_spaces: Dict[str, Dict[str, Any]],
        population_size: int = 10,
        generations: int = 8,
    ) -> Dict[str, Dict[str, Any]]:
        """Búsqueda evolutiva opcional con sklearn-genetic-opt."""
        try:
            from sklearn_genetic import GASearchCV
        except ImportError as exc:
            raise ImportError(
                "genetic_search requiere instalar 'sklearn-genetic-opt'. "
                "Puedes usar exhaustive_search sin dependencias extra."
            ) from exc

        results: Dict[str, Dict[str, Any]] = {}
        cv_strategy = self._default_cv()

        for name, config in model_spaces.items():
            estimator = config["estimator"]
            param_grid = config["param_grid"]

            searcher = GASearchCV(
                estimator=estimator,
                cv=cv_strategy,
                scoring=self.scoring,
                population_size=population_size,
                generations=generations,
                n_jobs=-1,
                verbose=False,
                criteria="max",
                algorithm="eaMuPlusLambda",
                param_grid=param_grid,
            )
            searcher.fit(self.X_train, self.y_train)
            results[name] = self._normalize_search_result(searcher)

        return results


# =============================================================================
# 5) NO SUPERVISADO (solo lógica; SIN plots)
# =============================================================================
class UnsupervisedRunner:
    """Embeddings / clustering, sin visualización.

    Para graficar usa Visualizer.unsup_*.
    """

    def __init__(self, name: str, X: pd.DataFrame, model, kind: str, scale_X: bool = True):
        self.name = name
        self.kind = kind.lower()
        self.model = model

        self.X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns) if scale_X else X.copy()
        self.metrics: Dict[str, object] = {}
        self.embedding_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None

        self._handlers = {
            "pca": self._fit_pca,
            "umap": self._fit_embedding,
            "tsne": self._fit_embedding,
            "kmeans": self._fit_cluster,
            "hac": self._fit_cluster,
            "cluster": self._fit_cluster,
        }

    def fit(self):
        fn = self._handlers.get(self.kind)
        if fn is None:
            raise ValueError("kind válido: pca/umap/tsne o kmeans/hac/cluster")
        fn()
        return self

    def _fit_pca(self):
        self.embedding_ = self.model.fit_transform(self.X)
        if hasattr(self.model, "explained_variance_ratio_"):
            v = np.asarray(self.model.explained_variance_ratio_)
            self.metrics.update(
                {
                    "varianza_total": float(v.sum()),
                    "varianza_pc1_pc2": float(v[:2].sum()),
                    "n_componentes": int(len(v)),
                }
            )

    def _fit_embedding(self):
        self.embedding_ = self.model.fit_transform(self.X)

    def _fit_cluster(self):
        if hasattr(self.model, "fit_predict"):
            self.labels_ = self.model.fit_predict(self.X)
        else:
            self.model.fit(self.X)
            self.labels_ = getattr(self.model, "labels_", None)

        if self.labels_ is not None:
            self.metrics["silhouette"] = float(silhouette_score(self.X, self.labels_))
        if hasattr(self.model, "inertia_"):
            self.metrics["inercia"] = float(self.model.inertia_)

    # --- operaciones numéricas para preparar visualización externa ---
    def ensure_2d_embedding(self):
        """Si no hay embedding_ (o no es 2D), crea una proyección PCA(2) para visualizar afuera."""
        if self.embedding_ is None or (isinstance(self.embedding_, np.ndarray) and self.embedding_.shape[1] != 2):
            self.embedding_ = PCA(n_components=2, random_state=0).fit_transform(self.X)
        return self.embedding_

    def evaluar_silhouette_en_embedding(self, n_clusters: int = 4):
        """Silhouette sobre embedding_ usando KMeans temporal."""
        if self.embedding_ is None:
            return None
        lab = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(self.embedding_)
        return float(silhouette_score(self.embedding_, lab))


# ==============================================================================
# 5) EDAExplorer (solo data; SIN plots)
# ==============================================================================
class EDAExplorer:
    """Carga + limpieza + transformación. (Sin visualización)

    Para graficar: usa Visualizer.eda_* y pásale eda.df.
    """

    def __init__(self, path: str, modo_csv: int = 1, num=None):
        if num is not None:
            modo_csv = num
        self._df = self._cargar_csv(path, modo_csv)

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        instance = cls.__new__(cls)
        instance._df = df.copy()
        return instance

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @df.setter
    def df(self, p_df: pd.DataFrame):
        self._df = p_df

    def _cargar_csv(self, path: str, modo_csv: int) -> pd.DataFrame:
        if modo_csv == 1:
            # Lee primero sin asumir índice. Solo promueve la primera columna a índice
            # cuando el CSV parece venir exportado con el índice de pandas.
            df = pd.read_csv(path, sep=",", decimal=".")
            first_col = str(df.columns[0]).strip() if len(df.columns) else ""
            if first_col.lower().startswith("unnamed:"):
                df = df.set_index(df.columns[0])
            return df
        if modo_csv == 2:
            return pd.read_csv(path, sep=";", decimal=".")
        raise ValueError("modo_csv debe ser 1 (sep=',', index_col=0) o 2 (sep=';').")

    # --- perfil / limpieza ---
    def tipo_datos(self):
        """Retorna dtypes (antes imprimía)."""
        return self._df.dtypes

    def solo_numericas(self):
        self._df = self._df.select_dtypes(include=["number"])
        return self

    def normalizar_columnas(self):
        self._df = self._df.copy()
        self._df.columns = self._df.columns.str.replace('"', '', regex=False).str.strip()
        return self

    def convertir_datetime(self, columnas):
        columnas = [columnas] if isinstance(columnas, str) else list(columnas)
        for col in columnas:
            if col not in self._df.columns:
                raise ValueError(f"'{col}' no existe en el DataFrame.")
            self._df[col] = pd.to_datetime(self._df[col], errors="coerce")
        return self

    def convertir_numerico(self, columnas):
        columnas = [columnas] if isinstance(columnas, str) else list(columnas)
        for col in columnas:
            if col not in self._df.columns:
                raise ValueError(f"'{col}' no existe en el DataFrame.")
            self._df[col] = pd.to_numeric(self._df[col], errors="coerce")
        return self

    def a_dummies(self, drop_first: bool = True):
        cols_cat = self._df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cols_cat:
            self._df = pd.get_dummies(self._df, columns=cols_cat, drop_first=drop_first)
            for c in self._df.columns:
                if self._df[c].dtype == bool:
                    self._df[c] = self._df[c].astype(int)
        return self

    def eliminar_columnas(self, columnas):
        idx_name = self._df.index.name
        columnas = list(columnas)
        if idx_name is not None and idx_name in columnas:
            self._df.reset_index(drop=True, inplace=True)
            columnas = [c for c in columnas if c != idx_name]
        self._df.drop(columns=columnas, inplace=True, errors="ignore")
        return self

    def renombrar_columnas(self, mapping):
        self._df.rename(columns=mapping, inplace=True)
        return self

    def valores_unicos(self, col: str):
        vc = self._df[col].value_counts(dropna=False)
        return vc

    def valores_faltantes(self):
        """Retorna conteo de nulos por columna."""
        return self._df.isna().sum()

    def eliminarDuplicados(self, columnas=None):
        subset = None if columnas is None else ([columnas] if isinstance(columnas, str) else list(columnas))
        self._df = self._df.drop_duplicates(subset=subset)
        return self

    def eliminarNulos(self):
        self._df = self._df.dropna()
        return self

    def eliminar_nulos_en(self, columnas):
        columnas = [columnas] if isinstance(columnas, str) else list(columnas)
        self._df = self._df.dropna(subset=columnas)
        return self

    def ordenar_por(self, columna: str, ascending: bool = True):
        if columna not in self._df.columns:
            raise ValueError(f"'{columna}' no existe en el DataFrame.")
        self._df = self._df.sort_values(columna, ascending=ascending).reset_index(drop=True)
        return self

    def resumen_estadistico(self) -> pd.DataFrame:
        """Retorna describe() para numéricas."""
        return self._df.describe(include="number")

    # --- features ---
    def ingenieria_tiempo(self, columna_tiempo: str):
        if columna_tiempo not in self._df.columns:
            return self
        self.convertir_datetime(columna_tiempo)
        self._df[f"{columna_tiempo}_Hour"] = self._df[columna_tiempo].dt.hour
        self._df[f"{columna_tiempo}_DayOfWeek"] = self._df[columna_tiempo].dt.dayofweek
        return self.eliminar_columnas([columna_tiempo])

    def preparar_serie_temporal(
        self,
        date_col: str,
        target_col: str,
        drop_duplicates: bool = False,
    ):
        """Normaliza un DataFrame para forecasting.

        - Limpia nombres de columnas.
        - Convierte la columna de fecha a datetime.
        - Convierte el target a numérico.
        - Elimina filas con nulos en fecha/target.
        - Ordena cronológicamente.
        - Opcionalmente elimina duplicados exactos.
        """
        self.normalizar_columnas()
        self.convertir_datetime(date_col)
        self.convertir_numerico(target_col)
        self.eliminar_nulos_en([date_col, target_col])
        if drop_duplicates:
            self.eliminarDuplicados()
        self.ordenar_por(date_col)
        return self

    def detectar_columnas_fecha(self, sample_size: int = 20, threshold: float = 0.7) -> List[str]:
        candidates = []
        for col in self._df.columns:
            if pd.api.types.is_datetime64_any_dtype(self._df[col]):
                candidates.append(col)
                continue

            sample = self._df[col].dropna().astype(str).head(sample_size)
            if sample.empty:
                continue

            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() >= threshold:
                candidates.append(col)
        return candidates

    # --- correlaciones (data) ---
    def correlaciones(self) -> pd.DataFrame:
        return self._df.corr(numeric_only=True)

    def correlacion_con_target(self, target_col: str) -> pd.Series:
        if target_col not in self._df.columns:
            raise ValueError(f"'{target_col}' no existe en el DataFrame.")
        corr = self._df.corr(numeric_only=True)[target_col].drop(target_col, errors="ignore")
        return corr.sort_values(ascending=False)

    # --- pipelines rápidos ---
    def analisisCompleto(self, convertir_dummies: bool = True, drop_first: bool = True):
        """Ejecuta un EDA rápido (calidad + dummies opcional)."""
        # No imprime: retorna self para chaining.
        self.eliminarDuplicados()
        if convertir_dummies:
            self.a_dummies(drop_first=drop_first)
        return self

    def analisis(self):
        """Vista rápida: retorna un dict con shape/head/dtypes."""
        return {
            "shape": self._df.shape,
            "head": self._df.head(),
            "dtypes": self._df.dtypes,
        }


# =============================================================================
# Funciones para comparar resultados fuera de los runners
# =============================================================================
def compare_unsupervised(models: List[UnsupervisedRunner], metrics: List[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"modelo": m.name, "tipo": m.kind, **{k: m.metrics.get(k, np.nan) for k in metrics}} for m in models]
    )

def pick_best(df: pd.DataFrame, metric: str, higher_is_better: bool = True) -> str:
    idx = df[metric].idxmax() if higher_is_better else df[metric].idxmin()
    return df.loc[idx, "modelo"]

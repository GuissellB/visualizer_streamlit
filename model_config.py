MODEL_DEFAULT_PARAMS = {
    "Regresión Logística": {
        "solver": "liblinear",
        "C": 1.0,
        "penalty": "l2",
        "max_iter": 100,
    },
    "Random Forest": {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
    "SVM": {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        "probability": True,
    },
    "XGBoost": {
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.3,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    },
    "LightGBM": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": -1,
        "num_leaves": 31,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    },
}

SHARED_PARAM_SCHEMA = {
    "n_estimators": {"label": "n_estimators", "type": "int", "help": "Número de estimadores si el modelo lo soporta."},
    "max_depth": {"label": "max_depth", "type": "int", "help": "Profundidad máxima si aplica al modelo."},
    "learning_rate": {"label": "learning_rate", "type": "float", "help": "Tasa de aprendizaje para boosters."},
    "subsample": {"label": "subsample", "type": "float", "help": "Proporción de muestras usadas por iteración."},
    "colsample_bytree": {"label": "colsample_bytree", "type": "float", "help": "Proporción de variables por árbol."},
}

MODEL_PARAM_SCHEMA = {
    "Regresión Logística": {
        "C": {"label": "C", "type": "float"},
        "solver": {"label": "solver", "type": "select", "options": ["liblinear", "lbfgs", "newton-cg", "saga"]},
        "penalty": {"label": "penalty", "type": "select", "options": ["l1", "l2", "elasticnet", "None"]},
        "max_iter": {"label": "max_iter", "type": "int"},
    },
    "Random Forest": {
        "min_samples_split": {"label": "min_samples_split", "type": "int"},
        "min_samples_leaf": {"label": "min_samples_leaf", "type": "int"},
        "max_features": {"label": "max_features", "type": "select", "options": ["sqrt", "log2", "None"]},
    },
    "SVM": {
        "C": {"label": "C", "type": "float"},
        "kernel": {"label": "kernel", "type": "select", "options": ["linear", "poly", "rbf", "sigmoid"]},
        "gamma": {"label": "gamma", "type": "select_or_text", "options": ["scale", "auto"]},
    },
    "XGBoost": {
        "eval_metric": {"label": "eval_metric", "type": "select", "options": ["logloss", "auc", "error"]},
    },
    "LightGBM": {
        "num_leaves": {"label": "num_leaves", "type": "int"},
    },
}

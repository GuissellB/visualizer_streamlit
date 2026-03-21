TARGET_COL = "result"
DATA_PATH = "fixed_values_ds.csv"
RANDOM_STATE = 42
N_SPLITS = 10
SEMILLAS = [1, 7, 21, 42, 99]
CLASS_WEIGHT = "balanced"

DEFAULT_BEST_MODEL_CRITERION = "ROC_AUC_CV_mean"
DEFAULT_NO_CV_CRITERION = "ROC_AUC_Global"

CV_CRITERION_OPTIONS = {
    "ROC_AUC_CV_mean": "ROC AUC (CV)",
    "F1_CV_mean": "F1 (CV)",
    "Accuracy_CV_mean": "Accuracy (CV)",
}

NO_CV_CRITERION_OPTIONS = {
    "ROC_AUC_Global": "ROC AUC (Global)",
    "F1_Global": "F1 (Global)",
    "Accuracy_Global": "Accuracy (Global)",
}

BALANCE_METHOD_OPTIONS = {
    "none": "Sin balanceo",
    "class_weight": "Pesos automáticos del modelo",
    "undersample": "Submuestreo (NearMiss)",
    "oversample": "Sobremuestreo (RandomOverSampler)",
    "smote_tomek": "SMOTE + Tomek",
}

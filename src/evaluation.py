from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold
)


import matplotlib.pyplot as plt

CV_FOLDS = 5
RANDOM_STATE = 42

def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Train and evaluate a binary classification model.
    """

    # ==========================
    # Train Model
    # ==========================
    model.fit(X_train, y_train)

    # ==========================
    # Predictions
    # ==========================
    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    # ==========================
    # Metrics
    # ==========================
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    roc_auc = roc_auc_score(y_test, y_prob)

    # ==========================
    # Print Metrics
    # ==========================

    print("=" * 60)
    print(model.named_steps["classifier"].__class__.__name__)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")

    # ==========================
    # Classification Report
    # ==========================

    print("\n")
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(classification_report(y_test, y_pred))

    # ==========================
    # Confusion Matrix
    # ==========================

    plt.figure(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        cmap="Blues"
    )

    plt.title("Confusion Matrix")

    plt.show()

    # ==========================
    # ROC Curve
    # ==========================

    RocCurveDisplay.from_predictions(
        y_test,
        y_prob
    )

    plt.title("ROC Curve")

    plt.show()

    # ==========================
    # Precision Recall Curve
    # ==========================

    PrecisionRecallDisplay.from_predictions(
        y_test,
        y_prob
    )

    plt.title("Precision-Recall Curve")

    plt.show()

    return {

    "model": model,

    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "f1": f1,

    "roc_auc": roc_auc,

    "predictions": y_pred,

    "probabilities": y_prob

}


def cross_validation_score(
    model,
    X_train,
    y_train
):

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    print("=" * 60)
    print("5-FOLD STRATIFIED CROSS VALIDATION")
    print("=" * 60)

    print(scores)

    print()

    print(f"Mean ROC-AUC : {scores.mean():.4f}")

    print(f"Std Dev      : {scores.std():.4f}")

    return scores



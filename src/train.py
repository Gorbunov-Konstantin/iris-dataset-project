import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json

def train_and_evaluate(train_file, test_file, model_file, metrics_file):
    # Загрузка данных
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    X_train, y_train = train.drop(columns=["species"]), train["species"]
    X_test, y_test = test.drop(columns=["species"]), test["species"]

    # Инициализация списка для метрик
    metrics = []
    max_depths = [1,2,3,4,5]
    n_estimators = 1
    # Обучение модели для различных значений max_depth
    for max_depth in max_depths:
        model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        # Предсказания и вычисление метрик
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Сохранение метрик в список
        metrics.append({
            "max_depth": max_depth,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

    # Сохранение всех метрик в CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(metrics_file, index=False)

    # Сохранение модели
    joblib.dump(model, model_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to training dataset")
    parser.add_argument("--test", required=True, help="Path to testing dataset")
    parser.add_argument("--model", required=True, help="Path to save the model")
    parser.add_argument("--metrics", required=True, help="Path to save metrics in JSON format")
    args = parser.parse_args()

    train_and_evaluate(args.train, args.test, args.model, args.metrics)

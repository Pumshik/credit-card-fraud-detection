# Credit Card Fraud Detection

Обнаружение мошеннических транзакций по банковским картам с использованием
методов машинного обучения.

## Данные

Использован датасет Credit Card Fraud Detection с Kaggle  
(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).  
Содержит 284 807 транзакций, из которых 492 (0.172%) являются мошенническими.  
Признаки V1–V28 получены методом PCA.  
Дополнительные признаки: Time (секунды от первой транзакции) и Amount (сумма).

## Задача и метрики

Бинарная классификация с экстремальным дисбалансом классов.  
Основная метрика оптимизации - Recall (полнота обнаружения мошенничества).  
Дополнительно отслеживались Precision, F1, ROC-AUC и AUPRC
(площадь под Precision-Recall кривой).

## Структура репозитория
notebooks/

    1_EDA_and_Preprocessing.ipynb
    
        Загрузка данных, разведочный анализ, масштабирование признаков,
        разделение на train/test.
        
    2_Model_Training_and_Tuning.ipynb
    
        Baseline-модели, борьба с дисбалансом (class_weight, SMOTE),
        
        XGBoost и LightGBM, RandomizedSearchCV, SHAP-интерпретация.

app/

    main.py
    
        FastAPI-приложение, эндпоинт /predict.
    requirements.txt
    
        Зависимости для запуска сервиса.
        
    best_model.pkl
    
        Сериализованная модель (генерируется ноутбуком 2).
    scaler_amount.pkl
    
        StandardScaler для признака Amount.
        
    scaler_time.pkl
    
        StandardScaler для признака Time.

.gitignore

README.md

## Установка и запуск API

1. Клонировать репозиторий, перейти в папку app/.
2. Создать виртуальное окружение (Python 3.12+):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Установить зависимости:
```bash
   pip install -r requirements.txt
```

4. Перенести файлы best_model.pkl, scaler_amount.pkl, scaler_time.pkl в папку app/ (они генерируются вторым ноутбуком).

5. Запустить сервер:
```bash
   python main.py
```

6. Swagger-документация:  ``` http://127.0.0.1:8000/docs ```

## Пример запроса к /predict
```bash

curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"Time": 123456, "V1": -1.2, "V2": 0.5, "V3": 1.4, "V4": -0.8, "V5": 0.3, "V6": -0.2, "V7": 0.1, "V8": 0.05, "V9": -0.1, "V10": 0.2, "V11": 0.03, "V12": 0.7, "V13": -0.3, "V14": -0.5, "V15": 0.4, "V16": 0.2, "V17": -0.1, "V18": 0.05, "V19": 0.01, "V20": -0.02, "V21": 0.1, "V22": 0.07, "V23": -0.03, "V24": 0.2, "V25": 0.04, "V26": -0.08, "V27": 0.02, "V28": 0.01, "Amount": 100.0}'
```

Ответ:

{"fraud_probability": 0.87, "is_fraud": true}

## Результаты моделирования
```
Модель                              Recall  Precision  AUPRC
LogisticRegression (baseline)       0.00    0.00       0.17
LogisticRegression + class_weight   0.61    0.75       0.74
RandomForest + class_weight         0.75    0.89       0.86
XGBoost + scale_pos_weight          0.78    0.90       0.88
LightGBM + is_unbalance             0.76    0.88       0.87
XGBoost (RandomizedSearchCV)        0.84    0.83       0.90
```

Лучшая модель – XGBoost после подбора гиперпараметров – обнаруживает 84% мошеннических транзакций при точности 83% (AUPRC 0.90). Интерпретация SHAP показала, что наибольший вклад в предсказание вносят признаки V14, V4, V12.

## Используемые технологии

Python 3.12, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Imbalanced-learn (SMOTE), SHAP, FastAPI, Uvicorn, Joblib.

## Возможные улучшения

- Добавить временные фичи (цикличность, час, день) вместо масштабирования Time.
- Построить ансамбль моделей с динамическим порогом для повышения Recall.
- Реализовать пайплайн предобработки в API через sklearn Pipeline.
- Упаковать в Docker и развернуть на облачном сервере.
- Внедрить мониторинг дрейфа признаков в реальном времени.

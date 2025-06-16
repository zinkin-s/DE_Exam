# etl/model_trainer.py

import numpy as np
import pandas as pd
import joblib
import json
import logging
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple, Dict

# Настройка логирования
logging.basicConfig(
    filename='logs/model_trainer.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BreastCancerModelTrainer:
    def __init__(self, train_data_path: str, model_output_path: str, metrics_output_path: str):
        """
        Инициализация тренера модели
        
        Args:
            train_data_path: путь к файлу с обучающими данными
            model_output_path: путь для сохранения обученной модели
            metrics_output_path: путь для сохранения метрик
        """
        self.train_data_path = train_data_path
        self.model_output_path = model_output_path
        self.metrics_output_path = metrics_output_path
        self.model = None
        self.metrics = {}
    
    def load_data(self) -> bool:
        """Загрузка обучающих данных"""
        try:
            if not os.path.exists(self.train_data_path):
                logger.error(f"Файл обучающих данных {self.train_data_path} не найден")
                return False
                
            # Загрузка данных
            data = pd.read_csv(self.train_data_path)
            
            # Разделение на признаки и целевую переменную
            self.X = data.drop('diagnosis', axis=1)
            self.y = data['diagnosis']
            
            logger.info(f"Данные загружены успешно. Форма данных: {self.X.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            return False
    
    def train_model(self, 
                   penalty: str = 'l2', 
                   C: float = 1.0, 
                   solver: str = 'lbfgs',
                   max_iter: int = 100,
                   random_state: int = 42) -> bool:
        """Обучение модели логистической регрессии
        
        Args:
            penalty: тип регуляризации
            C: обратная сила регуляризации
            solver: алгоритм оптимизации
            max_iter: максимальное число итераций
            random_state: зерно для случайных чисел
            
        Returns:
            bool: успешность обучения
        """
        try:
            # Инициализация модели
            self.model = LogisticRegression(
                penalty=penalty,
                C=C,
                solver=solver,
                max_iter=max_iter,
                random_state=random_state
            )
            
            # Обучение модели
            self.model.fit(self.X, self.y)
            
            # Сохранение параметров модели
            self.model_params = {
                'penalty': penalty,
                'C': C,
                'solver': solver,
                'max_iter': max_iter,
                'random_state': random_state
            }
            
            logger.info("Модель успешно обучена")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обучения модели: {e}")
            return False
    
    def evaluate_model(self, test_data_path: str) -> bool:
        """Оценка модели на тестовых данных
        
        Args:
            test_data_path: путь к файлу с тестовыми данными
            
        Returns:
            bool: успешность оценки
        """
        try:
            if not os.path.exists(test_data_path):
                logger.error(f"Файл тестовых данных {test_data_path} не найден")
                return False
                
            # Загрузка тестовых данных
            test_data = pd.read_csv(test_data_path)
            X_test = test_data.drop('diagnosis', axis=1)
            y_test = test_data['diagnosis']
            
            # Предсказание
            y_pred = self.model.predict(X_test)
            
            # Расчет метрик
            self.metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred)),
                'recall': float(recall_score(y_test, y_pred)),
                'f1_score': float(f1_score(y_test, y_pred)),
                'model_params': self.model_params,
                'test_samples': len(X_test),
                'train_samples': len(self.X)
            }
            
            logger.info(f"Модель оценена на {len(X_test)} тестовых образцах")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка оценки модели: {e}")
            return False
    
    def save_model(self) -> bool:
        """Сохранение модели в файл
        
        Returns:
            bool: успешность сохранения
        """
        try:
            # Создание директории, если она не существует
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            
            # Сохранение модели
            joblib.dump(self.model, self.model_output_path)
            
            logger.info(f"Модель сохранена в {self.model_output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка сохранения модели: {e}")
            return False
    
    def save_metrics(self) -> bool:
        """Сохранение метрик в файл
        
        Returns:
            bool: успешность сохранения
        """
        try:
            # Создание директории, если она не существует
            os.makedirs(os.path.dirname(self.metrics_output_path), exist_ok=True)
            
            # Сохранение метрик
            with open(self.metrics_output_path, 'w') as f:
                json.dump(self.metrics, f, indent=4)
            
            logger.info(f"Метрики сохранены в {self.metrics_output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка сохранения метрик: {e}")
            return False
    
    def train_and_evaluate(self, test_data_path: str) -> bool:
        """Обучение и оценка модели
        
        Args:
            test_data_path: путь к файлу с тестовыми данными
            
        Returns:
            bool: успешность выполнения
        """
        try:
            # Загрузка данных
            if not self.load_data():
                return False
            
            # Обучение модели
            if not self.train_model():
                return False
            
            # Оценка модели
            if not self.evaluate_model(test_data_path):
                return False
            
            # Сохранение модели и метрик
            if not self.save_model():
                return False
                
            if not self.save_metrics():
                return False
            
            logger.info("Модель успешно обучена и оценена")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка в процессе обучения и оценки: {e}")
            return False

if __name__ == "__main__":
    # Пример использования
    train_data_path = "data/processed/train_data.csv"
    test_data_path = "data/processed/test_data.csv"
    model_output_path = "models/logistic_regression_model.pkl"
    metrics_output_path = "metrics/logistic_regression_metrics.json"
    
    trainer = BreastCancerModelTrainer(train_data_path, model_output_path, metrics_output_path)
    
    if trainer.train_and_evaluate(test_data_path):
        print("Модель успешно обучена и оценена")
    else:
        print("Ошибка в процессе обучения и оценки")
        
# etl/data_preprocessor.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import logging
import os
import json

# Настройка логирования
logging.basicConfig(
    filename='logs/data_preprocessor.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, input_path, output_dir):
        """
        Инициализация процессора данных
        
        Args:
            input_path: путь к входному файлу с данными
            output_dir: директория для сохранения обработанных данных
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.label_encoder = None
        self.metadata = {
            'class_distribution': {},
            'feature_means': {},
            'feature_stds': {},
            'preprocessing_steps': []
        }
    
    def load_data(self):
        """Загрузка данных из CSV файла"""
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Данные успешно загружены из {self.data_path}")
            self.metadata['preprocessing_steps'].append("Данные загружены")
            return True
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {e}")
            return False
    
    def check_data_quality(self):
        """Проверка качества данных"""
        try:
            # Проверка дубликатов
            duplicates = self.data.duplicated().sum()
            if duplicates > 0:
                logger.warning(f"Найдено {duplicates} дубликатов в данных")
            
            # Проверка пропущенных значений
            missing_values = self.data.isnull().sum().sum()
            if missing_values > 0:
                logger.warning(f"Найдено {missing_values} пропущенных значений в данных")
            
            # Проверка наличия целевой колонки
            if 'diagnosis' not in self.data.columns:
                logger.error("Отсутствует целевая колонка 'diagnosis'")
                return False
            
            # Проверка, что целевая колонка содержит только 'M' и 'B'
            if not set(self.data['diagnosis'].unique()).issubset({'M', 'B'}):
                logger.error("Целевая колонка содержит неподдерживаемые значения")
                return False
            
            logger.info("Проверка качества данных завершена успешно")
            self.metadata['preprocessing_steps'].append("Проверка качества данных")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при проверке качества данных: {e}")
            return False
    
    def clean_data(self):
        """Очистка данных"""
        try:
            # Удаление дубликатов
            self.data.drop_duplicates(inplace=True)
            
            # Удаление колонки 'id' - не нужна для обучения модели
            if 'id' in self.data.columns:
                self.data.drop(columns=['id'], inplace=True)
            
            # Заполнение пропущенных значений средними значениями
            # Предположим, что пропущенных значений не должно быть, но на всякий случай
            self.data.fillna(self.data.mean(), inplace=True)
            
            logger.info("Данные успешно очищены")
            self.metadata['preprocessing_steps'].append("Данные очищены")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при очистке данных: {e}")
            return False
    
    def encode_target(self):
        """Кодирование целевой переменной"""
        try:
            # Создание label encoder
            self.label_encoder = LabelEncoder()
            
            # Кодирование целевой переменной
            self.data['diagnosis'] = self.label_encoder.fit_transform(self.data['diagnosis'])
            
            # Сохранение соответствия классов
            classes = dict(enumerate(self.label_encoder.classes_))
            self.metadata['class_mapping'] = {v: k for k, v in classes.items()}
            
            logger.info("Целевая переменная успешно закодирована")
            self.metadata['preprocessing_steps'].append("Целевая переменная закодирована")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при кодировании целевой переменной: {e}")
            return False
    
    def normalize_features(self):
        """Нормализация признаков"""
        try:
            # Разделение признаков и целевой переменной
            X = self.data.drop(columns=['diagnosis'])
            y = self.data['diagnosis']
            
            # Нормализация признаков
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Объединение признаков и целевой переменной
            self.data = pd.DataFrame(X_scaled, columns=X.columns)
            self.data['diagnosis'] = y
            
            # Сохранение статистики признаков
            self.metadata['feature_means'] = dict(zip(X.columns, self.scaler.mean_))
            self.metadata['feature_stds'] = dict(zip(X.columns, np.sqrt(self.scaler.var_)))
            
            logger.info("Признаки успешно нормализованы")
            self.metadata['preprocessing_steps'].append("Признаки нормализованы")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при нормализации признаков: {e}")
            return False
    
    def split_data(self, test_size=0.2, random_state=42):
        """Разделение данных на обучающую и тестовую выборки"""
        try:
            # Разделение признаков и целевой переменной
            X = self.data.drop(columns=['diagnosis'])
            y = self.data['diagnosis']
            
            # Разделение на обучающую и тестовую выборки
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Сохранение информации о разделении
            self.metadata['train_samples'] = len(self.X_train)
            self.metadata['test_samples'] = len(self.X_test)
            self.metadata['test_size'] = test_size
            
            # Сохранение информации о распределении классов
            self.metadata['class_distribution']['original'] = dict(y.value_counts())
            self.metadata['class_distribution']['train'] = dict(self.y_train.value_counts())
            self.metadata['class_distribution']['test'] = dict(self.y_test.value_counts())
            
            logger.info(f"Данные успешно разделены на обучающую и тестовую выборки (test_size={test_size})")
            self.metadata['preprocessing_steps'].append("Данные разделены")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при разделении данных: {e}")
            return False
    
    def save_split_data(self):
        """Сохранение разделенных данных в CSV файлы"""
        try:
            # Создание директории для сохранения, если ее нет
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Сохранение обучающей выборки
            pd.concat([self.X_train, self.y_train], axis=1).to_csv(
                os.path.join(self.output_dir, 'train_data.csv'), index=False
            )
            
            # Сохранение тестовой выборки
            pd.concat([self.X_test, self.y_test], axis=1).to_csv(
                os.path.join(self.output_dir, 'test_data.csv'), index=False
            )
            
            # Сохранение X_train, X_test, y_train, y_test отдельно
            self.X_train.to_csv(os.path.join(self.output_dir, 'X_train.csv'), index=False)
            self.X_test.to_csv(os.path.join(self.output_dir, 'X_test.csv'), index=False)
            self.y_train.to_csv(os.path.join(self.output_dir, 'y_train.csv'), index=False)
            self.y_test.to_csv(os.path.join(self.output_dir, 'y_test.csv'), index=False)
            
            logger.info(f"Разделенные данные успешно сохранены в {self.output_dir}")
            self.metadata['preprocessing_steps'].append("Данные сохранены")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении разделенных данных: {e}")
            return False
    
    def save_metadata(self):
        """Сохранение метаданных в JSON файл"""
        try:
            # Создание директории для сохранения, если ее нет
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Добавление информации о времени обработки
            self.metadata['processing_time'] = pd.Timestamp.now().isoformat()
            
            # Сохранение метаданных
            with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
                json.dump(self.metadata, f, indent=4)
                
            logger.info(f"Метаданные успешно сохранены в {self.output_dir}/metadata.json")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении метаданных: {e}")
            return False
    
    def preprocess(self, test_size=0.2, random_state=42):
        """Основной метод для выполнения всего процесса предобработки"""
        try:
            # Загрузка данных
            if not self.load_data():
                return False
            
            # Проверка качества данных
            if not self.check_data_quality():
                return False
            
            # Очистка данных
            if not self.clean_data():
                return False
            
            # Кодирование целевой переменной
            if not self.encode_target():
                return False
            
            # Нормализация признаков
            if not self.normalize_features():
                return False
            
            # Разделение данных
            if not self.split_data(test_size, random_state):
                return False
            
            # Сохранение разделенных данных
            if not self.save_split_data():
                return False
            
            # Сохранение метаданных
            if not self.save_metadata():
                return False
            
            logger.info("Процесс предобработки данных завершен успешно")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка в процессе предобработки: {e}")
            return False

if __name__ == "__main__":
    # Пример использования
    input_path = "data/raw/wdbc.csv"
    output_dir = "data/processed"
    
    preprocessor = DataPreprocessor(input_path, output_dir)
    
    if preprocessor.preprocess(test_size=0.25):
        print("Данные успешно предобработаны и сохранены")
    else:
        print("Ошибка при предобработке данных")
        
# dags/pipeline_dag.py

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago
from datetime import timedelta, datetime
import logging
import os

# Настройка пути к проекту
AIRFLOW_HOME = os.environ.get('AIRFLOW_HOME')
PROJECT_DIR = os.path.dirname(AIRFLOW_HOME)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
METRICS_DIR = os.path.join(PROJECT_DIR, 'metrics')
CREDENTIALS_PATH = os.path.join(PROJECT_DIR, 'credentials.json')

# Настройка логирования
logger = logging.getLogger(__name__)

# Импорты наших компонентов
try:
    from etl.data_uploader import RawDataUploader
    from etl.data_preprocessor import DataPreprocessor
    from etl.model_trainer import BreastCancerModelTrainer
except ImportError as e:
    logger.error(f"Ошибка импорта модулей: {e}")
    raise

# Параметры по умолчанию для DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email': ['airflow@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=60),
    'provide_context': True
}

# Инициализация DAG
dag = DAG(
    'breast_cancer_pipeline',
    default_args=default_args,
    description='ETL-процесс для диагностики рака груди',
    schedule_interval='@daily',
    catchup=False,
    tags=['breast_cancer', 'etl', 'ml']
)

# Функции для PythonOperators
def upload_raw_data(**kwargs):
    """Функция для загрузки сырых данных в Google Drive"""
    ti = kwargs['ti']
    
    try:
        # Пути к данным
        input_path = os.path.join(RAW_DATA_DIR, 'wdbc.csv')
        folder_name = 'Breast Cancer Raw Data'
        
        # Создание загрузчика
        uploader = RawDataUploader(credentials_file=CREDENTIALS_PATH)
        
        # Проверка валидности данных
        if not uploader.validate_raw_data(input_path):
            logger.error("Данные не прошли проверку валидности")
            raise ValueError("Данные не прошли проверку валидности")
        
        # Проверка, существует ли уже папка
        service = build('drive', 'v3', credentials=uploader.creds)
        results = service.files().list(
            q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'", 
            spaces='drive', 
            fields='files(id, name)'
        ).execute()
        
        folders = results.get('files', [])
        folder_id = folders[0].get('id') if folders else None
        
        if not folder_id:
            # Создание новой папки
            folder_id = uploader.create_folder(folder_name)
        
        # Загрузка файла
        file_id = uploader.upload_raw_data(
            input_path, 
            folder_id=folder_id,
            description="Сырые данные о раке груди (Breast Cancer Wisconsin Diagnostic Dataset)",
            overwrite=True
        )
        
        # Сохранение file_id в XCom
        ti.xcom_push(key='raw_data_file_id', value=file_id)
        logger.info(f"Сырые данные успешно загружены. File ID: {file_id}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка загрузки сырых данных: {e}")
        raise

def preprocess_data(**kwargs):
    """Функция для предобработки данных"""
    ti = kwargs['ti']
    
    try:
        # Пути к данным
        input_path = os.path.join(RAW_DATA_DIR, 'wdbc.csv')
        output_dir = os.path.join(PROCESSED_DATA_DIR, 'breast_cancer')
        
        # Создание процессора данных
        preprocessor = DataPreprocessor(input_path, output_dir)
        
        # Выполнение предобработки
        if not preprocessor.preprocess(test_size=0.25):
            logger.error("Ошибка при предобработке данных")
            raise ValueError("Ошибка при предобработке данных")
        
        # Сохранение информации о разделении данных
        ti.xcom_push(key='train_samples', value=preprocessor.metadata['train_samples'])
        ti.xcom_push(key='test_samples', value=preprocessor.metadata['test_samples'])
        ti.xcom_push(key='class_distribution', value=preprocessor.metadata['class_distribution'])
        
        logger.info("Данные успешно предобработаны")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка предобработки данных: {e}")
        raise

def train_model(**kwargs):
    """Функция для обучения модели"""
    ti = kwargs['ti']
    
    try:
        # Пути к данным
        train_data_path = os.path.join(PROCESSED_DATA_DIR, 'breast_cancer', 'train_data.csv')
        test_data_path = os.path.join(PROCESSED_DATA_DIR, 'breast_cancer', 'test_data.csv')
        model_output_path = os.path.join(MODELS_DIR, 'logistic_regression_model.pkl')
        metrics_output_path = os.path.join(METRICS_DIR, 'logistic_regression_metrics.json')
        
        # Создание тренера модели
        trainer = BreastCancerModelTrainer(train_data_path, model_output_path, metrics_output_path)
        
        # Обучение и оценка модели
        if not trainer.train_and_evaluate(test_data_path):
            logger.error("Ошибка при обучении и оценке модели")
            raise ValueError("Ошибка при обучении и оценке модели")
        
        # Сохранение метрик в XCom
        ti.xcom_push(key='model_metrics', value=trainer.metrics)
        ti.xcom_push(key='model_params', value=trainer.model_params)
        
        logger.info("Модель успешно обучена и оценена")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка обучения модели: {e}")
        raise

def upload_model(**kwargs):
    """Функция для выгрузки модели в Google Drive"""
    ti = kwargs['ti']
    model_path = os.path.join(MODELS_DIR, 'logistic_regression_model.pkl')
    
    try:
        # Получение информации о модели из XCom
        model_metrics = ti.xcom_pull(key='model_metrics', task_ids='train_model')
        model_params = ti.xcom_pull(key='model_params', task_ids='train_model')
        
        # Создание загрузчика
        uploader = RawDataUploader(credentials_file=CREDENTIALS_PATH)
        
        # Создание папки
        folder_name = 'Breast Cancer Models'
        folder_id = uploader.create_folder(folder_name)
        
        # Формирование описания
        description = (
            f"Логистическая регрессия для диагностики рака груди\n"
            f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"Параметры: {json.dumps(model_params, indent=2)}\n"
            f"Метрики: {json.dumps(model_metrics, indent=2)}"
        )
        
        # Загрузка модели
        file_id = uploader.upload_raw_data(
            model_path,
            folder_id=folder_id,
            description=description,
            overwrite=True
        )
        
        # Сохранение file_id в XCom
        ti.xcom_push(key='model_file_id', value=file_id)
        logger.info(f"Модель успешно загружена. File ID: {file_id}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        raise

def upload_metrics(**kwargs):
    """Функция для выгрузки метрик в Google Drive"""
    ti = kwargs['ti']
    
    try:
        # Получение информации о модели из XCom
        model_metrics = ti.xcom_pull(key='model_metrics', task_ids='train_model')
        model_file_id = ti.xcom_pull(key='model_file_id', task_ids='upload_model')
        
        # Создание файла метрик
        metrics_dir = METRICS_DIR
        metrics_file = os.path.join(metrics_dir, 'logistic_regression_metrics.json')
        
        # Сохранение метрик в файл
        with open(metrics_file, 'w') as f:
            json.dump(model_metrics, f, indent=4)
        
        # Создание загрузчика
        uploader = RawDataUploader(credentials_file=CREDENTIALS_PATH)
        
        # Получение папки модели
        if model_file_id:
            # Получаем папку, в которую была загружена модель
            service = build('drive', 'v3', credentials=uploader.creds)
            model_file = service.files().get(fileId=model_file_id, fields='parents').execute()
            folder_id = model_file.get('parents', [])[0] if model_file.get('parents') else None
            
            if folder_id:
                # Загрузка метрик в ту же папку
                file_id = uploader.upload_raw_data(
                    metrics_file,
                    folder_id=folder_id,
                    description="Метрики модели логистической регрессии",
                    overwrite=True
                )
                
                # Сохранение file_id в XCom
                ti.xcom_push(key='metrics_file_id', value=file_id)
                logger.info(f"Метрики успешно загружены. File ID: {file_id}")
                return True
        
        # Если не удалось определить папку модели, загружаем в отдельную папку
        folder_name = 'Breast Cancer Model Metrics'
        folder_id = uploader.create_folder(folder_name)
        
        file_id = uploader.upload_raw_data(
            metrics_file,
            folder_id=folder_id,
            description="Метрики модели логистической регрессии",
            overwrite=True
        )
        
        ti.xcom_push(key='metrics_file_id', value=file_id)
        logger.info(f"Метрики успешно загружены. File ID: {file_id}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка загрузки метрик: {e}")
        raise

# Определение задач
start = DummyOperator(
    task_id='start',
    dag=dag
)

end = DummyOperator(
    task_id='end',
    dag=dag
)

upload_raw_data_task = PythonOperator(
    task_id='upload_raw_data',
    python_callable=upload_raw_data,
    dag=dag
)

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

upload_model_task = PythonOperator(
    task_id='upload_model',
    python_callable=upload_model,
    dag=dag
)

upload_metrics_task = PythonOperator(
    task_id='upload_metrics',
    python_callable=upload_metrics,
    dag=dag
)

# Определение зависимостей
start >> upload_raw_data_task >> preprocess_data_task >> train_model_task
train_model_task >> upload_model_task >> upload_metrics_task >> end


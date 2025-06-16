# etl/cloud_uploader.py

import os
import logging
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Настройка логирования
logging.basicConfig(
    filename='logs/cloud_uploader.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GoogleDriveUploader:
    def __init__(self, credentials_file=None, token_file='token.pickle'):
        """
        Инициализация Google Drive загрузчика
        
        Args:
            credentials_file (str): Путь к файлу учетных данных Google API
            token_file (str): Путь к файлу токена для сохранения/восстановления сессии
        """
        self.credentials_file = credentials_file or os.getenv('GOOGLE_CREDENTIALS_PATH')
        self.token_file = token_file
        self.scopes = ['https://www.googleapis.com/auth/drive.file']
        self.creds = self._authenticate()
        
    def _authenticate(self):
        """Аутентификация с Google Drive API"""
        try:
            # Попытка загрузить существующий токен
            if os.path.exists(self.token_file):
                with open(self.token_file, 'rb') as token:
                    creds = pickle.load(token)
                    if creds and creds.valid:
                        return creds
                    
            # Если токен недействителен или отсутствует
            if not os.path.exists(self.credentials_file):
                logger.error(f"Файл учетных данных {self.credentials_file} не найден")
                raise FileNotFoundError(f"Файл учетных данных {self.credentials_file} не найден")
                
            flow = InstalledAppFlow.from_client_secrets_file(
                self.credentials_file, self.scopes)
            
            creds = flow.run_local_server(port=0)
            
            # Сохранение токена для последующего использования
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)
                
            return creds
            
        except Exception as e:
            logger.error(f"Ошибка аутентификации в Google Drive: {e}")
            raise

    def upload_file(self, file_path, folder_id=None):
        """
        Выгрузка файла в Google Drive
        
        Args:
            file_path (str): Путь к локальному файлу для загрузки
            folder_id (str): ID папки в Google Drive, куда загружать (опционально)
            
        Returns:
            str: ID загруженного файла в Google Drive
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"Файл {file_path} не найден")
                raise FileNotFoundError(f"Файл {file_path} не найден")
                
            service = build('drive', 'v3', credentials=self.creds)
            
            # Получение информации о файле
            file_name = os.path.basename(file_path)
            
            # Подготовка метаданных
            file_metadata = {
                'name': file_name,
                'description': 'Модель машинного обучения'
            }
            
            if folder_id:
                file_metadata['parents'] = [folder_id]
            
            media = MediaFileUpload(file_path, resumable=True)
            
            # Загрузка файла
            request = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id,webViewLink'
            )
            
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    logger.info(f"Загрузка {int(status.progress() * 100)}%")
                    
            file_id = response.get('id')
            file_link = response.get('webViewLink')
            
            logger.info(f"Файл успешно загружен: {file_link}")
            return file_id
            
        except Exception as e:
            logger.error(f"Ошибка загрузки файла в Google Drive: {e}")
            raise

    def create_folder(self, folder_name, parent_folder_id=None):
        """
        Создание папки в Google Drive
        
        Args:
            folder_name (str): Название папки
            parent_folder_id (str): ID родительской папки (опционально)
            
        Returns:
            str: ID созданной папки
        """
        try:
            service = build('drive', 'v3', credentials=self.creds)
            
            # Подготовка метаданных папки
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            if parent_folder_id:
                folder_metadata['parents'] = [parent_folder_id]
                
            folder = service.files().create(body=folder_metadata, fields='id').execute()
            folder_id = folder.get('id')
            
            logger.info(f"Папка создана: {folder_id}")
            return folder_id
            
        except Exception as e:
            logger.error(f"Ошибка создания папки в Google Drive: {e}")
            raise

if __name__ == "__main__":
    # Пример использования
    uploader = GoogleDriveUploader(credentials_file='credentials.json')
    
    # Создание папки для результатов
    results_folder_id = uploader.create_folder("Breast Cancer Models")
    
    # Загрузка модели
    model_file_path = "models/logistic_regression_model.pkl"
    model_id = uploader.upload_file(model_file_path, results_folder_id)
    
    # Загрузка метрик
    metrics_file_path = "metrics/model_metrics.json"
    metrics_id = uploader.upload_file(metrics_file_path, results_folder_id)
    
    logger.info("Процесс загрузки завершен")
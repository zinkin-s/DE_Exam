# etl/data_uploader.py

import os
import logging
from datetime import datetime
from typing import Optional
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Настройка логирования
logging.basicConfig(
    filename='logs/data_uploader.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RawDataUploader:
    def __init__(self, credentials_file: Optional[str] = None, token_file: str = 'token.pickle'):
        """
        Инициализация загрузчика сырых данных для Google Drive
        
        Args:
            credentials_file: Путь к файлу учетных данных Google API
            token_file: Путь к файлу токена для сохранения/восстановления сессии
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

    def upload_raw_data(self, 
                       file_path: str, 
                       folder_id: Optional[str] = None,
                       description: Optional[str] = None,
                       overwrite: bool = False) -> str:
        """
        Выгрузка сырых данных в Google Drive
        
        Args:
            file_path: Путь к файлу данных для загрузки
            folder_id: ID папки в Google Drive, куда загружать (опционально)
            description: Описание файла (опционально)
            overwrite: Перезаписывать ли существующий файл (если есть)
            
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
            
            # Проверка существования файла
            existing_file_id = self._check_file_exists(file_name, folder_id)
            
            if existing_file_id and not overwrite:
                logger.info(f"Файл {file_name} уже существует. Используйте overwrite=True для перезаписи")
                return existing_file_id
            
            # Подготовка метаданных
            file_metadata = {
                'name': file_name,
                'description': description or f"Сырые медицинские данные, загружены {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            }
            
            if folder_id:
                file_metadata['parents'] = [folder_id]
            
            media = MediaFileUpload(file_path, resumable=True)
            
            # Загрузка файла
            if existing_file_id and overwrite:
                # Обновление существующего файла
                request = service.files().update(
                    fileId=existing_file_id,
                    body=file_metadata,
                    media_body=media,
                    fields='id,webViewLink'
                )
            else:
                # Создание нового файла
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
            
            logger.info(f"Файл успешно {'обновлен' if existing_file_id and overwrite else 'загружен'}: {file_link}")
            return file_id
            
        except Exception as e:
            logger.error(f"Ошибка загрузки файла в Google Drive: {e}")
            raise

    def create_folder(self, folder_name: str, parent_folder_id: Optional[str] = None) -> str:
        """
        Создание папки в Google Drive
        
        Args:
            folder_name: Название папки
            parent_folder_id: ID родительской папки (опционально)
            
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

    def _check_file_exists(self, file_name: str, folder_id: Optional[str] = None) -> Optional[str]:
        """
        Проверка наличия файла в Google Drive
        
        Args:
            file_name: Имя файла для поиска
            folder_id: ID папки, где искать файл (опционально)
            
        Returns:
            Optional[str]: ID файла если существует, иначе None
        """
        try:
            service = build('drive', 'v3', credentials=self.creds)
            
            query = f"name='{file_name}'"
            if folder_id:
                query += f" and '{folder_id}' in parents"
                
            results = service.files().list(q=query, spaces='drive', fields='nextPageToken, files(id, name)').execute()
            files = results.get('files', [])
            
            return files[0].get('id') if files else None
            
        except Exception as e:
            logger.error(f"Ошибка проверки файла в Google Drive: {e}")
            raise

    def validate_raw_data(self, file_path: str) -> bool:
        """
        Проверка валидности сырых данных перед загрузкой
        
        Args:
            file_path: Путь к файлу данных для проверки
            
        Returns:
            bool: True если данные валидны
        """
        try:
            # Проверка, что файл существует
            if not os.path.exists(file_path):
                logger.error(f"Файл {file_path} не найден")
                return False
                
            # Проверка размера файла (минимум 1KB)
            if os.path.getsize(file_path) < 1024:
                logger.error(f"Файл {file_path} слишком маленький для медицинских данных")
                return False
                
            # Проверка расширения файла (CSV)
            if not file_path.lower().endswith('.csv'):
                logger.warning(f"Файл {file_path} не имеет расширения CSV")
                
            # Проверка содержимого (по необходимости)
            # TODO: Добавить проверку структуры файла, если требуется
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка проверки валидности данных: {e}")
            return False

if __name__ == "__main__":
    # Пример использования
    try:
        uploader = RawDataUploader(credentials_file='credentials.json')
        
        # Путь к сырым данным
        raw_data_path = "data/raw/wdbc.csv"
        
        # Проверка валидности данных
        if not uploader.validate_raw_data(raw_data_path):
            logger.error("Данные не прошли проверку валидности")
            exit(1)
        
        # Создание папки для данных
        data_folder_id = uploader.create_folder("Breast Cancer Data")
        
        # Загрузка сырых данных
        raw_file_id = uploader.upload_raw_data(
            raw_data_path,
            folder_id=data_folder_id,
            description="Сырые данные о раке груди (Breast Cancer Wisconsin Diagnostic Dataset)",
            overwrite=True
        )
        
        logger.info(f"Загрузка сырых данных завершена успешно. File ID: {raw_file_id}")
        
    except Exception as e:
        logger.error(f"Ошибка в процессе загрузки данных: {e}")
        exit(1)
        
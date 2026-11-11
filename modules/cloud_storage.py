"""
UDIP v3 — modules/cloud_storage.py
Cloud storage integration: AWS S3, Google Cloud Storage, Google Drive, Azure Blob Storage.
"""

import os
import io
import streamlit as st
from modules.core import log
import pandas as pd

class CloudStorageManager:
    """Unified interface for cloud storage operations."""
    
    def __init__(self):
        self.providers = {
            'aws': self._init_aws,
            'gcp': self._init_gcp,
            'google': self._init_gcp,
            'google_drive': self._init_google_drive,
            'azure': self._init_azure
        }
        self.clients = {}
    
    def connect(self, provider: str, credentials: dict):
        """Connect to cloud storage provider."""
        try:
            if provider in self.providers:
                client = self.providers[provider](credentials)
                if not client:
                    return False
                if hasattr(client, "authorization_url"):
                    return client
                self.clients[provider] = client
                return True
            return False
        except Exception as e:
            log.error(f"Failed to connect to {provider}: {e}")
            return False
    
    def _init_aws(self, credentials: dict):
        """Initialize AWS S3 client."""
        import boto3
        return boto3.client(
            's3',
            aws_access_key_id=credentials.get('access_key'),
            aws_secret_access_key=credentials.get('secret_key'),
            region_name=credentials.get('region', 'us-east-1')
        )
    
    def _init_gcp(self, credentials: dict):
        """Initialize Google Cloud Storage client."""
        from google.cloud import storage
        import json
        import tempfile
        
        # Create temporary credentials file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(credentials, f)
            creds_file = f.name
        
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_file
        return storage.Client()
    
    def _init_google_drive(self, credentials: dict):
        """Initialize Google Drive client with OAuth authentication."""
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
            import os
            
            # Check for credentials.json file
            creds_file = credentials.get('credentials_file')
            if not creds_file:
                # Try default location
                default_creds_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'google', 'credentials.json')
                if os.path.exists(default_creds_path):
                    creds_file = default_creds_path
            
            if not creds_file or not os.path.exists(creds_file):
                log.error("Google Drive credentials.json not found")
                return None
            
            # Check for token.json file
            token_file = credentials.get('token_file')
            if not token_file:
                default_token_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'google', 'token.json')
                token_file = default_token_path
            
            # Load or create credentials
            creds = None
            if os.path.exists(token_file):
                creds = Credentials.from_authorized_user_file(token_file)
                log.info(f"Loaded existing token from {token_file}")
            
            # If there are no (valid) credentials available, let the user log in.
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    log.info("Refreshing expired credentials")
                    creds.refresh(Request())
                    log.info("Credentials refreshed successfully")
                else:
                    log.info("Starting OAuth flow - no valid credentials found")
                    flow = InstalledAppFlow.from_client_secrets_file(
                        creds_file, 
                        scopes=['https://www.googleapis.com/auth/drive.readonly']
                    )
                    # Use local server flow for browser-based authentication
                    # Return the flow object so the auth URL can be displayed in UI
                    return flow
                
                # Save the credentials for the next run
                log.info(f"Saving credentials to {token_file}")
                try:
                    with open(token_file, 'w') as token:
                        token.write(creds.to_json())
                    log.info(f"Credentials saved successfully to {token_file}")
                except Exception as save_error:
                    log.error(f"Failed to save credentials to {token_file}: {save_error}")
                    raise
            
            # Build Drive service
            drive_service = build('drive', 'v3', credentials=creds)
            return drive_service
            
        except ImportError as e:
            log.error(f"Google Drive libraries not installed: {e}")
            log.error("Install: pip install google-api-python-client google-auth-oauthlib google-auth-httplib2")
            return None
        except Exception as e:
            log.error(f"Failed to initialize Google Drive: {e}")
            return None
    
    def _init_azure(self, credentials: dict):
        """Initialize Azure Blob Storage client."""
        from azure.storage.blob import BlobServiceClient
        return BlobServiceClient.from_connection_string(credentials.get('connection_string'))
    
    def list_files(self, provider: str, bucket: str, prefix: str = '') -> list:
        """List files in cloud storage."""
        try:
            client = self.clients.get(provider)
            if not client:
                return []
            
            if provider == 'aws':
                response = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
                return [obj['Key'] for obj in response.get('Contents', [])]
            elif provider == 'gcp':
                bucket_obj = client.bucket(bucket)
                blobs = bucket_obj.list_blobs(prefix=prefix)
                return [blob.name for blob in blobs]
            elif provider == 'google_drive':
                # List files from Google Drive
                results = client.files().list(q=f"'{bucket}' in parents and trashed=false", pageSize=100).execute()
                files = results.get('files', [])
                return [(f['id'], f['name'], f.get('mimeType', '')) for f in files]
            elif provider == 'azure':
                container_client = client.get_container_client(bucket)
                blobs = container_client.list_blobs(name_starts_with=prefix)
                return [blob.name for blob in blobs]
            
            return []
        except Exception as e:
            log.error(f"Failed to list files: {e}")
            return []
    
    def download_file(self, provider: str, bucket: str, key: str) -> bytes:
        """Download file from cloud storage."""
        try:
            client = self.clients.get(provider)
            if not client:
                return b''
            
            if provider == 'aws':
                response = client.get_object(Bucket=bucket, Key=key)
                return response['Body'].read()
            elif provider == 'gcp':
                bucket_obj = client.bucket(bucket)
                blob = bucket_obj.blob(key)
                return blob.download_as_bytes()
            elif provider == 'google_drive':
                # Download file from Google Drive
                file_id = key
                return client.files().get_media(fileId=file_id).execute()
            elif provider == 'azure':
                blob_client = client.get_blob_client(container=bucket, blob=key)
                return blob_client.download_blob().readall()
            
            return b''
        except Exception as e:
            log.error(f"Failed to download file: {e}")
            return b''
    
    def upload_file(self, provider: str, bucket: str, key: str, data: bytes) -> bool:
        """Upload file to cloud storage."""
        try:
            client = self.clients.get(provider)
            if not client:
                return False
            
            if provider == 'aws':
                client.put_object(Bucket=bucket, Key=key, Body=data)
            elif provider == 'gcp':
                bucket_obj = client.bucket(bucket)
                blob = bucket_obj.blob(key)
                blob.upload_from_string(data)
            elif provider == 'azure':
                blob_client = client.get_blob_client(container=bucket, blob=key)
                blob_client.upload_blob(data, overwrite=True)
            
            return True
        except Exception as e:
            log.error(f"Failed to upload file: {e}")
            return False

# Global instance
cloud_manager = CloudStorageManager()

def get_cloud_files(provider: str, bucket: str, prefix: str = '') -> list:
    """Get list of files from cloud storage."""
    return cloud_manager.list_files(provider, bucket, prefix)

def download_cloud_file(provider: str, bucket: str, key: str) -> bytes:
    """Download file from cloud storage."""
    return cloud_manager.download_file(provider, bucket, key)

def upload_cloud_file(provider: str, bucket: str, key: str, data: bytes) -> bool:
    """Upload file to cloud storage."""
    return cloud_manager.upload_file(provider, bucket, key, data)

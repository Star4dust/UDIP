"""
UDIP v3 — udip_pages/cloud_sources.py
Cloud storage integration page.
"""

import io
import uuid
import os

import streamlit as st
import pandas as pd
from modules.core import add_history, log, read_uploaded_file, save_dataset
from modules.cloud_storage import cloud_manager, get_cloud_files, download_cloud_file, upload_cloud_file

def cloud_sources_page():
    """Cloud storage sources page."""
    st.markdown("## ☁️ Cloud Data Sources")
    st.info("Cloud sources are currently quarantined from the scientific runtime to reduce blast radius. They remain available in the legacy module for later reactivation.")
    return

    # Cloud provider selection
    provider = st.selectbox(
        "Cloud Provider",
        ["Google Drive", "AWS S3", "Google Cloud Storage", "Azure Blob Storage"],
        help="Select your cloud storage provider"
    )

    provider_map = {
        "Google Drive": "google_drive",
        "AWS S3": "aws",
        "Google Cloud Storage": "google",
        "Azure Blob Storage": "azure",
    }
    provider_key = provider_map.get(provider, "aws")

    # Connection credentials
    st.markdown("### 🔐 Connection Setup")
    with st.expander("Configure Credentials", expanded=False):
        if provider_key == "google_drive":
            st.markdown("#### Google Drive OAuth Authentication")
            
            # Check if credentials.json exists
            import os
            config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'google')
            creds_file = os.path.join(config_dir, 'credentials.json')
            
            if not os.path.exists(creds_file):
                st.warning("⚠️ Google Drive credentials not found")
                with st.expander("📋 Setup Instructions", expanded=True):
                    st.markdown("""
                    **To enable Google Drive access:**
                    
                    1. Go to [Google Cloud Console](https://console.cloud.google.com/)
                    2. Create a new project or select existing one
                    3. Enable Google Drive API
                    4. Go to APIs & Services → Credentials
                    5. Create OAuth 2.0 Client ID → Desktop app
                    6. Download credentials.json
                    7. Place credentials.json in: `config/google/credentials.json`
                    
                    **Directory structure:**
                    ```
                    udip_v3/
                    ├── config/
                    │   └── google/
                    │       ├── credentials.json
                    │       └── token.json (auto-generated)
                    ```
                    """)
                
                # Allow manual upload of credentials.json
                uploaded_creds = st.file_uploader("Upload credentials.json", type=['json'])
                if uploaded_creds:
                    import json
                    try:
                        # Create config directory if it doesn't exist
                        os.makedirs(config_dir, exist_ok=True)
                        
                        # Save credentials.json
                        with open(creds_file, 'wb') as f:
                            f.write(uploaded_creds.getbuffer())
                        
                        st.success("✅ credentials.json saved successfully")
                        st.info("Click 'Connect to Google Drive' to complete authentication")
                    except Exception as e:
                        st.error(f"Failed to save credentials: {e}")
            else:
                st.success("✅ credentials.json found")
                st.info("Click 'Connect to Google Drive' to authenticate")
            
            if st.button("🔗 Connect to Google Drive", type="primary"):
                credentials = {'credentials_file': creds_file}
                st.info("🔄 Starting Google Drive authentication...")
                log.info(f"Attempting to connect to Google Drive with credentials: {creds_file}")
                
                # Try to connect - this will return a flow object if OAuth is needed
                connection_result = cloud_manager.connect(provider_key, credentials)
                log.info(f"Connection result: {connection_result}")
                
                # If connection_result is a flow object, we need to complete OAuth
                if hasattr(connection_result, 'authorization_url'):
                    # Store the flow in session state for later completion
                    st.session_state['google_drive_flow'] = connection_result
                    st.session_state['google_drive_token_file'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'google', 'token.json')
                    st.rerun()
                elif connection_result:
                    st.session_state[f"{provider_key}_connected"] = True
                    st.session_state[f"{provider_key}_credentials"] = credentials
                    log.info(f"Set session state {provider_key}_connected to True")
                    st.success(f"✅ Connected to Google Drive")
                    st.rerun()
                else:
                    log.error(f"Connection failed for Google Drive")
                    st.error(f"❌ Failed to connect to Google Drive")
            
            # Display auth URL if flow is in session state (persists across reruns)
            if 'google_drive_flow' in st.session_state:
                st.markdown("---")
                st.markdown("### 🔐 Google Authentication Required")
                st.info("Click the link below to authenticate with Google Drive:")
                
                try:
                    flow = st.session_state['google_drive_flow']
                    auth_url, _ = flow.authorization_url(prompt='consent')
                    st.markdown(f"[🔗 Authenticate with Google Drive]({auth_url})", unsafe_allow_html=True)
                    st.info("After authenticating, click 'Complete Authentication' below")
                except Exception as e:
                    st.error(f"Failed to generate auth URL: {e}")
                    log.error(f"Failed to generate auth URL: {e}")
                
                # Add button to complete authentication after user clicks the link
                if st.button("✅ Complete Authentication", type="primary"):
                    flow = st.session_state['google_drive_flow']
                    token_file = st.session_state['google_drive_token_file']
                    
                    try:
                        # Complete the OAuth flow
                        creds = flow.run_local_server(port=0)
                        
                        # Save the credentials
                        with open(token_file, 'w') as token:
                            token.write(creds.to_json())
                        
                        # Initialize the Drive service
                        from googleapiclient.discovery import build
                        drive_service = build('drive', 'v3', credentials=creds)
                        cloud_manager.clients['google_drive'] = drive_service
                        
                        # Clean up session state
                        del st.session_state['google_drive_flow']
                        del st.session_state['google_drive_token_file']
                        
                        # Set connection state
                        st.session_state[f"{provider_key}_connected"] = True
                        st.session_state[f"{provider_key}_credentials"] = credentials
                        log.info(f"Set session state {provider_key}_connected to True")
                        st.success(f"✅ Connected to Google Drive")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to complete authentication: {e}")
                        log.error(f"Failed to complete authentication: {e}")
        elif provider_key == "aws":
            access_key = st.text_input("AWS Access Key ID", type="password")
            secret_key = st.text_input("AWS Secret Access Key", type="password")
            region = st.text_input("AWS Region", value="us-east-1")
            credentials = {
                'access_key': access_key,
                'secret_key': secret_key,
                'region': region
            }
        elif provider_key == "google":
            creds_file = st.file_uploader("Google Cloud Credentials JSON", type=['json'])
            if creds_file:
                import json
                credentials = json.load(creds_file)
            else:
                credentials = {}
        elif provider_key == "azure":
            connection_string = st.text_input("Azure Connection String", type="password")
            credentials = {'connection_string': connection_string}

        if provider_key != "google_drive" and st.button("🔗 Connect", type="primary"):
            if cloud_manager.connect(provider_key, credentials):
                st.success(f"✅ Connected to {provider}")
                st.session_state[f"{provider_key}_connected"] = True
                st.session_state[f"{provider_key}_credentials"] = credentials
            else:
                st.error(f"❌ Failed to connect to {provider}")

    # Connection status display
    if st.session_state.get(f"{provider_key}_connected", False):
        st.success(f"✅ {provider} is connected")
        st.info("📁 Go to Upload page to import datasets from this source")
    else:
        st.warning(f"⚠️ {provider} is not connected")
        st.info("Configure credentials above and click Connect to enable")

    # Usage tips
    st.markdown("### 📘 Usage")
    with st.expander("How to use cloud sources"):
        st.markdown("""
        - **Google Drive**: Configure OAuth credentials, then import files from Upload page
        - **AWS S3**: Provide access key, secret key, and region, then import from Upload page
        - **Google Cloud**: Upload service account JSON credentials, then import from Upload page
        - **Azure**: Provide connection string, then import from Upload page
        
        After connecting, go to the **Upload** page to browse and import datasets from your cloud storage.
        """)

def render():
    cloud_sources_page()

if __name__ == "__main__":
    render()

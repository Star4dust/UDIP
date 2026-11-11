# Google Drive credentials

UDIP's Google Drive import needs an OAuth client file that is **not** included in this repository.

1. Open the Google Cloud Console, create a project, and enable the Google Drive API.
2. Create an OAuth client ID of type "Desktop app".
3. Download the JSON file and save it here as `config/google/credentials.json`.

`credentials.json` is listed in `.gitignore`. Never commit it or share it.

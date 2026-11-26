from alpha_mining.ollama import ollama_request
from alpha_mining.data_fetcher.operator_fetcher import brain_api_url
import json
import logging
import os
from getpass import getpass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from urllib.parse import urljoin

import requests

DEV = False
logger = logging.getLogger(__name__)
brain_api_url = 'https://api.worldquantbrain.com'


class Credentials:
    def __init__(self, username: str = "", password: str = ""):
        self.username = username
        self.password = password

    def to_dict(self) -> Dict[str, str]:
        return {"username": self.username, "password": self.password}

    def validate(self) -> bool:
        return bool(self.username and self.password and len(self.username) > 0 and len(self.password) > 0)


class CredentialManager:
    def __init__(self, base_path=None):
        self.base_path = base_path
        self.credentials: Optional[Credentials] = None
        self.authenticated = False
        self.session: Optional[requests.Session] = None

    @staticmethod
    def _default_credentials_file() -> Path:
        credentials_folder_path = Path.home() / "secrets"
        return credentials_folder_path / "platform-brain.json"

    def _resolve_path(self, path: Optional[Union[str, Path]] = None) -> Optional[Path]:
        if path is not None:
            candidate = Path(path)
            if candidate.exists():
                return candidate
            return candidate

        if self.base_path is not None:
            candidate = Path(self.base_path)
            if candidate.exists():
                return candidate
            return candidate

        return self._default_credentials_file()

    @staticmethod
    def _parse_credentials(data) -> Optional[Credentials]:
        if isinstance(data, dict):
            username = data.get("email") or data.get("username") or data.get("user") or ""
            password = data.get("password") or data.get("pwd") or ""
            if username and password:
                return Credentials(username=str(username), password=str(password))
        elif isinstance(data, (list, tuple)) and len(data) >= 2:
            return Credentials(username=str(data[0]), password=str(data[1]))
        return None

    def load_from_file(self, path: Optional[Union[str, Path]] = None) -> bool:
        """Load credentials from a JSON file if present."""
        credential_path = self._resolve_path(path)
        if credential_path is None:
            return False

        if not credential_path.exists() or credential_path.stat().st_size <= 2:
            return False

        try:
            with open(credential_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError):
            return False

        credentials = self._parse_credentials(payload)
        if credentials is None:
            return False

        self.credentials = credentials
        self.authenticated = True
        return True

    def validate_credentials(self) -> bool:
        if self.credentials is None:
            return False
        return self.credentials.validate()

    def get_session(self) -> Optional[requests.Session]:
        if self.session is not None:
            return self.session

        if not self.validate_credentials():
            return None

        session = requests.Session()
        session.auth = (self.credentials.username, self.credentials.password)
        self.session = session
        r = session.post(brain_api_url, '/authentication')
        logger.debug(f"New session created (ID: {id(session)}) with authentication response {r.status_code}, {r.json()}")
        if r.status_code == requests.status_codes.codes.unauthorized:
            if r.headers["WWW-Authenticate"] == 'persona':
                auth_link = urljoin(r.url)
        self.authenticated = True
        return session

    @staticmethod
    def get_credentials() -> Credentials:
        """Retrieve or prompt for platform credentials."""
        credential_email = os.environ.get("BRAIN_CREDENTIAL_EMAIL")
        credential_password = os.environ.get("BRAIN_CREDENTIAL_PASSWORD")

        credentials_file_path = CredentialManager._default_credentials_file()
        credentials_file_path.parent.mkdir(parents=True, exist_ok=True)

        if credentials_file_path.exists() and credentials_file_path.stat().st_size > 2:
            with open(credentials_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            parsed = CredentialManager._parse_credentials(data)
            if parsed is not None:
                return parsed

        if credential_email and credential_password:
            email = credential_email
            password = credential_password
        else:
            email = input("Email:\n")
            password = getpass(prompt="Password:")

        data = {"email": email, "password": password}
        with open(credentials_file_path, "w", encoding="utf-8") as file:
            json.dump(data, file)
        return Credentials(username=email, password=password)

   
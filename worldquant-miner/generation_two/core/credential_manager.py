"""
Secure Credential Manager
Handles WorldQuant Brain authentication securely

IMPORTANT: Credentials are NEVER embedded in code or executable.
All credentials are loaded from external files or user input only.
"""

import json
import os
import logging
import requests
from os.path import expanduser  
from pathlib import Path
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from getpass import getpass

logger = logging.getLogger(__name__)


@dataclass
class Credentials:
    """Secure credential container (never logged or stored in code)"""
    username: str
    password: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary (for API calls only)"""
        return {
            'username': self.username,
            'password': self.password
        }
    
    def validate(self) -> bool:
        """Basic validation"""
        return bool(self.username and self.password and len(self.username) > 0 and len(self.password) > 0)


class CredentialManager:
    """
    Secure credential manager
    
    Features:
    - Loads from credential.txt or credentials.txt
    - Prompts for login if file not found
    - Validates credentials before use
    - NEVER embeds credentials in code
    - Stores credentials only in memory
    """
    
    # Possible credential file names (checked in order)
    CREDENTIAL_FILE_NAMES = ['platform-brain.json', 'credential.txt', 'credentials.txt']
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize credential manager
        
        Args:
            base_path: Base directory to search for credential files
                      If None, searches current directory and parent directories
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.credentials: Optional[Credentials] = None
        self.authenticated = False
        self.session: Optional[requests.Session] = None
    
    def find_credential_file(self) -> Optional[Path]:
        """
        Find credential file in common locations
        
        Returns:
            Path to credential file if found, None otherwise
        """
        # Search locations (in order of priority):
        search_paths = [
            self.base_path,  # Current/base directory
            self.base_path.parent,  # Parent directory
            Path.home() / 'secrets',  # User secrets directory (for platform-brain.json)
            Path.home(),  # User home directory
            Path.cwd(),  # Current working directory
        ]
        
        for search_path in search_paths:
            for filename in self.CREDENTIAL_FILE_NAMES:
                credential_file = search_path / filename
                if credential_file.exists() and credential_file.is_file():
                    logger.info(f"Found credential file: {credential_file}")
                    return credential_file
        
        logger.warning("No credential file found in standard locations")
        return None
    
    def load_from_file(self, file_path: Optional[Path] = None) -> bool:
        """
        Load credentials from file
        
        Args:
            file_path: Path to credential file. If None, searches automatically.
        
        Returns:
            True if credentials loaded successfully, False otherwise
        """
        if file_path is None:
            file_path = self.find_credential_file()
        
        if file_path is None:
            logger.error("No credential file specified or found")
            return False
        
        if not file_path.exists():
            logger.error(f"Credential file not found: {file_path}")
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Try JSON format first: {"email": ..., "password": ...} or ["username", "password"]
            try:
                json_data = json.loads(content)
                if isinstance(json_data, dict):
                    username = json_data.get("email") or json_data.get("username")
                    password = json_data.get("password")
                elif isinstance(json_data, list) and len(json_data) >= 2:
                    username = json_data[0]
                    password = json_data[1]
                else:
                    raise ValueError("Invalid credential format")
                
                if not username or not password:
                    raise ValueError("Missing username or password in JSON credentials")
            except (json.JSONDecodeError, ValueError):
                # Try line-separated format: username\npassword
                lines = content.split('\n')
                if len(lines) >= 2:
                    username = lines[0].strip()
                    password = lines[1].strip()
                else:
                    raise ValueError("Invalid credential format")
            
            self.credentials = Credentials(username=username, password=password)
            
            if not self.credentials.validate():
                logger.error("Invalid credentials: empty username or password")
                self.credentials = None
                return False
            
            logger.info(f"✅ Credentials loaded from: {file_path}")
            logger.info(f"   Username: {self.credentials.username}")
            # NEVER log password
            return True
            
        except Exception as e:
            logger.error(f"Failed to load credentials from {file_path}: {e}")
            self.credentials = None
            return False
    
    def prompt_for_credentials(self) -> bool:
        """
        Prompt user for credentials (interactive)
        
        Returns:
            True if credentials entered, False if cancelled
        """
        try:
            print("\n" + "="*60)
            print("🔐 WORLDQUANT BRAIN AUTHENTICATION REQUIRED")
            print("="*60)
            print("Please enter your WorldQuant Brain credentials:")
            print()
            
            username = input("Username (email): ").strip()
            if not username:
                logger.warning("Username not provided")
                return False
            
            # Use getpass to hide password input
            password = getpass("Password: ").strip()
            if not password:
                logger.warning("Password not provided")
                return False
            
            self.credentials = Credentials(username=username, password=password)
            
            if not self.credentials.validate():
                logger.error("Invalid credentials: empty username or password")
                self.credentials = None
                return False
            
            logger.info(f"✅ Credentials entered (username: {self.credentials.username})")
            return True
            
        except (KeyboardInterrupt, EOFError):
            logger.warning("Credential entry cancelled by user")
            self.credentials = None
            return False
        except Exception as e:
            logger.error(f"Error prompting for credentials: {e}")
            self.credentials = None
            return False
    
    def validate_credentials(self) -> bool:
        """
        Validate credentials by attempting authentication.
        Handles Persona identity/biometric authentication if required.
        
        Returns:
            True if credentials are valid, False otherwise
        """
        if not self.credentials or not self.credentials.validate():
            logger.error("No valid credentials to validate")
            return False
        
        try:
            # Create a temporary session for validation
            test_session = requests.Session()
            
            from requests.auth import HTTPBasicAuth
            auth = HTTPBasicAuth(self.credentials.username, self.credentials.password)
            
            # Step 1: Try to restore active session from cache to avoid triggering authentication / biometrics
            if self.load_session(test_session):
                logger.info("Restored session from cache. Verifying validity...")
                try:
                    test_response = test_session.get('https://api.worldquantbrain.com/operators', params={'limit': 1}, timeout=10)
                    if test_response.status_code == 200:
                        logger.info("✅ Restored session is still valid! Bypassing authentication and biometrics.")
                        self.authenticated = True
                        self.session = test_session
                        self.session.auth = auth
                        return True
                    else:
                        logger.warning(f"Restored session returned status {test_response.status_code}. Starting fresh login.")
                        test_session.cookies.clear()
                        test_session.headers.clear()
                except Exception as e:
                    logger.warning(f"Failed to verify restored session: {e}. Starting fresh login.")
                    test_session.cookies.clear()
                    test_session.headers.clear()
            
            # Step 2: Attempt standard authentication POST if session cache is invalid or missing
            logger.info(f"Validating credentials for: {self.credentials.username}")
            response = test_session.post(
                'https://api.worldquantbrain.com/authentication',
                auth=auth,
                timeout=10
            )
            
            if response.status_code == 201:
                logger.info("✅ Credentials validated successfully")
                self.authenticated = True
                
                # Store session for reuse
                self.session = test_session
                self.session.auth = auth
                
                # Auto-save credentials and session cache
                self.save_credentials()
                self.save_session()
                
                return True
                
            elif response.status_code == 401 and (
                response.headers.get("WWW-Authenticate") == "persona" or
                "inquiry" in response.text
            ):
                from urllib.parse import urljoin
                import webbrowser
                
                location = response.headers.get("Location")
                if not location:
                    try:
                        inquiry = response.json().get("inquiry")
                        if inquiry:
                            location = f"/authentication/persona?inquiry={inquiry}"
                    except Exception:
                        pass
                
                if location:
                    verification_url = urljoin('https://api.worldquantbrain.com', location)
                    logger.warning(f"⚠️ Identity/Biometric authentication is required by WorldQuant Brain.")
                    logger.warning(f"Please complete verification in your browser: {verification_url}")
                    
                    try:
                        webbrowser.open(verification_url)
                    except Exception as e:
                        logger.error(f"Failed to open browser automatically: {e}")
                    
                    is_gui = False
                    try:
                        import tkinter as tk
                        if tk._default_root is not None:
                            is_gui = True
                    except Exception:
                        pass
                    
                    if is_gui:
                        from tkinter import messagebox
                        messagebox.showinfo(
                            "🔐 Biometric Authentication Required",
                            "WorldQuant Brain requires identity/biometric verification.\n\n"
                            "A browser window has been opened to complete the process. If not, open:\n"
                            f"{verification_url}\n\n"
                            "Click OK AFTER you have fully completed the verification in the browser."
                        )
                    else:
                        input("\nComplete biometrics authentication in your browser and press Enter to continue...\n")
                    
                    logger.info("Checking biometrics verification status...")
                    check_response = test_session.post(verification_url)
                    
                    while check_response.status_code != 201:
                        if is_gui:
                            from tkinter import messagebox
                            retry = messagebox.askretrycancel(
                                "Verification Pending",
                                "Verification is not complete yet.\n\n"
                                "Please make sure you finished the steps in the browser, then click 'Retry' to check again."
                            )
                            if not retry:
                                logger.error("Authentication cancelled by user.")
                                self.authenticated = False
                                return False
                        else:
                            input("Verification not complete. Press Enter to retry after completing steps in the browser...")
                        
                        check_response = test_session.post(verification_url)
                    
                    logger.info("✅ Biometrics authentication verified successfully!")
                    self.authenticated = True
                    self.session = test_session
                    self.session.auth = auth
                    
                    # Save credentials and session cache
                    self.save_credentials()
                    self.save_session()
                    return True
            
            elif response.status_code == 429 and "BIOMETRICS_THROTTLED" in response.text:
                error_msg = (
                    "❌ BIOMETRICS THROTTLED\n\n"
                    "WorldQuant Brain has temporarily rate-limited biometric authentication for your account.\n"
                    "This usually happens if authentication was requested too many times in a short period.\n\n"
                    "Please wait 10-15 minutes before launching the application again to let the lockout clear."
                )
                logger.error(error_msg)
                
                is_gui = False
                try:
                    import tkinter as tk
                    if tk._default_root is not None:
                        is_gui = True
                except Exception:
                    pass
                if is_gui:
                    from tkinter import messagebox
                    messagebox.showerror("Rate Limit Exceeded (429)", error_msg)
                
                self.authenticated = False
                return False
                
            logger.error(f"❌ Authentication failed: {response.status_code}")
            logger.error(f"   Response: {response.text[:200]}")
            self.authenticated = False
            return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Network error during credential validation: {e}")
            self.authenticated = False
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error during credential validation: {e}")
            self.authenticated = False
            return False

    def save_credentials(self) -> bool:
        """Save current credentials to ~/secrets/platform-brain.json"""
        if not self.credentials:
            return False
        try:
            credentials_folder_path = os.path.join(os.path.expanduser("~"), "secrets")
            credentials_file_path = os.path.join(credentials_folder_path, "platform-brain.json")
            os.makedirs(credentials_folder_path, exist_ok=True)
            data = {"email": self.credentials.username, "password": self.credentials.password}
            with open(credentials_file_path, "w") as file:
                json.dump(data, file)
            logger.info(f"Saved credentials to: {credentials_file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
            return False

    def save_session(self) -> bool:
        """Save current authenticated session cookies and headers to ~/secrets/platform-brain-session.json"""
        if not self.session or not self.credentials:
            return False
        try:
            session_folder_path = os.path.join(os.path.expanduser("~"), "secrets")
            session_file_path = os.path.join(session_folder_path, "platform-brain-session.json")
            os.makedirs(session_folder_path, exist_ok=True)
            
            cookies_dict = requests.utils.dict_from_cookiejar(self.session.cookies)
            session_token = self.session.headers.get('X-WQB-Session-Token')
            
            data = {
                "username": self.credentials.username,
                "cookies": cookies_dict,
                "headers": {
                    "X-WQB-Session-Token": session_token
                } if session_token else {}
            }
            with open(session_file_path, "w") as file:
                json.dump(data, file)
            logger.info(f"Saved authenticated session cache to: {session_file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save session cache: {e}")
            return False

    def load_session(self, session: requests.Session) -> bool:
        """Load session cookies and headers from ~/secrets/platform-brain-session.json if username matches"""
        if not self.credentials:
            return False
        try:
            session_folder_path = os.path.join(os.path.expanduser("~"), "secrets")
            session_file_path = os.path.join(session_folder_path, "platform-brain-session.json")
            
            if not Path(session_file_path).exists():
                return False
                
            with open(session_file_path, "r") as file:
                data = json.load(file)
                
            # Verify username matches
            if data.get("username") != self.credentials.username:
                return False
                
            cookies_dict = data.get("cookies", {})
            headers_dict = data.get("headers", {})
            
            session.cookies.update(requests.utils.cookiejar_from_dict(cookies_dict))
            session.headers.update(headers_dict)
            
            logger.info("Loaded session cache from file.")
            return True
        except Exception as e:
            logger.error(f"Failed to load session cache: {e}")
            return False
    
    def get_credentials(self) -> Optional[Credentials]:
        """
        Get current credentials (if authenticated/loaded).
        If not loaded, attempts to load them from file/env.
        
        Returns:
            Credentials object if available, None otherwise
        """
        if not self.credentials:
            # Try load from env first
            credential_email = os.environ.get("BRAIN_CREDENTIAL_EMAIL")
            credential_password = os.environ.get("BRAIN_CREDENTIAL_PASSWORD")
            if credential_email and credential_password:
                self.credentials = Credentials(username=credential_email, password=credential_password)
                self.authenticated = True
            else:
                self.load_from_file()
        return self.credentials
    
    def get_session(self) -> Optional[requests.Session]:
        """
        Get authenticated session
        
        Returns:
            Authenticated requests.Session if available, None otherwise
        """
        if self.authenticated and self.session:
            return self.session
        return None
    
    def authenticate(self, auto_load: bool = True, auto_prompt: bool = True) -> bool:
        """
        Complete authentication flow
        
        Args:
            auto_load: Automatically try to load from file
            auto_prompt: Automatically prompt if file not found
        
        Returns:
            True if authenticated, False otherwise
        """
        # Step 1: Try to load from environment variables first (similar to ace_lib.py)
        credential_email = os.environ.get("BRAIN_CREDENTIAL_EMAIL")
        credential_password = os.environ.get("BRAIN_CREDENTIAL_PASSWORD")
        if credential_email and credential_password:
            self.credentials = Credentials(username=credential_email, password=credential_password)
            if self.validate_credentials():
                return True
            else:
                logger.warning("Credentials from environment variables failed validation")

        # Step 2: Try to load from file
        if auto_load:
            if self.load_from_file():
                if self.validate_credentials():
                    return True
                else:
                    logger.warning("Credentials from file failed validation")
        
        # Step 3: Prompt user if file not found or validation failed
        if auto_prompt:
            if self.prompt_for_credentials():
                if self.validate_credentials():
                    return True
                else:
                    logger.error("Entered credentials failed validation")
        
        logger.error("❌ Authentication failed - cannot proceed without valid credentials")
        return False
    
    def clear_credentials(self):
        """Clear credentials from memory (security)"""
        if self.credentials:
            # Overwrite password in memory (best effort)
            self.credentials.password = "***CLEARED***"
        self.credentials = None
        self.authenticated = False
        self.session = None
        logger.info("Credentials cleared from memory")


def get_credential_manager(base_path: Optional[str] = None) -> CredentialManager:
    """
    Factory function to get credential manager instance
    
    Args:
        base_path: Base directory to search for credentials
    
    Returns:
        CredentialManager instance
    """
    return CredentialManager(base_path=base_path)

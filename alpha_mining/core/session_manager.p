import json
import os
import logging
import requests
from os.path import expanduser  
from pathlib import Path
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import threading
from getpass import getpass
from credential_manager import CredentialManager 

DEV = False
brain_api_url = os.environ.get("BRAIN_API_URL", "https://api.worldquantbrain.com")

class SingleSession(requests.Session):
    _instance = None 
    _lock = threading.Lock()
    _relogin_lock = threading.Lock()
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, *args, **kwargs):
        if not self._initialized:
            super(SingleSession, self).__init__(*args, **kwargs)
            self._initialized = True
    
    def get_relogin_lock(self):
        return self._relogin_lock


def setup_logger() -> logging.Logger:
    """
    This function sets up a logger that writes log messages to the console and, if the global variable DEV is set to True, also to a file name 

    Returns:
        logging.Logger: _description_
    """
    logger = logging.getLogger('ace')
    level = logging.DEBUG if DEV else logging.INFO
    
    logger.setLevel(level)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    file_handler = logging.FileHandler('ace.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
        
    return logger

logger = setup_logger()

def start_session() -> SingleSession:
    """
    Start a new session with WorldQuant Brain platform

    Returns:
        SingleSession: An authenticated session object
        
    Raises: 
        requests.exceptions.RequestException: If there's an error during authentication
        
    """
    s = SingleSession()
    s.auth = CredentialManager.get_credentials()
    r = s.post(brain_api_url + "/authentication")
    logger.debug(f"New session created (ID: {id(s)})")
import logging
import requests
import os 
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
import json

logger = logging.getLogger(__name__)
brain_api_url = os.environ.get("BRAIN_API_URL", "https://api.worldquantbrain.com")

class OperatorFetcher:
    def __init__(self, session: requests.Session, cache_dir: str = None):
        self.session = session
        if cache_dir is None:
            current_file = Path(__file__)
            alpha_mining = current_file.parent.parent
            cache_dir = str(alpha_mining / "constants")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "operatorRAW.json"
        self.operators: List[Dict] = []
    
    def fetch_operators(self, force_refresh: bool = False) -> List[Dict]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.operators = json.load(f)
                logger.info(f"Loaded {len(self.operators)} operators from {self.cache_file}")
                return self.operators
            except Exception as e:
                logger.error(f"Failed to load operators: {e}")
                return
        else:
            logger.warning(f"Operator file not found: {self.cache_file}")
            logger.info("Attempting to fetch from API as fallback")
            
            if not self.session:
                logger.error("No session available for fetching operators")
                return []
            try:
                logger.info("Fetching operators from WorldQuant Brain API...")
                response = self.session.get(f'{brain_api_url}/operators')
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if isinstance(data, List):
                        self.operators = data
                    elif 'results' in data:
                        self.operators = data['results']
                    elif 'items' in data:
                        self.operators = data['items']
                    else:
                        self.operators = []
                    
                    self._save_cache()
                    
                    logger.info(f"Fetched {len(self.operators)} operators from API")
                    return self.operators
                else:
                    logger.error(f"Failed to fetch operators: {response.status_code}")
                    return [] 
            except Exception as e:
                logger.error(f"Error fetching operators: {e}")
                return []
            
    # def fetch_operators(s: requests.Session) -> pd.DataFrame:
    #     """
    #     Fetches and processes the list of operators from the WorldQuant Brain API.

    #     This function retrieves the operators from the provided session `s`,
    #     explodes the 'scope' column (which contains lists) into separate rows,
    #     and returns the resulting DataFrame.

    #     Args:
    #     s (SingleSession): An authenticated session object.

    #     Returns:
    #     pd.DataFrame: A DataFrame containing the operators with each scope entry
    #     as a separate row.
    #     """
    #     df = pd.DataFrame(s.get(brain_api_url + "/operators").json())
        
    #     return df.explode('scope').reset_index(drop=True)
    
    def _save_cache(self):
        """Save operators to cache file"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.operators, f, indent=2, ensure_ascii=False)
            logger.info(f"Cached {len(self.operators)} operators to {self.cache_file}")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
            
    def get_opterator_by_category(self, category: str) -> List[Dict]:
        """Get operators filtered by category"""
        return [op for op in self.operators if op.get('category') == category]
    
    def get_operator_by_name(self, name: str) -> Optional[Dict]:
        """Get operator by name"""
        for op in self.operators:
            if op.get('name') == name:
                return op
        return None
    
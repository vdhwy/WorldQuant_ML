"""
Template Generator Module
Handles alpha expression generation using AI/LLM with smart Ollama integration
"""

import logging
import requests
import json
from typing import List, Dict, Optional
import time

from ..ollama import OllamaManager, RegionThemeManager
from ..ollama.duplicate_detector import DuplicateDetector
from ..data_fetcher import OperatorFetcher, DataFieldFetcher, SmartSearchEngine
from .template_validator import TemplateValidator

logger = logging.getLogger(__name__)


class TemplateGenerator:
    """
    Generates alpha templates using AI/LLM
    
    Separated from simulation and testing logic for modularity.
    """
    
    def __init__(
        self, 
        credentials_path: str = None, 
        credentials: List[str] = None,  # New: allow passing credentials directly
        session: requests.Session = None,  # New: allow passing an already authenticated session
        deepseek_api_key: str = None,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5-coder:1.5b",
        db_path: str = "generation_two_backtests.db"
    ):
        """
        Initialize template generator
        
        Args:
            credentials_path: Path to WorldQuant Brain credentials file
            credentials: Direct credentials as [username, password] (takes precedence over credentials_path)
            session: Direct requests.Session instance already authenticated (reused)
            deepseek_api_key: DeepSeek API key for LLM generation
            ollama_url: Ollama server URL
            ollama_model: Ollama model name
            db_path: Path to database for storing compiler knowledge
        """
        self.credentials_path = credentials_path
        self._stored_credentials = credentials  # Store credentials in memory for re-authentication
        self.deepseek_api_key = deepseek_api_key
        self.db_path = db_path
        
        # Use pre-existing authenticated session if provided, otherwise create a new one
        if session is not None:
            self.sess = session
        else:
            self.sess = requests.Session()
            self.sess.cookies.clear()  # Start fresh
        
        # Initialize Ollama manager (smart with fallback)
        self.ollama_manager = OllamaManager(
            base_url=ollama_url,
            model=ollama_model
        )
        
        # Initialize theme manager
        self.theme_manager = RegionThemeManager()
        
        # Initialize duplicate detector
        self.duplicate_detector = DuplicateDetector()
        
        # Initialize template validator with self-correcting AST
        self.template_validator = None  # Will be initialized after data fields are loaded
        
        # Initialize data fetchers (will be set up after authentication)
        self.operator_fetcher = None
        self.data_field_fetcher = None
        self.search_engine = None
        
        if session:
            # If session is provided, it is already authenticated. Setup data fetchers directly.
            self._setup_data_fetchers()
        elif credentials_path or credentials:
            self.setup_auth()
            self._setup_data_fetchers()
    
    def setup_auth(self):
        """Setup authentication for WorldQuant Brain API with session persistence"""
        try:
            from requests.auth import HTTPBasicAuth
            
            # Try to get credentials from memory first (for re-authentication)
            if self._stored_credentials:
                username = self._stored_credentials[0]
                password = self._stored_credentials[1]
            elif self.credentials_path:
                # Try to read from file
                try:
                    with open(self.credentials_path, 'r') as f:
                        credentials = json.loads(f.read().strip())
                    username = credentials[0]
                    password = credentials[1]
                    # Store in memory for future re-authentication
                    self._stored_credentials = [username, password]
                except FileNotFoundError:
                    # File was deleted (e.g., temp file), try to use stored credentials
                    if self._stored_credentials:
                        username = self._stored_credentials[0]
                        password = self._stored_credentials[1]
                        logger.warning("Credentials file not found, using stored credentials from memory")
                    else:
                        raise Exception("Credentials file not found and no stored credentials available")
            else:
                raise Exception("No credentials provided (neither credentials_path nor credentials)")
            
            # Log credentials (masked for security)
            logger.info(f"Authenticating with username: {username}")
            logger.debug(f"Password length: {len(password)} characters")
            
            # Set auth on session itself to persist (like stone_age does)
            # This ensures auth is maintained for all requests
            self.sess.auth = HTTPBasicAuth(username, password)
            
            # Authenticate with WorldQuant Brain using HTTPBasicAuth (same as v2)
            auth_response = self.sess.post(
                'https://api.worldquantbrain.com/authentication',
                auth=HTTPBasicAuth(username, password)  # Also pass explicitly for this request
            )
            
            if auth_response.status_code == 201:
                pass
            elif auth_response.status_code == 401 and (
                auth_response.headers.get("WWW-Authenticate") == "persona" or
                "inquiry" in auth_response.text
            ):
                from urllib.parse import urljoin
                import webbrowser
                
                location = auth_response.headers.get("Location")
                if not location:
                    try:
                        inquiry = auth_response.json().get("inquiry")
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
                            "A browser window has been opened. Please complete the verification there,\n"
                            "then click OK to check status."
                        )
                    else:
                        input("\nComplete biometrics authentication in your browser and press Enter to continue...\n")
                    
                    logger.info("Checking biometrics verification status...")
                    auth_response = self.sess.post(verification_url)
                    
                    while auth_response.status_code != 201:
                        if is_gui:
                            from tkinter import messagebox
                            retry = messagebox.askretrycancel(
                                "Verification Pending",
                                "Verification is not complete yet.\n\n"
                                "Please make sure you finished the steps in the browser, then click 'Retry' to check again."
                            )
                            if not retry:
                                raise Exception("Authentication cancelled by user during biometrics verification")
                        else:
                            input("Verification not complete. Press Enter to retry after completing steps in the browser...")
                        
                        auth_response = self.sess.post(verification_url)
                    
                    logger.info("✅ Biometrics authentication verified successfully!")
            else:
                logger.error(f"❌ Authentication failed: {auth_response.status_code}")
                logger.error(f"Response: {auth_response.text}")
                raise Exception(f"Authentication failed: {auth_response.status_code}")

            # Once authenticated successfully (either 201 directly or after verification)
            logger.info("✅ Session authenticated successfully")
            
            # Set session headers for all future requests (like v2 and stone_age do)
            self.sess.headers.update({
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            })
            
            # Verify cookies are set (for debugging)
            if self.sess.cookies:
                logger.info(f"Session cookies set: {len(self.sess.cookies)} cookie(s)")
                for cookie in self.sess.cookies:
                    logger.info(f"  Cookie: {cookie.name} (domain: {cookie.domain}, path: {cookie.path})")
            else:
                logger.warning("⚠️ No cookies set after authentication - session may not persist")
            
            # Check for session token in headers (like v2 does)
            session_token = auth_response.headers.get('X-WQB-Session-Token')
            if session_token:
                logger.info(f"Session token received: {session_token[:20]}...")
                # Store in session headers for future requests
                self.sess.headers.update({'X-WQB-Session-Token': session_token})
            
            # Verify session works by making a test GET request
            test_response = self.sess.get('https://api.worldquantbrain.com/operators', params={'limit': 1})
            if test_response.status_code == 200:
                logger.info("✅ Session verified - authenticated requests work")
            else:
                logger.warning(f"⚠️ Session verification failed: {test_response.status_code} - {test_response.text}")
                
        except Exception as e:
            logger.error(f"❌ Failed to setup authentication: {e}")
            raise
    
    def _setup_data_fetchers(self):
        """Setup data fetchers for cold start (matching generation_one approach)"""
        try:
            # Initialize fetchers
            self.operator_fetcher = OperatorFetcher(session=self.sess)
            self.data_field_fetcher = DataFieldFetcher(session=self.sess)
            
            # Load operators on cold start (from file, matching generation_one)
            logger.info("Loading operators on cold start...")
            operators = self.operator_fetcher.fetch_operators()
            
            # Note: Data fields are fetched on-demand per region/delay/universe
            # We don't pre-fetch all regions to avoid unnecessary API calls
            # They will be cached when first requested
            
            # Initialize smart search engine with operators only
            # Data fields will be added when fetched
            if operators:
                self.search_engine = SmartSearchEngine(operators, {})
                logger.info(f"Smart search engine initialized with {len(operators)} operators")
            else:
                logger.warning("Could not initialize search engine - missing operators")
            
            # Initialize template validator (AST disabled by default, using prompt engineering only)
            # Will be updated with operators and fields when loaded
            self.template_validator = TemplateValidator(
                operators=operators if operators else [],
                data_fields=[],  # Will be updated per region
                ollama_manager=self.ollama_manager,
                db_path=self.db_path,
                use_ast=False  # Disable AST by default, use prompt engineering and database knowledge only
            )
            logger.info("✅ Template validator initialized (AST disabled, using prompt engineering and database knowledge)")
                
        except Exception as e:
            logger.warning(f"Could not setup data fetchers: {e}")
            # Non-critical, continue without search engine
    
    def make_api_request(self, method: str, url: str, max_retries: int = 2, **kwargs):
        """
        Make API request with automatic 401 reauthentication and session persistence
        Similar to v2's make_api_request method
        
        Args:
            method: HTTP method (GET, POST, PUT, PATCH, DELETE)
            url: Request URL
            max_retries: Maximum retry attempts on 401
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            Response object
        """
        for attempt in range(max_retries):
            try:
                # Log request details for debugging
                logger.debug(f"Making {method} request to {url}")
                if self.sess.cookies:
                    logger.debug(f"Session has {len(self.sess.cookies)} cookie(s) - will be sent automatically")
                else:
                    logger.warning("⚠️ Session has no cookies - request may fail")
                
                if method.upper() == 'GET':
                    response = self.sess.get(url, **kwargs)
                elif method.upper() == 'POST':
                    response = self.sess.post(url, **kwargs)
                elif method.upper() == 'PUT':
                    response = self.sess.put(url, **kwargs)
                elif method.upper() == 'PATCH':
                    response = self.sess.patch(url, **kwargs)
                elif method.upper() == 'DELETE':
                    response = self.sess.delete(url, **kwargs)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                logger.debug(f"Response status: {response.status_code}")
                
                # Check for 401 error - session expired
                if response.status_code == 401:
                    if attempt < max_retries - 1:
                        logger.warning(f"🔐 401 Unauthorized - Re-authenticating (attempt {attempt + 1}/{max_retries})")
                        self.setup_auth()
                        continue
                    else:
                        logger.error(f"🔐 Authentication failed after {max_retries} attempts")
                        raise Exception("Authentication failed after retries")
                
                # Check for 405 error - might indicate session issue
                if response.status_code == 405:
                    logger.error(f"❌ 405 Method Not Allowed - This might indicate a session/authentication issue")
                    logger.error(f"Response: {response.text}")
                    if attempt < max_retries - 1:
                        logger.warning(f"Re-authenticating and retrying (attempt {attempt + 1}/{max_retries})")
                        self.setup_auth()
                        continue
                
                return response
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"🔐 API request failed, retrying: {e}")
                    time.sleep(1)
                    continue
                else:
                    raise e
        
        return None
    
    def generate_template_from_prompt(
        self, 
        prompt: str,
        region: str = "USA",
        use_ollama: bool = True
    ) -> Optional[str]:
        """
        Generate alpha template from a text prompt
        
        Args:
            prompt: Text prompt describing the alpha idea
            region: Region code for theme requirements
            use_ollama: Whether to try Ollama first
            
        Returns:
            Alpha expression string or None
        """
        # Get theme requirements
        theme_requirements = self.theme_manager.get_theme_requirements(region)
        dataset_categories = self.theme_manager.get_required_categories(region)
        
        # Try Ollama first (smart fallback)
        if use_ollama and self.ollama_manager.is_available:
            # Get avoidance context to prevent duplicates
            avoidance_context = self.duplicate_detector.get_avoidance_context(limit=10)
            
            # Get available operators and fields for enhanced prompt
            available_operators = None
            available_fields = None
            successful_patterns = None
            
            if self.operator_fetcher:
                available_operators = self.operator_fetcher.operators if hasattr(self.operator_fetcher, 'operators') else None
            
            if self.template_validator and self.template_validator.use_ast and self.template_validator.corrector:
                # Get AST-extracted patterns for better guidance
                successful_patterns = self.template_validator.corrector.get_successful_patterns(limit=5)
            
            # Get fields for the region if available
            try:
                region_fields = self.get_data_fields_for_region(region)
                if region_fields:
                    available_fields = region_fields
            except Exception as e:
                logger.debug(f"Could not fetch data fields for {region} (non-critical): {e}")

            template = self.ollama_manager.generate_template(
                prompt,
                region=region,
                dataset_categories=dataset_categories if dataset_categories else None,
                avoid_duplicates_context=avoidance_context,
                available_operators=available_operators,
                available_fields=available_fields,
                successful_patterns=successful_patterns
            )
            
            if template:
                # Check for duplicates
                if self.duplicate_detector.is_duplicate(template):
                    logger.warning(f"Generated duplicate template, retrying: {template[:50]}...")
                    # Retry once with stronger avoidance context
                    stronger_context = self.duplicate_detector.get_avoidance_context(limit=20)
                    # Get available operators and fields for retry
                    available_operators = None
                    available_fields = None
                    successful_patterns = None
                    
                    if self.operator_fetcher:
                        available_operators = self.operator_fetcher.operators if hasattr(self.operator_fetcher, 'operators') else None
                    
                    if self.template_validator and self.template_validator.use_ast and self.template_validator.corrector:
                        successful_patterns = self.template_validator.corrector.get_successful_patterns(limit=5)
                    
                    try:
                        region_fields = self.get_data_fields_for_region(region)
                        if region_fields:
                            available_fields = region_fields
                    except Exception as e:
                        logger.debug(f"Could not fetch data fields for {region} retry (non-critical): {e}")
                    
                    template = self.ollama_manager.generate_template(
                        prompt + "\n\nIMPORTANT: Avoid generating expressions similar to recent ones.",
                        region=region,
                        dataset_categories=dataset_categories if dataset_categories else None,
                        avoid_duplicates_context=stronger_context,
                        available_operators=available_operators,
                        available_fields=available_fields,
                        successful_patterns=successful_patterns
                    )
                
                if template and not self.duplicate_detector.is_duplicate(template):
                    # Register the new expression
                    self.duplicate_detector.register_expression(template, region)
                    logger.debug("Generated template using Ollama")
                    return template
                elif template:
                    logger.warning("Generated template is still a duplicate, skipping")
                    return None
        
        # Fallback to DeepSeek API
        if self.deepseek_api_key:
            return self._generate_with_deepseek(prompt, region, dataset_categories)
        
        # Final fallback
        logger.warning("No LLM available, using fallback generation")
        return self._generate_fallback_template(prompt, region)
        
        try:
            # Call DeepSeek API
            response = requests.post(
                'https://api.deepseek.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.deepseek_api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'deepseek-chat',
                    'messages': [
                        {
                            'role': 'system',
                            'content': 'You are an expert in quantitative finance. Generate WorldQuant Brain alpha expressions in FASTEXPR format.'
                        },
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ],
                    'temperature': 0.7,
                    'max_tokens': 500
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                template = result['choices'][0]['message']['content'].strip()
                # Extract code if wrapped in markdown
                if '```' in template:
                    lines = template.split('\n')
                    template = '\n'.join([l for l in lines if not l.strip().startswith('```')])
                return template
            else:
                logger.error(f"DeepSeek API error: {response.status_code}")
                return self._generate_fallback_template(prompt)
                
        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {e}")
            return self._generate_fallback_template(prompt)
    
    def _generate_with_deepseek(
        self, 
        prompt: str, 
        region: str,
        dataset_categories: List[str]
    ) -> Optional[str]:
        """Generate using DeepSeek API"""
        try:
            system_prompt = 'You are an expert in quantitative finance. Generate WorldQuant Brain alpha expressions in FASTEXPR format.'
            user_prompt = f"Region: {region}\n{prompt}"
            
            if dataset_categories:
                user_prompt += f"\nRequired categories: {', '.join(dataset_categories)}"
            
            response = requests.post(
                'https://api.deepseek.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.deepseek_api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'deepseek-chat',
                    'messages': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ],
                    'temperature': 0.7,
                    'max_tokens': 500
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                template = result['choices'][0]['message']['content'].strip()
                if '```' in template:
                    lines = template.split('\n')
                    template = '\n'.join([l for l in lines if not l.strip().startswith('```')])
                logger.debug("Generated template using DeepSeek")
                return template
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
        
        return None
    
    def _generate_fallback_template(self, prompt: str, region: str = "USA") -> str:
        """Fallback template generation"""
        # Simple fallback - would use more sophisticated logic in production
        if 'momentum' in prompt.lower():
            return "ts_rank(close, 20)"
        elif 'mean reversion' in prompt.lower():
            return "-ts_rank(close - ts_mean(close, 20), 10)"
        elif 'volume' in prompt.lower():
            return "ts_rank(volume, 20)"
        else:
            return "ts_rank(close, 20)"
    
    def generate_templates_batch(
        self, 
        prompts: List[str], 
        max_templates: int = 10
    ) -> List[str]:
        """
        Generate multiple templates from prompts
        
        Args:
            prompts: List of prompt strings
            max_templates: Maximum number of templates to generate
            
        Returns:
            List of alpha expressions
        """
        templates = []
        for prompt in prompts[:max_templates]:
            template = self.generate_template_from_prompt(prompt)
            if template:
                templates.append(template)
            time.sleep(0.5)  # Rate limiting
        
        logger.info(f"Generated {len(templates)} templates from {len(prompts)} prompts")
        return templates
    
    def generate_from_hypothesis(
        self, 
        hypothesis: str, 
        data_fields: List[str],
        region: str = "USA"
    ) -> List[str]:
        """
        Generate alpha expressions from a research hypothesis
        
        Args:
            hypothesis: Research hypothesis
            data_fields: Available data fields
            region: Region code
            
        Returns:
            List of alpha expressions
        """
        prompt = f"""
        Hypothesis: {hypothesis}
        Available data fields: {', '.join(data_fields[:10])}
        
        Generate 3 alpha expressions in FASTEXPR format that test this hypothesis.
        """
        
        templates = []
        for i in range(3):
            template = self.generate_template_from_prompt(prompt, region=region)
            if template:
                templates.append(template)
        
        return templates
    
    def get_data_fields_for_region(
        self,
        region: str,
        delay: int = 1,
        universe: str = None
    ) -> List[Dict]:
        """
        Get data fields for a specific region and delay with local caching
        (Matching generation_one approach)
        
        Args:
            region: Region code (USA, EUR, CHN, ASI, etc.)
            delay: Delay value (default: 1)
            universe: Universe code (e.g., "TOP3000", "MINVOL1M")
            
        Returns:
            List of data field dictionaries
        """
        if not self.data_field_fetcher:
            logger.warning("Data field fetcher not initialized")
            return []
        
        # Get universe from region configs if not provided
        if not universe:
            try:
                from .region_config import get_default_universe
                universe = get_default_universe(region)
            except ImportError:
                # Fallback if region_config not available
                universe_map = {
                    'USA': 'TOP3000',
                    'EUR': 'TOP2500',  # Fixed: was TOP3000
                    'CHN': 'TOP2000U',  # Fixed: was TOP3000
                    'ASI': 'MINVOL1M',
                    'GLB': 'TOP3000',
                    'IND': 'TOP500'
                }
                universe = universe_map.get(region, 'TOP3000')
        
        # Fetch data fields (will use cache if available)
        fields = self.data_field_fetcher.fetch_data_fields(
            region=region,
            delay=delay,
            universe=universe
        )
        
        # Store field types (including event input detection) in database
        if fields:
            self._store_field_types(fields, region, delay)
        
        # Update search engine with fetched fields
        if self.search_engine and fields:
            if region not in self.search_engine.data_fields:
                self.search_engine.data_fields[region] = []
            self.search_engine.data_fields[region] = fields
            # Rebuild index for this region
            self.search_engine._build_field_index_for_region(region, fields)
        
        return fields
    
    def _store_field_types(self, fields: List[Dict], region: str, delay: int):
        """Store field type information including event input detection"""
        try:
            from ..storage.backtest_storage import BacktestStorage
            storage = BacktestStorage()
            
            for field in fields:
                field_id = field.get('id', '')
                if not field_id:
                    continue
                
                # Detect event input fields
                field_type = field.get('type', 'REGULAR')
                category = field.get('category', {})
                category_name = category.get('name', '') if isinstance(category, dict) else str(category)
                
                # Check if field is event input
                is_event_input = (
                    field_type == 'EVENT' or
                    'event' in str(field_type).lower() or
                    'event' in category_name.lower() or
                    'event' in str(field.get('name', '')).lower() or
                    'event' in str(field.get('description', '')).lower()
                )
                
                # Store in database
                import sqlite3
                import json
                from ..storage.backtest_storage import BacktestStorage
                storage = BacktestStorage(self.db_path)
                conn = sqlite3.connect(storage.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO field_types
                    (field_id, region, delay, field_type, is_event_input, category, field_data, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    field_id,
                    region,
                    delay,
                    field_type,
                    1 if is_event_input else 0,
                    category_name,
                    json.dumps(field)
                ))
                
                conn.commit()
                conn.close()
            
            logger.debug(f"Stored {len(fields)} field types for {region} delay={delay}")
        except Exception as e:
            logger.debug(f"Failed to store field types: {e}")
    
    def generate_templates_for_region(
        self,
        region: str,
        num_templates: int = 10
    ) -> List[Dict]:
        """
        Generate templates for a specific region with self-correcting AST validation
        
        Args:
            region: Region code
            num_templates: Number of templates to generate
            
        Returns:
            List of template dictionaries with 'template' key
        """
        templates = []
        
        # Get data fields for region
        data_fields = self.get_data_fields_for_region(region)
        
        if not data_fields:
            logger.warning(f"No data fields available for {region}")
            return templates
        
        # Update validator with operators and fields
        if self.template_validator:
            validator = self.template_validator
            # Update operators if not already set
            if self.operator_fetcher:
                operators = self.operator_fetcher.operators
                if operators:
                    # Update parser if AST is enabled, otherwise update stored operators list
                    if validator.use_ast and validator.parser:
                        if not validator.parser.operators:
                            validator.parser.add_operators(operators)
                            logger.info(f"✅ Updated validator with {len(operators)} operators")
                    else:
                        validator.operators = operators
                        logger.info(f"✅ Updated validator with {len(operators)} operators (AST disabled)")
            
            # Update data fields for this region
            if validator.use_ast and validator.parser:
                validator.parser.add_data_fields(data_fields)
                logger.info(f"✅ Updated validator with {len(data_fields)} data fields for {region}")
            else:
                validator.data_fields = data_fields
                logger.info(f"✅ Updated validator with {len(data_fields)} data fields for {region} (AST disabled)")
        
        # Get storage instance for template similarity checking
        from ..storage.backtest_storage import BacktestStorage
        storage = BacktestStorage(self.db_path)
        
        # Don't load all templates upfront - use targeted queries instead (resource efficient)
        logger.info(f"📊 Template generation for {region} (using targeted database queries)")
        
        # Generate templates using Ollama with v2-style step-by-step approach
        max_retries_per_template = 5  # Maximum retries if template is too similar
        for i in range(num_templates):
            retry_count = 0
            template = None
            
            while retry_count < max_retries_per_template:
                try:
                    logger.info(f"🔄 Generating template {i+1}/{num_templates} for {region} (attempt {retry_count + 1})")
                    
                    # Use enhanced prompt engineering with operator+field combinations
                    prompt = f"""Generate a WorldQuant Brain FASTEXPR alpha expression for {region} region.

The expression should combine OPERATORS (functions like ts_rank, ts_delta, rank) with DATA FIELDS (variables like {region}.MCAP, {region}.RETURN).

Generate a valid FASTEXPR expression that uses operator(data_field, parameters) syntax."""
                    
                    # Get available operators and fields for this region
                    available_operators = None
                    available_fields = data_fields  # Already fetched above
                    successful_patterns = None
                    
                    if self.operator_fetcher:
                        available_operators = self.operator_fetcher.operators if hasattr(self.operator_fetcher, 'operators') else None
                    
                    if self.template_validator and self.template_validator.corrector:
                        # Get AST-extracted patterns for better guidance
                        successful_patterns = self.template_validator.corrector.get_successful_patterns(limit=5)
                    
                    # Add similarity avoidance context (query only recent templates for performance)
                    avoidance_context = ""
                    # Only fetch last 10 templates for avoidance context (not all templates)
                    recent_templates = storage.get_all_templates(region=region, limit=10)
                    if recent_templates:
                        from ..core.template_similarity import TemplateSimilarityChecker
                        similarity_checker = TemplateSimilarityChecker()
                        
                        # Extract operators and fields from recent templates for context
                        all_ops = set()
                        all_fields = set()
                        for existing in recent_templates:
                            all_ops.update(similarity_checker.extract_operators(existing))
                            all_fields.update(similarity_checker.extract_fields(existing))
                        
                        avoidance_context = f"\n\nAVOID SIMILARITY: Do NOT generate templates similar to these recent ones:\n"
                        avoidance_context += "\n".join([f"  - {t[:60]}..." for t in recent_templates[:5]])
                        avoidance_context += f"\n\nCommon operators already used: {', '.join(list(all_ops)[:10])}"
                        avoidance_context += f"\nCommon fields already used: {', '.join(list(all_fields)[:5])}"
                        avoidance_context += "\n\nGenerate a COMPLETELY DIFFERENT expression with different operators and fields!"
                    
                    # V2 Approach: Use placeholder-based generation to avoid misspelling
                    # Step 1: Randomly/weighted pick data fields by index
                    import random
                    selected_field_indices = self._select_fields_v2(data_fields, num_fields=random.randint(2, 4))
                    selected_fields = [data_fields[i] for i in selected_field_indices if i < len(data_fields)]
                    
                    # Step 2: Randomly/weighted pick operators
                    if available_operators:
                        selected_operator_indices = self._select_operators_v2(available_operators, num_operators=random.randint(2, 4))
                        selected_operators = [available_operators[i] for i in selected_operator_indices if i < len(available_operators)]
                    else:
                        selected_operators = available_operators
                    
                    # Step 3: Generate template with placeholders
                    template = self.generate_template_from_prompt(prompt, region=region, use_ollama=True)
                    
                    # If generation failed, try direct Ollama call with placeholder approach
                    if not template and self.ollama_manager.is_available:
                        template = self.ollama_manager.generate_template(
                            prompt,
                            region=region,
                            avoid_duplicates_context=avoidance_context,
                            available_operators=selected_operators,
                            available_fields=selected_fields,  # Pass selected fields for placeholder mapping
                            successful_patterns=successful_patterns,
                            use_placeholder_fields=True  # Enable V2 placeholder approach
                        )
                    
                    # Step 4: Replace placeholders with actual operator names and field IDs
                    if template and selected_operators:
                        template = self._replace_operator_placeholders(template, selected_operators)
                    if template and selected_fields:
                        template = self._replace_field_placeholders(template, selected_fields, region)
                    
                    if not template:
                        logger.warning(f"Failed to generate template {i+1}, attempt {retry_count + 1}")
                        retry_count += 1
                        continue
                    
                    # Check similarity with existing templates (query only when needed)
                    similar_templates = storage.check_template_similarity(
                        template, 
                        region=region,
                        similarity_threshold=0.7,
                        max_check=100  # Only check last 100 templates for performance
                    )
                    
                    if similar_templates:
                        logger.warning(f"⚠️ Template {i+1} is too similar to existing templates (similarity: {similar_templates[0][1]:.2f})")
                        logger.warning(f"   Similar to: {similar_templates[0][0][:60]}...")
                        logger.info(f"   Retrying with different template...")
                        retry_count += 1
                        template = None  # Reset to retry
                        continue
                    
                    # Template is unique enough, break retry loop
                    break
                    
                except Exception as e:
                    logger.error(f"Error generating template {i+1}, attempt {retry_count + 1}: {e}")
                    retry_count += 1
                    continue
            
            # If we still don't have a template after retries, skip
            if not template:
                logger.warning(f"⚠️ Failed to generate unique template {i+1} after {max_retries_per_template} attempts, skipping")
                continue
            
            # Validate and fix using self-correcting AST and compiler
            try:
                if self.template_validator:
                    # Use compiler for full compilation pipeline
                    compile_result = self.template_validator.compile_template(template, optimize=False)
                    
                    if compile_result.success:
                        # Compilation successful - use optimized expression if available
                        if compile_result.final_expression and compile_result.final_expression != template:
                            logger.debug(f"Compiler generated expression: {compile_result.final_expression[:80]}...")
                            template = compile_result.final_expression
                        
                        # Learn from successful compilation
                        self.template_validator.learn_from_success(template)
                        logger.info(f"✅ Template {i+1} compiled successfully (complexity: {compile_result.metadata.get('complexity', 'N/A')})")
                    else:
                        # Compilation failed - try traditional validation and fixing
                        logger.warning(f"⚠️ Template {i+1} compilation failed: {compile_result.errors[0].message if compile_result.errors else 'Unknown error'}")
                        is_valid, error_msg, suggested_fix = self.template_validator.validate_template(template, region)
                        
                        if not is_valid:
                            # Try to fix using both AST and prompt engineering
                            fixed_template, fixes = self.template_validator.fix_template(template, error_msg, region)
                            if fixes:
                                logger.info(f"🔧 Fixed template {i+1} with {len(fixes)} corrections: {fixes}")
                                template = fixed_template
                                # Re-compile to verify
                                compile_result = self.template_validator.compile_template(template, optimize=False)
                                if compile_result.success:
                                    logger.info(f"✅ Template {i+1} fixed and compiled")
                                    self.template_validator.learn_from_success(template)
                                else:
                                    logger.warning(f"⚠️ Template {i+1} still invalid after fixes, but including it")
                            else:
                                logger.warning(f"⚠️ Could not fix template {i+1}, skipping")
                                continue
                        else:
                            # Traditional validation passed
                            self.template_validator.learn_from_success(template)
                
                # Extract operators and fields for storage
                from ..core.template_similarity import TemplateSimilarityChecker
                similarity_checker = TemplateSimilarityChecker()
                operators_used = list(similarity_checker.extract_operators(template))
                fields_used = list(similarity_checker.extract_fields(template))
                
                # Store template in database
                storage.store_template(
                    template=template,
                    region=region,
                    operators_used=operators_used,
                    fields_used=fields_used
                )
                
                # No need to keep in memory - we'll query when needed (resource efficient)
                
                # Add to results
                templates.append({
                    'template': template,
                    'region': region,
                    'index': i + 1
                })
                logger.info(f"✅ Generated unique template {i+1}: {template[:80]}...")
                
            except Exception as e:
                logger.error(f"Error processing template {i+1}: {e}", exc_info=True)
                continue
        
        logger.info(f"✅ Generated {len(templates)}/{num_templates} templates for {region}")
        return templates
    
    def _select_fields_v2(self, data_fields: List[Dict], num_fields: int = 3) -> List[int]:
        """V2 Approach: Randomly/weighted select field indices (not names)"""
        import random
        
        if not data_fields or num_fields <= 0:
            return []
        
        # Simple random selection for now (can add weighting based on success later)
        num_fields = min(num_fields, len(data_fields))
        selected_indices = random.sample(range(len(data_fields)), num_fields)
        
        logger.debug(f"Selected field indices: {selected_indices}")
        return selected_indices
    
    def _select_operators_v2(self, operators: List[Dict], num_operators: int = 3) -> List[int]:
        """V2 Approach: Randomly/weighted select operator indices (not names)"""
        import random
        
        if not operators or num_operators <= 0:
            return []
        
        # Simple random selection for now (can add weighting based on success later)
        num_operators = min(num_operators, len(operators))
        selected_indices = random.sample(range(len(operators)), num_operators)
        
        logger.debug(f"Selected operator indices: {selected_indices}")
        return selected_indices
    
    def _replace_field_placeholders(self, template: str, selected_fields: List[Dict], region: str = None) -> str:
        """V2 Approach: Replace DATA_FIELD1, DATA_FIELD2, etc. with actual field IDs"""
        import re
        
        if not template or not selected_fields:
            return template
        
        # Create field mapping: DATA_FIELD1 -> first field, DATA_FIELD2 -> second field, etc.
        # Support both uppercase (DATA_FIELD1) and lowercase (data_field1) variants
        field_mapping = {}
        for i, field in enumerate(selected_fields[:10], start=1):  # Support up to DATA_FIELD10
            field_id = field.get('id', '')
            if field_id:
                # Map both uppercase and lowercase variants
                field_mapping[f'DATA_FIELD{i}'] = field_id
                field_mapping[f'data_field{i}'] = field_id
                field_mapping[f'Data_Field{i}'] = field_id
        
        # Replace placeholders with actual field IDs (word boundary to avoid partial matches)
        result = template
        replacements_made = []
        for placeholder, actual_field in field_mapping.items():
            # Use word boundary to ensure exact match, case-insensitive
            pattern = r'\b' + re.escape(placeholder) + r'\b'
            if re.search(pattern, result, flags=re.IGNORECASE):
                result = re.sub(pattern, actual_field, result, flags=re.IGNORECASE)
                replacements_made.append(f"{placeholder} -> {actual_field}")
                logger.debug(f"Replaced {placeholder} -> {actual_field}")
        
        if result != template:
            logger.info(f"✅ Field placeholder replacement ({len(replacements_made)} replacements): {template[:50]}... -> {result[:50]}...")
        
        return result
    
    def _replace_operator_placeholders(self, template: str, selected_operators: List[Dict]) -> str:
        """V3 Approach: Replace OPERATOR1, OPERATOR2, etc. with actual operator names"""
        import re
        
        if not template or not selected_operators:
            return template
        
        # Create operator mapping: OPERATOR1 -> first operator, OPERATOR2 -> second operator, etc.
        # Support both uppercase (OPERATOR1) and lowercase (operator1) variants
        operator_mapping = {}
        for i, op in enumerate(selected_operators[:10], start=1):  # Support up to OPERATOR10
            operator_name = op.get('name', '')
            if operator_name:
                # Map both uppercase and lowercase variants
                operator_mapping[f'OPERATOR{i}'] = operator_name
                operator_mapping[f'operator{i}'] = operator_name
                operator_mapping[f'Operator{i}'] = operator_name
        
        # Replace placeholders with actual operator names (word boundary to avoid partial matches)
        result = template
        replacements_made = []
        for placeholder, actual_operator in operator_mapping.items():
            # Use word boundary to ensure exact match, case-insensitive
            pattern = r'\b' + re.escape(placeholder) + r'\b'
            if re.search(pattern, result, flags=re.IGNORECASE):
                result = re.sub(pattern, actual_operator, result, flags=re.IGNORECASE)
                replacements_made.append(f"{placeholder} -> {actual_operator}")
                logger.debug(f"Replaced {placeholder} -> {actual_operator}")
        
        if result != template:
            logger.info(f"✅ Operator placeholder replacement ({len(replacements_made)} replacements): {template[:50]}... -> {result[:50]}...")
        
        return result






# from alpha_mining.core.region_config import logger
# import re
# from typing import Any, Dict, List, Optional, Literal

# from region_config import RegionConfig, get_region_config
# from alpha_mining.data_fetcher.data_field_fetcher import DataFieldFetcher



# class AlphaExpressionGenerator:
#     """Build a simple FASTEXPR-style alpha expression from region defaults."""

#     def __init__(
#         self,
#         region: str = "USA",
#         region_config: Optional[RegionConfig] = None,
#         fetcher: Optional[DataFieldFetcher] = None,
#         max_fields: int = 3,
#     ) -> None:
#         self.region = region
#         self.region_config = region_config or get_region_config(region)
#         self.fetcher = fetcher or DataFieldFetcher()
#         self.max_fields = max_fields

#     def generate_expression(self, field_limit: Optional[int] = None) -> str:
#         """Generate an alpha expression using the region defaults and fetched fields."""
#         fields = self._fetch_fields()
#         selected_fields = self._select_fields(fields, field_limit or self.max_fields)

#         if not selected_fields:
#             return self._fallback_expression()

#         if len(selected_fields) == 1:
#             base_field = selected_fields[0]
#             return f"{self.region_config.default_universe}_{self.region_config.default_neutralization.lower()}_rank({base_field})"

#         first_field = selected_fields[0]
#         second_field = selected_fields[1]
#         expression = f"{self.region_config.default_universe}_{self.region_config.default_neutralization.lower()}_rank(ts_zscore({first_field} - {second_field}, 10))"

#         if len(selected_fields) >= 3:
#             third_field = selected_fields[2]
#             expression = f"{self.region_config.default_universe}_{self.region_config.default_neutralization.lower()}_rank(ts_zscore(({first_field} - {second_field}) + {third_field}, 10))"

#         return expression

#     def generate_alpha(
#         regular: Optional[str] = None,
#         selection: Optional[str] = None,
#         combo: Optional[str] = None,
#         alpha_type: Literal["REGULAR", "SUPER"] = "REGULAR",
#         region: str = "USA",
#         universe: str = "TOP3000",
#         delay: Literal[0, 1] = 1,
#         decay: int = 0,
#         neutralization: str = "INDUSTRY",
#         truncation: float = 0.08,
#         pasteurization: Literal["ON", "OFF"] = "ON",
#         test_period: str = "P0Y0M0D",
#         unit_handling: Literal["VERIFY"] = "VERIFY",
#         nan_handling: Literal["ON", "OFF"] = "OFF",
#         max_trade: Literal["ON", "OFF"] = "OFF",
#         selection_handling: str = "POSITIVE",
#         selection_limit: int = 100,
#         visualization: bool = False,
#     ) -> dict:
#         """
#         Generate an alpha dictionary for simulation. If alpha_type='REGULAR',
#         function generates alpha dictionary using regular input. If alpha_type='SUPER',
#         function generates alpha dictionary using selection and combo inputs.

#         Args:
#             regular (str, optional): The regular alpha expression.
#             selection (str, optional): The selection expression for super alphas.
#             combo (str, optional): The combo expression for super alphas.
#             alpha_type (str, optional): The type of alpha ("REGULAR" or "SUPER"). Defaults to "REGULAR".
#             region (str, optional): The region for the alpha. Defaults to "USA".
#             universe (str, optional): The universe for the alpha. Defaults to "TOP3000".
#             delay (int, optional): The delay for the alpha. Defaults to 1.
#             decay (int, optional): The decay for the alpha. Defaults to 0.
#             neutralization (str, optional): The neutralization method. Defaults to "INDUSTRY".
#             truncation (float, optional): The truncation value. Defaults to 0.08.
#             pasteurization (str, optional): The pasteurization setting. Defaults to "ON".
#             test_period (str, optional): The test period. Defaults to "P0Y0M0D".
#             unit_handling (str, optional): The unit handling method. Defaults to "VERIFY".
#             nan_handling (str, optional): The NaN handling method. Defaults to "OFF".
#             max_trade (str, optional): The max trade method. Defaults to "OFF".
#             selection_handling (str, optional): The selection handling method for super alphas. Defaults to "POSITIVE".
#             selection_limit (int, optional): The selection limit for super alphas. Defaults to 100.
#             visualization (bool, optional): Whether to include visualization. Defaults to False.

#         Returns:
#             dict: A dictionary containing the alpha configuration for simulation.

#         Raises:
#             ValueError: If an invalid alpha_type is provided.
#         """

#         settings = {
#             "instrumentType": "EQUITY",
#             "region": region,
#             "universe": universe,
#             "delay": delay,
#             "decay": decay,
#             "neutralization": neutralization,
#             "truncation": truncation,
#             "pasteurization": pasteurization,
#             "testPeriod": test_period,
#             "unitHandling": unit_handling,
#             "nanHandling": nan_handling,
#             "maxTrade": max_trade,
#             "language": "FASTEXPR",
#             "visualization": visualization,
#         }
#         if alpha_type == "REGULAR":
#             simulation_data = {
#                 "type": alpha_type,
#                 "settings": settings,
#                 "regular": regular,
#             }
#         elif alpha_type == "SUPER":
#             simulation_data = {
#                 "type": alpha_type,
#                 "settings": {
#                     **settings,
#                     "selectionHandling": selection_handling,
#                     "selectionLimit": selection_limit,
#                 },
#                 "combo": combo,
#                 "selection": selection,
#             }
#         else:
#             logger.error("alpha_type should be REGULAR or SUPER")
#             return {}
#         return simulation_data

#     def build_alpha_payload(self, regular_expression: Optional[str] = None, **overrides: Any) -> Dict[str, Any]:
#         """Create an alpha payload compatible with the simulation helpers."""
#         try:
#             from ..lib import generate_alpha
#         except Exception:  # pragma: no cover - optional dependency path
#             generate_alpha = None

#         expression = regular_expression or self.generate_expression()
#         if generate_alpha is None:
#             return {
#                 "regular": expression,
#                 "region": self.region_config.region,
#                 "universe": self.region_config.default_universe,
#                 "delay": self.region_config.delay,
#                 "neutralization": self.region_config.default_neutralization,
#                 **overrides,
#             }

#         return generate_alpha(
#             regular=expression,
#             region=self.region_config.region,
#             universe=self.region_config.default_universe,
#             delay=self.region_config.delay,
#             neutralization=self.region_config.default_neutralization,
#             **overrides,
#         )

#     def _fetch_fields(self) -> List[Dict[str, Any]]:
#         return self.fetcher.fetch_data_fields(
#             region=self.region_config.region,
#             delay=self.region_config.delay,
#             universe=self.region_config.default_universe,
#             try_all_universes=True,
#         )

#     def _select_fields(self, fields: List[Dict[str, Any]], field_limit: int) -> List[str]:
#         selected: List[str] = []
#         seen = set()

#         for field in fields:
#             field_id = field.get("id") or field.get("name") or ""
#             normalized = self._normalize_field_id(field_id)
#             if not normalized or normalized in seen:
#                 continue

#             selected.append(normalized)
#             seen.add(normalized)
#             if len(selected) >= field_limit:
#                 break

#         return selected

#     def _normalize_field_id(self, field_id: Any) -> str:
#         text = str(field_id).strip()
#         if not text:
#             return ""

#         text = text.replace(".", "_").replace("-", "_").replace(" ", "_")
#         text = re.sub(r"[^A-Za-z0-9_]", "_", text)
#         text = re.sub(r"_+", "_", text).strip("_")
#         return text or "field"

#     def _fallback_expression(self) -> str:
#         return f"{self.region_config.default_universe}_{self.region_config.default_neutralization.lower()}_rank(ts_zscore(field, 10))"


# def generate_alpha_expression(
#     region: str = "USA",
#     region_config: Optional[RegionConfig] = None,
#     fetcher: Optional[DataFieldFetcher] = None,
#     max_fields: int = 3,
# ) -> str:
#     """Convenience wrapper that returns a generated expression using region defaults."""
#     generator = AlphaExpressionGenerator(
#         region=region,
#         region_config=region_config,
#         fetcher=fetcher,
#         max_fields=max_fields,
#     )
#     return generator.generate_expression()

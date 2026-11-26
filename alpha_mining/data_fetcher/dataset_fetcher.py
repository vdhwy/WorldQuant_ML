import logging
import os
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)
brain_api_url = os.environ.get("BRAIN_API_URL", "https://api.worldquantbrain.com")


class DatasetFetcher:
    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()

    def fetch_all_dataset_ids(
        self,
        instrument_type: str = "EQUITY",
        region: str = "USA",
        delay: int = 1,
        universe: str = "TOP3000",
        theme: Optional[bool] = None,
        limit: int = 200,
    ) -> List[str]:
        """Fetch all dataset ids from the WorldQuant Brain API."""
        if self.session is None:
            raise ValueError("A requests session is required to fetch dataset ids.")

        params = {
            "instrumentType": instrument_type,
            "region": region,
            "delay": delay,
            "universe": universe,
            "limit": limit,
            "offset": 0,
        }
        if theme is not None:
            params["theme"] = "true" if theme else "false"

        dataset_ids: List[str] = []
        while True:
            logger.debug("Fetching dataset ids with params: %s", params)
            response = self.session.get(f"{brain_api_url}/data-sets", params=params)
            if response.status_code != 200:
                logger.error(
                    "Failed to fetch dataset ids: %s %s",
                    response.status_code,
                    response.text[:200],
                )
                break

            payload = response.json()
            results = payload.get("results", [])
            if not results:
                break

            dataset_ids.extend([item.get("id") for item in results if item.get("id")])
            if len(results) < limit:
                break

            params["offset"] += limit

        # Preserve order while removing duplicates
        unique_ids = list(dict.fromkeys(dataset_ids))
        logger.info("Fetched %s unique dataset ids for region=%s delay=%s universe=%s", len(unique_ids), region, delay, universe)
        return unique_ids

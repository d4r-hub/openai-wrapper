from typing import Any, Dict, Optional
import requests
import time

class Client:
    def __init__(
        self,
        api_key: str,
        organization_id: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1"
    ):
        self.api_key = api_key
        self.organization_id = organization_id
        self.base_url = base_url.rstrip('/')
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
        if organization_id:
            self.session.headers.update({
                "OpenAI-Organization": organization_id
            })

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Make an HTTP request to the OpenAI API."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(max_retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json,
                    files=files
                )
                
                if response.status_code == 429:  # Rate limit
                    retry_after = int(response.headers.get('Retry-After', 1))
                    print(f"Rate limit hit. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Request failed. Retrying... ({attempt + 1}/{max_retries})")
                time.sleep(1)

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a GET request to the OpenAI API."""
        return self._make_request("GET", endpoint, params=params)

    def _post(self, endpoint: str, json: Optional[Dict[str, Any]] = None, files: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a POST request to the OpenAI API."""
        return self._make_request("POST", endpoint, json=json, files=files)

    def _delete(self, endpoint: str) -> Dict[str, Any]:
        """Make a DELETE request to the OpenAI API."""
        return self._make_request("DELETE", endpoint) 
import requests
import json
from typing import Dict, List, Optional
from constants import SENSOR_IDS

class APIClient:
    def __init__(self, base_url: str, email: str, password: str):
        """
        Initialize the API client with credentials.
        
        Args:
            base_url: Base URL of the API (e.g., 'https://iveg.ual.es:3790')
            email: Login email
            password: Login password
        """
        self.base_url = base_url
        self.email = email
        self.password = password
        self.token = None
        self.session = requests.Session()
        # Disable SSL verification warning (use with caution)
        requests.packages.urllib3.disable_warnings()
    
    def login(self) -> bool:
        """
        Authenticate with the API and store the token.
        
        Returns:
            True if login successful, False otherwise
        """
        url = f"{self.base_url}/api/login"
        
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'email': self.email,
            'password': self.password
        }
        
        try:
            response = self.session.post(
                url, 
                data=data, 
                headers=headers,
                verify=False  # Set to True in production with proper SSL cert
            )
            
            if response.status_code == 200:
                result = response.json()
                # The token might be in different fields depending on API response
                # Common locations: 'token', 'access_token', 'authToken', etc.
                self.token = result.get('token') or result.get('access_token') or result.get('authToken')
                
                if self.token:
                    print(f"✓ Login successful! Token: {self.token[:20]}...")
                    return True
                else:
                    print(f"⚠ Login returned 200 but no token found in response: {result}")
                    return False
            else:
                print(f"✗ Login failed with status {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Login error: {str(e)}")
            return False
    
    def get_last_data(self, sensor_ids: List[str]) -> Optional[Dict]:
        """
        Retrieve the last data for specified sensors.
        
        Args:
            sensor_ids: List of sensor IDs to query
            
        Returns:
            JSON response data or None if request fails
        """
        if not self.token:
            print("✗ Not authenticated. Please login first.")
            return None
        
        url = f"{self.base_url}/api/v1/data/last-data-combined/multiple-by-sensorIds"
        
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': self.token  # Some APIs use 'Bearer ' prefix
        }
        
        payload = {
            'sensorIds': sensor_ids
        }
        
        try:
            response = self.session.post(
                url,
                json=payload,
                headers=headers,
                verify=False
            )
            
            if response.status_code == 200:
                print(f"✓ Data retrieved successfully!")
                return response.json()
            elif response.status_code == 401:
                print(f"✗ Unauthorized (401). Token may have expired. Trying to re-login...")
                # Try to re-authenticate
                if self.login():
                    # Retry with new token
                    headers['Authorization'] = self.token
                    response = self.session.post(url, json=payload, headers=headers, verify=False)
                    if response.status_code == 200:
                        print(f"✓ Data retrieved successfully after re-login!")
                        return response.json()
                return None
            else:
                print(f"✗ Request failed with status {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"✗ Request error: {str(e)}")
            return None


def remote_read():
    # Configuration
    BASE_URL = "https://iveg.ual.es:3790"
    EMAIL = "paul@ual.es"
    PASSWORD = "Paul2024!"

    # Create client and authenticate
    client = APIClient(BASE_URL, EMAIL, PASSWORD)
    
    if client.login():
        # Fetch data
        data = client.get_last_data(SENSOR_IDS)
        
        # if data:
            # # Pretty print the response
            # print("\n" + "="*50)
            # print("SENSOR DATA:")
            # print("="*50)
            # print(json.dumps(data, indent=2))
            
            # # Optionally save to file
            # with open('sensor_data.json', 'w') as f:
            #     json.dump(data, f, indent=2)
            # print("\n✓ Data saved to sensor_data.json")
    else:
        print("\n✗ Failed to authenticate. Please check your credentials.")

    return [entry["measures"]["value"] for entry in data["data"]]


# Example usage
if __name__ == "__main__":
    # Configuration
    BASE_URL = "https://iveg.ual.es:3790"
    EMAIL = "paul@ual.es"
    PASSWORD = "Paul2024!"


    
    # Create client and authenticate
    client = APIClient(BASE_URL, EMAIL, PASSWORD)
    
    if client.login():
        # Fetch data
        data = client.get_last_data(SENSOR_IDS)
        
        if data:
            # Pretty print the response
            print("\n" + "="*50)
            print("SENSOR DATA:")
            print("="*50)
            print(json.dumps(data, indent=2))
            
            # Optionally save to file
            with open('sensor_data.json', 'w') as f:
                json.dump(data, f, indent=2)
            print("\n✓ Data saved to sensor_data.json")
    else:
        print("\n✗ Failed to authenticate. Please check your credentials.")
# poc_congress_api_client.py

# 1. Define a placeholder for the Congress.gov API key
CONGRESS_API_KEY = "YOUR_CONGRESS_API_KEY_HERE"

def fetch_bill_details(bill_congress: int, bill_type: str, bill_number: int):
    """
    Simulates fetching bill details from the Congress.gov API.
    This is a Proof of Concept and does not make actual API calls.
    """
    # 2a. Prints a message indicating it's a PoC
    print(f"[PoC] Attempting to fetch Congress.gov data for bill: {bill_congress}-{bill_type}-{bill_number}")

    # 2b. Defines a base URL
    base_url = f"https://api.congress.gov/v3/bill/{bill_congress}/{bill_type}/{bill_number}"

    # 2c. Shows how the API key would be included in headers or parameters
    # Congress.gov API typically uses an API key in the header
    headers = {
        "X-Api-Key": CONGRESS_API_KEY,
        "Accept": "application/json"
    }
    print(f"[PoC] Would make a GET request to: {base_url} with headers containing the API key.")
    # Alternatively, some APIs might take it as a query parameter:
    # params = {"api_key": CONGRESS_API_KEY}
    # print(f"[PoC] Or with params: {params}")


    # 2d. Includes a mock JSON response structure for a bill
    mock_response = {
        "bill": {
            "congress": bill_congress,
            "number": str(bill_number),
            "originChamber": "House",
            "title": f"A bill to authorize appropriations for fiscal year {bill_congress+1788} for military activities of the Department of Defense and for military construction, to prescribe military personnel strengths for such fiscal year, and for other purposes.",
            "type": bill_type.upper(),
            "introducedDate": f"{bill_congress+1787}-01-15",
            "sponsors": [
                {
                    "firstName": "Jane",
                    "lastName": "Doe",
                    "party": "D",
                    "state": "NY",
                    "district": 9
                }
            ],
            "latestAction": {
                "actionDate": f"{bill_congress+1787}-07-20",
                "text": "Became Public Law No: {bill_congress}-XX."
            },
            "policyArea": {
                "name": "Armed Forces and National Security"
            },
            "committees": {
                "houseCommittees": [
                    {
                        "name": "House Armed Services Committee",
                        "systemCode": "hsas00"
                    }
                ]
            }
        }
    }

    # 2e. Prints a message "Simulating API call..."
    print("Simulating API call...")

    # 2f. Returns the mock JSON response
    return mock_response

def parse_congress_bill_data(api_response: dict):
    """
    Parses the mock Congress.gov API response and prints key information.
    """
    print("\n--- Parsing Congress.gov Bill Data ---")
    if api_response and api_response.get("bill"):
        bill_info = api_response["bill"]
        # 3b. Extracts and prints key information
        print(f"Bill Title: {bill_info.get('title')}")
        if bill_info.get("sponsors"):
            sponsor = bill_info["sponsors"][0]
            print(f"Sponsor: {sponsor.get('firstName')} {sponsor.get('lastName')} ({sponsor.get('party')}-{sponsor.get('state')})")
        print(f"Introduced Date: {bill_info.get('introducedDate')}")
        if bill_info.get("latestAction"):
            print(f"Latest Action: {bill_info['latestAction'].get('text')} on {bill_info['latestAction'].get('actionDate')}")
        print(f"Policy Area: {bill_info.get('policyArea', {}).get('name')}")
        print("--- Congress.gov Data Parsed ---")
    else:
        print("No bill data found in API response.")

# 4. In a main block
if __name__ == "__main__":
    # 4a. Call fetch_bill_details
    sample_congress = 117
    sample_type = "hr"
    sample_number = 1
    mock_data = fetch_bill_details(sample_congress, sample_type, sample_number)

    # 4b. Call parse_congress_bill_data
    parse_congress_bill_data(mock_data)

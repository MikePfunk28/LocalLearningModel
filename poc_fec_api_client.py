# poc_fec_api_client.py

# 1. Define a placeholder for the FEC API key
FEC_API_KEY = "YOUR_FEC_API_KEY_HERE"

def fetch_fec_candidate_data(candidate_id: str):
    """
    Simulates fetching candidate data from the FEC API.
    This is a Proof of Concept and does not make actual API calls.
    """
    # 2a. Prints a message indicating it's a PoC
    print(f"[PoC] Attempting to fetch FEC data for candidate_id: {candidate_id}")

    # 2b. Defines a base URL
    base_url = f"https://api.open.fec.gov/v1/candidate/{candidate_id}/"

    # 2c. Shows how parameters (like the API key) would be added
    params = {
        "api_key": FEC_API_KEY,
        "sort_nulls_last": "false",
        "page": 1,
        "sort_hide_null": "false",
        "sort_null_only": "false",
        "per_page": 20
    }
    print(f"[PoC] Would make a request to: {base_url} with params: {params}")

    # 2d. Includes a mock JSON response structure
    mock_response = {
        "api_version": "1.0",
        "pagination": {
            "count": 1,
            "pages": 1,
            "per_page": 20,
            "page": 1
        },
        "results": [
            {
                "name": "SMITH, JOHN",
                "party": "DEM",
                "state": "CA",
                "candidate_id": candidate_id,
                "election_years": [2022, 2024],
                "principal_committees": [
                    {"committee_id": "C00123456", "name": "SMITH FOR AMERICA"}
                ],
                "load_date": "2023-10-26T10:00:00Z",
                "last_f5_date": "2023-10-15",
                "candidate_status": "C",
                "incumbent_challenge_full": "Challenger",
                "office_full": "President",
                "financial_summary": {
                    "receipts": 1500000.75,
                    "disbursements": 1200000.50,
                    "cash_on_hand_end_period": 300000.25,
                    "coverage_end_date": "2023-09-30"
                }
            }
        ]
    }

    # 2e. Prints a message "Simulating API call..."
    print("Simulating API call...")

    # 2f. Returns the mock JSON response
    return mock_response

def parse_fec_candidate_data(api_response: dict):
    """
    Parses the mock FEC API response and prints key information.
    """
    print("\n--- Parsing FEC Candidate Data ---")
    if api_response and api_response.get("results"):
        candidate_info = api_response["results"][0]
        # 3b. Extracts and prints a few key pieces of information
        print(f"Candidate Name: {candidate_info.get('name')}")
        print(f"Party: {candidate_info.get('party')}")
        print(f"State: {candidate_info.get('state')}")
        print(f"Office: {candidate_info.get('office_full')}")
        if candidate_info.get("financial_summary"):
            print(f"Total Receipts: ${candidate_info['financial_summary'].get('receipts'):,.2f}")
            print(f"Total Disbursements: ${candidate_info['financial_summary'].get('disbursements'):,.2f}")
        print("--- FEC Data Parsed ---")
    else:
        print("No results found in API response.")

# 4. In a main block
if __name__ == "__main__":
    # 4a. Call fetch_fec_candidate_data
    sample_candidate_id = "P12345678"
    mock_data = fetch_fec_candidate_data(sample_candidate_id)

    # 4b. Call parse_fec_candidate_data
    parse_fec_candidate_data(mock_data)

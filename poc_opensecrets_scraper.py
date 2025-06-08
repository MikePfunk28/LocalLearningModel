import csv
import io

# 1. Define a sample string variable containing a few lines of CSV data
SAMPLE_CSV_DATA = """Date,Recipient,Party,Amount,Source_Dataset_Name
2023-01-15,Friends of Democracy,Democratic,$500,OpenSecrets Q1 2023 Featured Donor Report
2023-02-20,Liberty Now PAC,Republican,$1200,OpenSecrets Q1 2023 Featured Donor Report
2023-03-10,Committee for Open Government,Independent,$250,OpenSecrets Q1 2023 Featured Donor Report
2022-11-05,Elon Musk for Doge,Libertarian,$420,OpenSecrets Q4 2022 Doge Donor Report
"""

# 2. Implement a function parse_opensecrets_donations(csv_data_string)
def parse_opensecrets_donations(csv_data_string: str):
    """
    Parses a CSV string of donation data and prints a structured representation.
    """
    # 3. Inside the function, use Python's csv module to parse the string.
    # Using io.StringIO to treat the string as a file
    csvfile = io.StringIO(csv_data_string)
    reader = csv.DictReader(csvfile) # Using DictReader for easier column access

    # 4. Iterate through the parsed rows and print out a structured representation
    print("--- OpenSecrets Donation Data ---")
    for row in reader:
        print(
            f"Date: {row['Date']}, "
            f"Recipient: {row['Recipient']}, "
            f"Party: {row['Party']}, "
            f"Amount: {row['Amount']}, " # Removed the extra $ here
            f"Source: {row['Source_Dataset_Name']}"
        )
    print("--- End of Report ---")

# 5. Include a main block that calls this function with the sample CSV data.
if __name__ == "__main__":
    parse_opensecrets_donations(SAMPLE_CSV_DATA)

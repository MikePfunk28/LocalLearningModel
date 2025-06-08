import sqlite3
import csv
import io

# 2. Define Database Name
DB_NAME = "poc_database.db"

# --- PoC Data Definitions ---
# 5. PoC Data Integration Logic - OpenSecrets PoC Data
OPENSECRETS_SAMPLE_CSV_DATA = """Date,Recipient,Party,Amount,Source_Dataset_Name
2023-01-15,Friends of Democracy,Democratic,$500,OpenSecrets Q1 2023 Featured Donor Report
2023-02-20,Liberty Now PAC,Republican,$1200,OpenSecrets Q1 2023 Featured Donor Report
2023-03-10,Committee for Open Government,Independent,$250,OpenSecrets Q1 2023 Featured Donor Report
2022-11-05,Elon Musk for Doge,Libertarian,$420,OpenSecrets Q4 2022 Doge Donor Report
"""

# 5. PoC Data Integration Logic - FEC PoC Data
FEC_MOCK_CANDIDATE_DATA = {
    "candidate_id": "P12345678", # Primary Key for politicians table
    "name": "SMITH, JOHN",
    "party": "DEM",
    "state": "CA",
    "office_full": "President",
    "financial_summary": {
        "receipts": 1500000.75
    }
}

# 5. PoC Data Integration Logic - Congress.gov PoC Data
CONGRESS_MOCK_BILL_DATA = {
    "bill_identifier": "117-hr-1", # Primary Key for bills table
    "congress": 117,
    "type": "hr",
    "number": 1,
    "title": "A bill to authorize appropriations for fiscal year 1905 for military activities...",
    "sponsors": [
        {
            "firstName": "Jane",
            "lastName": "Doe",
        }
    ],
    "introducedDate": "1904-01-15",
    "latestAction": {
        "text": "Became Public Law No: 117-XX."
    },
    "policyArea": {
        "name": "Armed Forces and National Security"
    }
}

# 3. Table Creation Functions
def create_politicians_table(conn):
    """Creates the politicians table."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS politicians (
                politician_id TEXT PRIMARY KEY,
                name TEXT,
                party TEXT,
                state TEXT,
                office TEXT,
                total_receipts REAL
            )
        """)
        conn.commit()
        print("Politicians table created successfully.")
    except sqlite3.Error as e:
        print(f"Error creating politicians table: {e}")

def create_donations_table(conn):
    """Creates the donations table."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS donations (
                donation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                recipient_name TEXT,
                party TEXT,
                amount REAL,
                source_dataset TEXT
            )
        """)
        conn.commit()
        print("Donations table created successfully.")
    except sqlite3.Error as e:
        print(f"Error creating donations table: {e}")

def create_bills_table(conn):
    """Creates the bills table."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bills (
                bill_identifier TEXT PRIMARY KEY,
                title TEXT,
                sponsor_name TEXT,
                introduced_date TEXT,
                latest_action_text TEXT,
                policy_area TEXT
            )
        """)
        conn.commit()
        print("Bills table created successfully.")
    except sqlite3.Error as e:
        print(f"Error creating bills table: {e}")

# 4. Data Insertion Functions
def insert_politician_data(conn, politician_data_dict):
    """Inserts data into the politicians table."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO politicians (politician_id, name, party, state, office, total_receipts)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            politician_data_dict['candidate_id'], # Corrected key
            politician_data_dict['name'],
            politician_data_dict['party'],
            politician_data_dict['state'],
            politician_data_dict['office_full'],
            politician_data_dict['financial_summary']['receipts']
        ))
        # conn.commit() # Commit will be handled in main after all insertions
        print(f"Inserted politician: {politician_data_dict['name']}")
    except sqlite3.Error as e:
        print(f"Error inserting politician data for {politician_data_dict.get('name')}: {e}")

def insert_donation_data(conn, donation_data_row):
    """Inserts data into the donations table."""
    try:
        cursor = conn.cursor()
        # Amount might have '$', remove it and convert to float
        amount_str = donation_data_row['Amount'].replace('$', '')
        amount_float = float(amount_str)
        cursor.execute("""
            INSERT INTO donations (date, recipient_name, party, amount, source_dataset)
            VALUES (?, ?, ?, ?, ?)
        """, (
            donation_data_row['Date'],
            donation_data_row['Recipient'],
            donation_data_row['Party'],
            amount_float,
            donation_data_row['Source_Dataset_Name']
        ))
        # conn.commit()
        print(f"Inserted donation to: {donation_data_row['Recipient']}")
    except sqlite3.Error as e:
        print(f"Error inserting donation data for {donation_data_row.get('Recipient')}: {e}")
    except ValueError as ve:
        print(f"Error converting amount for {donation_data_row.get('Recipient')}: {ve}")


def insert_bill_data(conn, bill_data_dict):
    """Inserts data into the bills table."""
    try:
        cursor = conn.cursor()
        sponsor_obj = bill_data_dict['sponsors'][0] if bill_data_dict['sponsors'] else {}
        sponsor_name = f"{sponsor_obj.get('firstName', '')} {sponsor_obj.get('lastName', '')}".strip()

        cursor.execute("""
            INSERT INTO bills (bill_identifier, title, sponsor_name, introduced_date, latest_action_text, policy_area)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            bill_data_dict['bill_identifier'],
            bill_data_dict['title'],
            sponsor_name,
            bill_data_dict['introducedDate'],
            bill_data_dict['latestAction']['text'],
            bill_data_dict['policyArea']['name']
        ))
        # conn.commit()
        print(f"Inserted bill: {bill_data_dict['title'][:30]}...")
    except sqlite3.Error as e:
        print(f"Error inserting bill data for {bill_data_dict.get('title')}: {e}")

# 5. PoC Data Integration Logic - Processing functions
def process_and_insert_opensecrets_data(conn):
    """Parses OpenSecrets CSV string and inserts data."""
    print("\nProcessing OpenSecrets Data...")
    csvfile = io.StringIO(OPENSECRETS_SAMPLE_CSV_DATA)
    reader = csv.DictReader(csvfile)
    for row in reader:
        insert_donation_data(conn, row)
    print("OpenSecrets data processed.")

def process_and_insert_fec_data(conn):
    """Processes FEC mock data and inserts it."""
    print("\nProcessing FEC Data...")
    insert_politician_data(conn, FEC_MOCK_CANDIDATE_DATA)
    print("FEC data processed.")

def process_and_insert_congress_data(conn):
    """Processes Congress.gov mock data and inserts it."""
    print("\nProcessing Congress.gov Data...")
    insert_bill_data(conn, CONGRESS_MOCK_BILL_DATA)
    print("Congress.gov data processed.")

# 6. Verification Function
def verify_data(conn):
    """Performs SELECT queries to verify data insertion."""
    print("\n--- Verifying Data Insertion ---")
    cursor = conn.cursor()

    tables = ["politicians", "donations", "bills"]
    for table_name in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"Table '{table_name}': {count} row(s).")

            if count > 0:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
                first_row = cursor.fetchone()
                print(f"First row in '{table_name}': {first_row}")
        except sqlite3.Error as e:
            print(f"Error verifying table {table_name}: {e}")
    print("--- Verification Complete ---")

# 6. Main Execution Block
if __name__ == "__main__":
    conn = None  # Initialize conn to None for finally block
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(DB_NAME)
        print(f"Connected to database: {DB_NAME}")

        # Call table creation functions
        create_politicians_table(conn)
        create_donations_table(conn)
        create_bills_table(conn)

        # Call functions to process and insert data
        process_and_insert_opensecrets_data(conn)
        process_and_insert_fec_data(conn)
        process_and_insert_congress_data(conn)

        # Commit all transactions
        conn.commit()
        print("\nAll data committed to the database.")

        # Verify data
        verify_data(conn)

    except sqlite3.Error as e:
        print(f"An SQLite error occurred: {e}")
        if conn:
            conn.rollback() # Rollback changes on error
            print("Rolled back database changes.")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

    print("\nDatabase setup script finished.")

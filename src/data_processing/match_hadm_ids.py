import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import csv
import re
import urllib.request
import urllib.error
import os
import getpass
from typing import List, Tuple, Dict, Optional


def parse_waveform_filename(filename: str) -> Tuple[str, datetime]:
    """
    Parse waveform filename to extract patient ID and datetime.

    Args:
        filename: Like "p000020-2183-04-28-17-47.hea"

    Returns:
        Tuple of (patient_id, datetime)
    """
    # Remove .hea extension if present
    name = filename.replace('.hea', '')

    # Pattern: p000020-2183-04-28-17-47
    match = re.match(r'(p\d+)-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})', name)

    if not match:
        raise ValueError(f"Cannot parse filename: {filename}")

    patient_id, year, month, day, hour, minute = match.groups()

    # Create datetime object
    dt = datetime(int(year), int(month), int(day), int(hour), int(minute))

    return patient_id, dt


def load_results_csv(csv_path: str = "results.csv") -> pd.DataFrame:
    """Load and parse the results CSV file."""
    print(f"Loading results from {csv_path}...")

    try:
        df = pd.read_csv(csv_path)
        print(f"Found {len(df)} records")

        # Parse each filename
        parsed_data = []
        for _, row in df.iterrows():
            try:
                patient_id, waveform_datetime = parse_waveform_filename(row['filename'])
                parsed_data.append({
                    'filename': row['filename'],
                    'patient_id': patient_id,
                    'waveform_datetime': waveform_datetime,
                    'original_patient_id': row.get('patient_id', patient_id)  # From CSV if available
                })
            except Exception as e:
                print(f"Warning: Could not parse {row['filename']}: {e}")

        parsed_df = pd.DataFrame(parsed_data)
        print(f"Successfully parsed {len(parsed_df)} records")

        return parsed_df

    except Exception as e:
        print(f"Error loading results CSV: {e}")
        return pd.DataFrame()


def connect_to_mimic_clinical():
    """
    Connect to MIMIC-III Clinical Database.
    You'll need to modify this based on your access method.
    """
    print("How do you access MIMIC-III Clinical Database?")
    print("1. Local ADMISSIONS.csv file (downloaded from PhysioNet)")
    print("2. PhysioNet remote access")
    print("3. Local PostgreSQL database")
    print("4. Google BigQuery")
    print("5. CSV files downloaded locally")
    print("6. Other (please specify)")

    choice = input("Enter choice (1-6): ").strip()

    if choice == "1":
        return setup_local_admissions_file()
    elif choice == "2":
        return setup_physionet_connection()
    elif choice == "3":
        return setup_postgresql_connection()
    elif choice == "4":
        return setup_bigquery_connection()
    elif choice == "5":
        return setup_csv_connection()
    else:
        print("Please modify the connection function for your specific setup")
        return None


def setup_local_admissions_file():
    """Setup connection to local ADMISSIONS.csv file."""
    print("Setting up local ADMISSIONS.csv file access...")

    # Common locations to check
    possible_locations = [
        "ADMISSIONS.csv",
        "./ADMISSIONS.csv",
        "mimic_cache/ADMISSIONS.csv",
        "mimic3/ADMISSIONS.csv",
        "data/ADMISSIONS.csv"
    ]

    # First, check if file exists in common locations
    found_file = None
    for location in possible_locations:
        if os.path.exists(location):
            found_file = location
            print(f"✓ Found ADMISSIONS.csv at: {location}")
            break

    if found_file:
        # Verify it's a valid ADMISSIONS.csv file
        try:
            df_test = pd.read_csv(found_file, nrows=1)
            expected_cols = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME']

            if all(col in df_test.columns for col in expected_cols):
                print(f"✓ File verified - contains expected columns")
                print(f"Columns: {list(df_test.columns)}")
                return found_file
            else:
                print(f"❌ File doesn't appear to be ADMISSIONS.csv")
                print(f"Expected columns: {expected_cols}")
                print(f"Found columns: {list(df_test.columns)}")
        except Exception as e:
            print(f"❌ Error reading file: {e}")

    # If not found automatically, ask user for path
    print("\nADMISSIONS.csv not found in common locations.")
    print("Please download ADMISSIONS.csv from PhysioNet and place it in this folder.")
    print("Or specify the full path to the file:")

    while True:
        file_path = input("Path to ADMISSIONS.csv (or 'exit' to quit): ").strip()

        if file_path.lower() == 'exit':
            return None

        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]  # Remove quotes

        if os.path.exists(file_path):
            try:
                # Test read the file
                df_test = pd.read_csv(file_path, nrows=1)
                expected_cols = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME']

                if all(col in df_test.columns for col in expected_cols):
                    print(f"✓ File verified: {file_path}")
                    print(f"Columns: {list(df_test.columns)}")
                    return file_path
                else:
                    print(f"❌ File doesn't appear to be ADMISSIONS.csv")
                    print(f"Expected: {expected_cols}")
                    print(f"Found: {list(df_test.columns)}")
                    print("Please try again with the correct ADMISSIONS.csv file.")
            except Exception as e:
                print(f"❌ Error reading file: {e}")
                print("Please try again.")
        else:
            print(f"❌ File not found: {file_path}")
            print("Please check the path and try again.")


def find_hadm_ids_local_file(admissions_file: str, parsed_df: pd.DataFrame) -> pd.DataFrame:
    """Find hadm_ids using local ADMISSIONS.csv file."""
    print(f"Loading ADMISSIONS.csv from {admissions_file} and matching hadm_ids...")

    try:
        # Load admissions data
        print("Reading ADMISSIONS.csv...")
        admissions_df = pd.read_csv(admissions_file)

        # Convert datetime columns
        print("Converting datetime columns...")
        admissions_df['ADMITTIME'] = pd.to_datetime(admissions_df['ADMITTIME'])
        admissions_df['DISCHTIME'] = pd.to_datetime(admissions_df['DISCHTIME'])

        print(f"✓ Loaded {len(admissions_df)} admissions")
        print(f"Date range: {admissions_df['ADMITTIME'].min()} to {admissions_df['ADMITTIME'].max()}")
        print(f"Unique patients: {admissions_df['SUBJECT_ID'].nunique()}")

        results = []

        print("\nMatching waveform records to admissions...")
        for i, row in parsed_df.iterrows():
            if i % 5 == 0:
                print(f"  Processing {i + 1}/{len(parsed_df)}: {row['patient_id']}")

            patient_id = row['patient_id']
            waveform_dt = pd.to_datetime(row['waveform_datetime'])

            # Convert patient_id format (p000020 -> 20)
            subject_id = int(patient_id[1:])

            # Find matching admissions for this patient
            patient_admissions = admissions_df[admissions_df['SUBJECT_ID'] == subject_id]

            if len(patient_admissions) == 0:
                print(f"    ⚠️  No admissions found for patient {patient_id} (subject_id {subject_id})")
                results.append({
                    'filename': row['filename'],
                    'patient_id': patient_id,
                    'subject_id': subject_id,
                    'hadm_id': None,
                    'waveform_datetime': waveform_dt,
                    'admittime': None,
                    'dischtime': None,
                    'match_found': False,
                    'reason': 'No admissions for this patient'
                })
                continue

            # Find admissions that contain the waveform datetime
            matching_admissions = patient_admissions[
                (patient_admissions['ADMITTIME'] <= waveform_dt) &
                (patient_admissions['DISCHTIME'] >= waveform_dt)
                ]

            if len(matching_admissions) > 0:
                # Take the best match (in case multiple, take first)
                best_match = matching_admissions.iloc[0]

                results.append({
                    'filename': row['filename'],
                    'patient_id': patient_id,
                    'subject_id': subject_id,
                    'hadm_id': best_match['HADM_ID'],
                    'waveform_datetime': waveform_dt,
                    'admittime': best_match['ADMITTIME'],
                    'dischtime': best_match['DISCHTIME'],
                    'match_found': True,
                    'reason': 'Exact time match - waveform within admission period'
                })
                print(f"    ✓ Exact match: hadm_id {best_match['HADM_ID']}")

            else:
                # No exact match - find closest admission
                patient_admissions['time_diff'] = abs(
                    (patient_admissions['ADMITTIME'] - waveform_dt).dt.total_seconds())
                closest_admission = patient_admissions.loc[patient_admissions['time_diff'].idxmin()]

                time_diff_hours = closest_admission['time_diff'] / 3600

                if time_diff_hours <= 48:  # Within 48 hours - reasonable match
                    results.append({
                        'filename': row['filename'],
                        'patient_id': patient_id,
                        'subject_id': subject_id,
                        'hadm_id': closest_admission['HADM_ID'],
                        'waveform_datetime': waveform_dt,
                        'admittime': closest_admission['ADMITTIME'],
                        'dischtime': closest_admission['DISCHTIME'],
                        'match_found': True,
                        'reason': f'Closest match - {time_diff_hours:.1f} hours from admission'
                    })
                    print(
                        f"    ~ Close match: hadm_id {closest_admission['HADM_ID']} ({time_diff_hours:.1f}h difference)")
                else:
                    results.append({
                        'filename': row['filename'],
                        'patient_id': patient_id,
                        'subject_id': subject_id,
                        'hadm_id': None,
                        'waveform_datetime': waveform_dt,
                        'admittime': None,
                        'dischtime': None,
                        'match_found': False,
                        'reason': f'No admission within 48h (closest: {time_diff_hours:.1f}h away)'
                    })
                    print(f"    ✗ No close match (closest admission: {time_diff_hours:.1f}h away)")

        return pd.DataFrame(results)

    except Exception as e:
        print(f"❌ Error processing local ADMISSIONS.csv: {e}")
        return pd.DataFrame()


def setup_physionet_connection():
    """Setup PhysioNet remote connection to MIMIC-III Clinical Database."""
    print("Setting up PhysioNet access to MIMIC-III Clinical Database...")

    try:
        # Get PhysioNet credentials
        print("\nPhysioNet Authentication Required")
        print("Enter your PhysioNet credentials:")
        username = input("PhysioNet Username: ").strip()

        import getpass
        password = getpass.getpass("PhysioNet Password: ")

        # Try different authentication approaches
        print("Testing PhysioNet access...")

        # Method 1: Try the main MIMIC-III page first
        print("Method 1: Testing main MIMIC-III access...")

        # Set up authentication with more robust handling
        password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()

        # Add credentials for multiple PhysioNet URLs
        physionet_urls = [
            "https://physionet.org/",
            "https://physionet.org/files/",
            "https://physionet.org/content/",
            "https://physionet.org/files/mimiciii/",
            "https://physionet.org/files/mimiciii/1.4/"
        ]

        for url in physionet_urls:
            password_mgr.add_password(None, url, username, password)

        # Create authentication handler
        auth_handler = urllib.request.HTTPBasicAuthHandler(password_mgr)

        # Also try digest authentication
        digest_handler = urllib.request.HTTPDigestAuthHandler(password_mgr)

        # Create opener with both authentication methods
        opener = urllib.request.build_opener(auth_handler, digest_handler)

        # Add headers that PhysioNet might expect
        opener.addheaders = [
            ('User-Agent', 'Mozilla/5.0 (Python MIMIC-III Script)'),
            ('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'),
            ('Accept-Language', 'en-US,en;q=0.5'),
            ('Accept-Encoding', 'gzip, deflate'),
            ('Connection', 'keep-alive'),
        ]

        urllib.request.install_opener(opener)

        # Test different URLs to find the right one
        test_urls = [
            "https://physionet.org/files/mimiciii/1.4/ADMISSIONS.csv",
            "https://physionet.org/content/mimiciii/1.4/files/ADMISSIONS.csv",
            "https://physionet.org/static/published-projects/mimiciii/mimic-iii-clinical-database-1.4/ADMISSIONS.csv"
        ]

        successful_url = None

        for test_url in test_urls:
            try:
                print(f"  Trying: {test_url}")
                response = urllib.request.urlopen(test_url)

                # Check if we got HTML (login page) vs CSV data
                content_type = response.headers.get('content-type', '')
                first_bytes = response.read(100)
                response.close()

                # Reset for actual download
                response = urllib.request.urlopen(test_url)

                if 'text/html' in content_type or b'<html' in first_bytes.lower():
                    print(f"    Got HTML (likely login page) - trying next URL")
                    continue
                elif 'csv' in content_type or first_bytes.startswith(b'ROW_ID') or b',' in first_bytes:
                    print(f"    ✓ Success! Got CSV data")
                    successful_url = test_url
                    break
                else:
                    print(f"    Got unknown content type: {content_type}")

            except urllib.error.HTTPError as e:
                print(f"    HTTP {e.code}: {e.reason}")
                if e.code == 401:
                    continue  # Try next URL
                elif e.code == 403:
                    print(f"    Access denied - you may not have permission for this specific URL")
                    continue
            except urllib.error.URLError as e:
                print(f"    URL Error: {e.reason}")
                continue
            except Exception as e:
                print(f"    Error: {e}")
                continue

        if not successful_url:
            print("\n❌ Could not access ADMISSIONS.csv with any method")
            print("\n🔍 Debugging suggestions:")
            print("1. Verify you can access https://physionet.org/content/mimiciii/1.4/ in your browser")
            print("2. Make sure you see 'Access the files' button (not 'Request Access')")
            print("3. Try clicking that button and accessing ADMISSIONS.csv manually")
            print("4. Check if you need to accept any additional agreements")
            return None

        # If we got here, we have a working URL
        print(f"✓ Successfully accessed MIMIC-III Clinical Database via {successful_url}")

        # Read header to confirm
        response = urllib.request.urlopen(successful_url)
        header_line = response.readline().decode('utf-8').strip()
        print(f"ADMISSIONS.csv header: {header_line}")
        response.close()

        # Download to local cache for faster processing
        cache_dir = "mimic_cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        admissions_cache = os.path.join(cache_dir, "ADMISSIONS.csv")

        if not os.path.exists(admissions_cache):
            print("Downloading ADMISSIONS.csv for local processing...")
            print("(This may take a few minutes)")

            response = urllib.request.urlopen(successful_url)

            # Get file size if available
            content_length = response.headers.get('content-length')
            if content_length:
                file_size_mb = int(content_length) / (1024 * 1024)
                print(f"File size: {file_size_mb:.1f} MB")

            with open(admissions_cache, 'wb') as f:
                total_size = 0
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    total_size += len(chunk)

                    # Show progress every MB
                    if total_size % (1024 * 1024) < 8192:  # First chunk of each MB
                        mb_downloaded = total_size / (1024 * 1024)
                        print(f"  Downloaded {mb_downloaded:.1f} MB...")

            total_mb = total_size / (1024 * 1024)
            print(f"✓ Downloaded ADMISSIONS.csv ({total_mb:.1f} MB) to {admissions_cache}")
        else:
            print(f"✓ Using cached ADMISSIONS.csv from {admissions_cache}")

        return admissions_cache

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("\n🔍 Please try:")
        print("1. Verify your PhysioNet credentials are correct")
        print("2. Check that you have active access to MIMIC-III Clinical Database")
        print("3. Try accessing the data manually in your browser first")
        return None


def find_hadm_ids_physionet(admissions_file: str, parsed_df: pd.DataFrame) -> pd.DataFrame:
    """Find hadm_ids using PhysioNet-downloaded ADMISSIONS.csv."""
    print("Loading ADMISSIONS.csv from PhysioNet cache and matching hadm_ids...")

    try:
        # Load admissions data
        print("Reading ADMISSIONS.csv...")
        admissions_df = pd.read_csv(admissions_file)

        # Convert datetime columns
        print("Converting datetime columns...")
        admissions_df['ADMITTIME'] = pd.to_datetime(admissions_df['ADMITTIME'])
        admissions_df['DISCHTIME'] = pd.to_datetime(admissions_df['DISCHTIME'])

        print(f"Loaded {len(admissions_df)} admissions")
        print(f"Date range: {admissions_df['ADMITTIME'].min()} to {admissions_df['ADMITTIME'].max()}")

        results = []

        print("Matching waveform records to admissions...")
        for i, row in parsed_df.iterrows():
            if i % 5 == 0:
                print(f"  Processing {i + 1}/{len(parsed_df)}: {row['patient_id']}")

            patient_id = row['patient_id']
            waveform_dt = pd.to_datetime(row['waveform_datetime'])

            # Convert patient_id format (p000020 -> 20)
            subject_id = int(patient_id[1:])

            # Find matching admissions for this patient
            patient_admissions = admissions_df[admissions_df['SUBJECT_ID'] == subject_id]

            if len(patient_admissions) == 0:
                print(f"    No admissions found for patient {patient_id} (subject_id {subject_id})")
                results.append({
                    'filename': row['filename'],
                    'patient_id': patient_id,
                    'subject_id': subject_id,
                    'hadm_id': None,
                    'waveform_datetime': waveform_dt,
                    'admittime': None,
                    'dischtime': None,
                    'match_found': False,
                    'reason': 'No admissions for patient'
                })
                continue

            # Find admissions that contain the waveform datetime
            matching_admissions = patient_admissions[
                (patient_admissions['ADMITTIME'] <= waveform_dt) &
                (patient_admissions['DISCHTIME'] >= waveform_dt)
                ]

            if len(matching_admissions) > 0:
                # Take the best match (closest admit time)
                best_match = matching_admissions.iloc[0]

                results.append({
                    'filename': row['filename'],
                    'patient_id': patient_id,
                    'subject_id': subject_id,
                    'hadm_id': best_match['HADM_ID'],
                    'waveform_datetime': waveform_dt,
                    'admittime': best_match['ADMITTIME'],
                    'dischtime': best_match['DISCHTIME'],
                    'match_found': True,
                    'reason': 'Exact time match'
                })
                print(f"    ✓ Matched to hadm_id {best_match['HADM_ID']}")
            else:
                # No exact match - find closest admission
                patient_admissions['time_diff'] = abs(
                    (patient_admissions['ADMITTIME'] - waveform_dt).dt.total_seconds())
                closest_admission = patient_admissions.loc[patient_admissions['time_diff'].idxmin()]

                time_diff_hours = closest_admission['time_diff'] / 3600

                if time_diff_hours <= 24:  # Within 24 hours
                    results.append({
                        'filename': row['filename'],
                        'patient_id': patient_id,
                        'subject_id': subject_id,
                        'hadm_id': closest_admission['HADM_ID'],
                        'waveform_datetime': waveform_dt,
                        'admittime': closest_admission['ADMITTIME'],
                        'dischtime': closest_admission['DISCHTIME'],
                        'match_found': True,
                        'reason': f'Closest match ({time_diff_hours:.1f}h difference)'
                    })
                    print(f"    ~ Closest match: hadm_id {closest_admission['HADM_ID']} ({time_diff_hours:.1f}h diff)")
                else:
                    results.append({
                        'filename': row['filename'],
                        'patient_id': patient_id,
                        'subject_id': subject_id,
                        'hadm_id': None,
                        'waveform_datetime': waveform_dt,
                        'admittime': None,
                        'dischtime': None,
                        'match_found': False,
                        'reason': f'No admission within 24h (closest: {time_diff_hours:.1f}h)'
                    })
                    print(f"    ✗ No close match (closest: {time_diff_hours:.1f}h away)")

        return pd.DataFrame(results)

    except Exception as e:
        print(f"Error processing PhysioNet data: {e}")
        return pd.DataFrame()


def setup_postgresql_connection():
    """Setup PostgreSQL connection to local MIMIC-III database."""
    try:
        import psycopg2
        import sqlalchemy

        # Get connection details
        host = input("PostgreSQL host (default: localhost): ").strip() or "localhost"
        port = input("PostgreSQL port (default: 5432): ").strip() or "5432"
        database = input("Database name (default: mimic): ").strip() or "mimic"
        username = input("Username: ").strip()
        password = input("Password: ").strip()

        # Create connection
        connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
        engine = sqlalchemy.create_engine(connection_string)

        # Test connection
        with engine.connect() as conn:
            result = conn.execute("SELECT COUNT(*) FROM admissions")
            count = result.fetchone()[0]
            print(f"✓ Connected to PostgreSQL. Found {count} admissions.")

        return engine

    except ImportError:
        print("Error: psycopg2 and sqlalchemy required for PostgreSQL")
        print("Install with: pip install psycopg2-binary sqlalchemy")
        return None
    except Exception as e:
        print(f"PostgreSQL connection error: {e}")
        return None


def setup_bigquery_connection():
    """Setup Google BigQuery connection."""
    try:
        from google.cloud import bigquery

        project_id = input("Google Cloud Project ID: ").strip()

        # Initialize client
        client = bigquery.Client(project=project_id)

        # Test connection
        query = "SELECT COUNT(*) as count FROM `physionet-data.mimiciii_clinical.admissions`"
        result = client.query(query).result()
        count = list(result)[0].count
        print(f"✓ Connected to BigQuery. Found {count} admissions.")

        return client

    except ImportError:
        print("Error: google-cloud-bigquery required")
        print("Install with: pip install google-cloud-bigquery")
        return None
    except Exception as e:
        print(f"BigQuery connection error: {e}")
        return None


def setup_csv_connection():
    """Setup connection to local CSV files."""
    csv_path = input("Path to MIMIC-III CSV files directory: ").strip()

    try:
        admissions_file = f"{csv_path}/ADMISSIONS.csv"
        df = pd.read_csv(admissions_file, nrows=1)  # Test read
        print(f"✓ Found ADMISSIONS.csv with columns: {list(df.columns)}")
        return csv_path
    except Exception as e:
        print(f"CSV connection error: {e}")
        return None


def find_hadm_ids_postgresql(engine, parsed_df: pd.DataFrame) -> pd.DataFrame:
    """Find hadm_ids using PostgreSQL connection."""
    print("Querying PostgreSQL for hadm_ids...")

    results = []

    for _, row in parsed_df.iterrows():
        patient_id = row['patient_id']
        waveform_dt = row['waveform_datetime']

        # Convert patient_id format (p000020 -> 20)
        subject_id = int(patient_id[1:])  # Remove 'p' and convert to int

        # Query for admissions within reasonable time window
        query = """
        SELECT hadm_id, subject_id, admittime, dischtime
        FROM admissions 
        WHERE subject_id = %(subject_id)s
        AND admittime <= %(waveform_time)s
        AND dischtime >= %(waveform_time)s
        ORDER BY ABS(EXTRACT(EPOCH FROM (admittime - %(waveform_time)s)))
        LIMIT 1
        """

        try:
            with engine.connect() as conn:
                result = conn.execute(query, {
                    'subject_id': subject_id,
                    'waveform_time': waveform_dt
                })
                admission = result.fetchone()

                if admission:
                    results.append({
                        'filename': row['filename'],
                        'patient_id': patient_id,
                        'subject_id': subject_id,
                        'hadm_id': admission.hadm_id,
                        'waveform_datetime': waveform_dt,
                        'admittime': admission.admittime,
                        'dischtime': admission.dischtime,
                        'match_found': True
                    })
                else:
                    results.append({
                        'filename': row['filename'],
                        'patient_id': patient_id,
                        'subject_id': subject_id,
                        'hadm_id': None,
                        'waveform_datetime': waveform_dt,
                        'admittime': None,
                        'dischtime': None,
                        'match_found': False
                    })

        except Exception as e:
            print(f"Error querying for {patient_id}: {e}")
            results.append({
                'filename': row['filename'],
                'patient_id': patient_id,
                'subject_id': subject_id,
                'hadm_id': None,
                'waveform_datetime': waveform_dt,
                'admittime': None,
                'dischtime': None,
                'match_found': False,
                'error': str(e)
            })

    return pd.DataFrame(results)


def find_hadm_ids_bigquery(client, parsed_df: pd.DataFrame) -> pd.DataFrame:
    """Find hadm_ids using BigQuery."""
    print("Querying BigQuery for hadm_ids...")

    results = []

    for _, row in parsed_df.iterrows():
        patient_id = row['patient_id']
        waveform_dt = row['waveform_datetime']

        # Convert patient_id format (p000020 -> 20)
        subject_id = int(patient_id[1:])

        query = f"""
        SELECT hadm_id, subject_id, admittime, dischtime
        FROM `physionet-data.mimiciii_clinical.admissions`
        WHERE subject_id = {subject_id}
        AND admittime <= '{waveform_dt}'
        AND dischtime >= '{waveform_dt}'
        ORDER BY ABS(TIMESTAMP_DIFF(admittime, '{waveform_dt}', SECOND))
        LIMIT 1
        """

        try:
            result = client.query(query).result()
            admissions = list(result)

            if admissions:
                admission = admissions[0]
                results.append({
                    'filename': row['filename'],
                    'patient_id': patient_id,
                    'subject_id': subject_id,
                    'hadm_id': admission.hadm_id,
                    'waveform_datetime': waveform_dt,
                    'admittime': admission.admittime,
                    'dischtime': admission.dischtime,
                    'match_found': True
                })
            else:
                results.append({
                    'filename': row['filename'],
                    'patient_id': patient_id,
                    'subject_id': subject_id,
                    'hadm_id': None,
                    'waveform_datetime': waveform_dt,
                    'admittime': None,
                    'dischtime': None,
                    'match_found': False
                })

        except Exception as e:
            print(f"Error querying for {patient_id}: {e}")
            results.append({
                'filename': row['filename'],
                'patient_id': patient_id,
                'subject_id': subject_id,
                'hadm_id': None,
                'waveform_datetime': waveform_dt,
                'admittime': None,
                'dischtime': None,
                'match_found': False,
                'error': str(e)
            })

    return pd.DataFrame(results)


def find_hadm_ids_csv(csv_path: str, parsed_df: pd.DataFrame) -> pd.DataFrame:
    """Find hadm_ids using local CSV files."""
    print("Loading ADMISSIONS.csv and matching hadm_ids...")

    try:
        # Load admissions data
        admissions_file = f"{csv_path}/ADMISSIONS.csv"
        admissions_df = pd.read_csv(admissions_file)

        # Convert datetime columns
        admissions_df['ADMITTIME'] = pd.to_datetime(admissions_df['ADMITTIME'])
        admissions_df['DISCHTIME'] = pd.to_datetime(admissions_df['DISCHTIME'])

        print(f"Loaded {len(admissions_df)} admissions")

        results = []

        for _, row in parsed_df.iterrows():
            patient_id = row['patient_id']
            waveform_dt = pd.to_datetime(row['waveform_datetime'])

            # Convert patient_id format (p000020 -> 20)
            subject_id = int(patient_id[1:])

            # Find matching admissions
            patient_admissions = admissions_df[
                (admissions_df['SUBJECT_ID'] == subject_id) &
                (admissions_df['ADMITTIME'] <= waveform_dt) &
                (admissions_df['DISCHTIME'] >= waveform_dt)
                ]

            if len(patient_admissions) > 0:
                # Take the closest admission by admit time
                best_match = patient_admissions.iloc[0]

                results.append({
                    'filename': row['filename'],
                    'patient_id': patient_id,
                    'subject_id': subject_id,
                    'hadm_id': best_match['HADM_ID'],
                    'waveform_datetime': waveform_dt,
                    'admittime': best_match['ADMITTIME'],
                    'dischtime': best_match['DISCHTIME'],
                    'match_found': True
                })
            else:
                results.append({
                    'filename': row['filename'],
                    'patient_id': patient_id,
                    'subject_id': subject_id,
                    'hadm_id': None,
                    'waveform_datetime': waveform_dt,
                    'admittime': None,
                    'dischtime': None,
                    'match_found': False
                })

        return pd.DataFrame(results)

    except Exception as e:
        print(f"Error processing CSV files: {e}")
        return pd.DataFrame()


def main():
    """Main function to match waveform records to hadm_ids."""

    print("MIMIC-III Waveform to HADM_ID Matcher")
    print("=" * 40)

    # Load results CSV
    parsed_df = load_results_csv()
    if parsed_df.empty:
        print("No data to process. Exiting.")
        return

    print(f"\nSample parsed data:")
    print(parsed_df.head())

    # Connect to clinical database
    print(f"\nConnecting to MIMIC-III Clinical Database...")
    connection = connect_to_mimic_clinical()

    if connection is None:
        print("Could not establish database connection. Exiting.")
        return

    # Find hadm_ids based on connection type
    if hasattr(connection, 'execute'):  # PostgreSQL
        results_df = find_hadm_ids_postgresql(connection, parsed_df)
    elif hasattr(connection, 'query'):  # BigQuery
        results_df = find_hadm_ids_bigquery(connection, parsed_df)
    elif isinstance(connection, str):  # File path
        # Check if it's a single ADMISSIONS.csv file or directory
        if connection.endswith('.csv') or 'ADMISSIONS.csv' in connection:
            # Single ADMISSIONS.csv file (local or PhysioNet cache)
            results_df = find_hadm_ids_local_file(connection, parsed_df)
        else:
            # CSV directory path
            results_df = find_hadm_ids_csv(connection, parsed_df)
    else:
        print("Unknown connection type")
        return

    # Save results
    output_file = "../../data/mimic_3_data/results_with_hadm_id.csv"
    results_df.to_csv(output_file, index=False)

    # Summary
    total_records = len(results_df)
    matched_records = len(results_df[results_df['match_found'] == True])

    print(f"\n{'=' * 50}")
    print(f"MATCHING COMPLETE")
    print(f"{'=' * 50}")
    print(f"Total records: {total_records}")
    print(f"Successfully matched: {matched_records}")
    print(f"No match found: {total_records - matched_records}")
    print(f"Results saved to: {output_file}")

    if matched_records > 0:
        print(f"\nFirst few matches:")
        matched_df = results_df[results_df['match_found'] == True]
        display_cols = ['filename', 'patient_id', 'hadm_id', 'admittime']
        if 'reason' in matched_df.columns:
            display_cols.append('reason')
        print(matched_df[display_cols].head())

    if matched_records < total_records:
        print(f"\nRecords without matches:")
        no_match = results_df[results_df['match_found'] == False]
        display_cols = ['filename', 'patient_id', 'waveform_datetime']
        if 'reason' in no_match.columns:
            display_cols.append('reason')
        print(no_match[display_cols].head())


if __name__ == "__main__":
    main()
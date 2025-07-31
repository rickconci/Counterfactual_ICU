import wfdb
import numpy as np
import pandas as pd
import csv
import urllib.request
from typing import List, Tuple
import re
import time


def get_files_in_directory(directory_path: str) -> List[str]:
    """
    Get list of files in a PhysioNet directory.

    Args:
        directory_path: PhysioNet directory path like "mimic3wdb-matched/1.0/p00/p000020"

    Returns:
        List of filenames in that directory
    """
    try:
        # Try to get directory listing from PhysioNet
        base_url = "https://physionet.org/files"
        url = f"{base_url}/{directory_path}/"

        import urllib.request
        response = urllib.request.urlopen(url)
        html_content = response.read().decode('utf-8')

        # Parse HTML to find .hea files (simple approach)
        import re
        hea_files = re.findall(r'href="([^"]*\.hea)"', html_content)

        # Filter for waveform records (should have date pattern)
        waveform_files = []
        for filename in hea_files:
            # Look for pattern like p000020-2183-04-28-17-47.hea
            if re.match(r'p\d+-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}\.hea', filename):
                # Remove .hea extension to get record name
                record_name = filename.replace('.hea', '')
                waveform_files.append(record_name)

        return waveform_files

    except Exception as e:
        print(f"Error listing directory {directory_path}: {e}")
        return []


def get_mimic3_matched_records(limit: int = None, debug: bool = False) -> List[Tuple[str, str]]:
    """
    Get list of MIMIC-III matched subset records from PhysioNet.
    Now with smart scanning - only scans directories until we have enough records.

    Args:
        limit: Limit number of records for testing (None for all)
        debug: Show debug information about record parsing

    Returns:
        List of tuples (record_name, pn_dir) for remote access
    """
    print("Getting MIMIC-III matched subset record list from PhysioNet...")

    try:
        # Get the RECORDS file from PhysioNet (contains directory paths)
        url = "https://physionet.org/files/mimic3wdb-matched/1.0/RECORDS"
        response = urllib.request.urlopen(url)
        records_content = response.read().decode('utf-8')

        lines = records_content.strip().split('\n')
        total_dirs = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        print(f"Total directory entries in RECORDS file: {total_dirs}")

        # Smart scanning: scan directories until we have enough records
        target_records = limit if limit else 1000  # Default to 1000 records if no limit
        records = []
        processed_dirs = 0
        failed_dirs = 0

        print(f"🔍 Scanning directories to find ~{target_records} waveform records...")
        print("(This finds actual record files in each patient directory)")

        scan_start_time = time.time()
        last_update_time = scan_start_time

        for line_num, line in enumerate(lines):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Line should be like: p00/p000020/
            if not line.endswith('/'):
                continue

            # Show progress every 10 directories or every 5 seconds
            current_time = time.time()
            if (processed_dirs % 10 == 0) or (current_time - last_update_time > 5):
                elapsed = current_time - scan_start_time
                rate = processed_dirs / (elapsed / 60) if elapsed > 0 else 0

                progress_pct = (processed_dirs / min(total_dirs, 500)) * 100  # Estimate based on 500 dirs max
                print(f"\r📁 Scanning: {processed_dirs} dirs | {len(records)} records found | "
                      f"{rate:.1f} dirs/min | {progress_pct:.1f}%", end='', flush=True)
                last_update_time = current_time

            # Remove trailing slash and build full path
            dir_path = line.rstrip('/')
            full_dir_path = f"mimic3wdb-matched/1.0/{dir_path}"

            if debug and processed_dirs < 3:
                print(f"\nChecking directory: {full_dir_path}")

            # Get files in this directory
            waveform_files = get_files_in_directory(full_dir_path)

            if len(waveform_files) == 0:
                failed_dirs += 1
            else:
                # Add each waveform file as a record
                for record_name in waveform_files:
                    records.append((record_name, full_dir_path))

                    # Stop early if we have enough records
                    if len(records) >= target_records:
                        break

            if debug and processed_dirs < 3:
                print(f"  Found {len(waveform_files)} waveform files: {waveform_files[:3]}...")

            processed_dirs += 1

            # Stop conditions
            if len(records) >= target_records:
                print(f"\n✅ Found {len(records)} records from {processed_dirs} directories")
                break

            # Safety limit - don't scan more than 500 directories unless specifically requested
            if not limit and processed_dirs >= 500:
                print(f"\n⏹️  Stopped after scanning {processed_dirs} directories (safety limit)")
                print(f"Found {len(records)} records - this should be plenty for analysis")
                break

        scan_time = time.time() - scan_start_time
        print(f"\n📊 Directory scan complete:")
        print(f"  ⏱️  Time: {scan_time / 60:.1f} minutes")
        print(f"  📁 Directories scanned: {processed_dirs}")
        print(f"  📁 Directories with records: {processed_dirs - failed_dirs}")
        print(f"  📄 Total records found: {len(records)}")

        if debug and len(records) > 0:
            print(f"\nFirst few records found:")
            for i, (rec_name, pn_dir) in enumerate(records[:5]):
                print(f"  {i + 1}: {rec_name} in {pn_dir}")

        return records

    except Exception as e:
        print(f"\nError in directory scanning: {e}")

        # If we have some records despite the error, use them
        if 'records' in locals() and len(records) > 0:
            print(f"However, we did find {len(records)} records before the error occurred.")
            print("Continuing with the records we found...")
            return records

        print("Using fallback sample records...")

        # Fallback sample records for testing
        return [
            ("p000020-2183-04-28-17-47", "mimic3wdb-matched/1.0/p00/p000020"),
            ("p000030-2172-10-16-12-22", "mimic3wdb-matched/1.0/p00/p000030"),
            ("p000033-2180-07-19-15-37", "mimic3wdb-matched/1.0/p00/p000033")
        ]


def get_record_segments(record_name: str, pn_dir: str) -> List[Tuple[str, str]]:
    """
    Get segments for a multi-segment record.

    Args:
        record_name: Record name (e.g., "p000020-2183-04-28-17-47")
        pn_dir: PhysioNet directory path

    Returns:
        List of tuples (segment_name, pn_dir)
    """
    try:
        # Read the master record header using PhysioNet directory
        header = wfdb.rdheader(record_name, pn_dir=pn_dir)

        # Check if it's a multi-segment record
        if hasattr(header, 'seg_name') and header.seg_name:
            # Multi-segment record
            segments = []
            for seg_name in header.seg_name:
                if seg_name and seg_name != '~':  # Skip gaps and None values
                    segments.append((seg_name, pn_dir))
            return segments
        else:
            # Single segment record or no segments found
            # Try to use the main record name as a segment
            return [(record_name, pn_dir)]

    except Exception as e:
        # Handle 404 and other errors gracefully
        if "404" in str(e) or "Not Found" in str(e):
            # Record doesn't exist - this is common in MIMIC
            return []
        else:
            print(f"Error getting segments for {record_name}: {e}")
            # Try fallback: use main record as single segment
            return [(record_name, pn_dir)]


def check_segment_has_abp(segment_name: str, pn_dir: str) -> bool:
    """
    Check if a segment has ABP Mean signal.

    Args:
        segment_name: Segment name
        pn_dir: PhysioNet directory path

    Returns:
        True if ABP signal found
    """
    try:
        # Read segment header only (faster than full record)
        header = wfdb.rdheader(segment_name, pn_dir=pn_dir)

        # Check if sig_name exists and is not None
        if not hasattr(header, 'sig_name') or header.sig_name is None:
            return False

        # ABP signal name variations
        abp_patterns = ['ABP', 'ABP_MEAN', 'ABP MEAN', 'ABPMEAN', 'ABP Mean', 'ABPMean']

        for sig_name in header.sig_name:
            if sig_name:  # Make sure sig_name is not None
                for pattern in abp_patterns:
                    if pattern.upper() in sig_name.upper():
                        return True

        return False

    except Exception as e:
        # If we can't read the header, assume no ABP
        return False


def analyze_abp_in_segment(segment_name: str, pn_dir: str, threshold: float = 70.0) -> Tuple[bool, float]:
    """
    Analyze ABP values in a segment remotely.

    Args:
        segment_name: Segment name
        pn_dir: PhysioNet directory path
        threshold: BP threshold in mmHg

    Returns:
        Tuple of (has_low_bp, min_bp_value)
    """
    try:
        # Read the segment data from PhysioNet
        record = wfdb.rdrecord(segment_name, pn_dir=pn_dir)

        if record.p_signal is None or len(record.p_signal) == 0:
            return False, np.nan

        # Check if sig_name exists and is not None
        if not hasattr(record, 'sig_name') or record.sig_name is None:
            return False, np.nan

        # Find ABP signal index
        abp_patterns = ['ABP', 'ABP_MEAN', 'ABP MEAN', 'ABPMEAN', 'ABP Mean', 'ABPMean']

        abp_idx = None
        for i, sig_name in enumerate(record.sig_name):
            if sig_name:  # Make sure sig_name is not None
                for pattern in abp_patterns:
                    if pattern.upper() in sig_name.upper():
                        abp_idx = i
                        break
                if abp_idx is not None:
                    break

        if abp_idx is None:
            return False, np.nan

        # Get ABP values
        abp_values = record.p_signal[:, abp_idx]

        # Filter valid values (remove only clearly impossible values)
        # Keep very low values (could be real critical events, procedures, or sensor issues)
        # Just filter out NaN and completely impossible values
        valid_mask = (~np.isnan(abp_values)) & (abp_values > 0) & (abp_values < 300)
        valid_abp = abp_values[valid_mask]

        if len(valid_abp) == 0:
            return False, np.nan

        min_abp = np.min(valid_abp)
        has_low_bp = min_abp < threshold

        return has_low_bp, min_abp

    except Exception as e:
        print(f"Error analyzing {segment_name}: {e}")
        return False, np.nan


def process_patient_record(record_name: str, pn_dir: str, threshold: float = 70.0) -> Tuple[bool, dict]:
    """
    Process a patient record to check for low ABP.

    Args:
        record_name: Record name
        pn_dir: PhysioNet directory path
        threshold: BP threshold in mmHg

    Returns:
        Tuple of (has_low_bp, analysis_info)
    """
    result = {
        'record': f"{pn_dir}/{record_name}",
        'segments_checked': 0,
        'segments_with_abp': 0,
        'min_abp_overall': np.inf,
        'low_bp_segments': []
    }

    try:
        # Get all segments for this record
        segments = get_record_segments(record_name, pn_dir)
        result['segments_checked'] = len(segments)

        has_low_bp = False

        for segment_name, seg_pn_dir in segments:
            # First check if segment has ABP (faster)
            if check_segment_has_abp(segment_name, seg_pn_dir):
                result['segments_with_abp'] += 1

                # Analyze ABP values in this segment
                segment_has_low_bp, min_bp = analyze_abp_in_segment(segment_name, seg_pn_dir, threshold)

                if not np.isnan(min_bp):
                    result['min_abp_overall'] = min(result['min_abp_overall'], min_bp)

                if segment_has_low_bp:
                    has_low_bp = True
                    result['low_bp_segments'].append({
                        'segment': segment_name,
                        'min_bp': min_bp
                    })

        # Handle case where no ABP found
        if result['min_abp_overall'] == np.inf:
            result['min_abp_overall'] = np.nan

        return has_low_bp, result

    except Exception as e:
        print(f"Error processing record {record_name}: {e}")
        return False, result


def extract_patient_id(record_name: str) -> str:
    """
    Extract patient ID from record name.

    Args:
        record_name: Record name like "p000020-2183-04-28-17-47"

    Returns:
        Patient ID like "p000020"
    """
    return record_name.split('-')[0]


def main_analysis(test_mode: bool = False, max_records: int = None, threshold: float = 70.0, max_patients: int = 30):
    """
    Main function to analyze MIMIC-III for low ABP using remote access.

    Args:
        test_mode: If True, run with limited records and verbose output
        max_records: Maximum number of records to process (None for all)
        threshold: BP threshold in mmHg
        max_patients: Stop after finding this many unique patients with low BP
    """
    print("MIMIC-III Remote ABP Analysis")
    print("=" * 40)
    print(f"Threshold: {threshold} mmHg")
    print(f"Test mode: {test_mode}")
    print(f"Target: {max_patients} patients with low ABP")

    # Test PhysioNet access
    print("\nTesting PhysioNet access...")
    try:
        test_record_name = "p000020-2183-04-28-17-47"
        test_pn_dir = "mimic3wdb-matched/1.0/p00/p000020"
        test_header = wfdb.rdheader(test_record_name, pn_dir=test_pn_dir)
        print(f"✓ PhysioNet access successful!")

        # Check signal names safely
        if hasattr(test_header, 'sig_name') and test_header.sig_name:
            print(f"  Test record has {len(test_header.sig_name)} signals: {test_header.sig_name[:3]}...")
        else:
            print(f"  Test record header loaded but no signal names found")

        # Check if it's a multi-segment record
        if hasattr(test_header, 'seg_name') and test_header.seg_name:
            print(f"  Multi-segment record with {len(test_header.seg_name)} segments")
        else:
            print(f"  Single segment record")

    except Exception as e:
        print(f"✗ PhysioNet access failed: {e}")
        print("Please check your PhysioNet credentials and internet connection.")
        return

    # Get record list
    limit = 20 if test_mode else max_records
    debug_records = test_mode
    records = get_mimic3_matched_records(limit, debug=debug_records)

    if not records:
        print("No records found!")
        return

    print(f"\nProcessing records to find {max_patients} patients with low ABP...")

    # Results tracking
    low_bp_records = []  # All records with low BP
    unique_patients_with_low_bp = set()  # Unique patient IDs with low BP
    all_results = []
    error_count = 0
    not_found_count = 0
    records_processed = 0

    for i, (record_name, pn_dir) in enumerate(records):
        try:
            # Check if we've found enough patients
            if len(unique_patients_with_low_bp) >= max_patients:
                print(f"\n🎯 TARGET REACHED! Found {max_patients} patients with ABP < {threshold} mmHg")
                print(f"Stopping after processing {records_processed} records")
                break

            if i % 25 == 0 or test_mode:
                print(
                    f"Progress: {records_processed + 1} records processed, {len(unique_patients_with_low_bp)} patients with low ABP found")
                if not test_mode:
                    print(f"  Current record: {record_name}")

            # Process this record
            has_low_bp, result_info = process_patient_record(record_name, pn_dir, threshold)
            records_processed += 1

            # Check if record was actually processed (not 404)
            if result_info['segments_checked'] == 0:
                not_found_count += 1
                if test_mode:
                    print(f"  ⚠️  Record not found (404)")
                continue

            all_results.append(result_info)

            if has_low_bp:
                # Extract patient ID
                patient_id = extract_patient_id(record_name)

                # Add record to results
                filename = record_name + ".hea"
                low_bp_records.append(filename)

                # Track unique patient
                is_new_patient = patient_id not in unique_patients_with_low_bp
                unique_patients_with_low_bp.add(patient_id)

                if test_mode or is_new_patient:
                    print(f"  ✓ LOW BP FOUND: {filename}")
                    print(f"    Patient: {patient_id} ({'NEW' if is_new_patient else 'additional record'})")
                    print(f"    Min ABP: {result_info['min_abp_overall']:.1f} mmHg")
                    print(f"    Segments with ABP: {result_info['segments_with_abp']}")
                    if is_new_patient:
                        print(f"    📊 Total unique patients found: {len(unique_patients_with_low_bp)}/{max_patients}")
            elif test_mode:
                min_bp_str = f"{result_info['min_abp_overall']:.1f}" if not np.isnan(
                    result_info['min_abp_overall']) else "N/A"
                print(f"  - No low BP: Min ABP = {min_bp_str} mmHg, ABP segments: {result_info['segments_with_abp']}")

        except Exception as e:
            error_count += 1
            if test_mode:
                print(f"  ✗ Error processing {record_name}: {e}")
            elif error_count % 20 == 0:
                print(f"  Note: {error_count} errors encountered so far")

    # Save results
    output_file = "../../data/processed_data/results.csv"
    print(f"\nSaving results to {output_file}...")

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'patient_id'])
        for filename in low_bp_records:
            # Extract patient ID from filename
            record_name = filename.replace('.hea', '')
            patient_id = extract_patient_id(record_name)
            writer.writerow([filename, patient_id])

    # Summary
    print(f"\n{'=' * 60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Target: {max_patients} patients with ABP < {threshold} mmHg")
    print(f"Records processed: {records_processed}")
    print(f"Records not found (404): {not_found_count}")
    print(f"Records successfully analyzed: {len(all_results)}")
    print(f"Other errors: {error_count}")
    print(f"📊 UNIQUE PATIENTS with low ABP: {len(unique_patients_with_low_bp)}")
    print(f"📄 TOTAL RECORDS with low ABP: {len(low_bp_records)}")
    print(f"Records with ABP data: {sum(1 for r in all_results if r['segments_with_abp'] > 0)}")
    print(f"Results saved to: {output_file}")

    if low_bp_records:
        print(f"\nFirst 10 files with low ABP:")
        for filename in low_bp_records[:10]:
            record_name = filename.replace('.hea', '')
            patient_id = extract_patient_id(record_name)
            print(f"  {filename} (Patient: {patient_id})")

    if len(unique_patients_with_low_bp) >= max_patients:
        print(f"\n🎯 SUCCESS: Found {max_patients} patients with ABP < {threshold} mmHg!")
        print(f"Patient IDs: {sorted(list(unique_patients_with_low_bp))[:10]}..." if len(
            unique_patients_with_low_bp) > 10 else f"Patient IDs: {sorted(list(unique_patients_with_low_bp))}")
    else:
        print(f"\n⚠️  Only found {len(unique_patients_with_low_bp)} patients with low ABP (target was {max_patients})")
        print("Consider running with more records or lowering the threshold.")

    if test_mode and all_results:
        print(f"\nDetailed statistics:")
        abp_records = [r for r in all_results if not np.isnan(r['min_abp_overall'])]
        if abp_records:
            min_bps = [r['min_abp_overall'] for r in abp_records]
            print(f"  Overall min ABP: {min(min_bps):.1f} mmHg")
            print(f"  Average min ABP: {np.mean(min_bps):.1f} mmHg")
            print(f"  Total segments analyzed: {sum(r['segments_with_abp'] for r in all_results)}")
        else:
            print("  No ABP data found in test records")


def debug_test_record(record_name: str, pn_dir: str):
    """
    Debug function to test a single record and show detailed information.
    """
    print(f"\n🔍 DEBUG: Testing record {record_name}")
    print(f"   Directory: {pn_dir}")

    try:
        # Test header reading
        header = wfdb.rdheader(record_name, pn_dir=pn_dir)
        print(f"   ✓ Header loaded successfully")

        # Check attributes
        print(f"   Has sig_name: {hasattr(header, 'sig_name')}")
        if hasattr(header, 'sig_name'):
            print(f"   sig_name is None: {header.sig_name is None}")
            if header.sig_name:
                print(f"   Number of signals: {len(header.sig_name)}")
                print(f"   Signal names: {header.sig_name[:5]}...")  # First 5

        print(f"   Has seg_name: {hasattr(header, 'seg_name')}")
        if hasattr(header, 'seg_name'):
            print(f"   seg_name is None: {header.seg_name is None}")
            if header.seg_name:
                print(f"   Number of segments: {len(header.seg_name)}")
                print(f"   Segment names: {header.seg_name[:5]}...")  # First 5

        # Test getting segments
        segments = get_record_segments(record_name, pn_dir)
        print(f"   ✓ Found {len(segments)} segments")

        # Test first segment if available
        if segments:
            seg_name, seg_pn_dir = segments[0]
            print(f"   Testing first segment: {seg_name}")

            # Try to read segment header
            try:
                seg_header = wfdb.rdheader(seg_name, pn_dir=seg_pn_dir)
                if hasattr(seg_header, 'sig_name') and seg_header.sig_name:
                    print(f"   ✓ Segment has {len(seg_header.sig_name)} signals: {seg_header.sig_name}")

                    # Check for ABP
                    has_abp = check_segment_has_abp(seg_name, seg_pn_dir)
                    print(f"   ABP found: {has_abp}")

                else:
                    print(f"   ✗ Segment has no signal names")
            except Exception as e:
                print(f"   ✗ Error reading segment: {e}")

    except Exception as e:
        print(f"   ✗ Error: {e}")


def quick_test():
    """Quick test with a few records and detailed debug info."""
    print("Running quick test with debug information...")

    # ALWAYS show debug for quick test
    print("Debugging RECORDS file parsing (now looking IN directories)...")
    records = get_mimic3_matched_records(limit=10, debug=True)

    if not records:
        print("Still no records found!")
        return

    if records:
        print(f"\n{'=' * 50}")
        print(f"SUCCESS! Found {len(records)} records. Testing first one...")

        # Test the first record
        record_name, pn_dir = records[0]
        debug_test_record(record_name, pn_dir)

        print(f"\n{'=' * 50}")
        print("Now running normal analysis...")

        # Use the records we found instead of re-parsing
        print("MIMIC-III Remote ABP Analysis")
        print("=" * 40)
        print(f"Threshold: 70.0 mmHg")
        print(f"Test mode: True")

        # Test PhysioNet access with our found record
        print("\nTesting PhysioNet access with found record...")
        try:
            test_record_name, test_pn_dir = records[0]
            test_header = wfdb.rdheader(test_record_name, pn_dir=test_pn_dir)
            print(f"✓ PhysioNet access successful with {test_record_name}!")

            # Check signal names safely
            if hasattr(test_header, 'sig_name') and test_header.sig_name:
                print(f"  Record has {len(test_header.sig_name)} signals: {test_header.sig_name[:3]}...")
            else:
                print(f"  Record header loaded but no signal names found")

            # Check if it's a multi-segment record
            if hasattr(test_header, 'seg_name') and test_header.seg_name:
                print(f"  Multi-segment record with {len(test_header.seg_name)} segments")
            else:
                print(f"  Single segment record")

        except Exception as e:
            print(f"✗ PhysioNet access failed: {e}")
            return

        # Now analyze a few records
        print(f"\nProcessing {min(len(records), 5)} records...")

        low_bp_records = []
        all_results = []

        for i, (record_name, pn_dir) in enumerate(records[:5]):
            try:
                print(f"Progress: {i + 1}/5 - {record_name}")

                # Process this record
                has_low_bp, result_info = process_patient_record(record_name, pn_dir, 70.0)

                # Check if record was actually processed (not 404)
                if result_info['segments_checked'] == 0:
                    print(f"  ⚠️  Record not found (404)")
                    continue

                all_results.append(result_info)

                if has_low_bp:
                    filename = record_name + ".hea"
                    low_bp_records.append(filename)

                    print(f"  ✓ LOW BP FOUND: {filename}")
                    print(f"    Min ABP: {result_info['min_abp_overall']:.1f} mmHg")
                    print(f"    Segments with ABP: {result_info['segments_with_abp']}")
                    print(f"    Low BP segments: {len(result_info['low_bp_segments'])}")
                else:
                    min_bp_str = f"{result_info['min_abp_overall']:.1f}" if not np.isnan(
                        result_info['min_abp_overall']) else "N/A"
                    print(
                        f"  - No low BP: Min ABP = {min_bp_str} mmHg, ABP segments: {result_info['segments_with_abp']}")

            except Exception as e:
                print(f"  ✗ Error processing {record_name}: {e}")

        # Save results
        output_file = "../../data/processed_data/results.csv"
        print(f"\nSaving results to {output_file}...")

        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename'])
            for filename in low_bp_records:
                writer.writerow([filename])

        # Summary
        print(f"\n{'=' * 50}")
        print(f"TEST ANALYSIS COMPLETE")
        print(f"{'=' * 50}")
        print(f"Records processed: {len(all_results)}")
        print(f"Records with ABP data: {sum(1 for r in all_results if r['segments_with_abp'] > 0)}")
        print(f"Records with ABP < 70.0 mmHg: {len(low_bp_records)}")
        print(f"Results saved to: {output_file}")

        if low_bp_records:
            print(f"\nFiles with low ABP:")
            for filename in low_bp_records:
                print(f"  {filename}")

        if all_results:
            print(f"\nDetailed statistics:")
            abp_records = [r for r in all_results if not np.isnan(r['min_abp_overall'])]
            if abp_records:
                min_bps = [r['min_abp_overall'] for r in abp_records]
                print(f"  Overall min ABP: {min(min_bps):.1f} mmHg")
                print(f"  Average min ABP: {np.mean(min_bps):.1f} mmHg")
                print(f"  Total segments analyzed: {sum(r['segments_with_abp'] for r in all_results)}")
            else:
                print("  No ABP data found in test records")
    else:
        print("No records found!")


def full_analysis():
    """Full analysis to find 30 patients with low ABP."""
    print("This will search through MIMIC-III records until 30 patients with ABP < 70 mmHg are found.")
    response = input("Continue? (y/n): ")
    if response.lower() == 'y':
        main_analysis(test_mode=False, max_records=None, threshold=70.0, max_patients=30)
    else:
        print("Analysis cancelled.")


if __name__ == "__main__":
    print("MIMIC-III Remote ABP Analysis Tool")
    print("No downloads required - uses PhysioNet remote access!")
    print("=" * 50)

    print("\nOptions:")
    print("1. Quick test (5 records)")
    print("2. Medium test (50 records)")
    print("3. Find 30 patients with low ABP (stops when target reached)")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        quick_test()
    elif choice == "2":
        print("Running medium test...")
        main_analysis(test_mode=True, max_records=50, threshold=70.0, max_patients=10)  # Find 10 patients in test
    elif choice == "3":
        full_analysis()
    else:
        print("Invalid choice. Running quick test...")
        quick_test()
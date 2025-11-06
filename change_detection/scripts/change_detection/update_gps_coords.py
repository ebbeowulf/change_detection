import pandas as pd
import numpy as np
import argparse

try:
    parser = argparse.ArgumentParser()
    parser.add_argument('gps_log',type=str,help='location of the gps_log.csv')
    parser.add_argument('output_log',type=str,help='location of the output gps_log.csv')
    args = parser.parse_args()

    # --- Load Data ---
    file_path = args.gps_log
    # Read the CSV, specifying no header and providing column names
    df = pd.read_csv(
        file_path, 
        header=None, 
        names=['sensor_id', 'timestamp', 'latitude', 'longitude']
    )
    
    # Store the original row order by creating an 'index' column
    df = df.reset_index()
    
    # --- Process Timestamps ---
    # Convert the timestamp column from text to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort the entire DataFrame by timestamp. This is essential for merge_asof
    df_sorted = df.sort_values(by='timestamp')
    
    # --- Create GPS Lookup Table ---
    # Filter for only the GPS rows, which have valid lat/long data
    # Drop any potential NaN rows from this lookup to keep it clean
    gps_lookup = df_sorted[df_sorted['sensor_id'] == 'GPS'][
        ['timestamp', 'latitude', 'longitude']
    ].dropna()
    
    # --- Merge Data (Fill Missing Values) ---
    # Use merge_asof to find the correct GPS data for *every* row.
    # We merge the full sorted DataFrame (left) with our GPS lookup (right).
    df_filled = pd.merge_asof(
        # Left DataFrame: All rows, but we drop the original lat/long
        # columns, as they will be replaced by the merged ones.
        df_sorted.drop(columns=['latitude', 'longitude']), 
        
        # Right DataFrame: Our clean GPS lookup table
        gps_lookup,                                        
        
        # Key to merge on
        on='timestamp',                                    
        
        # Find the last GPS reading on or *before* the current row's timestamp
        direction='backward'                               
    )
    
    # --- Clean Up and Save ---
    # Restore the original file order using the saved 'index'
    df_final = df_filled.sort_values(by='index')
    
    # Drop the temporary 'index' column
    df_final = df_final.drop(columns=['index'])

    # Reorder columns to match the original [sensor_id, timestamp, latitude, longitude]
    df_final = df_final[['sensor_id', 'timestamp', 'latitude', 'longitude']]
    
    # Save the updated DataFrame to a new CSV
    output_file_path = args.output_log
    
    # Save without the pandas index or header
    # Format latitude/longitude to 6 decimal places
    # Format timestamp to match the input
    df_final.to_csv(
        output_file_path, 
        index=False, 
        header=False, 
        float_format='%.6f',
        date_format='%Y-%m-%d %H:%M:%S'
    )
    
    print(f"Successfully processed data and saved to '{output_file_path}'.")
    
    # Display the first 10 rows of the new file to show the result
    print("\nHead of updated DataFrame:")
    print(df_final.head(10))
    
    # Check for any remaining NaN values
    nan_count = df_final['latitude'].isna().sum()
    print(f"\nRemaining rows with NaN GPS data: {nan_count}")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

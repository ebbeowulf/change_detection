import sqlite3

def list_colmap_images(database_path):
    # Connect to the COLMAP database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Query the images table
    cursor.execute("SELECT image_id, name FROM images")
    images = cursor.fetchall()

    # Print the results
    print("Image ID\tImage Name")
    for image_id, name in images:
        print(f"{image_id}\t{name}")

    # Clean up
    conn.close()

# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="List images in COLMAP database")
    parser.add_argument('database_path', type=str, help='Path to the COLMAP database    .db file')
    args = parser.parse_args()
    list_colmap_images(args.database_path)
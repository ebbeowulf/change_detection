import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_csv', type=str, help='Input CSV file with image names and flattened 4x4 pose matrices')
parser.add_argument('output_csv', type=str, help='Output CSV file with image names and x, y, z positions')
args = parser.parse_args()

with open(args.input_csv, 'r', newline='') as infile, open(args.output_csv, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile, delimiter=' ')

    # Write header
    writer.writerow(['image_name', 'x', 'y', 'z'])

    for row in reader:
        image_name = row[0]
        # Extract translation components from the flattened 4x4 matrix
        x = float(row[4])   # first column is the name of the image... need to offset by 1
        y = float(row[8])
        z = float(row[12])
        writer.writerow([image_name, x, y, z])


#!/bin/bash

# Usage: ./inject_gps.sh /path/to/images coords.csv

IMG_DIR="$1"
COORDS_FILE="$2"

if [[ ! -d "$IMG_DIR" || ! -f "$COORDS_FILE" ]]; then
  echo "Usage: $0 /path/to/images coords.csv"
  exit 1
fi

# Step 1: Convert PNGs to JPGs
#echo "Converting PNGs to JPGs..."
cd "$IMG_DIR"
for img in *.png; do
  base="${img%.png}"
  jpg_out="$base.jpg"
  if [[ -f ${jpg_out} ]];then
	  echo "file exists - $jpg_out"
  else
	  echo "creating file - $jpg_out" 
	  convert "$img" "$base.jpg"
  fi
done

# Step 2: Prepare EXIF injection
echo "Injecting GPS EXIF data..."
while IFS=',' read -r filename lat lon alt; do
  # Ensure filename matches converted JPG
  jpg_file="${filename%.png}.jpg"
  echo $IFS
  if [[ -f "$jpg_file" ]]; then
    exiftool \
      -GPSLatitude="$lat" \
      -GPSLatitudeRef=$(awk "BEGIN {print ($lat >= 0) ? \"N\" : \"S\"}") \
      -GPSLongitude="$lon" \
      -GPSLongitudeRef=$(awk "BEGIN {print ($lon >= 0) ? \"E\" : \"W\"}") \
      -GPSAltitude="$alt" \
      -GPSAltitudeRef=0 \
      -overwrite_original "$jpg_file"
    echo "Tagged $jpg_file"
    #echo "Tagged $cmd"
  else
    echo "Warning: $jpg_file not found"
  fi
done < "$COORDS_FILE"

echo "Done."

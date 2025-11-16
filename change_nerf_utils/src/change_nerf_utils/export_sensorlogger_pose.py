import argparse
import os
import numpy as np
import pdb
from geopy.distance import geodesic
from pyproj import Proj, transform

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir',type=str,help='location of image directory')
    parser.add_argument('location_csv',type=str,help='location of location csv file')
    parser.add_argument('-image_type',type=str,default='jpg',help='type of image (default = png)')
    args = parser.parse_args()

    # Get the list of images
    files = [int(f.split(".")[0]) for f in os.listdir(args.image_dir) if f.endswith(args.image_type)]
    tstamp_length=len(str(files[0]))

    # Load the location csv
    with open(args.location_csv, 'r') as file:
        lines = file.readlines()
    
    fields=lines[0].split(",")
    time_idx=fields.index('time')
    alt_idx=fields.index('altitude')
    lat_idx=fields.index('latitude')
    long_idx=fields.index('longitude')

    loc_str=[ f.split(",") for f in lines[1:]]
    loc_arr=np.array([[int(ll[time_idx][:tstamp_length]),float(ll[lat_idx]),float(ll[long_idx]),float(ll[alt_idx])] for ll in loc_str])

    new_lat=np.interp(files,loc_arr[:,0],loc_arr[:,1])
    new_long=np.interp(files,loc_arr[:,0],loc_arr[:,2])
    new_alt=np.interp(files,loc_arr[:,0],loc_arr[:,3])

    # Define the WGS84 (lat/lon) and UTM projection
    wgs84 = Proj(proj="latlong", datum="WGS84")
    utm = Proj(proj="utm", zone=17, datum="WGS84")  # Replace '33' with your UTM zone
    pdb.set_trace()

    for idx in range(len(files)):

        print(f"{files[idx]}.{args.image_type} {new_lat[idx]} {new_long[idx]} {new_alt[idx]}")


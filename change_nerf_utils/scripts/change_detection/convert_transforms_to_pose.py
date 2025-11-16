import argparse
import os
import numpy as np
import pdb
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('transforms_file',type=str,help='location of image directory')
    args = parser.parse_args()

    # Load the location csv
    with open(args.transforms_file, 'r') as fin:
        A=json.load(fin)

    import pdb
    all_frames=np.zeros((0,3),dtype=float)
    for frame in A['frames']:
        fname=frame['file_path'].split('/')[-1]
        tmatrix=np.array(frame['transform_matrix'])
        print(f"{fname} {tmatrix[0][3]} {tmatrix[1][3]} {tmatrix[2][3]}")
        all_frames=np.vstack((all_frames,tmatrix[:3,3]))
    
    pdb.set_trace()

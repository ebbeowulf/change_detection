import argparse
import json
import os
import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file1',type=str,help='location of annotation file 1')
    parser.add_argument('file2',type=str,help='location of annotation file 1')
    parser.add_argument('outFile',type=str,help='location of output file')
    args = parser.parse_args()

    if os.path.exists(args.file1):
        with open(args.file1,'r') as handle:
            annotationsA=json.load(handle)
    else:
        raise Exception(f"Error opening {args.file1}")        

    if os.path.exists(args.file2):
        with open(args.file2,'r') as handle:
            annotationsB=json.load(handle)
    else:
        raise Exception(f"File {args.file2} does not exist")

    annotations_out=annotationsA
    for key in annotationsB:
        if len(annotationsB[key])>0:
            if key not in annotations_out or len(annotations_out[key])==0:
                annotations_out[key]=annotationsB[key]
            else:
                pdb.set_trace()

    print(annotations_out)

    with open(args.outFile,'w') as fout:
        json.dump(annotations_out, fout)

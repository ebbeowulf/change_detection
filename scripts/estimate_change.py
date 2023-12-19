from image_set import read_image_csv
import argparse
import matplotlib.pyplot as plt
import pdb
import numpy as np
import tf
from image_set import get_neighboring_images

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_initial',type=str,help='location of initial pose csv file to process')
    parser.add_argument('clip_initial',type=str,help='initial clip csv file to process')
    parser.add_argument('images_change',type=str,help='location of changed pose csv file to process')
    parser.add_argument('clip_change',type=str,help='changed clip csv file to process')
    args = parser.parse_args()

    
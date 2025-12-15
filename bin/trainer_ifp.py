import argparse
import sys
import numpy as np

import os
import logging
from interocsvm.trainer_gui import get_vector_gui, get_kernelvector_gui

from tqdm import tqdm

import pickle
import pandas as pd

from pyichem import ifp
        

def main(args):
    ifps = list()
    ifp_format = None
    columns = None

    logging.info('Reading IFPs')

    for i,file in enumerate(args.file):
        with open(file, 'rb') as f:
            ifp_obj = pickle.load(f)
        if ifp_format:
            if ifp_format != ifp_obj.ifp_format:
                raise ValueError('The two files are based on different IFP formats, impossible to train a model')
        else:
            ifp_format = ifp_obj.ifp_format

        if not hasattr(ifp_obj, 'fingerprints'):
            ifp_tmp = ifp_obj.read_ifp()
        else:
            if args.simplify:
                ifp_obj.simplify_hb(in_place = True)
            ifp_tmp = ifp_obj.fingerprints
        ifp_tmp = ifp.filter_ifp(ifp_tmp, args.residues, args.interactions)
        ifp_tmp = ifp.remove_empty_ints(ifp_tmp)
        if columns is not None:
            if (columns != ifp_tmp.columns).all():
                raise ValueError('The two files present different interactions or residues, impossible to train a model')
        else:
            columns = ifp_tmp.columns
        ifps.append(ifp_tmp)

    ifps = pd.concat(ifps)

    ifp_info = {'Format' : ifp_format,
                'Columns': columns,
                'Interactions' : args.interactions,
                'Residues': args.residues}
           

    logging.info('IFPs read')

    if args.kernel:
        root = get_kernelvector_gui(ifps.sparse.to_dense(), args.name, ifp_info)
    else:
        root = get_vector_gui(ifps, args.name, ifp_info)
        
    root.mainloop()

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-f', '--file', nargs = '+', help='Ints object containing the interactions', required = True)
    parser.add_argument('-n', '--name', help='Name of the models', required = True)
    parser.add_argument('-i', '--interactions', nargs = '+', default = [], help='Available interaction types: HBD, HBA, ETF, FTF, HYD, CAT, ANI, HBI')
    parser.add_argument('-r', '--residues', default = '', help = 'Select specific residues for interaction determination')
    parser.add_argument('-s', '--simplify', default = False, help = 'Unify hydrogen bonds', action = 'store_true')
    parser.add_argument('-k', '--kernel', default = False, help = 'Use kernel to estimate similarity between inputs', action = 'store_true')
    

    parser.set_defaults(func=main)
    args=parser.parse_args()
    status = args.func(args)
    sys.exit(status)


    
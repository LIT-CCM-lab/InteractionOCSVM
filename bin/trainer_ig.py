import argparse
import sys
import numpy as np
import os
import logging
from interocsvm.trainer_gui import get_graphkernel_gui
import pickle
from pyichem.ints import Ints
        

def main(args):
    graphs = list()
    ref_ipa_info = {}
    ref_ig_info = {'threshold': args.thresh,
                   'subgraph': args.subgraph,
                   'round_val': 1,
                   'simplify': args.simplify,
                   'mode': args.mode}

    logging.info('Generating interaction graphs from IPAs')

    for i,file in enumerate(args.file):
        with open(file, 'rb') as f:
            ints = pickle.load(f)
            if i == 0:
                ref_ipa_info = ints.ipa_info
            else:
                if ref_ipa_info != ints.ipa_info and args.subgraph != 'ELEC':
                    logging.warning(f'Skippining file {file} since it uses different IPAs from the first file')
                    continue
            graphs.append(ints.compute_graphs(subgraph = args.subgraph,
                                              simplify = args.simplify,
                                              mode = args.mode,
                                              threshold = args.thresh))

    g = np.concatenate(graphs)
    ng = np.array([gt.n for gt in g])
    mask = ng >2
    g = g[mask]

    logging.info('Interaction graphs generated')

    root = get_graphkernel_gui(g, args.name, ref_ig_info, ref_ipa_info)
    root.mainloop()

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-f', '--file', nargs = '+', help='Ints object containing the interactions', required = True)
    parser.add_argument('-n', '--name', help='Name of the models', required = True)
    parser.add_argument('-t', '--type', default = 'MERG', help='Type of interactions used in interaction detection.\navailble types: MERG, CENT, LIG, PROT')
    parser.add_argument('-sg', '--subgraph', default = 'MERG', help='Available subgraph types: CENT, LIG, PROT, ELEC')
    parser.add_argument('-s', '--simplify', default = False, help = 'apply simplification for hydrogen bonds', action = 'store_true')
    parser.add_argument('-mo', '--mode', default = 'grakel', help = 'type of interaction graph')
    parser.add_argument('-th', '--thresh', default = None, help = 'distance threshold for edge in interaction graph', type = float)

    parser.set_defaults(func=main)
    args=parser.parse_args()
    status = args.func(args)
    sys.exit(status)


    
import argparse
import sys
import pdb
import subprocess
import os
import pickle
import numpy as np
import logging
from pyichem.ints import Ints
from pyichem.ifp import Ifp
from pyichem.tifp import Tifp


def get_files(folder):
		if os.path.isdir(folder):
			list_file=[]
			files = os.listdir(folder)
			for f in files:
				list_file.append(f'{folder}/{f}')
			return sorted(list_file)
		else:
			return None

def main(args):
	receptor_folder = args.receptor if args.receptor else 'ichem_outputs/structures/receptor'
	ligand_folder = args.ligand if args.ligand else 'ichem_outputs/structures/ligand'
	
	receptor_file = get_files(receptor_folder)
	ligand_file = get_files(ligand_folder)

	logging.info('Starting calculations')

	if args.new:
		logging.info('Calculating interactions using the Newhyd defintion of hydrophobic contacts')
		ints_newhyd = Ints(ipa_type = 'MERG', newhyd = True, ichem_path = args.ichem)
		ints_newhyd.change_rules({'-D_Ar': 5.0, '-d_Hb': 2.0})
		ints_newhyd.calculate(receptor_file, ligand_file, input_file = 'interactions_newhyd.in')
		with open('ints_newhyd.pkl', 'wb') as outfile:
			pickle.dump(ints_newhyd, outfile)
		

	if args.default:
		logging.info('Calculating interactions using the default definitions')
		ints_hyd = Ints(ipa_type = 'MERG', newhyd = False, ichem_path = args.ichem)
		ints_hyd.change_rules({'-D_Ar': 5.0, '-d_Hb': 2.0})
		ints_hyd.calculate(receptor_file, ligand_file, input_file = 'interactions.in')
		with open('ints_hyd.pkl', 'wb') as outfile:
			pickle.dump(ints_hyd, outfile)

	if args.ifp:
		logging.info('Compute IFP')
		ifp_d = Ifp(ichem_path = args.ichem)
		ifp_d.change_rules({'-D_Ar': 5.0, '-d_Hb': 2.0})
		ifp_d.calculate(receptor_file, ligand_file)
		ifp_d.read_ifp()
		with open('ifp.pkl', 'wb') as outfile:
			pickle.dump(ifp_d, outfile)		

	logging.info('Calculation completed')
	

		
	

if __name__ == "__main__":
	parser=argparse.ArgumentParser()
	parser.add_argument('-ic', '--ichem', help = 'absolut path to the locally installed IChem executable', required = True)
	parser.add_argument('-d', '--default', default = True, help = 'Skip IPA detection with default defintion of hydrophobic contacts', action = 'store_false')
	parser.add_argument('-n', '--new', default = True, help = 'Skip IPA detection with the Newhyd defintion of hydrophobic contacts', action = 'store_false')
	parser.add_argument('-i', '--ifp', default = True, help = 'Skip IFP detection', action = 'store_false')
	parser.add_argument('-r', '--receptor', default = None, help = 'Folder containing the receptor structures')
	parser.add_argument('-l', '--ligand', default = None, help = 'Folder containing the ligand structures')

	parser.set_defaults(func=main)
	args=parser.parse_args()
	status = args.func(args)
	sys.exit(status)
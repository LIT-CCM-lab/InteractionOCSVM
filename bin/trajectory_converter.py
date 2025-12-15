import argparse
import sys
import pdb
import pytraj as pt
import os
import numpy as np
import logging
from mol2df.trajmol2 import TrajToMol2, check_topology


LIGAND_PATH='ichem_outputs/structures/ligand'
RECEPTOR_PATH = 'ichem_outputs/structures/receptor'
STRUCTURES_PATH = 'ichem_outputs/structures'

logging.getLogger().setLevel(logging.INFO)

AMINOACID_MASK = 'ARG,HIS,LYS,ASP,GLU,SER,THR,ASN,GLN,CYS,SEC,GLY,PRO,ALA,VAL,\
					ILE,LEU,MET,PHE,TYR,TRP,HSE,HSD,HSP,CYX,HIE,HIP,HID,HSD,HSE.\
					HSP,ASH,LYN,GLH,ACE,NME,TYS'

def main(args):

	#pdb.set_trace()
	
	for tr in args.trajectory:
		logging.info(f'Trajectory file: {tr}')
	logging.info(f'Topology file: {args.topology}')

	r_mask = f':{AMINOACID_MASK}'
	if args.receptor:
		r_mask = r_mask + f'&:{args.receptor}'

	top = pt.load_topology(args.topology)[f':{AMINOACID_MASK}|:{args.ligand}']

	check_topology(top, args.force_field, to_sybyl = True)
	
	logging.info('Loading trajectory ...')

	traj = pt.load(args.trajectory, args.topology, mask = f':{AMINOACID_MASK}|:{args.ligand}', stride = args.skip_frames)

	reference = pt.load(args.reference, mask = f':{AMINOACID_MASK}|:{args.ligand}')

	logging.info(f'Trajectory loaded with {traj.n_frames} frames')

	traj = traj.autoimage()
	traj = traj.superpose(mask=f'@CA&:{args.alignment}', ref=reference, ref_mask=f'@CA&:{args.alignment}')

	rmsd = pt.rmsd(traj, mask=f'@CA&:{args.alignment}', ref=reference, ref_mask=f'@CA&:{args.alignment}', nofit = True)
	max_rmsd = np.amax(rmsd)

	logging.info(f'Trajectory aligned with max RMSD {max_rmsd}')

	logging.info('Conversion trajectory to mol2 files')

	if not os.path.isdir(LIGAND_PATH):
			os.makedirs(LIGAND_PATH)
	if not os.path.isdir(RECEPTOR_PATH):
			os.makedirs(RECEPTOR_PATH)

	converter_receptor = TrajToMol2(force_field = args.force_field)
	converter_receptor.load_traj(traj[r_mask], args.receptor_name, 'PROTEIN')
	converter_receptor.to_sybyl(c_cat = args.receptor_cations)
	converter_receptor.write_traj(outfile = f'{RECEPTOR_PATH}/{args.receptor_name}_structure', multimol = False)

	converter_receptor = TrajToMol2(force_field = args.force_field)
	converter_receptor.load_traj(traj[f':{args.ligand}'], args.ligand_name, 'SMALL MOLECULE')
	converter_receptor.to_sybyl(c_cat = args.ligand_cations)
	converter_receptor.write_traj(outfile = f'{LIGAND_PATH}/{args.ligand_name}_structure', multimol = False)

	if args.pdb:
		if not os.path.isdir(RECEPTOR_PATH+'_pdb'):
			os.makedirs(RECEPTOR_PATH+'_pdb')
		n = 0
		n_atoms = traj.n_frames
		while True:
			n += 1
			n_atoms = n_atoms // 10
			if n_atoms == 0:
				break

		pt.write_traj(f'{RECEPTOR_PATH}/receptor', traj=traj, format='pdb', options='multi pdbv3')
		for i in range(1,traj.n_frames+1):
			os.rename(f'{RECEPTOR_PATH}/receptor.{i}', f'{RECEPTOR_PATH}/receptor_{str(i-1).zfill(n)}.pdb')


	logging.info('Process completed')		
	

if __name__ == "__main__":
	parser=argparse.ArgumentParser()
	parser.add_argument('-top', '--topology', help='Input topology file', required = True)
	parser.add_argument('-traj', '--trajectory', nargs='+', help='Input trajectory file(s)', required = True)
	parser.add_argument('-r', '--receptor', help='Residues of the receptor', required = False)
	parser.add_argument('-l', '--ligand', help='Residues of the ligand', required = True)
	parser.add_argument('-ref', '--reference', help='Reference file', required = True)
	parser.add_argument('-a', '--alignment', help='Residues to use for the alignment', required = True)
	parser.add_argument('-ln', '--ligand_name', help='Name of the ligand', required=True)
	parser.add_argument('-rn', '--receptor_name', help='Name of the ligand', required=True)
	parser.add_argument('-p', '--pdb', default = False, action = 'store_true', help = 'convert the receptor also to pdb file')
	parser.add_argument('-ar', '--alignment_ref', default=None, help='Residues to use for the alignment of the reference structure')
	parser.add_argument('-sf', '--skip_frames',default=None, help='Stride between the frames converted to mol2 files')
	parser.add_argument('-ff', '--force_field', default = 'amber', help = 'Force field in which the trajectory atom types are defined.\n Set to charmm if CHARMM is used, default option considers AMBER atom types')
	parser.add_argument('-lcat', '--ligand_cations', default = [], nargs = '+', help = 'List of carbon atoms to convert to the C.cat atom type', type = int)
	parser.add_argument('-rcat', '--receptor_cations', default = [], nargs = '+', help = 'List of carbon atoms to convert to the C.cat atom type', type = int)
	

	parser.set_defaults(func=main)
	args=parser.parse_args()
	status = args.func(args)
	sys.exit(status)
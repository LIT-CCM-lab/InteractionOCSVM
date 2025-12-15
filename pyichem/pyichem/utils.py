import pymol
import numpy as np
from pyichem import ints
from tqdm import tqdm

COLORS = {'tv_yellow': ['GLC', 'GLP', 'GLL'],
		  'tv_red': ['ALC', 'ALP', 'ALL'],
		  'tv_blue': ['SEC', 'SEP', 'SEL'],
		  #'red': ['ASC', 'ASP', 'ASL'],
		  'tv_orange': ['PHC', 'PHL', 'PHP']}

def generate_pymol_ipa_traj(ints,
							output_file,
							colors = COLORS,
							receptor_res = []):
	'''
	Generate a pymol session with IPA integrated in the structures
	:param ints: File containg relevant interaction pseudoatoms (IPAs) information
	:type ints: pyichem.ints.Ints
	:param output_file: File containing the pymol session
	:type output_file: str
	:param colors: Color of interaction pseduoatoms,
					certain IPAs are not colored due to a non unique naming convention
	:type colors: dict
	:param receptor_res: Residues in the receptor structure to be shown as sticks
	:type receptor_res: list
	'''

	pymol.finish_launching()

	for receptor, ligand, inter in tqdm(zip(ints.receptor_mol2, ints.ligand_mol2, ints.get_output_files())):
		pymol.cmd.load(f'{ints.root}/{receptor}', 'receptor')
		pymol.cmd.load(f'{ints.root}/{ligand}', 'ligand')
		pymol.cmd.load(inter, 'ints')

	for c, res in colors.items():
		pymol.cmd.color(c, ' '.join([f'resname {r}' for r in res]))


	if len(receptor_res) > 0:
		pymol.cmd.show('sticks', f"resi {'+'.join([str(r) for r in receptor_res])}")
	pymol.cmd.hide('sticks', 'hydrogens')

	pymol.cmd.hide('spheres', 'all')
	pymol.cmd.hide('nb_spheres', 'all')

	pymol.cmd.show('spheres', 'resname ALC resname ALP resname ALL resname SEC resname SEL resname SEP resname GLC resname GLL resname GLP resname PHC resname PHL resname PHP')
	pymol.cmd.set('sphere_scale', 0.2)

	pymol.cmd.bg_color('white')

	pymol.cmd.save(f'{output_file}.pse')
	pymol.cmd.quit()
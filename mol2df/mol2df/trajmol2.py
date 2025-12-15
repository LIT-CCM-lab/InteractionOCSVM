import numpy as np
import pandas as pd
import os
import pdb
from tqdm import tqdm
import logging
import yaml

from mol2df.mol2df import (Mol2Df,
							ATOM_COLUMNS,
							SUBST_COLUMNS,
							BACKBONE_ATOMS,
							atom_card,
							load_conversion_file)

BOND_PATH = os.path.dirname(os.path.realpath(__file__)) + '/bond_file/'

RES_NAME_UPDATE = {'CYX': 'CYS',
				   'HIE': 'HIS',
				   'HIP': 'HIS',
				   'HID': 'HIS',
				   'HSD': 'HIS',
				   'HSE': 'HIS',
				   'HSP': 'HIS',
				   'ASH': 'ASP',
				   'LYN': 'LYS',
				   'GLH': 'GLU'}

FF_FILES = {'charmm': [f'{BOND_PATH}charmm_bonds.yml',
					   f'{BOND_PATH}charmm_sp3.yml',],
			'amber' : [f'{BOND_PATH}amber_bonds.yml' ,
					   f'{BOND_PATH}amber_sp3.yml',],
			'sybyl' : [f'{BOND_PATH}sybyl_bonds.yml' ,
					   f'{BOND_PATH}sybyl_sp3.yml']}

def load_ff_files(force_field):
	files = FF_FILES.get(force_field)
	if not files:
		raise ValueError(f"Undetermined force field {force_field}, please select one between: amber, charmm, sybyl")
	with open(files[0]) as c2s:
		bond_dict = yaml.load(c2s, Loader=yaml.FullLoader)
	with open(files[1]) as c2s:
		sp3_atoms = yaml.load(c2s, Loader=yaml.FullLoader)

	return bond_dict, sp3_atoms

def check_topology(top, force_field, to_sybyl = False):
	bond_dict, sp3_atoms = load_ff_files(force_field)
	missing_atoms = set()
	missing_bonds = set()
	atom_types = [atom.type for atom in top.atoms]

	for i,j in top.bond_indices:
		ati = atom_types[i]
		atj = atom_types[j]
		if ati in sp3_atoms or atj in sp3_atoms:
			continue
		key_1 = f"{ati} {atj}"
		key_2 = f"{atj} {ati}"
		if not key_1 in bond_dict and not key_2 in bond_dict:
			missing_bonds.add(key_1)

	if to_sybyl:
		conversion_dict = load_conversion_file(force_field)
		for at in atom_types:
			if at not in conversion_dict:
				missing_atoms.add(at)

	error_message = ''
	if len(missing_atoms) != 0:
		error_message += '\n'
		error_message += 'Missing atoms for conversion to sybyl:\n'
		error_message += '\t'.join(missing_atoms)
	if len(missing_bonds) != 0:
		error_message += '\n'
		error_message += 'Missing bond types:\n'
		error_message += '\n'.join(missing_bonds)
	if error_message != '':
		raise ValueError(error_message)





class TrajToMol2():
	'''
	Object defining a molecular dynamics trajectory as a group of dataframes describing the different blocks.
	The coordinates of the trajectory are not considered in the dataframe but can be written to single mol2 files.

	'''
	def __init__(self, force_field = 'amber'):
		'''
		Constructor method

		:param traj: trajectory file
		:type traj: pytraj trajectory
		:param force_field: force field of the original atom types and bond types
		:type force_field: string
		:param to_sybyl: convert the atom types to the sybyl atom type
		:type to_sybyl: bool
		:param c_cat: atoms with the sybyl atom type C.cat since automatic recognition is not possible
		:type c_cat: list of str
		:param outfile_name: root name of the output files
		:type outfile_name: str
		:param moltype: tripos molecule type
		:type moltype: string

		'''
		self.ff = force_field
		self.read_files()


	def read_files(self):
		'''
		Function loading the necessary file for atom type conversion and for bond order determination
		'''
		self.bond_dict, self.sp3_atoms = load_ff_files(self.ff)
			
	def load_traj(self, traj, name, moltype):
		self.mol2_df = Mol2Df()
		self.traj = traj
		n_bonds = len([0 for _ in traj.top.bonds])
		self.mol2_df.load_mol_data(name,
				f'{self.traj.top.n_atoms}',
				f'{n_bonds}',
				f'{self.traj.top.n_residues}',
				moltype,
				'USER_CHARGES')

		self.build_atom()
		self.build_bonds()
		self.build_substructure()

	def build_atom(self):
		'''
		Function to create a dataframe with a mol2 like structure containing information on the atoms:
		- atom name
		- atom type
		- residue id
		- residue name
		- charge
		- status_bit
		'''
		atom_block = {col: [] for col in ATOM_COLUMNS}
		for i,atom in enumerate(self.traj.top.atoms, 1):
			atom_block['atom_type'].append(atom.type)
			atom_block['atom_name'].append(atom.name)
			atom_block['res_id'].append(atom.resid+1)
			#atom_block['res_name'].append(atom.resname+str(atom.resid+1))
			atom_block['charge'].append(atom.charge)
			bb = 'BACKBONE' if atom.name in BACKBONE_ATOMS else ''
			atom_block['status_bit'].append(bb)
			atom_block['atom_number'].append(i)
			atom_block['x'].append(0)
			atom_block['y'].append(0)
			atom_block['z'].append(0)

		for res in self.traj.top.residues:
			for _ in range(res.n_atoms):
				if res.name in RES_NAME_UPDATE:
					name = RES_NAME_UPDATE[res.name]
				else:
					name = res.name
				atom_block['res_name'].append(name+str(res.original_resid))



		self.mol2_df.generate_atom_df(atom_block)


	def build_bonds(self):
		'''
		Function to create a dataframe with a mol2 like structure containing information on the bonds:
		- atom 1
		- atom 2
		- bond type
		'''
		bond_block = {'atom_1' : self.traj.top.bond_indices[:,0]+1,
					  'atom_2' : self.traj.top.bond_indices[:,1]+1,
					  'bond_type' : list(),
					  'bond_number': list(),
					  'status': list(),}

		for idx, (i,j) in enumerate(self.traj.top.bond_indices, 1):	
			bond_block['bond_number'].append(idx)
			ati = self.mol2_df.atom_df.iloc[i]['atom_type']
			atj = self.mol2_df.atom_df.iloc[j]['atom_type']
			bond_block['status'].append(None)
			if ati in self.sp3_atoms or atj in self.sp3_atoms:
				bond_block['bond_type'].append(1)
				continue
			key_1 = f"{ati} {atj}"
			key_2 = f"{atj} {ati}"
			if key_1 in self.bond_dict:
				bond_block['bond_type'].append(self.bond_dict[key_1])
			elif key_2 in self.bond_dict:
				bond_block['bond_type'].append(self.bond_dict[key_2])
			else:
				raise ValueError(f'The bond type between atom types: {key_1}, has not been defined yet. Please update the file with the bond definition.')



		self.mol2_df.generate_bond_df(bond_block)


	def build_substructure(self):
		'''
		Function to create a dataframe with a mol2 like structure containing information on the substructuress:
		- residue name
		- root atom
		- substructure type
		'''
		substructure_block = {col: [] for col in SUBST_COLUMNS}

		for i,res in enumerate(self.traj.top.residues, 1):
			if res.name in RES_NAME_UPDATE:
				name = RES_NAME_UPDATE[res.name]
			else:
				name = res.name
			substructure_block['resname'].append(name+str(res.original_resid))
			substructure_block['root_atom'].append(res.first+1)
			substructure_block['sub_type'].append(name)
			substructure_block['subst_number'].append(i)

			substructure_block['subst_type'].append('RESIDUE')
			substructure_block['dict_type'].append(1)
			substructure_block['chain'].append('A')
			substructure_block['inter_bonds'].append(0)

		self.mol2_df.generate_subst_df(substructure_block)

	def write_traj(self, outfile, multimol = False):
		'''
		Function to write the trajectory frames coordinates to mol2 files.

		:param multimol: write the output to a single mol2 file or to multiple mol2 files
		:type multimol: bool
		'''

		write_mode = 'a' if multimol else 'w'
		last_line = '\n\n\n' if multimol else ''
		molecule_name = outfile if multimol else None
		if multimol and os.path.isfile(f'{outfile}.mol2'):
			os.remove(f'{outfile}.mol2')
		n = 0
		n_atoms = self.traj.n_frames
		while True:
			n += 1
			n_atoms = n_atoms // 10
			if n_atoms == 0:
				break

		out_strings = self.mol2_df.get_block_string()

		print(f'Writing trajectory to {outfile}')

		for i, coords in enumerate(tqdm(self.traj.xyz)):
			tmp_df = self.mol2_df.atom_df.copy()
			tmp_df['x'] = coords[:,0]
			tmp_df['y'] = coords[:,1]
			tmp_df['z'] = coords[:,2]
			
			molecule_name = f'{outfile}_frame_{str(i).zfill(n)}' if not multimol else molecule_name

			with open(f'{molecule_name}.mol2', write_mode) as out:
				atom_lines = [l+'\n' for l in tmp_df.to_string(
												columns = list(ATOM_COLUMNS.keys())[1:],
												header = False,
												index_names = False).split('\n')]
				atom_lines.insert(0, atom_card)
				out.writelines(out_strings['molecule_block']
								+atom_lines
								+out_strings['bond_block']
								+out_strings['subst_block'])

	def write_topology(self, outfile):
		self.mol2_df.write(outfile)

	def to_sybyl(self, c_cat = []):
		self.mol2_df.to_sybyl(self.ff, c_cat)
import numpy as np
import pandas as pd
import os
import pdb
import logging
import yaml


ATOM_PATH = os.path.dirname(os.path.realpath(__file__)) + '/conversion_file/'

BACKBONE_ATOMS = ['C', 'CA', 'O', 'N', 'H', 'HA', 'HA1', 'HA2', 'HA3']

subst_card = '@<TRIPOS>SUBSTRUCTURE\n'
atom_card= '@<TRIPOS>ATOM\n'
bond_card = '@<TRIPOS>BOND\n'
molecule_card = '@<TRIPOS>MOLECULE\n'
comment_card = '@<TRIPOS>COMMENT\n'

CARDS = [molecule_card, atom_card, bond_card, subst_card, comment_card]

ATOM_COLUMNS = {'atom_number':int,
				'atom_name':str,
				'x':float,
				'y':float,
				'z':float,
				'atom_type':str,
				'res_id':int,
				'res_name':str,
				'charge':float,
				'status_bit':str}
BOND_COLUMNS = {'bond_number':int,
				'atom_1':int,
				'atom_2':int,
				'bond_type':str,
				'status': str}
SUBST_COLUMNS = {'subst_number':int,
				 'resname':str,
				 'root_atom':int,
				 'subst_type':str,
				 'dict_type':str,
				 'chain':str,
				 'sub_type':str,
				 'inter_bonds':str}

CONVERSION_FILES = {'charmm': f'{ATOM_PATH}charmm2sybyl.yml',
					'amber' : f'{ATOM_PATH}amber2sybyl.yml' ,}

COLUMN_TYPES = {'atom_number': [int, str, float, float, float, str, int, str, float, str],
				'bond_number': [int, int, int, str],
				'subst_number': [int, str, int, str, str, str, str, str]}

def generate_block_df(input_data, columns, index):
	if isinstance(input_data, list):
		new_input_data = []
		for inp_d in input_data:
			if len(inp_d) > 0:
				if str(inp_d[0]).startswith('#'):
					continue
			while len(inp_d) < len(columns):
				inp_d.append(None)
			if len(inp_d) > len(columns):
				inp_d = inp_d[:len(columns)]
			new_input_data.append(inp_d)
		out = pd.DataFrame(new_input_data, columns=list(columns.keys())).dropna(how = 'all')
	elif isinstance(input_data, dict):
		out = pd.DataFrame.from_dict(input_data)
	out = out.fillna('')
	out = out.astype(columns)
	return out.set_index(index)


class Mol2Df():
	'''
	Object describing a single mol2 file as a collection of multiple tables describing the different blocks
	'''
	def __init__(self):
		pass

	def get_block_string(self):
		atom_lines = [l+'\n' for l in self.atom_df.to_string(
									   		columns = list(ATOM_COLUMNS.keys())[1:],
									   		header = False,
									   		index_names = False,
									   		na_rep = '').split('\n')]
		atom_lines.insert(0, atom_card)
		if hasattr(self, 'bond_df'):
			bond_lines = [l+'\n' for l in self.bond_df.to_string(
										   		columns = list(BOND_COLUMNS.keys())[1:],
										   		header = False,
										   		index_names = False,
										   		na_rep = '').split('\n')]
		else:
			bond_lines = []
		bond_lines.insert(0, bond_card)
		if hasattr(self, 'subst_df'):
			subst_lines = [l+'\n' for l in self.subst_df.to_string(
												columns = list(SUBST_COLUMNS.keys())[1:],
												header = False,
												index_names = False,
												na_rep = '').split('\n')]
		else:
			subst_lines = []
		subst_lines.insert(0, subst_card)

		return {'molecule_block': [''.join([molecule_card,
										   f'{self.molname}\n',
										   f'{self.n_atoms}',
										   f'\t{self.n_bonds}',
										   f'\t{self.n_subst}\t0\t0\n',
										   f'{self.moltype}\n'
										   f'{self.charges}\n\n'])],
				'atom_block': atom_lines,
				'bond_block': bond_lines,
				'subst_block': subst_lines}

	def write(self, out_file):
		if not out_file.endswith('.mol2'):
			out_file += '.mol2'

		out_strings = self.get_block_string()

		with open(out_file, 'w') as outfile:
			outfile.writelines(out_strings['molecule_block']
										  +out_strings['atom_block']
										  +out_strings['bond_block']
										  +out_strings['subst_block'])

	def write_multimol(self, open_out_file):
		out_strings = self.get_block_string()
		open_out_file.writelines(out_strings['molecule_block']
										  +out_strings['atom_block']
										  +out_strings['bond_block']
										  +out_strings['subst_block'])

	def generate_atom_df(self, input_data):
		self.atom_df = generate_block_df(input_data,
										ATOM_COLUMNS,
										'atom_number')

	def generate_bond_df(self, input_data):
		self.bond_df = generate_block_df(input_data,
										BOND_COLUMNS,
										'bond_number')

	def generate_subst_df(self, input_data):
		self.subst_df = generate_block_df(input_data,
										SUBST_COLUMNS,
										'subst_number')

	def generate_subst_df_from_data(self):
		input_data = []
		for index, (i, r) in enumerate(self.atom_df.drop_duplicates(['res_name', 'res_id']).iterrows(), 1):
			input_data.append([index, r.res_name, i, 'RESIDUE', 1, 'A', r.res_name[:3], 0])

		self.generate_subst_df(input_data)

	def molecule_info(self, lines):
		'''
		Function to read the molecular informations contained in the Molecule block
		'''
		if len(lines[1]) == 2:
			lines[1] = lines[1] + ' 1'
		if len(lines[0]) == 0:
			lines[0].append('')
		self.load_mol_data(
			lines[0][0],
			*lines[1][:3],
			lines[2][0],
			lines[3][0])

	def renumber_residues(self, new_numbering, fix_resname = True):
		self.atom_df['res_id'] = self.atom_df['res_id'].replace(new_numbering)
		if fix_resname:
			self.atom_df['res_name'] = [self.atom_df.loc[i+1]['res_name'][:3]+str(self.atom_df.loc[i+1]['res_id']) for i in range(self.atom_df.shape[0])]
			self.subst_df['resname'] = self.atom_df.query("atom_name == 'CA'")['res_name'].to_list()

	def load_mol_data(self,
			molname,
			n_atoms,
			n_bonds,
			n_subst,
			moltype,
			charges):
		self.molname = molname
		self.n_atoms = n_atoms
		self.n_bonds = n_bonds
		self.n_subst = n_subst
		self.moltype = moltype
		self.charges = charges

	def to_sybyl(self, force_field,  c_cat = []):
		logging.info(f'Converting atom types from {force_field} to sybyl')
		conversion_dict = load_conversion_file(force_field)
		self.atom_df['atom_type'] = self.atom_df['atom_type'].replace(conversion_dict)
		self.atom_df.loc[c_cat, 'atom_type'] = 'C.cat'

	def add_backbone(self):
		bb = []
		for row in self.atom_df.iterrows():
			status = 'BACKBONE' if row.atom_name in BACKBONE_ATOMS else ''
			bb.append(status)
		self.atom_df['status_bit'] = bb

	def get_unique_res_names(self):
		resnames = []
		substnames = []
		for i, r in self.atom_df.iterrows():
			resnames.append(f"{r['res_name']}{r['res_id']}")
		for i,r in self.subst_df.iterrows():
			substnames.append(f"{r['resname']}{i}")
		self.atom_df['res_name'] = resnames
		self.subst_df['resname'] = substnames

	def read_comment(self, lines):
		self.comment_lines = lines


def gold_comment_parser(lines):
	fields = {}
	headers = []
	for line in lines:
		if len(line) == 0:
			continue
		if line[0] == '>':
			fields[line[1][1:-1]] = []
			headers.append(line[1][1:-1])
		else:
			fields[headers[-1]].append(line)
	return fields

def gold_hydrogen_coords(comments):
	data = gold_comment_parser(comments)
	h_dict = {}
	for h_data in data['Gold.Protein.RotatedAtoms']:
		if h_data[3] != 'LP':
			h_dict[h_data[18]] = {'x': h_data[0], 'y': h_data[1], 'z': h_data[2]}
	return h_dict


def plants_comment_parser(lines):
	fields = {}
	headers = []
	fileds['original_name'] = lines[0][0]
	for l in lines[1:]:
		if l[0].startswith('PLANTS'):
			fields[l[0]] = []
			headers.append(l[0])
		elif l[0] == 'LINES':
			continue
		else:
			fields[header[-1]].append(l)
	return


def load_conversion_file(force_field):
	files = CONVERSION_FILES.get(force_field)
	if not files:
		logging.error(f'Unrecognized force field {force_field}')
		raise KeyError(f'Unrecognized force field {force_field}')
	with open(files) as c2s:
		conversion_dict = yaml.load(c2s, Loader=yaml.FullLoader)

	return conversion_dict


class Mol2Reader():
	'''
	Object used to read the content of a mol2 file to multiple mol2_df objects.
	'''
	def __init__(self):
		'''
		Constructor method
		:param multimol: indicate if the mol2 file contains multiple molecules
		:type multimol: bool
		'''
		
		pass
		

	def read_mol2(self, file, atom_only=False):
		'''
		Function reading the content of a mol2 file
		:param file: mol2 file to read
		:type file: str
		:return: objects with dataframes containing all the information contained in the original mol2 file
		:rtype: mol2_df or list of mol2_df
		'''
		self.atom_only = atom_only
		with open(file, 'r') as inp:
			text = inp.read()

		
		return self._generate_dfs(text)
		

	def _generate_df(self, text):
		'''
		Function to generate the mol2_df object for a single molecule
		:return: object with all the information contained in the mol2 file
		:rtype: mol2_df
		'''
		out = Mol2Df()

		block_functions = {'MOLECULE':out.molecule_info,
							'ATOM':out.generate_atom_df,
							'BOND':out.generate_bond_df,
							'SUBSTRUCTURE':out.generate_subst_df,
							'COMMENT':out.read_comment,}
		blocks = text.split('@<TRIPOS>')
		for b in blocks:
			lines = b.split('\n')
			function = block_functions.get(lines[0])
			if function:
				function([l.split() for l in lines[1:]])
			if self.atom_only and lines[0] == 'ATOM':
				return out

		return out

	def _generate_dfs(self, text):
		'''
		Function to generate the mol2_df object for a file containing multiple molecules
		:return: list of objects with all the information contained in the mol2 file
		:rtype: list of mol2_df
		'''
		files = text.split(CARDS[0])[1:]
		out = list()
		for file in files:
			out.append(self._generate_df('MOLECULE\n'+file))
		return out
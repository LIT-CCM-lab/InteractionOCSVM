import pandas as pd
import numpy as np
from pyichem.base_models import BatchCalculation
import pdb
import os

GRIM_PATH = 'ichem_outputs/GRIM'

class Grim(BatchCalculation):
	def __init__(
			self,
			values = True,
			inter = 'MERG',
			match = 'MERG',
			score = 'FCT',
			ichem_path = 'IChem',
			restart_file = None):
		super().__init__(folder=GRIM_PATH,
					  output_p='',
					  software='grim',
					  output_f = False,
					  opt = '',
					  ichem_path = ichem_path,)
		self.inter = inter
		self.match = match
		self.score = score
		self.values = values
		self.restart_file = restart_file
		self.option = self._check_option()

	def _check_option(self):
		option = ''
		allowed_values = ['MERG', 'LIG', 'CENT', 'PROT']

		if self.values:
			option = option + '--values '
		if self.inter in allowed_values:
			option = option + f'-outInt {self.inter} '
		else:
			raise ValueError(
				f'Unrecognized value for the output: {self.inter}'
				)
		if self.match in allowed_values:
			option = option + f'-match {self.match} '
		else:
			raise ValueError(
				f'Unrecognized value for the alignment: {self.align}'
				)
		if self.score == 'STD':
				option = option + '-score STD '
		elif self.score != 'FCT':
			raise ValueError(
				f'Unrecognized value for the score: {self.score}'
				)

		return option

	def read_results(self):
		self.grim_rescoring = pd.read_csv(f'{self.root}/Grifp_res.csv', 
										  sep = '/t',)
	def delete_outfile(self):
		os.remove(f'{self.root}/Grifp_res.csv')


class GrimStructures(Grim):
	'''
	Class calling the Grim tool of IChem.
	

	:param receptor_mol2: files containing the structure of the protein to use in IChem calculations
	:type receptor_mol2: list of str, optional
	:param ligand_mol2: files containing the structure of the ligands to use in IChem calculations
	:type ligand_mol2: list of str, optional
	:param ifp_format: format of the IFP
	:type ifp-format: str, optional
	:param output_file: name of the file where all the generated ifp are stored
	:type output_file: str
	'''

	def __init__(
			self,
			ref_receptor_mol2,
			ref_ligand_mol2,
			values = True,
			inter = 'MERG',
			match = 'MERG',
			score = 'FCT',
			ref_name = None,
			):
		'''Constructor method'''
		
		self.ref_receptor_mol2 = ref_receptor_mol2
		self.ref_ligand_mol2 = ref_ligand_mol2
		self.ref_name = ref_name if ref_name else [None for _ in ref_receptor_mol2]
		super().__init__(values, inter, match, score)
		

	def _write_input(self, input_file):
		'''
		Write the input file given to IChem.

		:param input_file: name of the input file
		:type input_file: str
		'''

		with open(f'{self.folder}/{input_file}', 'w') as input_f:
			for receptor, ligand, c_name in zip(self.receptor_mol2,
													self.ligand_mol2,
													self.comp_name):
				for ref_receptor, ref_ligand, r_name in zip(
												self.ref_receptor_mol2,
												self.ref_ligand_mol2,
												self.ref_name):
					line = [self.options]
					if r_name:
						line.append(f'-rn {r_name.strip()}')
					if c_name:
						line.append(f'-cn {c_name.strip()}')
					line.append(f'{self.software} {ref_receptor} {ref_ligand} {receptor} {ligand}\n')
					input_f.write(' '.join(line))
		self.input_file = input_file

	def calculate(
			self,
			receptor_mol2,
			ligand_mol2,
			input_file = 'ichem_input.in',
			stdout_c = True,
			stderr_c = True,
			comp_name = None
			):
		self.comp_name = comp_name if comp_name else [None for _ in receptor_mol2]
		BatchCalculation.calculate(receptor_mol2,
								   ligand_mol2,
								   input_file ,
								   stdout_c ,
								   stderr_c)

	def calculate_lbl(
			self,
			receptor_mol2,
			ligand_mol2,
			input_file = 'ichem_input.in',
			stdout_c = True,
			stderr_c = True,
			comp_name = None
			):
		self.comp_name = comp_name
		BatchCalculation.calculate_lbl(receptor_mol2,
								       ligand_mol2,
									   input_file ,
									   stdout_c ,
									   stderr_c )

class GrimIPA(Grim):

	'''
	Class calling the Grim tool of IChem.
	

	:param receptor_mol2: files containing the structure of the protein to use in IChem calculations
	:type receptor_mol2: list of str, optional
	:param ligand_mol2: files containing the structure of the ligands to use in IChem calculations
	:type ligand_mol2: list of str, optional
	:param ifp_format: format of the IFP
	:type ifp-format: str, optional
	:param output_file: name of the file where all the generated ifp are stored
	:type output_file: str
	'''

	def __init__(
			self,
			ref_ipa,
			values = True,
			inter = 'MERG',
			match = 'MERG',
			score = 'FCT',
			ref_name = None,
			ichem_path='IChem',
			restart_file=None,
			):
		'''Constructor method'''
		
		self.ref_ipa = ref_ipa
		self.ref_name = ref_name if ref_name else [None for _ in ref_ipa]

		super().__init__(values = values, inter=inter, match=match, score=score, ichem_path=ichem_path, restart_file=restart_file)
		

	def calculate(
			self,
			comp_ipa,
			input_file = 'ichem_input.in',
			stdout_c = True,
			stderr_c = True,
			comp_name = None,
			):
		self.comp_ipa = comp_ipa
		self.comp_name = comp_name if comp_name else [None for _ in comp_ipa]
		super().calculate([],
						   [],
						   input_file ,
						   stdout_c ,
						   stderr_c)

	def calculate_lbl(self,
					  comp_ipa,
					  input_file = 'ichem_input.in',
					  stdout_c = True,
					  stderr_c = True,
					  comp_name = None):
		self.comp_ipa = comp_ipa
		self.comp_name = comp_name if comp_name is not None else [None for _ in comp_ipa]
		super().calculate_lbl([],
						   [],
						   input_file ,
						   stdout_c ,
						   stderr_c )

	def _write_input(self, input_file):
		'''
		Write the input file given to IChem.

		:param input_file: name of the input file
		:type input_file: str
		'''
		self.preparation(self.folder)

		restart_info = set()

		if self.restart_file is not None:
			if os.path.isfile(f'{self.root}/{self.restart_file}'):
				data = pd.read_csv(f'{self.root}/{self.restart_file}', sep = '\t')
				restart_info = {(ref, comp) for ref, comp in zip(data['Ref'].to_list(), data['Comp'].to_list())}

		with open(input_file, 'w') as input_f:
			if self.ref_name is not None and self.comp_name is not None:
				for ipa, c_name in zip(self.comp_ipa, self.comp_name):
					for ripa, r_name in zip(self.ref_ipa,
											self.ref_name):
						line = [self.options]
						if (r_name, c_name) in restart_info:
							continue
						if r_name:
							line.append(f'-rn {r_name.strip()}')
						if c_name:
							line.append(f'-cn {c_name.strip()}')
						line.append(f'{self.software} {ripa} {ipa}\n')
						input_f.write(' '.join(line))
		self.input_file = input_file	
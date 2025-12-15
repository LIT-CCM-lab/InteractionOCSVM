import pandas as pd
import numpy as np
from pyichem.base_models import BatchCalculation
import logging
import os
from scipy.sparse import coo_matrix

TIFP_PATH = 'ichem_outputs/TIFP'

class Tifp(BatchCalculation):
	'''
	Class calling the Tifp tool of IChem.
	The generated TIFP can be stored using different formats, look at the IChem manual for more information

	:param receptor_mol2: files containing the structure of the protein to use in IChem calculations
	:type receptor_mol2: list of str, optional
	:param ligand_mol2: files containing the structure of the ligands to use in IChem calculations
	:type ligand_mol2: list of str, optional
	:param ifp_format: format used to store the TIFP
	:type ifp-format: str, optional
	:param small: calculate small tifp
	:type small: bool, optional
	'''

	def __init__(self, ichem_path, ifp_format = 'STD', small = True):
		'''Constructor method'''
		self.format = self.__check_format(ifp_format)

		opt = f'-fgps {self.format}'
		opt = '--small '+ opt if small else opt
		self.fp_len = 211 if small else 20000

		super().__init__(TIFP_PATH,
						 f'{TIFP_PATH}/out_tifp_',
						 'ints',
						 ichem_path,
						 opt = f'--small -fgps {self.format}',
						 output_s = '.tifp')
		
		
	def __check_format(self, ifp_format):
		'''
		Controls that the given TIFP format is between the accepted ones.

		:param ifp_format: format used to store the TIFP
		:type ifp_format: str
		:return: the TIFP format given as input if it is acceptable
		:rtype: str
		:raises :class:'ValueError': should the tifp format not being one of the implemented ones
		'''

		available_formats = ['STD', 'SVM', 'CMP']
		if ifp_format in available_formats:
			return ifp_format
		else:
			raise ValueError(f'Fingerprint format {ifp_format} not correct')

	def new_read_tifp(self):
		fp = sparse_tifp_reader([os.path.join(self.root, file) for file in self.output_location],
								self.format,
								self.fp_len)
		self.fingerprints = pd.DataFrame.sparse.from_spmatrix(out_data)

		return self.fingerprints

	def read_tifp(self):
		'''
		Reads the generated TIFP files in a single pandas DataFrame.
		In case the output file is missing a NaN row is given as the result.

		:return: fingerprints attribute of the TIFP object
		:rtype: pandas DataFrame
		'''
		fp = []
		for file in self.output_location:
			if os.path.isfile(os.path.join(self.root, file)):
				fp.append(tifp_reader(os.path.join(self.root, file), self.format))
			else:
				fp.append(np.full(self.fp_len, np.nan))
				logging.warning(f'Missing file: {file}')

		self.fingerprints = pd.DataFrame(data = fp)

		return self.fingerprints

def sparse_tifp_reader(files, tifp_type, n_cols = 210):

	row_count = 0
	row = []
	col = []
	values = []

	for file in files:
		with open(file, 'r') as input_file:
			for line in input_file:
				col_idx = 0
				for i,v in enumerate(line.split()[1:]):
					if tifp_type == 'SVM':
						pos, val = v.split(':')
						row.append(row_count)
						col.append(pos)
						values.append(int(val))
					elif tifp_type == 'CMP':
						if v.startswith('['):
							col_idx += int(v[1:])
						else:
							row.append(row_count)
							col.append(col_idx)
							calues.append(int(v))
							col_idx+=1
					if v != '0':
						row.append(row_count)
						col.append(i)
						values.append(int(v))
				row_count += 1

	return coo_matrix((values, (row, col)), shape = (row_count, n_cols))


def tifp_reader(file, tifp_type):
	'''
	Read a TIFP file

	:param file: file containing the TIFP
	:type file: str
	:param tifp_type: format in which the TIFP is saved
	:type tifp_type: str
	:return: binary TIFP
	:rtype: numpy Array
	'''
	
	with open(file, 'r') as input_file:
			for line in input_file:
				values = line.split()[1:]

	if tifp_type == 'SVM':
		return _convert_SVM(values)
	elif tifp_type == 'CMP':
		return _convert_CMP(values)
	else:
		return np.array(list(map(int, values)))

def _convert_CMP(character_list):
	'''
	Convert CMP TIFP to binary TIFP

	:param character_list: CMP TIFP set of characters
	:type character_list: str
	:return: binary TIFP
	:rtype: str
	'''
	pointer = 0
	fp = np.zeros(fp_lenght)
	for symbol in character_list:
		if symbol.startswith('['):
			pointer += int(symbol[1:])
		else:
			fp[pointer] = int(symbol)
			pointer += 1

	return fp

def _convert_SVM(character_list, fp_lenght):
	'''
	Convert SVM TIFP to binary TIFP

	:param character_list: SVM TIFP set of characters
	:type character_list: str
	:return: binary TIFP
	:rtype: str
	'''
	fp = np.zeros(fp_lenght)
	for couple in character_list:
		position, value = couple.split(':')
		fp[int(position)] = int(value)

	return fp
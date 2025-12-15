import pandas as pd
import numpy as np
import subprocess
import sys
import pdb
import os
from pyichem.base_models import BatchCalculation
from mol2df import Mol2Df, Mol2Reader
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from itertools import combinations, product
from scipy.spatial.distance import pdist
from scipy.sparse import coo_matrix
import logging

IFP_PATH = 'ichem_outputs/IFP'

class Ifp(BatchCalculation):
    '''
    Class calling the Ifp tool of IChem.
    Different types of interactions can be included in the IFP,
    look at the IChem manual for more information.

    :param receptor_mol2: files containing the structure of the protein
    :type receptor_mol2: list of str, optional
    :param ligand_mol2: files containing the structure of the ligands
    :type ligand_mol2: list of str, optional
    :param ifp_format: format of the IFP
    :type ifp-format: str, optional
    :param output_file: name of the file where all the generated ifp are stored
    :type output_file: str
    '''

    def __init__(
            self,
            ichem_path,
            ifp_format = 'regular',
            output_file = 'ligands.ifp'):
        '''Constructor method'''
        formats = {'regular': '',
                   'polar': '--polar',
                   'extended': '--extended'}
        self.ifp_format = ifp_format
        if ifp_format in formats:
            self.ifp_option = formats[ifp_format]
        else:
            raise ValueError(f'Invalid fingerprint type for {ifp_format}')
        super().__init__(
            IFP_PATH,
            f'{IFP_PATH}/ligands.ifp',
            'IFP',
            ichem_path,
            output_f = False,
            stdout_name = f'{IFP_PATH}/{output_file}',
            opt = self.ifp_option)

    def read_ifp(self):
        '''
        Reads the generated IFP files in a single pandas DataFrame.
        The function reads both a series of file or a single file.
        If a file is missing or it is empty NaN values are used to fill the IFP

        :returns: fingerprints attribute of the IFP object
        :rtype: pandas DataFrame
        '''

        if not hasattr(self, 'output_location'):
            self.fingerprints = ifp_reader(os.path.join(self.root,
                                                        self.stdout),
                                           self.ifp_format)
        else:
            for file in self.output_location:
                if hasattr(self, 'fingerprints'):
                        self.fingerprints = self.fingerprints.append(
                            ifp_reader(os.path.join(self.root,file),
                                        self.ifp_format),
                            ignore_index = True)
                else:
                    self.fingerprints = ifp_reader(os.path.join(
                                                        self.root,file
                                                        ),
                                                   self.ifp_format)
            

        return self.fingerprints

    def fp_interaction(self, interactions):
        '''
        Filters the IFP table selecting only the specified interactions types.

        :param interactions: selected interaction(s)
        :type interactions: str or list of str
        :return: table containing the selected interactions
        :rtype: pandas DataFrame
        '''
        if not isinstance(interactions, list):
            interactions = [interactions]
        return filter_interaction(self.fingerprints, interactions)

    def fp_residues(self, residues):
        '''
        Filters the IFP table selecting only the specified residues.

        :param residues: selected residue(s)
        :type residues: str or list of str
        :return: table containing the selected residues
        :rtype: pandas DataFrame
        '''
        if not isinstance(residues, list):
            residues = [residues]
        return filter_residues(self.fingerprints, residues)

    def calculate_lbl(self,
        receptor_mol2,
        ligand_mol2,
        input_file = 'ichem_ifp_input.in',
        stdout_c = True,
        stderr_c = True):
        '''
        Generates the input file, then launch the calculations line by line,
        this is not the optimal way to perform calculations but it is a
        workaround for 'Segmentation fault (core dumped)' errors observed
        when launching from file.

        :param input_file: name of the input file
        :type input_file: str
        :param stdout_c: capture the stdout stream from IChem
        :type stdout_c: bool
        :param stderr_c: capture the stderr stream from IChem
        :type stderr_c: bool
        '''

        self.receptor_mol2 = receptor_mol2
        self.ligand_mol2 = ligand_mol2

        self._write_input(input_file)

        with open(self.input_file, 'r') as inpt:
            for i,line in enumerate(inpt):
                counter = 0
                msg = ''
                while msg == '' and counter < 10:
                    command = line.split()
                    command.insert(0, self.ichem_path)
                    try:
                        process_output = subprocess.run(command,
                            stdout = subprocess.PIPE if stdout_c else None,
                            stderr = subprocess.PIPE if stderr_c else None)
                    except FileNotFoundError:
                        raise FileNotFoundError(
                            f'The IChem path was incorrect: {self.ichem_path}.'
                            )
                    msg = process_output.stdout.decode('utf-8')
                    counter += 1
                if process_output.stdout is not None:
                    with open(f'{IFP_PATH}/ligands.ifp', 'a') as out:
                        out.write(f'\nIFP from {line}\n')
                        msg = process_output.stdout.decode('utf-8')
                        if msg.split('\n')[-2][0] in ['0', '1']:
                            out.write(msg)
                        else:
                            error_message = f'|WARNING IChem was not' \
                            f'able to calculate IFP for line: {line}\n'
                            out.write(error_message)
                            logging.warning(error_message)
                if process_output.stderr is not None:
                    with open('ichem_stderr.txt', 'a') as err:
                        err.write(
                            process_output.stderr.decode('utf-8'))


    def plot_ifps(self,
                interactions = [],
                residues = '',
                mol_labels = None,
                vlines = None,
                hlines = None,
                entries = None):
        '''
        Generate a graphical representation of the stored IFP.

        :param interactions: Which interaction types to show in the plot
        ('HYD', 'HBA', 'HBD', 'CAT', 'ANI', 'ETF', 'FTF')
        :type interactions: list
        :param residues: Which residues to show in the plot, indicated by one
                        letter code and residue number (eg. A100)
        :type residues: list
        :param mol_labels: Labels for the entries in the plot,
                            if None then the ligand file name is shown
        :type mol_labels: list
        :param vlines: Show a vertical lines at the given positions
        :type vlines: list
        :param hlines: Show a horizontal lines at the given positions
        :type hlines: list
        :param entries: Show only the IFPs in the given positions
        :type entries: list
        '''
        plot_ifp = filter_ifp(self.fingerprints, residues, interactions)
        non_empty_bits = plot_ifp.columns[plot_ifp.sum() > 0]
        plot_ifp = plot_ifp[non_empty_bits]

        if lines is not None:
            plot_ifp = plot_ifp.loc[entries]
        if mol_labels is None:
            if lines is not None:
                mol_labels = [self.ligand_mol2[i] for i in entries]
            else:
                mol_labels = self.ligand_mol2

        fig, ax = plt.subplots()
        ax.imshow(plot_ifp)
        ax.set_xticks(np.arange(len(non_empty_bits)))
        ax.set_xticklabels(non_empty_bits, rotation = 90)
        ax.set_yticks(np.arange(len(mol_labels)))
        ax.set_yticklabels(mol_labels)

        if vlines is not None:
            for v in vlines:
                ax.axvline(v-0.5)
        if hlines is not None:
            for h in hlines:
                ax.axhline(h-0.5)

        return fig

    def sort_rows(self, idxs):
        '''
        Order the entries based on the new given indexes eg. [3 4 5 1 2 0]
        :param idxs: Index positions to be moved
        :type idxs: list
        '''
        self.fingerprints = self.fingerprints.loc[idxs]
        self.receptor_mol2 = np.array(self.receptor_mol2)[idxs].tolist()
        self.ligand_mol2 = np.array(self.ligand_mol2)[idxs].tolist()

    def compute_graphs(self, threshold = 4):
        '''
        Generate interaction graphs from the interaction information of the IFP
        :param threshold: distance threshold to determine the presence of an
                            edge between nodes in the graph
        :type threshold: float
        '''
        if not hasattr(self, 'output_location'):
            inter_data = interaction_reader(os.path.join(self.root,
                                                            self.stdout))
        else:
            inter_data = []
            for file in self.output_location:
                inter_data = inter_data + interaction_reader(
                                                os.path.join(self.root, file)
                                                )
        graphs = []
        graph_generator = IFPGraph()
        for lig, i_data in zip(self.ligand_mol2, inter_data):
            graphs.append(graph_generator.compute_graphs(i_data,
                                                        os.path.join(
                                                            self.root,
                                                            lig),
                                                        threshold))

        return graphs

    def get_residues(self):
        '''
        Obtain all residues names in the IFP
        '''
        residues = set()
        for bit in self.fingerprints.columns:
            residues.add(re.search('[A-Z]+[0-9]+', bit).group())

        return residues

    def simplify_hb(self, in_place = False):

        '''
        Simplify hydrogen bond representation by merging the donor and
        acceptor bits into a single value HBI
        '''

        residues = self.get_residues()
        fp = self.fingerprints.copy()

        hbd_cols = [c for c in fp.columns if 'HBD' in c]
        hba_cols = [c for c in fp.columns if 'HBA' in c]

        hb_matrix = fp[hbd_cols].to_numpy() + fp[hba_cols].to_numpy()
        hb_df = pd.DataFrame(hb_matrix,
                            columns = [c[:-1]+'I' for c in hba_cols])
        fp = fp.drop(hbd_cols, axis = 1)
        fp = fp.drop(hba_cols, axis = 1)
        fp = pd.concat([fp, hb_df], axis = 1)

        if in_place:
            self.fingerprints = fp

        return fp






PHARMACOPHORES = {'Hydrophobic': 'HYD',
                    'HBond_LIG': 'HBD',
                    'HBond_PROT': 'HBA',
                    'Ionic_LIG': 'CAT',
                    'Ionic_PROT': 'ANI'}
'''

class IFPGraph():
    def __init__(self):
        import networkx as nx

        self.graph = nx.Graph
        self.set_node_attribute = nx.set_node_attributes
        self.set_edge_attribute = nx.set_edge_attributes

        self.reader = Mol2Reader()

    def compute_graphs(self,interaction_data, ligand_file, threshold):

        lig_data = self.reader.read_mol2(ligand_file, atom_only = True)[0]
        edge_index = []
        n_labels = {}

        node_encoder = LabelEncoder().fit(interaction_data['lig']+interaction_data['prot'])
        lig_atom_labels = node_encoder.transform(interaction_data['lig'])
        lig_atoms = np.unique(interaction_data['lig'])

        n_nodes = np.unique(interaction_data['lig']+interaction_data['prot']).shape[0]-1
        
        for prot, lig, inter in zip(interaction_data['prot'],
                                    lig_atom_labels,
                                    interaction_data['inter']):
            p = node_encoder.transform([prot])[0]
            edge_index.append((lig, p))
            edge_index.append((p,lig))
            n_labels[lig] = PHARMACOPHORES[inter]
            n_labels[p] = prot

        atoms = lig_data.atom_df.query("atom_name in @lig_atoms")

        for a_ns in combinations(lig_atoms, 2):
            coords = atoms.query("atom_name in @a_ns")[['x', 'y', 'z']]
            if coords.shape[0] < 2:
                return self.graph
            d = pdist(coords, metric = 'euclidean')[0]
            if d < threshold:
                i,j = node_encoder.transform(a_ns)
                origin = i
                for s in np.arange(0,round(threshold+1e-5,0)+1):
                    if round(d, 0) <= s:
                        edge_index.append((origin,j))
                        edge_index.append((j,origin))
                        break
                    else:
                        n_nodes += 1
                        p_i = n_labels[i]
                        p_j = n_labels[j]
                        n_labels[n_nodes] = '-'.join(np.sort([p_i, p_j]))
                        edge_index.append((origin,n_nodes))
                        edge_index.append((n_nodes,origin))
                        origin = n_nodes

        out_graph = self.graph(edge_index)
        self.set_node_attribute(out_graph, n_labels, name = 'label')
        self.set_edge_attribute(out_graph, {i: 'i' for i in (edge_index)}, name = 'label')
                
        return out_graph
'''

class IFPGraph():
    def __init__(self):
        from grakel import Graph

        self.graph = Graph
        self.reader = Mol2Reader()

    def compute_graphs(self,interaction_data, ligand_file, threshold = 4.5):

        lig_data = self.reader.read_mol2(ligand_file, atom_only = True)[0]
        edges = {}
        edge_labels = {}
        n_labels = {}

        node_encoder = LabelEncoder().fit(
                            interaction_data['lig']+interaction_data['prot']
                            )
        lig_atom_labels = node_encoder.transform(interaction_data['lig'])
        lig_atoms = np.unique(interaction_data['lig'])
        
        for prot, lig, inter, dist in zip(interaction_data['prot'],
                                    lig_atom_labels,
                                    interaction_data['inter'],
                                    interaction_data['dist']):
            p = node_encoder.transform([prot])[0]
            edges[(lig, p)] = 1
            edges[(p, lig)] = 1
            edge_labels[(p, lig)] = f'{PHARMACOPHORES[inter]} {prot} \
                                        {round(dist)}'
            n_labels[lig] = PHARMACOPHORES[inter]
            n_labels[p] = prot

        atoms = lig_data.atom_df.query("atom_name in @lig_atoms")

        for a_ns in combinations(lig_atoms, 2):
            coords = atoms.query("atom_name in @a_ns")[['x', 'y', 'z']]
            if coords.shape[0] < 2:
                return self.graph
            d = pdist(coords, metric = 'euclidean')[0]
            if d < threshold:
                i,j = node_encoder.transform(a_ns)
                edges[(i,j)] = 1
                edges[(j,i)] = 1
                p_i = n_labels[i]
                p_j = n_labels[j]
                edge_labels[(i,j)] = f'{p_i} {p_j} {round(d)}'
                edge_labels[(j,i)] = f'{p_j} {p_i} {round(d)}'
                
        return self.graph(edges,
                            edge_labels = edge_labels,
                            node_labels = n_labels)


def interaction_reader(file):
    output = []
    data = {'inter': [], 'prot': [], 'lig': [], 'dist' : []}
    with open(file, 'r') as ifp_file:
        for line in ifp_file:
            if line.startswith('|WARNING'):
                output.append(data)
                data = {'inter': [], 'prot': [], 'lig': [], 'dist' : []}
            elif line.startswith('Hydrophobic') or line.startswith('Ionic') or line.startswith('HBond'):
                info = line.split()
                if len(info) == 11:
                    prot_i = 3
                    lig_i = 5
                    dist_i = 9
                    data['inter'].append(info[0])
                elif len(info) == 12:
                    prot_i = 4
                    lig_i = 6
                    dist_i = 10
                    data['inter'].append('_'.join([info[0], info[1]]))
                data['lig'].append(info[lig_i])
                data['prot'].append(f'{info[prot_i][1:]} {info[prot_i+1]}')
                data['dist'].append(float(info[dist_i]))
            elif line.startswith('0') or line.startswith('1'):
                output.append(data)
                data = {'inter': [], 'prot': [], 'lig': [], 'dist' : []}
    return output


def ifp_reader(file, ifp_type = 'regular'):
    '''
    Read an IFP file containing one or multiple IFPs

    :param file: file containing the IFPs
    :type file: str
    :param ifp_type: type of IFP
    :type tifp_type: str
    :return: table containing all the detected IFPs
    :rtype: pandas DataFrame
    :raises :class:'Exception': should there be a difference in the length of 
                                the IFPs stored in the same output file
    :raises :class:'ValueError': should be the type of IFP not implemented
    '''
    values = []
    header = None
    counter = 0

    if ifp_type == 'regular':
        interactions = ['HYD',
                        'FTF',
                        'ETF',
                        'HBD',
                        'HBA',
                        'CAT',
                        'ANI']
    elif ifp_type == 'polar':
        interactions = ['HBD',
                        'HBA',
                        'CAT',
                        'ANI',
                        'MCO']
    elif ifp_type == 'extended':
        interactions = ['HYD',
                        'FTF',
                        'ETF',
                        'HBD',
                        'HBA',
                        'CAT',
                        'ANI',
                        'PCI',
                        'MCO']
    else:
        raise Exception('Fingerprint format non-available')

    #pdb.set_trace()

    headers = {}
    header_idx = {}
    c = 0
    row_count = 0
    known_headers = set()
    row = []
    col = []

    with open(file, 'r') as ifp_file:
        for line in ifp_file:
            if line.startswith('|WARNING'):
                row_count += 1
            elif line.startswith('|') and not line.startswith('|ERROR'):
                curr_headers = [
                f'{h.strip()} {code}' for h, code in product(
                                                        line.split('|')[1:-1],
                                                        interactions)
                    ]
                for h in curr_headers:
                    if h in known_headers:
                        continue
                    else:
                        known_headers.add(h)
                    headers[h] = c
                    header_idx[c] = h
                    c += 1

            elif line.startswith('0') or line.startswith('1'):
                for bit, inter in zip(line[:-1], curr_headers):
                    if bit == '1':
                        row.append(row_count)
                        col.append(headers[inter])
                row_count += 1

    out_data = coo_matrix((np.ones(len(col)),
                            (row, col)),
                            shape = (row_count, c))

    return pd.DataFrame.sparse.from_spmatrix(out_data,
                    columns = [header_idx[i] for i in range(len(headers))])
        

def ifp_reader_old(file, ifp_type = 'regular'):
    '''
    Read an IFP file containing one or multiple IFPs

    :param file: file containing the IFPs
    :type file: str
    :param ifp_type: type of IFP
    :type tifp_type: str
    :return: table containing all the detected IFPs
    :rtype: pandas DataFrame
    :raises :class:'Exception': should there be a difference in the length of the IFPs stored in the same output file
    :raises :class:'ValueError': should be the type of IFP not implemented
    '''
    values = []
    header = None
    counter = 0

    if ifp_type == 'regular':
        interactions = ['HYD',
                        'FTF',
                        'ETF',
                        'HBD',
                        'HBA',
                        'CAT',
                        'ANI']
    elif ifp_type == 'polar':
        interactions = ['HBD',
                        'HBA',
                        'CAT',
                        'ANI',
                        'MCO']
    elif ifp_type == 'extended':
        interactions = ['HYD',
                        'FTF',
                        'ETF',
                        'HBD',
                        'HBA',
                        'CAT',
                        'ANI',
                        'PCI',
                        'MCO']
    else:
        raise Exception('Fingerprint format non-available')

    ifps = []

    #pdb.set_trace()

    with open(file, 'r') as ifp_file:
        for line in ifp_file:
            if line.startswith('|') and not line.startswith('|ERROR'):
                if line.startswith('|WARNING'):
                    raise ValueError('There was an error during the calculations, check the results')
                else:
                    data = {'residues': [], 'values': []}
                    header_residues = line.split('|')
                    data['residues'] = header_residues[1:-1]
                
            elif line.startswith('0') or line.startswith('1'):
                try:
                    data['values'] = [int(char) for char in line[:-1]]
                    ifps.append(data)
                except ValueError as ve:
                    pdb.set_trace()

    out_data = {}
    final_res = []
    for d in ifps:
        for res in d['residues']:
            if res not in final_res:
                final_res.append(res)
                for code in interactions:
                    out_data[f'{res} {code}'] = np.zeros(len(ifps))

    ifp_len = len(interactions)
    for ifp_pos, d in enumerate(ifps):
        for i, res in enumerate(d['residues']):
            bits = d['values'][i*ifp_len:(i+1)*ifp_len]
            for j, code in enumerate(interactions):
                    out_data[f'{res} {code}'][ifp_pos] = bits[j]

    return pd.DataFrame.from_dict(data = out_data)
        
recognized_interactions = ['HYD',
                               'FTF',
                               'ETF',
                               'HBD',
                               'HBA',
                               'CAT',
                               'ANI',
                               'PCI',
                               'MCO']

def filter_interactions(fingerprint, interactions):
    '''
    Filters the interaction type selecting only the bit associated
    to the selected interaction types.

    :param fingerprint: table containing IFPs
    :type fingerprint: pandas DataFrame
    :param interactions: interaction(s) to select from the IFP
    :type interactions: str of list of str
    :returns: filtered IFPs
    :rtype: pandas DataFrame
    :raises :class:'ValueError': Should the interaction not being
                                    between the recognized ones
    '''

    if not isinstance(interactions, list):
        interactions = [interactions]
        
    
    inter_columns = list()

    for interaction in interactions:
        if interaction not in recognized_interactions:
            raise ValueError(f'Unrecognized interaction {interaction}')

    for index in fingerprint.columns:
        if index[-3:] in interactions:
            inter_columns.append(index)

    return fingerprint[inter_columns]

def filter_residues(fingerprint, residues):
    '''
    Filters the interaction type selecting only the bit associated
    with the selected residues.
    residues are indicated by the one letter code for the aminoacid and their
    sequence number

    :param fingerprint: table containing IFPs
    :type fingerprint: pandas DataFrame
    :param residues: residue(s) to select from the IFP
    :type residues: str of list of str
    :returns: filtered IFPs
    :rtype: pandas DataFrame
    '''
    if not isinstance(fingerprint, pd.DataFrame):
        raise TypeError('The given fingerprint should be a DataFrame instance')
    singles = residues.split(',')
    res = []
    for s in singles:
        if '-' in s:
            min_res, max_res = s.split('-')
            for i in range(int(min_res), int(max_res)+1):
                res.append(str(i))
        else:
            res.append(s)
    columns = list()
    for bit in fingerprint.columns:
        if re.search('[0-9]+', bit).group() in residues:
            columns.append(bit)

    return fingerprint[columns]

def filter_ifp(fingerprint, residues, interactions):
    '''
    Filters the interaction type selecting only the bit associated to the
    selected residues and interactions.
    residues are indicated by the one letter code for the aminoacid and their
    sequence number

    :param fingerprint: table containing IFPs
    :type fingerprint: pandas DataFrame
    :param residues: residue(s) to select from the IFP
    :type residues: str of list of str
    :returns: filtered IFPs
    :rtype: pandas DataFrame
    :raises :class:'ValueError': Should the interaction not being between the
    recognized ones
    '''
    if not isinstance(fingerprint, pd.DataFrame):
        raise TypeError('The given fingerprint should be a DataFrame instance')

    for interaction in interactions:
        if interaction not in recognized_interactions:
            raise ValueError(f'Unrecognized interaction {interaction}')

    singles = residues.split(',')
    res = []
    for s in singles:
        if '-' in s:
            min_res, max_res = s.split('-')
            for i in range(int(min_res), int(max_res)+1):
                res.append(str(i))
        elif s != '':
            res.append(s)
    columns = []
    for bit in fingerprint.columns:
        if re.search('[0-9]+', bit).group() not in res and res != []:
            continue
        if bit[-3:] not in interactions and interactions != []:
            continue
        columns.append(bit)

    return fingerprint[columns]

def remove_empty_ints(fingerprint):
    '''
    Remove from the IFP interactions which are not phisically meaningfull:
    - pi-pi interactions for residues without aromatic rings
    - cations in salt bridges for residues without a positively charged sidechain
    - anion in salt bridges for residues without a negatively charged sidechain
    :param fingerprint: Ifp object to modify
    :type fingerprint: pyichem.ifp.Ifp
    '''
    aro_residues = ['H', 'F', 'W', 'Y']
    pos_residues = ['R', 'H', 'K']
    neg_residues = ['D', 'E']
    columns = []
    for bit in fingerprint.columns:
        if bit[-3:] in ['ETF', 'FTF']:
            if re.search('[A-Z][0-9]+', bit).group()[0] in aro_residues:
                columns.append(bit)
        elif bit[-3:]  == 'CAT':
            if re.search('[A-Z][0-9]+', bit).group()[0] in pos_residues:
                columns.append(bit)
        elif bit[-3:]  == 'ANI':
            if re.search('[A-Z][0-9]+', bit).group()[0] in neg_residues:
                columns.append(bit)
        else:
            columns.append(bit)

    return fingerprint[columns]

def concatenate_ifp_objects(ifps):
    '''
    Concatenate multiple Ifp objects
    :param ifps: list of IFPs to concatenate
    :type ifps: list of pyichem.ifp.Ifp
    '''
    ref_ifp_format = ifps[0].ifp_format
    if not all([fp.ifp_format == ref_ifp_format for fp in ifps]):
        raise TypeError('Cannot concatenate IFPs in different formats')

    new_ifp = Ifp(None, ref_ifp_format, None)
    new_receptor_mol2 = []
    new_ligand_mol2 = []
    for fp in ifps:
        for rmol2, lmol2 in zip(fp.receptor_mol2, fp.ligand_mol2):
            new_receptor_mol2.append(f'{fp.root}/{rmol2}')
            new_ligand_mol2.append(f'{fp.root}/{lmol2}')

    new_ifp.receptor_mol2 = np.array(new_receptor_mol2)
    new_ifp.ligand_mol2 = np.array(new_ligand_mol2)

    new_ifp.fingerprints = pd.concat(
                                    [fp.fingerprints for fp in ifps]
                                    ).fillna(0)
    new_ifp.root = ''

    return new_ifp

def split_ifp_object(input_ifp, labels):
    '''
    Split an Ifp object in n different Ifp objects
    :param input_ifp: Initial Ifp object
    :type input_file: pyichem.ifp.Ifp
    :param labels: labels indicating the groups in which the IFP is splitted
    :type labels: np.array
    '''
    if len(labels) != input_ifp.fingerprints.shape[0]:
        raise Exception('The number of labels is insufficient')

    out_ifps = {}

    for i in np.unique(labels):
        new_ifp = Ifp(None, input_ifp.ifp_format, None)
        new_ifp.root = ''
        new_ifp.receptor_mol2 = input_ifp.receptor_mol2[labels == i]
        new_ifp.ligand_mol2 = input_ifp.receptor_mol2[labels == i]
        new_ifp.fingerprints = input_ifp.fingerprints[labels == i]
        out_ifps[i] = new_ifp

    return out_ifps

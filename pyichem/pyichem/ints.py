import pandas as pd
import numpy as np
from pyichem.base_models import BatchCalculation
import sys
from mol2df import Mol2Df, Mol2Reader
from scipy.spatial.distance import squareform, pdist
import os.path
import os
import decimal
from itertools import permutations, combinations
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.spatial import distance

import pdb

INTERACTIONS_PATH='ichem_outputs/interactions'

subst = {'CENT' : ['SEC1', 'ALC2', 'LYC3', 'ASC4', 'GLC5', 'PHC6'],
        'LIG'   : ['SEL1', 'ALL2', 'LYL3', 'ASL4', 'GLL5', 'PHL6'],
        'PROT'  : ['SEP1', 'ALP2', 'LYP3', 'ASP4', 'GLP5', 'PHP6'],
        'MERG'  : ['SEC1', 'ALC2', 'LYC3', 'ASC4', 'GLC5', 'PHC6',
                    'SEL1', 'ALL2', 'LYL3', 'ASL4', 'GLL5', 'PHL6',
                    'SEP1', 'ALP2', 'LYP3', 'ASP4', 'GLP5', 'PHP6',],
        'ELEC'  : ['SEC1', 'ALC2', 'LYC3', 'ASC4', 'PHC6',
                    'SEL1', 'ALL2', 'LYL3', 'ASL4', 'PHL6',
                    'SEP1', 'ALP2','LYP3', 'ASP4', 'PHP6',],
        'HB'  : ['SEC1', 'ALC2', 'LYC3', 'ASC4'
                    'SEL1', 'ALL2', 'LYL3', 'ASL4'
                    'SEP1', 'ALP2','LYP3', 'ASP4'],
        'PL'    : ['SEL1', 'ALL2', 'LYL3', 'ASL4', 'GLL5', 'PHL6',
                    'SEP1', 'ALP2', 'LYP3', 'ASP4', 'GLP5', 'PHP6',],
        'CENT_S' : ['LYC3', 'ASC4', 'GLC5', 'PHC6', 'HBC0'],
        'LIG_S'   : ['LYL3', 'ASL4', 'GLL5', 'PHL6', 'HBL0'],
        'PROT_S'  : ['LYP3', 'ASP4', 'GLP5', 'PHP6', 'HBP0'],
        'MERG_S'  : ['LYC3', 'ASC4', 'GLC5', 'PHC6', 'HBC0',
                    'LYL3', 'ASL4', 'GLL5', 'PHL6', 'HBL0',
                    'LYP3', 'ASP4', 'GLP5', 'PHP6', 'HBP0'],
        'ELEC_S'  : ['LYC3', 'ASC4', 'PHC6', 'HBC0',
                    'LYL3', 'ASL4', 'PHL6', 'HBL0',
                    'LYP3', 'ASP4', 'PHP6', 'HBP0'],
        'PL_S'    : ['LYL3', 'ASL4', 'GLL5', 'PHL6', 'HBL0',
                    'LYP3', 'ASP4', 'GLP5', 'PHP6', 'HBP0']}

DISTANCES = {('SEL1', 'SEP1'): 3.5,
             ('ALL2', 'ALP2'): 3.5,
             ('LYL3', 'LYP3'): 4.0,
             ('ASL4', 'ASP4'): 4.0,
             ('GLL5', 'GLP5'): 4.5,
             ('PHL6', 'PHP6'): 5.0,}

SPACERS = {('SEL1', 'SEP1'): 'SES1',
             ('ALL2', 'ALP2'): 'ALS2',
             ('LYL3', 'LYP3'): 'LYS3',
             ('ASL4', 'ASP4'): 'ASS4',
             ('GLL5', 'GLP5'): 'GLS5',
             ('PHL6', 'PHP6'): 'PHS6',}

hb_simplification = {'SEC1': 'HBC0', 'ALC2': 'HBC0',
                    'SEP1': 'HBP0', 'ALP2': 'HBP0',
                    'SEL1': 'HBL0', 'ALL2': 'HBL0'}

subst_order = ['CENT', 'PROT', 'LIG', 'PL', 'ELEC', 'MERG']


class Ints(BatchCalculation):
    '''
    Class calling the Ints tool of IChem.

    WARNING !!! Metal chelation is not implemented in graph generation

    The supported definiton of IPAs during detection are available:
    MERG, IPAs placed on ligand, protein, and center
    CENT, IPAs placed in the middle point between the ligand interacting atom
            and the protein interacting atom
    PROT, IPAs placed on the protein interacting atom
    LIG, IPAs placed on the ligand iteracting atom

    From the IPAs different different interaction subgraphs can be generating,
    excluding some of the detected IPAs:
    CENT, IPAs placed in the middle point between the ligand interacting atom
            and the protein interacting atom
    PROT, IPAs placed on the protein interacting atom
    LIG, IPAs placed on the ligand iteracting atom
    ELEC, IPAs representing only polar interactions
            (hydrogen bonds, ionic bonds, pi-pi interactions)
    Pharma, IPAs placed on the ligand iteracting atom representing only polar
            interactions (hydrogen bonds, ionic bonds, pi-pi interactions)

    :param receptor_mol2: files containing the structure of the protein
    :type receptor_mol2: list of str, optional
    :param ligand_mol2: files containing the structure of the ligands
    :type ligand_mol2: list of str, optional
    :param type_int: Type of protein-ligand interaction to detect.
    :type type_int: str, optional
    :param new_hyd: Use the NewHyd definition of hydrophobic contacts rather
                    than the default one.
    :type new_hyd: bool, optional
    '''

    def __init__(
        self,
        ipa_type = 'MERG',
        newhyd = False,
        no_merge = False,
        ichem_path = 'IChem'):
        opt = f'-type {ipa_type}'
        self.ipa_info = {'ipa_type': ipa_type,
                         'newhyd': newhyd,
                         'no_merge': no_merge,}
        if newhyd:
            opt = opt + ' --newH'
            self.newhyd = True
        if no_merge:
            opt = opt + ' --noMerge'
            self.no_merge = True

        self.ipa_type = ipa_type[0]

        super().__init__(folder=INTERACTIONS_PATH,
                            output_p=f'{INTERACTIONS_PATH}/out_ints_',
                            software='ints',
                            opt = opt,
                            ichem_path = ichem_path)

    def compute_graphs(self,
                        threshold = None,
                        subgraph = 'MERG',
                        round_val = 1,
                        simplify = False,
                        mode = 'grakel',
                        filter_out = []):
        '''
        Computes interaction graphs.
        Currently the output is given only as grakel graphs.

        :param graph_type: graph format used for the outputs, currently only grakel graphs are generated
        :type graph_type: str, optional
        :param threshold: distance threshold for edge deinition, currently not implemented in this version
        :type threshold: float, optional
        :param subgraph: extract a specific subgraph from the IPA data
        :type subgraph: bool
        :param round_val: closest value to which the Euclidean distance is approximated to define the edge weight
        :type round_val: float
        :param simplify: simplify the description of hydrogen bonds by treating hydrogen bond donor and acceptors as the same interaction type.
        :type simplify: bool
        :returns: Array containing the generated graphs
        :rtype: numpy array
        '''
        graphs = list()
        if simplify:
            subgraph = subgraph+'_S'
        graph_function = GENERATOR_FUNCTION[mode](subgraph)
        
        for file in self.get_output_files():
            g = graph_generator(file,
                    graph_function,
                    threshold = threshold,
                    subgraph = subgraph,
                    round_val = round_val,
                    simplify = simplify,
                    filter_out = filter_out)
            if g:
                graphs.append(g)
        if mode == 'grakel':
            graphs = np.array(graphs)
        return graphs

    def compute_graphs_batch(self,
                        threshold = None,
                        subgraph = 'MERG',
                        round_val = 1,
                        simplify = False,
                        mode = 'grakel',
                        filter_out = [],
                        batch_size = 100):
        graphs = []
        if simplify:
            subgraph = subgraph+'_S'
        graph_function = GENERATOR_FUNCTION[mode](subgraph)
        
        for i,file in enumerate(self.get_output_files()):
            g = graph_generator(file,
                    graph_function,
                    threshold = threshold,
                    subgraph = subgraph,
                    round_val = round_val,
                    simplify = simplify,
                    filter_out = filter_out)
            if g:
                graphs.append(g)

            if len(graphs) == batch_size:
                if mode == 'grakel':
                    graphs = np.array(graphs)
                yield graphs
                graphs = []

        if len(graphs) != 0:
            if mode == 'grakel':
                graphs = np.array(graphs)
            yield graphs

    def get_output_files(self):
        return [f'{self.root}/{file}_INTS_{self.ipa_type}.mol2' for file in self.output_location]

def graph_reader(file, subgraph = None, simplify = False):
    '''
    Function converting the IPAs in the .mol2 file to a list of labels and a distance matrix.
    :param subgraph: extract a specific subgraph from the IPA data
    :type subgraph: bool
    :param round_val: closest value to which the Euclidean distance is approximated to define the edge weight
    :type round_val: float
    :param simplify: simplify the description of hydrogen bonds by treating hydrogen bond donor and acceptors as the same interaction type.
    :type simplify: bool
    :returns: the upper triangular distance matrix as a 1D array, the list of node labels
    :rtype: numpy array, list
    '''
    if os.path.isfile(file):
        pmol = Mol2Reader().read_mol2(file, atom_only = True)[0]
        if simplify and pmol.atom_df.shape[0] > 0:
            pmol.atom_df = pmol.atom_df.replace(hb_simplification)
            #subgraph = subgraph+'_S'
        sg = subst.get(subgraph)
        if subgraph in subst:
            pmol.atom_df = pmol.atom_df.query("res_name in @sg")
        elif subgraph is not None:
            raise ValueError(f'The subgraph type {subgraph} not supported')
        labels_list = pmol.atom_df['res_name'].tolist()
        pos = pmol.atom_df[['x', 'y', 'z']].to_numpy(dtype = float)
    else:
        raise Exception(f'File {file} does not exist')
    return pos, labels_list

class IGGrakel():
    '''
    Generate interaction graphs compatible with the grakel python module
    :param subgraph: Indicate which positions should be considered for graph generation
    :param type: str
    '''
    def __init__(self, subgraph):

        from grakel import Graph

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(subst[subgraph])
        self.graph_type = Graph

    def compute_graphs(self, pos, node_labels, threshold, round_val):
        dist = pdist(pos)
        if round_val:
            dist = np.rint(dist/round_val)*round_val
            dist[dist == 0] = round_val*0.1
        if threshold:
            dist[dist > threshold] = 0

        node_labels = self.label_encoder.transform(node_labels)
        n_labels = {i:lab for i,lab in enumerate(node_labels)}
        
        e_labels = dict()
        permutation = combinations(node_labels, 2)
        idxs = combinations(np.arange(len(node_labels)), 2)
        for n_ls, d, idx in zip(permutation, dist, idxs):
            if d > 0:
                e_labels[(idx[0], idx[1])] = ' '.join([str(n_ls[0]), str(n_ls[1]), str(d)])
                e_labels[(idx[1], idx[0])] = ' '.join([str(n_ls[1]), str(n_ls[0]), str(d)])

        return self.graph_type(
            squareform(dist),
            node_labels = n_labels,
            edge_labels = e_labels)

class IGTorch():
    '''
    Generate interaction graphs compatible with the torch_geometric python module
    :param subgraph: Indicate which positions should be considered for graph generation
    :param type: str
    '''
    def __init__(self, subgraph):
        from torch_geometric.nn import radius_graph
        from torch_geometric.data import Data
        import torch 

        self.tensor = torch.tensor
        self.torch_float = torch.float
        self.radius_graph = radius_graph
        self.data = Data

        self.label_encoder = LabelEncoder()
        base_label = self.label_encoder.fit_transform(subst[subgraph])
        self.node_feature_encoder = OneHotEncoder(sparse_output = False)
        self.node_feature_encoder.fit(np.reshape(base_label, (-1,1)))

    def compute_graphs(self, pos, node_labels, threshold, round_val=None):
        n_labels = self.label_encoder.transform(node_labels)
        node_features = self.node_feature_encoder.transform(np.reshape(n_labels, (-1, 1)))
        x = self.tensor(node_features,dtype=self.torch_float)
        dist = pdist(pos)
        pos = self.tensor(pos,dtype=self.torch_float)
        edge_index = [[], []]
        edge_weight = []
        for d, (i, j) in zip(dist, combinations(np.arange(len(node_labels)), 2)):
            l_i = node_labels[i]
            l_j = node_labels[j]
            if DISTANCES.get((l_i, l_j)):
                th = DISTANCES.get((l_i, l_j))
            elif DISTANCES.get((l_j, l_i)):
                th = DISTANCES.get((l_j, l_i))
            elif l_i[2] == l_j[2]:
                th = threshold
            else:
                continue
            if d <= th:
                edge_index[0].append(i)
                edge_index[1].append(j)
                edge_index[0].append(j)
                edge_index[1].append(i)
                
                edge_weight.append(th-d)
                edge_weight.append(th-d)
        edge_index = self.tensor(edge_index)
        edge_weight = self.tensor(edge_weight,dtype=self.torch_float)

        return self.data(x=x, edge_index=edge_index, pos=pos, edge_weight=edge_weight)

    def compute_graphs_only_threshold(self, pos, node_labels, threshold, round_val):
        n_labels = self.label_encoder.transform(node_labels)
        node_features = self.node_feature_encoder.transform(np.reshape(n_labels, (-1, 1)))
        x = self.tensor(node_features,dtype=self.torch_float)
        pos = self.tensor(pos,dtype=self.torch_float)
        edge_index = self.radius_graph(pos, r = threshold)
        edge_weight = list()
        for e_i, e_j in zip(edge_index[0], edge_index[1]):
            edge_weight.append([distance.euclidean(pos[e_i], pos[e_j])])
        edge_weight = self.tensor(edge_weight,dtype=self.torch_float)
        
        return self.data(x=x, edge_index=edge_index, pos=pos, edge_weight=edge_weight)

class IGNetworkX():
    '''
    Generate interaction graphs compatible with the networkx python module
    :param subgraph: Indicate which positions should be considered for graph generation
    :param type: str
    '''
    def __init__(self, subgraph):
        import networkx as nx 

        self.graph = nx.Graph
        self.set_attribute = nx.set_node_attributes

        self.label_encoder = LabelEncoder()
        base_label = self.label_encoder.fit_transform(subst[subgraph])

    def compute_graphs(self, pos, node_labels, threshold, round_val):
        n_labels = self.label_encoder.transform(node_labels)
        n_labels = {i: nl for i, nl in enumerate(n_labels)}
        dist = pdist(pos)
        edge_index = []

        for d, (i, j) in zip(dist, combinations(np.arange(len(node_labels)), 2)):
            l_i = node_labels[i]
            l_j = node_labels[j]
            if DISTANCES.get((l_i, l_j)):
                th = DISTANCES.get((l_i, l_j))
            elif DISTANCES.get((l_j, l_i)):
                th = DISTANCES.get((l_j, l_i))
            elif l_i[2] == l_j[2]:
                th = threshold
            else:
                continue
            if d <= th:
                edge_index.append((i,j))
                edge_index.append((j,i))

        out_graph = self.graph(edge_index)
        self.set_attribute(out_graph, n_labels, name = 'label')
                
        return out_graph

class IGCCKNetworkX():
    '''
    Generate interaction graphs compatible with the networkx python module.
    Distances are represented by introducing spacer nodes between the original interaction nodes.
    :param subgraph: Indicate which positions should be considered for graph generation
    :param type: str
    '''
    def __init__(self, subgraph):
        import networkx as nx

        self.graph = nx.Graph
        self.set_node_attribute = nx.set_node_attributes
        self.set_edge_attribute = nx.set_edge_attributes

        self.label_encoder = LabelEncoder()
        base_label = self.label_encoder.fit_transform(subst['PL']+['s'])

    def compute_graphs(self, pos, node_labels, threshold, round_val):
        #n_labels = self.label_encoder.transform(node_labels)
        n_labels=node_labels
        n_labels = {i: nl for i, nl in enumerate(n_labels)}
        dist = pdist(pos)
        edge_index = []

        n_nodes = len(n_labels)-1

        for d, (i, j) in zip(dist, combinations(np.arange(len(node_labels)), 2)):
            l_i = node_labels[i]
            l_j = node_labels[j]
            if DISTANCES.get((l_i, l_j)):
                th = DISTANCES.get((l_i, l_j))
                spacer = SPACERS.get((l_i, l_j))
            elif DISTANCES.get((l_j, l_i)):
                th = DISTANCES.get((l_j, l_i))
                spacer = SPACERS.get((l_j, l_i))
            elif l_i[2] == l_j[2]:
                th = threshold
                spacer = f'{l_i[2]}S'
            else:
                continue
            if d <= th:
                origin = i
                for s in np.arange(0,round(th+1e-5,0)+1):
                    if round(d, 0) <= s:
                        edge_index.append((origin,j))
                        edge_index.append((j,origin))
                        break
                    else:
                        n_nodes += 1
                        #n_labels[n_nodes] = self.label_encoder.transform(['s'])[0]
                        n_labels[n_nodes] = spacer
                        edge_index.append((origin,n_nodes))
                        edge_index.append((n_nodes,origin))
                        origin = n_nodes

        out_graph = self.graph(edge_index)
        self.set_node_attribute(out_graph, n_labels, name = 'label')
        self.set_edge_attribute(out_graph, {i: 'i' for i in (edge_index)}, name = 'label')
                
        return out_graph


GENERATOR_FUNCTION = {'grakel': IGGrakel,
                        'torch': IGTorch,
                        'networkx': IGNetworkX,
                        'CCK': IGCCKNetworkX}

def graph_generator(file,
                    function,
                    threshold = None,
                    subgraph = None,
                    round_val = None,
                    simplify = False,
                    filter_out = []):
    '''
    Generates an interaction graph from a single .mol2 containing IPAs.
    Three different type of graphs are currently supported:
    node, only atom labels are set and edges are weighted
    edge, only edge labels are set
    node_edge, both node and edge labels are set 

    :param file: .mol2 file containing the IPAs location and types
    :type file: str
    :param mode: Describe the position of the labels in the graph
    :type mode: str
    :param threshold: distance threshold for edge deinition, currently not implemented in this version
    :type threshold: float, optional 
    :param subgraph: extract a specific subgraph from the IPA data
    :type subgraph: bool
    :param round_val: closest value to which the Euclidean distance is approximated to define the edge weight
    :type round_val: float
    :param simplify: simplify the description of hydrogen bonds by treating hydrogen bond donor and acceptors as the same interaction type.
    :type simplify: bool
    :param graph_type: graph format used for the outputs, currently only grakel graphs are generated
    :type graph_type: str, optional
    :returns: An interaction graph with the desired characteristics
    :rtype: graph
    '''
    
    pos, labels = graph_reader(file, subgraph, simplify)
    for fo in filter_out:
        if fo in labels:
            return
    return function.compute_graphs(pos, labels, threshold, round_val)
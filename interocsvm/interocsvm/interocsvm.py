import numpy as np
import pdb
import logging
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist

class InteractionOCSVM():
    '''
    Base class containing common features of different methods combining
    interaction similarity and OCSVM model for docking pose classification.
    :param name: Name of the model
    :type data: str
    :param ig_info: Information regarding the interactions for model training
    :type ig_info: dict
    :param ipa_info: Information regarding the interaction pseudo atoms (IPA)
                     used to train the model,if the model is trained on IFP
                     this value is set to None
    :type ipa_info: dict
    :param trainer_info: Information on the used training heuristic
    :type trainer_info: dict
    :param ocsvm: OneClassSVM object from sklearn
    :type ocsvm: object
    '''
    def __init__(self, name, ig_info, ipa_info, trainer_info, ocsvm):
        self.trainer_info = trainer_info
        self.ig_info = ig_info
        self.ipa_info = ipa_info
        self.ocsvm = ocsvm

    def classify(self, inter):
        '''
        Give the class (1=inlier, 0=outlier) to the evalauted interactions.
        :param inter: interactions (IFPs or IGs) to be evalauted by the model
        :type inter: list or np.array
        '''
        scores = self.score(inter)
        if not scores:
            return
        results = np.zeros(len(scores))
        results[scores > 0] = 1
        return results

    def score(self, inter):
        '''
        Score the evalauted interactions as inliers (score > 0)
        or outliers (score < 0)
        :param inter: interactions (IFPs or IGs) to be evalauted by the model
        :type inter: list or np.array
        '''
        pass

    def check_graphs(self, ig_info, ipa_info):
        '''
        Compare the interactions given for inference with the interactions
        used to train the model.
        This function assures that only interactions obtained in similar
        conditions as the training set are evalauted by the model.
        :param ig_info: Information regarding the interactions used to train
                        the model
        :type ig_info: dict
        :param ipa_info: Information regarding the IPAs used for training
        :type ipa_info: dict
        '''
        error_message = 'The evalauted Interactions are not compatible with \
                            the model, evaluation will be skipped'
        if self.ipa_info != ipa_info:
            if ig_info['subgraph'] != 'ELEC' or self.ig_info['subgraph'] != 'ELEC':
                for key, val in self.ipa_info.items():
                    if val != ipa_info.get(key):
                        error_message += f'\n{key}: {val} {ipa_info.get(key)}'
                logging.error(error_message)
                return False
        if self.ig_info != ig_info:
            for key, val in self.ig_info.items():
                    if val != ig_info.get(key):
                        error_message += f'\n{key}: {val} {ig_info.get(key)}'
            logging.error(error_message)
            return False

        return True


class IGOCSVM(InteractionOCSVM):
    '''
    Interaction Graph (IG) OCSVM model.
    Model based on an IG representation of protein-ligand interactions.
    Method first described in 10.1186/s13321-022-00654-z
    :param name: Name of the model
    :type data: str
    :param ig_info: Information regarding the interactions used for training
    :type ig_info: dict
    :param ipa_info: Information regarding the interaction pseudo atoms (IPA)
    :type ipa_info: dict
    :param trainer_info: Information concenrning the used training heuristic
    :type trainer_info: dict
    :param kernel: Graph kernel used to measure the similarity between graphs
    :type kernel: graph kernel
    :param ocsvm: OneClassSVM object from sklearn
    :type ocsvm: object
    '''
    def __init__(
            self,
            name,
            ig_info,
            ipa_info,
            trainer_info,
            kernel,
            ocsvm):
        norm = 'Norm' if trainer_info['Normalization'] else 'NotNorm'
        self.description = '_'.join([name,
                                    trainer_info['Kernel'].replace(' ', ''),
                                    norm,
                                    trainer_info['Training method'],
                                    str(trainer_info['Nu'])[2:5],])
        self.kernel = kernel
        super().__init__(name, ig_info, ipa_info, trainer_info, ocsvm)

    def score(self, graphs, ig_info, ipa_info):
        '''
        Score the evalauted interactions as inliers (score > 0)
        or outliers (score < 0)
        :param graphs: interaction graphs (IG) to be evalauted by the model
        :type graphs: list or np.array
        :param ig_info: Information regarding the interactions to evaluate
        :type ig_info: dict
        :param ipa_info: Information regarding the IPAs to evalaute
        :type ipa_info: dict
        '''
        if not self.check_graphs(ig_info, ipa_info):
            return
        dist = self.kernel.transform(graphs)
        dist[np.isnan(dist)] = 0
        return self.ocsvm.decision_function(dist)

class IFPOCSVM(InteractionOCSVM):
    '''
    Interaction FingerPrint (IFP) OCSVM model.
    Model based on an IFP representation of protein-ligand interactions.
    :param name: Name of the model
    :type data: str
    :param ifp_info: Information regarding the IFPs used to train the model
    :type ifp_info: dict
    :param trainer_info: Information concenrning the used training heuristic
    :type trainer_info: dict
    :param training_points: Original IFP used to train the model
    :type training_points: array
    :param ocsvm: OneClassSVM object from sklearn
    :type ocsvm: object
    '''
    def __init__(
            self,
            name,
            ifp_info,
            trainer_info,
            training_points,
            ocsvm):
        self.description = '_'.join([name,
                                    trainer_info['Metric'].replace(' ', ''),
                                    trainer_info['Training method'],
                                    str(trainer_info['Nu'])[2:5],])
        self.training_points = training_points
        super().__init__(name, ifp_info, None, trainer_info, ocsvm)

    def score(self, ifps):
        '''
        Score the evalauted interactions as inliers (score > 0)
        or outliers (score < 0)
        :param ifps: interaction fingeprints (IFP) to be evalauted by the model
        :type ifps: array
        '''
        if self.trainer_info['Metric'] == 'cosine':
            data = normalize(ifps)
        else:
            data = 1-cdist(ifps, self.training_points,
                            metric = self.trainer_info['Metric'])
        return self.ocsvm.decision_function(data)

class KIFPOCSVM(InteractionOCSVM):
    '''
    Interaction FingerPrint (IFP) OCSVM model.
    Model based on an IFP representation of protein-ligand interactions.
    Unlike IFPOCSVM, which calculates the Gram matrix to perform
    classification, this implementation uses one of the built-in kernel of 
    OneClassSVM from sklearn (‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’)
    :param name: Name of the model
    :type data: str
    :param ifp_info: Information regarding the IFPs used to train the model
    :type ifp_info: dict
    :param trainer_info: Information concenrning the used training heuristic
    :type trainer_info: dict
    :param ocsvm: OneClassSVM object from sklearn 
    :type ocsvm: object
    '''
    def __init__(
            self,
            name,
            ifp_info,
            trainer_info,
            ocsvm):
        self.description = '_'.join([name,
                                    trainer_info['Kernel'].replace(' ', ''),
                                    trainer_info['Training method'],
                                    str(trainer_info['Nu'])[2:5],])
        super().__init__(name, ifp_info, None, trainer_info, ocsvm)

    def score(self, ifps):

        '''
        Score the evalauted interactions as inliers (score > 0)
        or outliers (score < 0)
        :param ifps: IFPs to be evalauted by the model
        :type ifps: array
        '''
        
        return self.ocsvm.decision_function(ifps)


class DIGOCSVM(InteractionOCSVM):
    '''
    Deep Interaction Graph (DIG) OCSVM model.
    The graph is represented by a single vector obtained after the use of a 
    graph autoencoder,and an adequate pooling function.
    :param name: Name of the model
    :type data: str
    :param ig_info: Information regarding the interactions used for training
    :type ig_info: dict
    :param ipa_info: Information regarding the IPAs used for model training
    :type ipa_info: dict
    :param trainer_info: Information concenrning the used training heuristic
    :type trainer_info: dict
    :param auto_encoder: Graph autoencoder used to encode the interactions.
    :type auto_encoder: object
    :param ocsvm: OneClassSVM object from sklearn
    :type ocsvm: object
    :param pooling_function: Pooling function to obtain a single vector
                                representation of the graph.
    :type pooling_function: object
    '''
    def __init__(
            self,
            name,
            ig_info,
            ipa_info,
            trainer_info,
            auto_encoder,
            ocsvm,
            pooling_function):
        from torch_geometric.loader import DataLoader
        self.description = '_'.join([name,
                                    trainer_info['Kernel'].replace(' ', ''),
                                    trainer_info['Training method'],
                                    str(trainer_info['Nu'])[2:5]])
        self.auto_encoder = auto_encoder
        self.pooling_function = pooling_function
        self.dataloader_func = DataLoader
        super().__init__(name, ig_info, ipa_info, trainer_info, ocsvm)

    def score(self, graphs, ig_info, ipa_info):
        '''
        Score the evalauted interactions as inliers (score > 0)
        or outliers (score < 0)
        :param ig_info: Information regarding the interactions to evaluate
        :type ig_info: dict
        :param ipa_info: Information regarding the IPAs to evalaute
        :type ipa_info: dict
        '''
        if not self.check_graphs(ig_info, ipa_info):
            return
        embs = []
        node_loss = []
        edge_loss = []
        self.auto_encoder.eval()

        dataloader = self.dataloader_func(graphs, batch_size = 500)

        for batch in dataloader:
            e = self.auto_encoder.gae.encode(batch.cuda(0))
            embs.append(
                self.pooling_function(e,
                                    batch.batch).cpu().detach().numpy()
                )
        return self.ocsvm.decision_function(np.concatenate(embs))

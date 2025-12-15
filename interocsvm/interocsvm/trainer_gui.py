import tkinter as tk
from tkinter import ttk
import pdb
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)
import sys
import numpy as np
import os
import logging

from sklearn.manifold import MDS, Isomap
from sklearn.svm import OneClassSVM as OCSVM
import matplotlib.pyplot as plt
import yaml
from scipy.stats import median_abs_deviation
import pickle
from interocsvm.qms import Qms1, Qms2, plot_knee_normalized, QmsDBSCAN
from interocsvm.mad_knn import MadKnn, plot_madknn
from interocsvm.interocsvm import IGOCSVM, DIGOCSVM, IFPOCSVM, KIFPOCSVM
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import polynomial_kernel, sigmoid_kernel, rbf_kernel, linear_kernel

training_methods = ('QMS2', 'QMS1', 'MAD', 'SetNu', 'DBSCAN')
COLORS = ['b', 'g']

def get_graphkernel_gui(g, name, ref_ipa_info, ref_ig_info):
    root = TrainerGraphKernel(g, name, ref_ipa_info, ref_ig_info)
    root.geometry("1200x750")
    root.columnconfigure(0,weight=1, minsize = 600)
    root.columnconfigure(1,weight=1, minsize = 600)
    root.rowconfigure(0,minsize=700)
    root.title('IG-OCSVM trainer')

    return root

def get_vector_gui(ifp, name, ref_ifp_info):
    root = TrainerVector(ifp, name,ref_ifp_info)
    root.geometry("1200x750")
    root.columnconfigure(0,weight=1, minsize = 600)
    root.columnconfigure(1,weight=1, minsize = 600)
    root.rowconfigure(0,minsize=700)
    root.title('IFP-OCSVM trainer')

    return root

def get_kernelvector_gui(ifp, name, ref_ifp_info):
    root = TrainerVectorKernel(ifp, name,ref_ifp_info)
    root.geometry("1200x750")
    root.columnconfigure(0,weight=1, minsize = 600)
    root.columnconfigure(1,weight=1, minsize = 600)
    root.rowconfigure(0,minsize=700)
    root.title('Kernel-IFP-OCSVM trainer')

    return root

def get_ae_gui(g, name, ref_ipa_info, ref_ig_info, autoencoder):
    root = TrainerAE(g, name, ref_ipa_info, ref_ig_info, autoencoder)
    root.geometry("1200x750")
    root.columnconfigure(0,weight=1, minsize = 600)
    root.columnconfigure(1,weight=1, minsize = 600)
    root.rowconfigure(0,minsize=700)
    root.title('DIG-OCSVM trainer')

    return root

class TrainerGUI(tk.Tk):
    def __init__(self, graphs, name, ig_info, ipa_info):
        super().__init__()
        self.description = name
        self.ig_info = ig_info
        self.ipa_info = ipa_info
        self.graphs = graphs
        self.pooling_method = tk.StringVar(self)
        self.kernel_type = tk.StringVar(self)
        self.kernel_norm = tk.BooleanVar(self, True)
        self.create_widgets()

    def create_widgets(self):

        #Frame selecting the type of kernel and the projection of similarity
        self.frm_sim = tk.Frame(master=self,)
        self.frm_sim.grid(row=0, column=0)
        self.frm_sim.rowconfigure(0,minsize = 30, pad = 5)
        self.frm_sim.rowconfigure(1,minsize = 30, pad = 5)
        self.frm_sim.rowconfigure(2,minsize = 30, pad = 5)
        self.frm_sim.rowconfigure(3,minsize = 500, pad = 5)
        self.frm_sim.rowconfigure(4,minsize = 110, pad = 5)
        self.display_sim()

        self.frm_train = tk.Frame(master=self,)
        self.frm_train.grid(row=0, column=1)
        self.frm_train.rowconfigure(0,minsize = 30, pad = 5)
        self.frm_train.rowconfigure(3,minsize = 60, pad = 5)
        self.frm_train.rowconfigure(2,minsize = 130, pad = 5)
        self.frm_train.rowconfigure(1,minsize = 480, pad = 5)
    
    def display_sim(self):
        pass

    def _compute_sim(self):
        pass

    def compute_sim(self):
        self.outlier_mask = np.array(['b' for _ in range(len(self.graphs))])
        self._compute_sim()
        self.plot_sim()
        self.display_training()

    def display_training(self, *args):
        self.training_method = tk.StringVar(self)
        self.nu = tk.StringVar(self, '0.010')
        for widget in self.frm_train.winfo_children():
            widget.destroy()
        label_tm = tk.Label(self.frm_train, text="Select training method")
        label_tm.grid(row=0, column=0)
        training_menu = ttk.OptionMenu(self.frm_train,
                                       self.training_method,
                                       None,
                                       *training_methods,
                                       command=self.training_opt)
        training_menu.grid(row=0, column=1)
        button_t = ttk.Button(self.frm_train, text='Train model', command=self.train_model)
        button_t.grid(row=3,column=0, columnspan=2)
        #insert thing for matplotlib plot
        knee = Figure(figsize = (5, 5),
                 dpi = 100)
        knee.tight_layout(pad = 5)
        self.canvas_knee = FigureCanvasTkAgg(
                                knee,
                                master = self.frm_train)
        self.canvas_knee.draw()
        self.canvas_knee.get_tk_widget().grid(row=1, column=0, columnspan=2)
        self.ax_knee = knee.add_subplot()

    def training_opt(self, *args):
        for widget in self.frm_train.winfo_children():
            if str(type(widget)) == "<class 'tkinter.Frame'>":
                widget.destroy()
        tmp_frame = tk.Frame(self.frm_train)
        tmp_frame.grid(row=2, column=0, columnspan=2)

        if not hasattr(self, 'grham_matrix'):
            label=ttk.Label(tmp_frame,
                            text = 'Compute the kernel before selecting the training method').grid(row=7, column=0)
            return
        
        if self.training_method.get() in ['QMS2', 'DBSCAN']:
            self.display_qms(tmp_frame, self.training_method.get())
        elif self.training_method.get() == 'MAD':
            self.display_mad(tmp_frame)
            return
        elif self.training_method.get() == 'QMS1':
            self.display_qms(tmp_frame, self.training_method.get())
            return
        else:
            self.outlier_mask = np.array(['b' for _ in range(len(self.graphs))])
            self.plot_sim()
            self.ax_knee.clear()
            self.canvas_knee.draw()
        
        label=ttk.Label(tmp_frame, text = 'Set nu value').grid(row=6, column=0)
        spinbox = ttk.Spinbox(tmp_frame,from_=0,to=1,textvariable=self.nu, increment=0.001)
        spinbox.grid(row=6, column=1)

    def display_qms(self, tmp_frame, method):
        interp_method = tk.StringVar(self, 'interp1d')
        knee_sensibility = tk.IntVar(self, 0)
        polynomial_degree = tk.IntVar(self, 7)

        label=ttk.Label(tmp_frame, text = 'Sensibility').grid(row=3, column=0)
        spinbox = tk.Scale(tmp_frame,
                            from_=0,
                            to=10,
                            command=lambda args: self.kneedle(knee_sensibility.get(),
                                                                interp_method.get(),
                                                                polynomial_degree.get()),
                            variable=knee_sensibility,
                            showvalue=True,
                            orient='horizontal',
                            resolution=1)
        spinbox.grid(row=3, column=1)
        label_2=ttk.Label(tmp_frame,
                            text = 'Interpolation method').grid(row=4, column=0)
        interp_menu = ttk.OptionMenu(tmp_frame,
            interp_method,
            'interp1d',
            'interp1d','polynomial',
            command=lambda args: self.kneedle(knee_sensibility.get(),
                                                interp_method.get(),
                                                polynomial_degree.get())).grid(row=4, column=1)
        ttk.Label(tmp_frame,
                    text = 'Polynomial degree').grid(row=5, column=0)
        tk.Scale(tmp_frame,
            from_=2,to=12,
            variable=polynomial_degree,
            showvalue=True,
            orient='horizontal',
            resolution=1,
            command=lambda args: self.kneedle(knee_sensibility.get(),
                                                interp_method.get(),
                                                polynomial_degree.get())).grid(row=5, column=1)
        if method == 'QMS2':
            self.trainer = Qms2(self.grham_matrix, self.mode)
        elif method == 'DBSCAN':
            self.trainer = QmsDBSCAN(self.grham_matrix, self.mode)
        elif method == 'QMS1':
            self.trainer = Qms1(self.grham_matrix, self.mode)
        self.kneedle(knee_sensibility.get(), interp_method.get(), polynomial_degree.get())

    def display_mad(self, tmp_frame):
        if not self.kernel_norm.get():
            ttk.Label(tmp_frame,
                text = 'The MAD heuristic is incompatible with non-normalized kernels').grid(row=2, column=0)
            self.ax_knee.clear()
            self.canvas_knee.draw()
            return
        self.trainer = MadKnn(self.grham_matrix)
        mask = self.trainer.fit()
        plot_madknn(self.ax_knee, mask, self.trainer.vals)
        self.canvas_knee.draw()
        self.outlier_mask = np.array(['b' for _ in range(len(self.graphs))])
        self.outlier_mask[mask] = 'r'
        self.plot_sim()

    def plot_sim(self):
        self.ax_proj.clear()
        self.ax_proj.set_xticks([])
        self.ax_proj.set_yticks([])
        self.ax_proj.scatter(self.mds_coords[:,0], self.mds_coords[:,1], c = self.outlier_mask)
        f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
        self.ax_proj.legend([f('s', 'b'), f('s', 'r')],['Inliers', 'Outliers'])
        self.canvas_proj.draw()

    def train_model(self):
        pass

    def kneedle(self, sensibility, interp_method, polynomial_degree):
        if not hasattr(self, 'trainer'):
            return
        mask = self.trainer.fit(find_n = True,
                                direction = self.direction,
                                curve = self.concavity,
                                S_knee = sensibility,
                                interp_method = interp_method,
                                polynomial_degree = polynomial_degree)
        self.ax_knee.clear()
        plot_knee_normalized(self.ax_knee, self.trainer.knee_finder)
        self.canvas_knee.draw()
        self.outlier_mask = np.array(['r' for _ in range(len(self.graphs))])
        for i, c in enumerate(COLORS):
            self.outlier_mask[mask == i] = c
        self.plot_sim()

class TrainerGraphKernel(TrainerGUI):
    def __init__(self, graphs, name, ig_info, ipa_info):
        self.direction = 'increasing'
        self.concavity = 'concave'
        self.mode = 'similarity'
        from grakel import ShortestPath, EdgeHistogram, WeisfeilerLehman
        self.grakel_kernel = {'Shortest Path': ShortestPath,
                                'Edge Histogram': EdgeHistogram,
                                'Vertex WL': WeisfeilerLehman}
        super().__init__(graphs, name, ig_info, ipa_info)
        

    def display_sim(self):
        kernels=list(self.grakel_kernel.keys())
        label_km = tk.Label(self.frm_sim,
                            text="Select the type of kernel")
        label_km.grid(row=0, column=0)
        kernel_menu = ttk.OptionMenu(self.frm_sim, self.kernel_type, kernels[0],*kernels)
        kernel_menu.grid(row=0, column=1)
        label_kn = tk.Label(self.frm_sim,
                            text="Normalize kernel")
        label_kn.grid(row=1, column=0)
        r1 = ttk.Radiobutton(self.frm_sim,
                            text='True',
                            value=True,
                            variable=self.kernel_norm)
        r1.grid(row=1, column=1)
        r2 = ttk.Radiobutton(self.frm_sim,
                            text='False',
                            value=False,
                            variable=self.kernel_norm)
        r2.grid(row=1, column=2)

        proj = Figure(figsize = (5, 5),
                 dpi = 100)
        proj.tight_layout()
        plt.tight_layout()
        self.canvas_proj = FigureCanvasTkAgg(proj,
                               master = self.frm_sim)
        self.canvas_proj.draw()
        self.canvas_proj.get_tk_widget().grid(row=3, column=0, columnspan=2)
        self.ax_proj = proj.add_subplot()
        self.ax_proj.set_xticks([])
        self.ax_proj.set_yticks([])

        button = ttk.Button(self.frm_sim,
                            text='Run kernel',
                            command=self.compute_sim)
        button.grid(row=2,column=0)

    def _compute_sim(self):
        if self.kernel_type.get() == 'Vertex WL':
            self.base_kernel = self.grakel_kernel[self.kernel_type.get()](normalize=self.kernel_norm.get(), n_iter = 1)
            self.dist_kernel = self.grakel_kernel[self.kernel_type.get()](normalize=self.kernel_norm.get(), n_iter = 1)
        else:
            self.base_kernel = self.grakel_kernel[self.kernel_type.get()](normalize=self.kernel_norm.get())
            self.dist_kernel = self.grakel_kernel[self.kernel_type.get()](normalize=self.kernel_norm.get())

        self.grham_matrix = self.dist_kernel.fit_transform(self.graphs)
        msa_dist = 1-self.grham_matrix if self.kernel_norm.get() else 1/self.grham_matrix
        if len(self.graphs) > 2500:
            p = 2500/len(self.graphs)
            sample = np.random.choice(a = [True, False], size = len(self.graphs), p = (p, 1-p))
        else:
            sample = [True for _ in self.graphs]
        #self.mds_coords = MDS(dissimilarity='precomputed', metric = True).fit_transform(msa_dist)
        isomap = Isomap(n_neighbors = int(len(self.graphs)*0.01),metric = 'precomputed')
        isomap.fit(msa_dist[:, sample][sample, :])
        self.mds_coords =isomap.transform(msa_dist[:, sample])

    def train_model(self):
        if not hasattr(self, 'grham_matrix') or not self.training_method.get():
            if str(type(self.frm_train.winfo_children()[-1])) == "<class 'tkinter.Label'>":
                self.frm_train.winfo_children()[-1].destroy()
            tk.Label(self.frm_train, text=f'Select a heuristic before training').grid(row=4, column=0)
            return

        nu = self.trainer.nu if self.training_method.get() in ['MAD', 'QMS1'] else float(self.nu.get())

        if self.training_method.get() == 'QMS2':
            grham_matrix = self.base_kernel.fit_transform(self.graphs[self.outlier_mask == 'b'])
            self.train_ocsvm(grham_matrix, self.base_kernel, nu)
        elif self.training_method.get() == 'DBSCAN':
            for i,color in enumerate(COLORS):
                if color not in self.outlier_mask:
                    break
                grham_matrix = self.base_kernel.fit_transform(self.graphs[self.outlier_mask == color])
                self.train_ocsvm(grham_matrix, self.base_kernel, nu, f'_{i}')

        else:
            self.train_ocsvm(self.grham_matrix, self.dist_kernel, nu)
        
    def train_ocsvm(self, grham_matrix, dist_kernel, nu, appendix = ''):
        logging.info(f'Training IG-OCSVM\nKernel: {self.kernel_type.get()} ' \
                     f'Normalization: {self.kernel_norm.get()}\n' \
                     f'Training IG-OCSVM using the {self.training_method.get()} method\nNu value set to {nu}')
        training_info = {'Kernel': self.kernel_type.get(),
                         'Normalization': self.kernel_norm.get(),
                         'Training method': self.training_method.get()+appendix,
                         'Nu': nu}

        ocsvm = OCSVM(kernel = 'precomputed', nu = nu)
        ocsvm.fit(grham_matrix)
        igocsvm = IGOCSVM(self.description, self.ig_info, self.ipa_info, training_info, dist_kernel, ocsvm)
        with open(f'{igocsvm.description}.pkl', 'wb') as out:
            pickle.dump(igocsvm, out)
        logging.info(f'IG-OCSVM saved as {igocsvm.description}.pkl')
        if str(type(self.frm_train.winfo_children()[-1])) == "<class 'tkinter.Label'>":
            self.frm_train.winfo_children()[-1].destroy()
        tk.Label(self.frm_train, text=f'Model saved as: {igocsvm.description}.pkl').grid(row=4, column=0)


class TrainerVector(TrainerGUI):
    def __init__(self, graphs, name, vector_info):
        self.direction = 'increasing'
        self.concavity = 'concave'
        self.mode = 'similarity'
        super().__init__(graphs, name, vector_info, None)
        

    def display_sim(self):
        metrics = ['jaccard', 'cosine', 'dice']
        self.metric = tk.StringVar(self)
        label_km = tk.Label(self.frm_sim,
                            text="Select the similarity measure")
        label_km.grid(row=0, column=0)
        kernel_menu = ttk.OptionMenu(self.frm_sim, self.metric, metrics[0],*metrics)
        kernel_menu.grid(row=0, column=1)

        proj = Figure(figsize = (5, 5),
                 dpi = 100)
        proj.tight_layout()
        plt.tight_layout()
        self.canvas_proj = FigureCanvasTkAgg(proj,
                               master = self.frm_sim)
        self.canvas_proj.draw()
        self.canvas_proj.get_tk_widget().grid(row=3, column=0, columnspan=2)
        self.ax_proj = proj.add_subplot()
        self.ax_proj.set_xticks([])
        self.ax_proj.set_yticks([])

        button = ttk.Button(self.frm_sim,
                            text='Calculate similarity',
                            command=self.compute_sim)
        button.grid(row=2,column=0)

    def _compute_sim(self):

        self.grham_matrix = squareform(pdist(self.graphs, metric = self.metric.get()))
        if len(self.graphs) > 2500:
            p = 2500/len(self.graphs)
            sample = np.random.choice(a = [True, False], size = len(self.graphs), p = (p, 1-p))
        else:
            sample = [True for _ in self.graphs.iterrows()]
        #self.mds_coords = MDS(dissimilarity='precomputed', metric = True).fit_transform(msa_dist)
        if len(self.graphs) <= 100:
            n = len(self.graphs)-1
        else:
            n = 2
        isomap = Isomap(n_neighbors = max(int(len(self.graphs)*0.01), 3),metric = 'precomputed')
        isomap.fit(self.grham_matrix[:, sample][sample, :])
        self.mds_coords = isomap.transform(self.grham_matrix[:, sample])
        self.grham_matrix = 1-self.grham_matrix

    def train_model(self):
        if not hasattr(self, 'grham_matrix') or not self.training_method.get():
            if str(type(self.frm_train.winfo_children()[-1])) == "<class 'tkinter.Label'>":
                self.frm_train.winfo_children()[-1].destroy()
            tk.Label(self.frm_train, text=f'Select a heuristic before training').grid(row=4, column=0)
            return

        nu = self.trainer.nu if self.training_method.get() in ['MAD', 'QMS1'] else float(self.nu.get())

        if self.training_method.get() == 'QMS2':
            self.train_ocsvm(self.grham_matrix[:, self.outlier_mask == 'b'][self.outlier_mask == 'b', :],
                             self.graphs[self.outlier_mask == 'b'],
                             nu)
        elif self.training_method.get() == 'DBSCAN':
            for i,color in enumerate(COLORS):
                if color not in self.outlier_mask:
                    break
                self.train_ocsvm(self.grham_matrix[:, self.outlier_mask == color][self.outlier_mask == color, :],
                             self.graphs[self.outlier_mask == color],
                             nu, f'_{i}')
        else:
            self.train_ocsvm(self.grham_matrix, self.graphs, nu)  

    def train_ocsvm(self, grham_matrix, graphs, nu, appendix=''):
        logging.info(f'Training IFP-OCSVM\nKernel: {self.metric.get()} ' \
                     f'Training IFP-OCSVM using the {self.training_method.get()} method\nNu value set to {nu}')
        training_info = {'Metric': self.metric.get(),
                         'Training method': self.training_method.get()+appendix,
                         'Nu': nu}
        if self.metric.get() == 'cosine':
            ocsvm = OCSVM(kernel = 'linear', nu = nu)
            ocsvm.fit(normalize(graphs))
        else:
            ocsvm = OCSVM(kernel = 'precomputed', nu = nu)
            ocsvm.fit(grham_matrix)
        igocsvm = IFPOCSVM(self.description, self.ig_info, training_info, graphs, ocsvm)
        with open(f'{igocsvm.description}.pkl', 'wb') as out:
            pickle.dump(igocsvm, out)
        logging.info(f'IG-OCSVM saved as {igocsvm.description}.pkl')
        if str(type(self.frm_train.winfo_children()[-1])) == "<class 'tkinter.Label'>":
            self.frm_train.winfo_children()[-1].destroy()
        tk.Label(self.frm_train, text=f'Model saved as: {igocsvm.description}.pkl').grid(row=4, column=0)


class TrainerVectorKernel(TrainerGUI):
    def __init__(self, graphs, name, vector_info):
        self.direction = 'increasing'
        self.concavity = 'convex'
        self.mode = 'distance'
        self.sklearn_kernels = {'linear': linear_kernel,
                                'rbf': rbf_kernel,
                                'sigmoid': sigmoid_kernel,
                                'poly': polynomial_kernel}
        super().__init__(graphs, name, vector_info, None)
        

    def display_sim(self):
        kernels=list(self.sklearn_kernels.keys())
        self.kernel = tk.StringVar(self)
        label_km = tk.Label(self.frm_sim,
                            text="Select the kernel")
        label_km.grid(row=0, column=0)
        kernel_menu = ttk.OptionMenu(self.frm_sim, self.kernel, kernels[0],*kernels)
        kernel_menu.grid(row=0, column=1)

        proj = Figure(figsize = (5, 5),
                 dpi = 100)
        proj.tight_layout()
        plt.tight_layout()
        self.canvas_proj = FigureCanvasTkAgg(proj,
                               master = self.frm_sim)
        self.canvas_proj.draw()
        self.canvas_proj.get_tk_widget().grid(row=3, column=0, columnspan=2)
        self.ax_proj = proj.add_subplot()
        self.ax_proj.set_xticks([])
        self.ax_proj.set_yticks([])

        button = ttk.Button(self.frm_sim,
                            text='Calculate similarity',
                            command=self.compute_sim)
        button.grid(row=2,column=0)

    def _compute_sim(self):

        #self.grham_matrix = squareform(pdist(self.graphs, metric = 'euclidean'))
        self.grham_matrix = 1/self.sklearn_kernels[self.kernel.get()](self.graphs)
        self.grham_matrix[self.grham_matrix == np.inf] = max(self.grham_matrix[self.grham_matrix != np.inf]) * 1.1
        if len(self.graphs) > 2500:
            p = 2500/len(self.graphs)
            sample = np.random.choice(a = [True, False], size = len(self.graphs), p = (p, 1-p))
        else:
            sample = [True for _ in self.graphs.iterrows()]
        if len(self.graphs) <= 100:
            n = len(self.graphs)-1
        else:
            n = 2
        isomap = Isomap(n_neighbors = max(int(len(self.graphs)*0.01), 3),metric = 'precomputed')
        isomap.fit(self.grham_matrix[:, sample][sample, :])
        self.mds_coords = isomap.transform(self.grham_matrix[:, sample])

    def train_model(self):
        if not hasattr(self, 'grham_matrix') or not self.training_method.get():
            if str(type(self.frm_train.winfo_children()[-1])) == "<class 'tkinter.Label'>":
                self.frm_train.winfo_children()[-1].destroy()
            tk.Label(self.frm_train, text=f'Select a heuristic before training').grid(row=4, column=0)
            return

        if self.training_method.get() == 'QMS2':
            data = self.graphs[self.outlier_mask == 'b']
        else:
            data = self.graphs
        
        nu = self.trainer.nu if self.training_method.get() in ['MAD', 'QMS1'] else float(self.nu.get())
        gamma = self.trainer.gamma if self.training_method.get() in ['QMS2'] else 'auto'

        logging.info(f'Training IG-OCSVM\nKernel: {self.kernel.get()} ' \
                     f'Training IG-OCSVM using the {self.training_method.get()} method\nNu value set to {nu}')
        training_info = {'Kernel': self.kernel.get(),
                         'Training method': self.training_method.get(),
                         'Nu': nu}

        ocsvm = OCSVM(kernel = self.kernel.get(), nu = nu, gamma = gamma)
        ocsvm.fit(data)
        igocsvm = KIFPOCSVM(self.description, self.ig_info, training_info, ocsvm)
        with open(f'{igocsvm.description}.pkl', 'wb') as out:
            pickle.dump(igocsvm, out)
        logging.info(f'KIFP-OCSVM saved as {igocsvm.description}.pkl')
        if str(type(self.frm_train.winfo_children()[-1])) == "<class 'tkinter.Label'>":
            self.frm_train.winfo_children()[-1].destroy()
        tk.Label(self.frm_train, text=f'Model saved as: {igocsvm.description}.pkl').grid(row=4, column=0)

class TrainerAE(TrainerGUI):
    def __init__(self, graphs, name, ig_info, ipa_info, auto_encoder):
        from torch_geometric.nn.pool import global_add_pool, global_max_pool
        from torch_geometric.loader import DataLoader
        self.pooling_funcs = {'Add': global_add_pool, 'Max': global_max_pool}
        self.auto_encoder = auto_encoder
        self.direction = 'increasing'
        self.concavity = 'convex'
        self.mode = 'distance'
        self.dataloader = DataLoader(graphs, batch_size = 500)
        super().__init__(graphs, name, ig_info, ipa_info)
        

    def display_sim(self):
        pool = list(self.pooling_funcs.keys())
        label_km = tk.Label(self.frm_sim,
                            text="Encode graphs using an autoencoder")
        label_km.grid(row=0, column=0)

        label_km = tk.Label(self.frm_sim,
                            text="Select the pooling method")
        label_km.grid(row=1, column=0)
        kernel_menu = ttk.OptionMenu(self.frm_sim, self.pooling_method, pool[0],*pool)
        kernel_menu.grid(row=1, column=1)

        button = ttk.Button(self.frm_sim,
                            text='Run AutoEncoder',
                            command=self.compute_sim)
        button.grid(row=2,column=0)

        proj = Figure(figsize = (5, 5),
                 dpi = 100)
        proj.tight_layout()
        plt.tight_layout()
        self.canvas_proj = FigureCanvasTkAgg(proj,
                               master = self.frm_sim)
        self.canvas_proj.draw()
        self.canvas_proj.get_tk_widget().grid(row=3, column=0, columnspan=2)
        self.ax_proj = proj.add_subplot()
        self.ax_proj.set_xticks([])
        self.ax_proj.set_yticks([])

    def _compute_sim(self):
        #embs = [self.auto_encoder(g.cuda(0)).sum(0).cpu().detach().numpy() for g in self.graphs]
        #embs = [self.auto_encoder(g.cuda(0)).max(0).values.cpu().detach().numpy() for g in self.graphs]
        #embs = [self.auto_encoder(g.cuda(0)).mean(0).cpu().detach().numpy() for g in self.graphs]
        if not hasattr(self, 'node_embs'):
            self.node_embs = [self.auto_encoder(batch.cuda(0)).cpu().detach() for batch in self.dataloader]
        embs = [self.pooling_funcs[self.pooling_method.get()](ne, batch.batch).numpy() for ne, batch in zip(self.node_embs, self.dataloader)]
        self.n_embs = np.concatenate(embs)
        self.grham_matrix = squareform(pdist(self.n_embs))
        msa_dist = self.grham_matrix
        #msa_dist[msa_dist < 0] = 0
        if len(self.graphs) > 2500:
            p = 2500/len(self.graphs)
            sample = np.random.choice(a = [True, False], size = len(self.graphs), p = (p, 1-p))
        else:
            sample = [True for _ in self.graphs]
        #self.mds_coords = MDS(dissimilarity='precomputed', metric = True).fit_transform(msa_dist)
        isomap = Isomap(n_neighbors = int(len(self.graphs)*0.01),metric = 'precomputed')
        isomap.fit(msa_dist[:, sample][sample, :])
        self.mds_coords =isomap.transform(msa_dist[:, sample])

    def train_model(self):

        if self.training_method.get() in ['QMS2', 'DBSCAN']:
            embs = []
            for c in COLORS:
                if c in self.outlier_mask:
                    embs.append(self.n_embs[self.outlier_mask == c])
        else:
            embs = [self.n_embs]
        nu = self.trainer.nu if self.training_method.get() in ['MAD', 'QMS1'] else float(self.nu.get())

        logging.info(f'Training IG-OCSVM\nAuto encoder' \
                     f'Training IG-OCSVM using the {self.training_method.get()} method\nNu value set to {nu}')
        training_info = {'Kernel': 'AutoEncoder',
                         'Normalization': None,
                         'Training method': self.training_method.get(),
                         'Nu': nu}
        for i, emb in enumerate(embs):
            ocsvm = OCSVM(kernel = 'linear', nu = nu)
            ocsvm.fit(emb)
            if len(embs) > 1:
                description = '_'.join([self.description, self.pooling_method.get(), f'clst_{i}'])
            else:
                description = '_'.join([self.description, self.pooling_method.get()])
            igocsvm = DIGOCSVM(
                                description,
                                self.ig_info,
                                self.ipa_info,
                                training_info,
                                self.auto_encoder,
                                ocsvm,
                                self.pooling_funcs[self.pooling_method.get()]
                            )
            with open(f'{igocsvm.description}.pkl', 'wb') as out:
                pickle.dump(igocsvm, out)
            logging.info(f'IG-OCSVM saved as {igocsvm.description}.sav')
            if str(type(self.frm_train.winfo_children()[-1])) == "<class 'tkinter.Label'>":
                self.frm_train.winfo_children()[-1].destroy()
            tk.Label(self.frm_train, text=f'Model saved as: {igocsvm.description}.pkl').grid(row=4, column=0)
        return

        
        
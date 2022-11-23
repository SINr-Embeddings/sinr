import matplotlib.pyplot as plt
import networkit as nk
from scipy.sparse import csr
import numpy as np
import networkx as nx
from copy import deepcopy

# Graph Import from URL
import urllib.request
import io
import zipfile


# Matrix Factorization
from tqdm.auto import tqdm
import torch
import numpy as np

import torch.nn as nn
import torch.optim as optim
import pandas as pd
import sys
from pathlib import Path

from numpy.random import choice



# Visualize
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Evaluate
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import normalized_mutual_info_score, homogeneity_score, adjusted_rand_score
from scipy.spatial.distance import cosine

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pandas as pd
from copy import deepcopy

from karateclub.node_embedding.neighbourhood import HOPE

import sys
sys.path.append("/lium/home/tprouteau/git/sinr/src/sinr/")
from sparse_nfm import get_nfm_embeddings

# Graph loading

def load_graph(url):
    sock = urllib.request.urlopen(url)  # open URL
    s = sock.read()  # read into BytesIO "file"
    sock.close()
    gml = s.decode()  # read gml data
    G = nx.parse_gml(gml)  # parse gml data

    G=nx.relabel_nodes(G, {val:idx for idx,val in enumerate(G.nodes())})

    return G

def get_membership(G, kw_membership="gt"):
    try:
        membership = [int(G.nodes[u][kw_membership]) for u in G.nodes]
    except:
        membership = [G.nodes[u][kw_membership] for u in G.nodes]
        gt_2_int = {gt:idx for idx, gt in enumerate(sorted(list(set(membership))))}
        membership = [gt_2_int[gt] for gt in membership]
    return membership

def plot_graph_coms(G, gt_membership, pos=None, with_labels=False):
    Blues = plt.get_cmap('spring')
    for_colors = [com / max(gt_membership) for com in gt_membership]
    colors=[Blues(x) for x in for_colors]
    if not pos:
        pos = nx.spring_layout(G, seed=1969)  # Seed for reproducible layout
    nx.draw(G, pos, node_color=colors, node_size=50, linewidths= 0, width= 0.1, with_labels=with_labels)
    return pos

import community
import partition_networkx

from networkx.algorithms.core import core_number
from collections import namedtuple

import sknetwork as skn
def get_ensemble_coms(G, algorithm=skn.clustering.Louvain, gamma=10, nb_run=10):
    min_weight = 0.05
    W = {k:0 for k in G.edges()}
    louvain=algorithm(modularity='newman',shuffle_nodes=True)
    ## Ensemble of level-1 Louvain 
    partition_size=[]
    for i in range(nb_run):
        adj = nx.convert_matrix.to_scipy_sparse_matrix(G)
        l = louvain.fit_transform(adj)
        partition_size.append(len(set(l)))
        for e in G.edges():
            W[e] += int(l[e[0]] == l[e[1]])
    ## vertex core numbers
    core = core_number(G)
    ## set edge weights
    for e in G.edges():
        m = min(core[e[0]],core[e[1]])
        if m > 1:
            W[e] = min_weight + (1-min_weight)*W[e]/nb_run
        else:
            W[e] = min_weight

    nx.set_edge_attributes(G, W, 'weight')
    adj = nx.convert_matrix.to_scipy_sparse_matrix(G)
    final_louvain=skn.clustering.Louvain(modularity='newman', resolution=gamma)
    part = final_louvain.fit_transform(adj)
    #print("Mean size of partition", np.mean(partition_size))
    return part
    
def get_coms(G, algorithm=nk.community.ParallelLeiden, gamma=10, nb_run=10):
    G_nk= nk.nxadapter.nx2nk(G, weightAttr=None)
    mod=nk.community.Modularity()
    best_communities=[]
    best_modularity=0
    for i in range(nb_run):
        communities = nk.community.detectCommunities(G_nk, algo=algorithm(G_nk,  gamma=gamma), inspect=False)
        communities.compact()
        modularity=mod.getQuality(communities, G_nk)
        if modularity > best_modularity:
            best_modularity=modularity
            best_communities=communities
    communities=best_communities
    nk.community.inspectCommunities(communities, G_nk)
    return G_nk,communities.getVector()
    

#### Matrix Factorization
    
class MatrixFactorization(nn.Module):
    def __init__(self, v, d, C, device="cpu"):
        super(MatrixFactorization, self).__init__()

        self.X = nn.Parameter(torch.zeros(v, d, requires_grad=True).to(device), requires_grad=True)
        # self.X = nn.Parameter(X)#torch.zeros(v, d, requires_grad=True))
        self.C = C
        #self.X.to(device)
        self.C = self.C.to(device)


    def forward(self):
        return torch.matmul(self.X, self.C)
    
#squared allows to get the A² matrix which is for example used in HOPE. It is less sparse and more informative, it thus may be useful in some cases
def get_torch_adjacency(G, squared=False):
    adjacency_graph = nx.adjacency_matrix(G)
    if squared:
        adjacency_graph =adjacency_graph.dot(adjacency_graph)
    adjacency_graph = torch.Tensor(adjacency_graph.todense())
    return adjacency_graph

def get_torch_coms(communities):
    print(communities)
    adjacency_coms = np.zeros((len(set(communities)),len(communities)))
    for idx, com in enumerate(communities):
        adjacency_coms[com,idx]=1
    return torch.Tensor(adjacency_coms)

def train_mf(adjacency, communities, device="cpu", lr=50e-5, momentum=0.9, criterion=nn.L1Loss, criterion_reduction="sum", optimizer=optim.SGD, n_epoch=3000, nb_epochs_print=100, norm=None, lr_scheduler=None):
    d, v = communities.shape
    adjacency = adjacency.to(device)
    model = MatrixFactorization(v, d, communities, device)
    criterion = criterion(reduction=criterion_reduction)
    loss_acc = []
    modelloss=[] #collect loss
    best_accuracy = 0.0
    best_model = None
    best_df = None
    best_epoch = 0
    # optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.1)#, momentum=MOMENTUM)
    optimizer = optimizer(model.parameters(), lr=lr, momentum=momentum)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,70,90,110,130], gamma=0.1)
    running_loss = 0.0

    for epoch in tqdm(range(n_epoch), total=n_epoch):
        model.train()
        optimizer.zero_grad()

        Y = model()
        loss = criterion(Y, adjacency) #+ 0.1 * torch.norm(model.X, p=2)

        # loss = criterion(Y, A)
        loss.backward()

        # scheduler.step()
        optimizer.step()
        # print(model.X)

        running_loss = loss.item()
        loss_acc.append((epoch, running_loss))
        if epoch % nb_epochs_print == 0:
            print("Epoch : ", epoch, "; Loss : ", running_loss)
        # if epoch % 1 == 0:    # print every 2000 mini-batches
            # print(f'[{epoch + 1}] loss: {running_loss:.3f}', end="\r")
    return model

##### Visualize

def get_colors(communities):
    Blues = plt.get_cmap('spring')
    for_colors = [com / max(communities) for com in communities]
    return [Blues(x) for x in for_colors]

def draw_comgraph(G, communities, pos):
    if pos:
        nx.draw(G, node_color=get_colors(communities),node_size=50, linewidths= 0, width= 0.1, pos=pos)
    else:
        nx.draw_spring(G, node_color=get_colors(communities),node_size=50, linewidths= 0, width= 0.1)



def plot_pca(embeddings, names, gt_membership, n_components=2, nb_cols=2, show_labels=True):
    colors = get_colors(gt_membership)
    if isinstance(embeddings, list) and len(embeddings)>1:
        fig, axs = plt.subplots(int(np.ceil(len(embeddings)/nb_cols)), nb_cols, figsize=(15,15))
        x,y = 0, 0
        for embedding, name in zip(embeddings, names):
            axs[x,y].set_title(name)
            pca = PCA(n_components=2)
            X_2=pca.fit_transform(embedding)
            axs[x,y].scatter(X_2[:,0], X_2[:,1], color=colors)
            if show_labels:
                for i in range(X_2.shape[0]):
                    axs[x,y].text(X_2[i,0], X_2[i,1], i)
            y+=1
            if y==nb_cols:
                x+=1
                y=0
        fig.show()
    else:
        pca = PCA(n_components=2)
        X_2=pca.fit_transform(embeddings[0])
        plt.scatter(X_2[:,0], X_2[:,1], color=colors)
        for i in range(X_2.shape[0]):
            if show_labels:
                plt.text(X_2[i,0], X_2[i,1], i)
        plt.show()
###### Evaluate

def evaluate_spectral_clustering(embeddings, names,  gt_membership, n_clusters=2, similarity_thresh=0.01):
    results = {}
    for name, embedding in zip(names, embeddings):
        similarities=cosine_similarity(embedding)
        similarities[similarities < similarity_thresh]=0
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',assign_labels='discretize').fit(similarities)
        results[name] =  {
            "NMI":normalized_mutual_info_score(clustering.labels_, gt_membership),
            "Homogeneity":homogeneity_score(clustering.labels_, gt_membership),
            "Rand Index": adjusted_rand_score(clustering.labels_, gt_membership)
        }
    return pd.DataFrame(results)


    


##### Graph reconstruction

def reconstruct_several(G, Xs, names):
    plt.figure(figsize=(15, 15))
    for X, name in zip(Xs, names):
        reconstruct(G,X,name)
    plt.axvline(x=G.number_of_edges(), c="grey", ls="--") #Vertical line at the limit in number of edges
    plt.legend(loc='upper right', frameon=False)
    plt.show()
def reconstruct(G, X, name, runs=10):
    precision=[]
    pairs_of_nodes = set(G.edges)
    pairs_of_nodes_dict = {(p[0],p[1]):cosine(X[p[0]], X[p[1]]) for p in pairs_of_nodes}
    for ri in range(runs):
        
        no_pairs = choice(max(list(G.nodes())),size=(len(pairs_of_nodes)*10,2), replace=True)
        no_pairs = {(p[0],p[1]):cosine(X[p[0]], X[p[1]]) for p in no_pairs if (p[0],p[1]) not in pairs_of_nodes}
        all_pairs = pairs_of_nodes_dict | no_pairs
        sorted_pairs= {k: v for k, v in sorted(all_pairs.items(), key=lambda item: item[1])}
        cpt_print=0
        precision_ri=[]
        nb_ok=0
        nb_tot=0
        for (k,v) in sorted_pairs:
            if (k,v) in pairs_of_nodes:
                nb_ok+=1
            nb_tot+=1
            precision_ri.append(float(nb_ok/nb_tot))
            if nb_tot == (len(pairs_of_nodes) * 3):
                break
            #if nb_ok == len(pairs_of_nodes):
            #    break
        precision.append(precision_ri)
    # print(np.mean(precision,axis=0))
    plt.plot(np.mean(precision,axis=0), label=name)

###### Regression
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def evaluate_regression(X, y, cv=5, plot=True, tofit="Node feature"):
    y_predicted,mse = evaluate_linear(X,y,cv)
    y_predicted_sq,mse_sq = evaluate_polysq(X,y,cv)
    print("Linear model, mse :", mse)
    print("Second order polynom, mse :", mse_sq)
    plot_regression(y, y_predicted, tofit)
    plot_regression(y, y_predicted_sq, tofit)

def evaluate_linear(X,y,cv):
    reg=LinearRegression()
    print(X.shape)
    #y_predicted = cross_val_predict(reg, X,y)#, cv=cv)
    res = cross_validate(reg, X, y, cv=cv, return_estimator=True)
    print(y, y_predicted)
    return y_predicted, mean_squared_error(y, y_predicted)
    
def evaluate_polysq(X,y,cv):
    polyreg=make_pipeline(PolynomialFeatures(2),LinearRegression())
    y_predicted = cross_val_predict(polyreg, X,y, cv=cv)
    print(y, y_predicted)
    return y_predicted, mean_squared_error(y, y_predicted)
    
def plot_regression(y, y_predicted, tofit):
    plt.figure()
    n,bins,p=plt.hist(y, color='blue', alpha=0.5, label='true')
    plt.hist(y_predicted, bins=bins, color='red', alpha=0.5, label='estimated')
    plt.legend(loc='upper right')
    plt.xlabel(tofit)
    plt.show()


def evaluate_multiple_regression(embeddings, names, y, cv, tofit="degree"):
    if isinstance(embeddings, list) and len(embeddings)>1:
        res = {}
        nb_cols=2
        fig, axs = plt.subplots(len(embeddings), nb_cols, figsize=(15,15))
        fig.tight_layout(pad=3.0)
        x = 0
        for X, name in zip(embeddings, names):
            axs[x,0].set_title(name)
            axs[x,1].set_title(f"{name}_sq")
            print(name)
            y_predicted,mse = evaluate_linear(X,y,cv)
            y_predicted_sq,mse_sq = evaluate_polysq(X,y,cv)
            # print("Linear model, mse :", mse)
            # print("Second order polynom, mse :", mse_sq)
            res[name] = {"Linear model MSE":mse,
                         "Second order polynom, MSE":mse_sq}
            n,bins,p=axs[x,0].hist(y, color='blue', alpha=0.5, label='true')
            axs[x,0].hist(y_predicted, bins=bins, color='red', alpha=0.5, label='estimated')
            n,bins,p=axs[x,1].hist(y, color='blue', alpha=0.5, label='true')
            axs[x,1].hist(y_predicted_sq, bins=bins, color='red', alpha=0.5, label='estimated')
            axs[x,0].legend(loc='upper right')
            axs[x,1].legend(loc='upper right')
            axs[x,0].set(xlabel=tofit)
            axs[x,1].set(xlabel=tofit)
            x+=1
        return fig, pd.DataFrame(res)


###### Link Prediction

def _prepare_lp_sets(G, prop_train=0.80):
    G_lp = deepcopy(G) # In order to keep the original graph intact
    nb_edges = G_lp.number_of_edges()
    nb_nodes = len(G_lp.nodes())
    edges_set = set(G_lp.edges())
    edges = np.random.permutation(list(G_lp.edges))

    train_size = int(np.round(prop_train*nb_edges))
    test_size = nb_edges - train_size    

    # Do not diconnect the graph
    nb_comps = nx.number_connected_components(G_lp)
    test_set_yes = []
    for u,v in edges:
        if len(test_set_yes)<test_size:
            edge_data = G_lp.get_edge_data(u, v)
            G_lp.remove_edge(u,v) # Remove and check number of components
            if nx.number_connected_components(G_lp)>nb_comps: 
                G_lp.add_edge(u,v)
                G_lp[u][v].update(edge_data)
            else:
                test_set_yes.append((u,v))
                
    train_set_yes = {(u,v):1 for u,v in np.random.permutation(list(G_lp.edges()))[:train_size]} # Existing edges train set
    test_set_yes = {(u, v):1 for u,v in test_set_yes} # Existing edges test set
    

    no_pairs = np.random.choice(max(list(G.nodes())),size=(nb_edges*20,2), replace=True) # Generate many random edges
    no_pairs = [(e[0],e[1]) for e in no_pairs if e[0]<=e[1] and (e[0],e[1]) not in set(G.edges()) ] # Non-existing edges pool
    
    
    train_set_no = {(u, v):0 for u, v in list(no_pairs)[:len(train_set_yes)]} # Non-existing edges train set
    test_set_no = {(u, v):0 for u,v in list(no_pairs)[len(train_set_yes):len(train_set_yes)+len(test_set_yes)]} # Non-existing edges train set

    train = train_set_yes | train_set_no # Test set dict((u,v):class)
    train_Xuv = np.random.permutation(list(train.keys()))  # no.array of shuffled (u,v)
    train_Y = [train[(k[0], k[1])] for k in train_Xuv] # classes aligned with train_Xuv

    test = test_set_yes | test_set_no # Test set dict((u,v):class)
    test_Xuv = np.random.permutation(list(test.keys())) # no.array of shuffled (u,v)
    test_Y = [test[(k[0],k[1])] for k in test_Xuv] # classes aligned with train_Xuv
    # print(f"tr_Xuv {train_Xuv}, tr_Y {train_Y}, train {train}, test_Xuv {test_Xuv}, test_Y {test_Y}, test {test}")
    
    return G_lp, train_Xuv, train_Y, train, test_Xuv, test_Y, test

def run_multiple_lp(G, methods, nb_runs, device="cuda:0", aggregate=False):
    dfs = []
    for i in range(10):
        lp = link_prediction(G=G, methods=methods, device=device)
        dfs.append(lp)
    res = pd.concat([pd.concat(j) for j in dfs])
    if aggregate:
        res = res.groupby(level=[0,1]).mean().round(decimals=2)
    return res

def link_prediction(G, classifier=XGBClassifier(use_label_encoder=False), prop_train=0.80, methods=[], device="cpu"):
    device = "cuda:0" if torch.cuda.is_available() and "cuda" in device else "cpu"
    print(device)
    G_lp, train_Xuv, train_Y, train, test_Xuv, test_Y, test = _prepare_lp_sets(G, prop_train)  
    df = {}
    idx = 0
    print(methods)
    for method, params in methods:
        print(method, params)
        embeddings = get_embeddings(G_lp, method, params)
        #train classifier
        print(type(embeddings))
        train_X = [embeddings[e[0]] * embeddings[e[1]] for e in train_Xuv]
        test_X = [embeddings[e[0]] * embeddings[e[1]] for e in test_Xuv]

        classifier.fit(np.array(train_X), np.array(train_Y))

        #evaluate

        test_res = classifier.predict(test_X)
        # print(test_res)
        result = classification_report(test_Y, test_res, output_dict=True)
        method = f"{method}_squared" if "squared" in params and params["squared"] == True else method
        method = f"{method}_{params['algo']}"
        df[method] = pd.DataFrame(result)
        print(pd.DataFrame(result))
        idx += 1
    
    return df#pd.DataFrame(result)

def get_embeddings(G, method, args):
    # args["G"] = G
    if method == "SINrMF":
        return get_mf_embeddings(G, **args)
    elif method == "HOPE":
        return get_HOPE_embeddings(G, **args)
    elif method == "SINr":
        return get_sinr_embeddings(G, **args)
def get_mf_embeddings(G, **kwargs):#G, gamma=10, device="cuda:0"):
    print("Getting embeddings")
    gamma = kwargs["gamma"]
    if "algo" not in kwargs:
        kwargs["algo"] = "louvain"
    if kwargs["algo"] == "louvain":
        G_nk, communities = get_coms(G, gamma=gamma)
    elif kwargs["algo"] == "ensemble":
         G_nk= nk.nxadapter.nx2nk(G, weightAttr=None)
         if "nb_run" not in kwargs:
             kwargs["nb_run"] = 16
         G_egc = deepcopy(G)
         G_egc.remove_edges_from(nx.selfloop_edges(G_egc))
         communities = get_ensemble_coms(G_egc, gamma=gamma, nb_run=kwargs["nb_run"])

    # _ , communities = get_coms(G, gamma=gamma)
    A = get_torch_adjacency(G, kwargs["squared"])
    C = get_torch_coms(communities)
    C = torch.Tensor(C)
    model = train_mf(A, C, device=kwargs["device"], lr=kwargs["lr"], n_epoch=kwargs["n_epoch"], nb_epochs_print=kwargs["nb_epochs_print"])
    if model.X.is_cuda:
        X = model.X.detach().cpu().numpy()
    else:
        X = model.X.detach().numpy()
    return X

def get_sinr_embeddings(G, **kwargs):
    gamma = kwargs["gamma"]
    if "algo" not in kwargs:
        kwargs["algo"] = "louvain"
    if kwargs["algo"] == "louvain":
        G_nk, communities = get_coms(G, gamma=gamma)
    elif kwargs["algo"] == "ensemble":
         G_nk= nk.nxadapter.nx2nk(G, weightAttr=None)
         if "nb_run" not in kwargs:
             kwargs["nb_run"] = 16
         G_egc = deepcopy(G)
         G_egc.remove_edges_from(nx.selfloop_edges(G_egc))
         communities = get_ensemble_coms(G_egc, gamma=gamma, nb_run=kwargs["nb_run"])
    elif 
    return np.array(get_nfm_embeddings(G_nk, communities, len(set(communities)), len(communities))[2].todense())

def get_HOPE_embeddings(G, **kwargs):#G, dimensions=5):
    hope = HOPE(dimensions=kwargs["dimensions"])
    hope.fit(graph=G)
    return hope.get_embedding()



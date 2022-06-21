# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

import networkx as nx
import tensorflow as tf

from utils import normt_spm, spm_to_tensor, get_khops_neighbors
from tqdm import tqdm

from joblib import Parallel, delayed
import multiprocessing
from p_tqdm import p_map

import time

# architecture used from DGCNN https://github.com/muhanzhang/DGCNN
class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, relu=True):
        super().__init__()

        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None

        self.w = nn.Parameter(torch.empty(in_channels, out_channels))
        self.b = nn.Parameter(torch.zeros(out_channels))
        xavier_uniform_(self.w)

        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = None

    def forward(self, inputs, adj_set = None, att = None):
        
        if self.dropout is not None:
            inputs = self.dropout(inputs)
            
        # Sean: changed from torch.mm to torch.matmul for vector to matrix 
        # multiplication             
        support = torch.matmul(inputs, self.w) + self.b 
        outputs = None
        
        if adj_set is not None:
            for i, adj in enumerate(adj_set):
                # Sean: changed from torch.mm to torch.matmul for vector to 
                # matrix multiplication
                y = torch.matmul(adj, support) * att[i]
                if outputs is None:
                    outputs = y
                else:
                    outputs = outputs + y   
        # Sean: changed to allow for the second layer to complete
        # due to the vector matrix multiplication messing with the matrix 
        # dimensions that would be otherwise anticipated
        else:
            outputs = support
            
        return outputs


class GCN_Dense_Att(nn.Module):

    # sleon010 An piece of future work not yet complete. It was working towards building
    # a model which could capture more features.
    # def __init__(self, n, graph : nx.Graph, in_channels, out_channels, hidden_layers):
    #     super().__init__()
    #     self.n = n
    #     #self.d = len(edges_set)

    #     self.a_adj_set = []
    #     self.r_adj_set = []

    #     # for edges in edges_set:
    #     #     edges = np.array(edges)
    #     #     adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
    #     #                         shape=(n, n), dtype='float32')
    #     #     a_adj = spm_to_tensor(normt_spm(adj, method='in'))
    #     #     r_adj = spm_to_tensor(normt_spm(adj.transpose(), method='in'))
    #     #     self.a_adj_set.append(a_adj)
    #     #     self.r_adj_set.append(r_adj)

    #     hl = hidden_layers.split(',')
    #     if hl[-1] == 'd':
    #         dropout_last = True
    #         hl = hl[:-1]
    #     else:
    #         dropout_last = False

    #     self.a_att = nn.Parameter(torch.ones(self.d))
    #     self.r_att = nn.Parameter(torch.ones(self.d))
        
    #     #print( self.d, len( self.a_att ))
        
    #     # sean leonard ------------------------------------
    #     #architecture used from DGCNN https://github.com/muhanzhang/DGCNN 
    #     number_of_node_features = len( graph.nodes[ 0 ][ 1 ] )
    #     print( len( graph.nodes[ 0 ][ 1 ] ) )
    #     # used as defaults for the DGCNN model
    #     #latent_dim=[32, 32, 32, 1]
    #     #conv1d_channels=[16, 32]
    #     #conv1d_kws=[0, 5]
        
    #     layer1 = nn.Linear   ( number_of_node_features, 32      )
    #     layer2 = nn.Linear   ( 32                     , 32      )
    #     layer3 = nn.Linear   ( 32                     , 32      )
    #     layer4 = nn.Linear   ( 32                     , 1       )
    #     layer5 = nn.GraphConv( 1                      , 16, dropout_last, relu=False)
        
    #     dense_dim = ( int( (( 30 - 2 ) / 2 ) + 1 ) - 4 ) * 32
        
    #     layer6 = nn.Linear( dense_dim, out_channels)
    #     output_layer = nn.ReLU( out_channels, out_channels )
        
    #     self.add_module('layer1'     , layer1 )
    #     self.add_module('layer2'     , layer2 )
    #     self.add_module('layer3'     , layer3 )
    #     self.add_module('layer4'     , layer4 )
    #     self.add_module('layer5'     , layer5 )
    #     self.add_module('layer6'     , layer6 )
    #     self.add_module('output relu', output_layer )
        
    #     layers = []
        
    #     layers.append( layer1 )
    #     layers.append( layer2 )
    #     layers.append( layer3 )
    #     layers.append( layer4 )
    #     layers.append( layer5 )
    #     layers.append( layer6 )
    #     layers.append( output_layer )
        
    #     #two models one for the src gene another for the dest gene
    #     #NOTE that the src and dest terms do not imply a direction, but rather
    #     # follow a common parlance. 
    #     self.layers = [ layers, layers ]
        
    #     # -------------------------------------

    # work of the selected paper
    def __init__(self, n, graph, in_channels, out_channels):
        super().__init__()

        self.graph = graph
        self.n = n
        self.hops = 2
        
        self.a_adj_set = []
        self.r_adj_set = []
            
        self.att = nn.Parameter(torch.ones(self.hops + 1))

        i = 0
        layers = []
        last_c = in_channels
        
        # The network is currently very small and dropout will penalize an 
        # already weak network
        dropout = False

        conv = GraphConv(last_c, out_channels, relu=False, dropout=dropout)
        self.add_module('conv-last', conv)
        layers.append(conv)

        self.layers = layers
        
    # A change of the foward propogation funciton to allow for a input
    # gene expression vector to build a list of j-hop neighbors with j being 
    # one neighborhood of all neighbors h-hops away and the list being the size
    # N where N is the number of hops plus 1
    # the addition of the one is to include the self-loop of the target node. 
    def forward(self, edge, graph, is_dest : bool):
 
        def sigmoid( x ):
            return 1 / ( 1 + math.exp( -x ) )
        
        # src side
        # The paper this model finds its inspiration from uses weights from the 
        # model to learn weights for the different hops starting at the 0-hop
        # neighbors (source node) to the n-hop neighbors
        #
        # [ 0-hop_neighbors_weight, 1-hop_neighbors_weight ...]
        att = self.att
        att = F.softmax( att, dim = 0 )
         
        src_id  = edge[ 0 ]  
        dest_id = edge[ 1 ]             
        
        #start_time = time.time()
        adj_set = get_khops_neighbors( src_id, dest_id, self.hops, self.graph )             
         
        output     = self.layers[ 0 ]( torch.from_numpy( np.array( 
                     self.graph.nodes[ src_id ][ "expression" ] ) ).float(), 
                                                                    adj_set, 
                                                                       att ) 
        prediction = self.layers[ 1 ]( output )
          
        if is_dest:
            return  [ dest_id, src_id, [ sigmoid( prediction[ 0 ] ), 
                                         sigmoid( prediction[ 1 ] ) ] ]
        else:
            return  [ src_id, dest_id, [ sigmoid( prediction[ 0 ] ), 
                                         sigmoid( prediction[ 1 ] ) ] ]

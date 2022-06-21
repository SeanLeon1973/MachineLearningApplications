import os
import os.path as osp
import shutil

import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import networkx as nx

import time

# Used from DGCNN https://github.com/muhanzhang/DGCNN
def normt_spm(mx, method='in'):
    if method == 'in':
        mx = mx.transpose()
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    if method == 'sym':
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = mx.dot(r_mat_inv).transpose().dot(r_mat_inv)
        return mx

# Used from DGCNN https://github.com/muhanzhang/DGCNN
def spm_to_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack(
            (sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# Sean Leonard -------------------------------
# takes two files ( connections and features )
# connections
# src_gene dest_gene 
# 
# features
# gene_id chrom start end gene_name sample1, sample2 ...
#
# returns a networkx graph with the node features and nodes being in their correct locations
def load_graph( connection_file_path, features_file_path, TF_file_path ):
    
    connections = pd.read_csv( connection_file_path, header = 0, sep = "\t" ) 
    features    = pd.read_csv( features_file_path  , header = 0, sep = "\t" ) 
    TFs         = pd.read_csv( TF_file_path        , header = None,  sep = "\t" )
    
    graph = nx.Graph()
    
    adjacency_matrix = []
    
    gene_to_node_id = {}
    TF_ids = []
    for node_id in range( 0, features.shape[1]): 

        if features.columns[ node_id ] in list( TFs.iloc[ : , 0 ] ):
            
            graph.add_node( node_id , 
                            TF = 1, 
                            expression = list( features.iloc[ :, node_id ] )
                          )
            TF_ids.append( node_id )
            graph.add_edge( node_id, node_id  )
            
        else:
            
            graph.add_node( node_id, 
                            TF = 0, 
                            expression = list( features.iloc[ :, node_id ] )
                          )
        gene_to_node_id[ features.columns[ node_id ] ] = node_id 
        
    for row_id in range( 0, connections.shape[0] ):
        src  = connections.iloc[ row_id, 0 ] 
        dest = connections.iloc[ row_id, 1 ] 
        # for now the graph will be undirected
        graph.add_edge( gene_to_node_id[ src ], gene_to_node_id[ dest ]  )
        
    return graph, features, TF_ids

# The function to get the neighbors of a source node. The dest node is 
# necessary as the link between the two nodes is considered off to all for the 
# context between the two to be unique. Otherwise, there would be a large 
# amount of overlap between the source and destination nodes neighborhoods 
def get_khops_neighbors(  src_id : int, 
                         dest_id : int, 
                         hops    : int, 
                         graph   : nx.Graph) -> list:
    
    neighborhood_nodes = []
    
    number_of_nodes = graph.number_of_nodes()
    graph_copy = graph.copy()
    
    neighbors = set( [ src_id ] )
    
    # Ensure a self-loop so that the src node is incorporated into the set of 
    # features to be used to make the prediction
    edges = np.array( [ [ src_id, src_id ] ] )
    
    adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                    shape=(number_of_nodes, number_of_nodes), dtype='float32')
    adj = spm_to_tensor(normt_spm(adj, method='in'))
    
    print( adj )
    
    # The edges_set is added for the sake of the attention weights as they 
    # are for their respective k-hop neighbors of the source node.
    # 
    # [ 0-hop_neighbors, 1-hop_neighbors, 2-hop_neighbors ... ]
    edges_set = [ adj ]
    for k in range( 1, hops + 1 ):
        
        edges = [] 
                
        previous_neighbors = neighbors.copy()
        
        # start_time = time.time() 
                
        for n in previous_neighbors:
            for nbr in graph_copy[ n ]:
                neighbors.add( nbr )
                edges.append( [ n, nbr ] )
                
        # Remove the destination node to allow for a more unique context for 
        # each side
        if k == 1:
                    
            if graph.has_edge( src_id, dest_id ):
                neighbors.remove( dest_id )
            
       # print( (time.time() - start_time ) )           

        # If the new neighbors contain new links to preivous neighbors they are
        # removed
        neighbors = set( previous_neighbors ).symmetric_difference( 
                                                       set( neighbors ) )
        
        if( neighbors == set() ):
            break
       
        edges = np.array( edges )
        adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                             shape=(number_of_nodes, number_of_nodes), 
                             dtype='float32')

        adj = spm_to_tensor(normt_spm(adj, method='in'))
  
        edges_set.append( adj )

    return edges_set 

# A function to update the graph as new predictions are made if a link's 
# prediction on the src or dest side has a high probability of no link[ 0 ] 
# then the link is removed. Only in the case when both the src and dest 
# networks say that an edge should be formed does it get created. This was 
# chosen as a constraint to reduce the number of false positives a predicted 
# GRN has. 
def update_graph( graph : nx.Graph, predictions : list):
    
    src_predictions  = predictions[ 0 ]
    dest_predictions = predictions[ 1 ]
    
    assert( len(src_predictions).__eq__( len( dest_predictions ) ) )
    
    for edge in range( 0, len( src_predictions )):
        
        no_link = False
        
        # If the source node or the dest node predicts that a link is not 
        # present then the links is removed from the edges used as input
        # This is so after a while the ending group of edges will be the 
        # predicted GRN.
        if(  src_predictions[ edge ][ 2 ][ 0 ] + 0.1 >  
             src_predictions[ edge ][ 2 ][ 1 ] )      or \
          ( dest_predictions[ edge ][ 2 ][ 0 ] + 0.1 > 
            dest_predictions[ edge ][ 2 ][ 1 ] ):
              
              if graph.has_edge( src_predictions[ edge ][ 0 ], 
                                 src_predictions[ edge ][ 1 ] ):
                  graph.remove_edge( src_predictions[ edge ][ 0 ], 
                                     src_predictions[ edge ][ 1 ] )
                  
              if graph.has_edge( src_predictions[ edge ][ 1 ], 
                                 src_predictions[ edge ][ 0 ] ):
                  graph.remove_edge( src_predictions[ edge ][ 1 ], 
                                     src_predictions[ edge ][ 0 ] )  
              
        else:
              graph.add_edge( src_predictions[ edge ][ 0 ], 
                              src_predictions[ edge ][ 1 ] )   

    return [ [ edge[ 0 ], edge[ 1 ] ] for edge in graph.edges()]
            
# ---------------------------------------------
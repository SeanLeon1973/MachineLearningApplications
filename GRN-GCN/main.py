# -*- coding: utf-8 -*-

import json
import random
import os.path as osp

import torch
import torch.nn.functional as F

from utils import load_graph
from GCN import GCN_Dense_Att
import numpy as np
import tensorflow as tf
from utils import normt_spm, spm_to_tensor, get_khops_neighbors, update_graph
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import networkx as nx
import sys

import tqdm

import matplotlib.pyplot as plt

# Future Work: implement a weighting scheme that weights the positive instances
# higher the negative instances insuring the model does not try to overfit 
# to a simple set all of the links to negative answer which it currently does

# L2 Loss function -------------------------
# predictions : the predictions made by the model
# labels      : the ground-truth labels
def graph_loss( predictions, labels ):
    
    loss        = 0
    numerator   = 0
    denominator = len(predictions) * 2
    
    for prediction in predictions:
        
        predicted_no_link = prediction[ 2 ][ 0 ]
        predicted_link    = prediction[ 2 ][ 1 ]
        actual            = labels[ prediction[ 0 ] ][ prediction[ 1 ] ]
    
        numerator += ( predicted_no_link - actual )**2
        numerator += ( predicted_link    - actual )**2
    
    if not denominator == 0:
        return ( numerator/denominator ).clone().detach().requires_grad_(True)
    else:
        return 0
    
# Build a fully connected graph between all TFs and all TFs and target genes
# The ground_truth_graph is required as it is used to get the gene expression
# values, but is likely that this is not an optimal solution. However, it is 
# only run once over the course of the program.
# NOTE: self-edges are not added, so G1 -- G1 is never considered as having an
# edge
def build_fully_connected( TFs, genes, ground_truth_graph ):
    
    fully_connected_graph = nx.Graph()
    # for usage as a means to update edges
    edges = []
    
    for TF in TFs:
        for gene_id  in genes:
            
            if TF != gene_id:
                # gather the gene expression values for the src and dest
                # genes 
                src_expr  = ground_truth_graph.nodes[ TF ]\
                    [ "expression" ].copy()
                dest_expr = ground_truth_graph.nodes[ gene_id ]\
                    [ "expression" ].copy()
                    
                edges.append( [ TF, gene_id ] )
                
                fully_connected_graph.add_node( TF ,  
                                                expression = src_expr
                                              )
                fully_connected_graph.add_node( gene_id ,  
                                                expression = dest_expr
                                              )
                
                # This is reqiured as we want to avoid having in edges between
                # both the src and dest nodes and the dest and src nodes
                # and would rather only have edges between the src and dest.
                # The learning outcome could be affected as it would treat each
                # edge as a new observation. That could weight the edges 
                # between TFs unfairly and result in a bias that is not 
                # intended
                if not fully_connected_graph.has_edge( gene_id, TF ):
                    
                    fully_connected_graph.add_edge( TF, gene_id )

    return fully_connected_graph, edges

def main():
    # Sean Leonard -------------------------------
    save_path = "output/"
    data_path = "data/" 
    
    print( "loading data" )
    
    # Gold standard GRNs to be utilized S. Aureus(training) and 
    # E. coli (testing)
    # Data is in the format: 
    #
    # training_graph (518 rows)
    #    Source Gene | Destination Gene | Presence of Link
    # 1  G(some_num)   G(some_other_num)  1      
    # 2  G(some_num)   G(some_other_num)  1   
    # ...
    # 518
    #
    # NOTE: Data deviates from raw source data because the absence of a link is 
    # removed in favor of only having the links that are present be represented
    # and the links that are not present are assumed to not be links
    # NOTE: There are no self-pointing edges where a gene can point to itself 
    # in this dataset
    
    # train_input (2810 Columns, 161 rows)
    #         G1 | G2 | G3 | ... | G2810
    # Cell1   Quantity of reads in Cell1 for each Gene based on Column
    # Cell2   Quantity of reads in Cell2 for each Gene based on Column
    # Cell3   Quantity of reads in Cell3 for each Gene based on Column
    #  ...
    # Cell161 Quantity of reads in Cell3 for each Gene based on Column

    # train_IF_ids (99 rows)
    # Transcription Factor
    # G1
    # G2
    # G3
    # ...
    # G99

    training_graph, train_input, train_TF_ids = load_graph( 
        data_path + "DREAM5_NetworkInference_GoldStandard_Network2" + \
                                                 " - S. aureus.txt", 
        data_path + "net2_expression_data.tsv", 
        data_path + "net2_transcription_factors.tsv" 
        )
    
    # test_graph, test_TF_ids, test_input = load_graph( 
    #     data_path + "DREAM5_NetworkInference_GoldStandard_Network3 " + \
    #     "- E. coli.tsv", 
    #     data_path + "net3_expression_data.tsv",
    #     data_path + "net3_transcription_factors.tsv" )  
    
    TFs   = list( range( 0, len( train_TF_ids         ) ) ).copy()
    genes = list( range( 0, len( train_input.columns  ) ) ).copy()
    
    # Take 200 genes which can be either a TF or a target gene
    number_of_genes = 200
    TFs = TFs[ 0:10 ]
    genes = genes[ 0:number_of_genes ]
    
    # NOTE: The above subset was taken to show a proof of function and thus the 
    # graph size was reduced. Orignally the graph was over ~200,000 links which 
    # took upwards of 200 hours (src: tqdm) to compute for one epoch without 
    # parallel computing and with parallel computing took 26 hours (src: ptqdm)
    # with 15 cores for one epoch. As such, I reduced the size and left 
    # optimization improvements for future work. I got CUDA to work, but this 
    # did not show improvement due to the majority of the time being spent 
    # finding neighbors (~1 second spent on finding neighbors for each 
    # forward pass). 
    
    # A 2D matrix of TFs (y-axis) and all genes (x-axis)
    #    G1  G2  G3 ... G2810
    # G1 0   1   0
    # G2 1   0   0
    # G3 0   0   0
    # ...
    # G99
    labels_vector = torch.zeros( 
        [ len( train_TF_ids ), len( train_input.columns  ) ], 
        dtype=torch.float32 )
    
    # This shuffle was commented out to make sure results are repeatable while
    # debugging the model. Given the performance constraints laid out below,
    # I wanted to take a subset of a rich area of true positives in this graph 
    # which is present largely in the TFs so I removed the shuffle in order to 
    # avoid situations where the graph would be only made up of false 
    # positives. 
    #random.shuffle( genes )
    #random.shuffle( TFs   )

    # Not truly fully connected as target genes cannot link with other target 
    # genes as a result all TFs will be linked to either a TF or a target gene 
    # and in that sense the graph is fully connected as it connects at all
    # opportunities       
    fully_connected_graph, edges = build_fully_connected( TFs, 
                                                          genes, 
                                                          training_graph 
                                                        )
    
    # update the labels_vector with the new false positive edges
    for TF in range( 0, len( train_TF_ids ) ):
        for gene_id in range( 0, len( train_input.columns  ) ):
            labels_vector[ TF ][ gene_id ] = training_graph.has_edge( TF, 
                                                                     gene_id )
    
    gene_expression_vectors = torch.tensor( 
                              train_input[ 0:number_of_genes ].to_numpy() 
                                          )
    
    # Normalize the gene expression values to avoid skewing values later on
    # in the machine learning process.
    gene_expression_vectors = F.normalize( gene_expression_vectors )
      
    # A gcn for the source node
    src_gcn = GCN_Dense_Att( fully_connected_graph.number_of_nodes(),
                         fully_connected_graph,
                         len( training_graph.nodes[ 0 ][ "expression" ] ),
                         2
                         )
    
    # A gcn for the destination node
    # It is the case that the destination node is either a target gene or a TF
    # and the source node is always a TF and as a result unique features may be 
    # present within those instances and thus having a seperate network could 
    # make it more likely that those features would be captured and utilized
    dest_gcn = GCN_Dense_Att( fully_connected_graph.number_of_nodes(),
                         fully_connected_graph,
                         len( training_graph.nodes[ 0 ][ "expression" ] ),
                         2
                         )
     
    
    # An optimizer for each model
    src_optimizer  = torch.optim.Adam(  src_gcn.parameters(), lr = 0.001, 
                                        weight_decay = 0.0005 )
    dest_optimizer = torch.optim.Adam( dest_gcn.parameters(), lr = 0.001, 
                                        weight_decay = 0.0005 )
    
    training_set = edges
    
    # losses on the src_gcn and dest_gcn's predictions
    src_losses  = []
    dest_losses = []
    
    max_epochs = 100
    
    # The exceptions are to catch the graph zeroing out and having all of its 
    # links removed. This was added to show that the application's lack of 
    # function after a few epochs was noted and understood.
    try: 
        for epoch in range(1, max_epochs):
    
            # Set pytorch models to training mode
            src_gcn.train()
            dest_gcn.train()
            
            print( "forward" )
    
            src_predictions  = []
            dest_predictions = []
            
            # It is put into future work to consider allowing the model to 
            # learn from the abscene of links, but this would have to be done 
            # cautiously as there are significantly more absent links then 
            # present links. In addition, moving this to a mini-batch format
            # would likely yield better results and a quicker convergence.
            
            # Go through each link
            for pair_id in range( 0, len( edges )):
                
                # Use the neighbors of the source nodes as context
                src_prediction  =  src_gcn(   edges[ pair_id ]      , 
                                              fully_connected_graph ,
                                              is_dest=False
                                          )
                
                # Use the neighbors of the destination node as context
                dest_prediction = dest_gcn( [ edges[ pair_id ][ 1 ] , 
                                              edges[ pair_id ][ 0 ] ], 
                                              fully_connected_graph  , 
                                              is_dest=True 
                                          )
                
                src_predictions.append (  src_prediction )
                dest_predictions.append( dest_prediction )
                
            # predictions = Parallel(n_jobs=num_cores)(
            # delayed(forward_propogation)\
            #         ( pair_id ) for pair_id in tqdm( \
            #      range( 0, len( self.edges )) ) )
                    
            print( "backward" )
            src_loss  = graph_loss(  src_predictions, labels_vector )
            dest_loss = graph_loss( dest_predictions, labels_vector )
            print( src_loss )
            src_optimizer.zero_grad()
            dest_optimizer.zero_grad()
            
            if src_loss == 0 or dest_loss == 0:
                raise ZeroDivisionError
    
            src_loss.backward()
            dest_loss.backward()
            
            src_optimizer.step()
            dest_optimizer.step()
    
            print( "evaluation")
            src_gcn.eval()
            dest_gcn.eval()
            
            src_predictions_eval  = []
            dest_predictions_eval = []
            
            for pair_id in range( 0, len( training_set )):
                
                src_prediction  = src_gcn (   training_set[ pair_id ], 
                                              fully_connected_graph  ,
                                              is_dest=False
                                               )
                
                dest_prediction = dest_gcn( [ training_set[ pair_id ][ 1 ], 
                                              training_set[ pair_id ][ 0 ] ] , 
                                                       fully_connected_graph , 
                                                                is_dest=True )
                
                src_predictions_eval.append(   src_prediction )
                dest_predictions_eval.append( dest_prediction )
            
            src_train_loss  = graph_loss( src_predictions_eval , 
                                                 labels_vector )
            dest_train_loss = graph_loss( dest_predictions_eval, 
                                                 labels_vector )
            src_losses.append( src_train_loss )
            dest_losses.append( dest_train_loss )
            
            print('epoch {}, src_train_loss={:.4f} dest_train_loss={:.4f}'
                  .format(epoch, src_train_loss, dest_train_loss ))
    
            if epoch > 50:
                
                edges = update_graph( fully_connected_graph, 
                                     [ src_predictions, dest_predictions] 
                                     )
            
            if len( edges ) == 0:
                raise ZeroDivisionError
    
            pred_obj = None
        
    except ZeroDivisionError:
        
        print( "Edges are empty")
        pass
    
    

    # Plot a simple line chart
    plt.plot(range( 1, len( src_losses ) + 1 ), 
                                    src_losses, 
                                           'g', 
                             label='src losses')
    
    # Plot another line on the same chart/graph
    plt.plot(range( 1, len( dest_losses ) + 1 ), 
                                    dest_losses, 
                                            'r', 
                            label='dest losses')
    
    plt.legend()
    plt.show()
    
    # -----------------------------  
if __name__=="__main__":
    main()
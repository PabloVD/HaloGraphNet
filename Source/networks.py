#----------------------------------------------------------------------
# Definition of the neural network architectures
# Author: Pablo Villanueva Domingo
# Last update: 10/11/21
#----------------------------------------------------------------------

import torch
from torch.nn import Sequential, Linear, ReLU, ModuleList
from torch_geometric.nn import MessagePassing, GCNConv, PPFConv, MetaLayer, EdgeConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_cluster import knn_graph, radius_graph
from torch_scatter import scatter_mean, scatter_sum, scatter_max, scatter_min
import numpy as np

#------------------------------
# Architectures considered:
#   DeepSet
#   PointNet
#   EdgeNet
#   EdgePointLayer (a mix of the two above)
#   Convolutional Graph Network
#   Metalayer (graph network)
#
# See pytorch-geometric documentation for more info
# pytorch-geometric.readthedocs.io/
#-----------------------------

#--------------------------------------------
# Message passing architectures
#--------------------------------------------

# PointNet layer
class PointNetLayer(MessagePassing):
    def __init__(self, in_channels, mid_channels, out_channels, use_mod=1):
        # Message passing with "max" aggregation.
        super(PointNetLayer, self).__init__('max')

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3, or 1 if only modulus is used).
        if use_mod:
            self.mlp = Sequential(Linear(in_channels+1, mid_channels),
                                  ReLU(),
                                  Linear(mid_channels, mid_channels),
                                  ReLU(),
                                  Linear(mid_channels, out_channels))
        else:
            self.mlp = Sequential(Linear(in_channels+3, mid_channels),
                                  ReLU(),
                                  Linear(mid_channels, mid_channels),
                                  ReLU(),
                                  Linear(mid_channels, out_channels))

        self.messages = 0.
        self.input = 0.
        self.use_mod = use_mod

    def forward(self, x, pos, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_j, pos_j, pos_i):
        # x_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]

        input = pos_j - pos_i  # Compute spatial relation.
        if self.use_mod:
            input = input[:,0]**2.+input[:,1]**2.+input[:,2]**2.
            input = input.view(input.shape[0], 1)

        if x_j is not None:
            # In the first layer, we may not have any hidden node features,
            # so we only combine them in case they are present.
            input = torch.cat([x_j, input], dim=-1)

        self.input = input
        self.messages = self.mlp(input)

        return self.messages


# Edge convolution layer
class EdgeLayer(MessagePassing):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(EdgeLayer, self).__init__(aggr='max') #  "Max" aggregation.
        self.mlp = Sequential(Linear(2 * in_channels, mid_channels),
                       ReLU(),
                       Linear(mid_channels, mid_channels),
                       ReLU(),
                       Linear(mid_channels, out_channels))
        self.messages = 0.
        self.input = 0.

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        input = torch.cat([x_i, x_j - x_i], dim=-1)  # tmp has shape [E, 2 * in_channels]

        self.input = input
        self.messages = self.mlp(input)

        return self.messages


# Mix of EdgeNet and PointNet, using only modulus of the distance between neighbors
class EdgePointLayer(MessagePassing):
    def __init__(self, in_channels, mid_channels, out_channels, use_mod=1):
        # Message passing with "max" aggregation.
        super(EdgePointLayer, self).__init__('max')

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3, or 1 if only modulus is used).
        self.mlp = Sequential(Linear(2*in_channels-2, mid_channels),
                              ReLU(),
                              Linear(mid_channels, mid_channels),
                              ReLU(),
                              Linear(mid_channels, out_channels))

        self.messages = 0.
        self.input = 0.
        self.use_mod = use_mod

    def forward(self, x, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]

        pos_i, pos_j = x_i[:,:3], x_j[:,:3]

        input = pos_j - pos_i  # Compute spatial relation.
        input = input[:,0]**2.+input[:,1]**2.+input[:,2]**2.
        input = input.view(input.shape[0], 1)
        input = torch.cat([x_i, x_j[:,3:], input], dim=-1)

        self.input = input
        self.messages = self.mlp(input)

        return self.messages


# Node model for the MetaLayer
class NodeModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels):
        super(NodeModel, self).__init__()
        #self.node_mlp_1 = Sequential(Linear(in_channels,hidden_channels),  LeakyReLU(0.2), Linear(hidden_channels,hidden_channels),LeakyReLU(0.2), Linear(mid_channels,out_channels))
        #self.node_mlp_2 = Sequential(Linear(303,500), LeakyReLU(0.2), Linear(500,500),LeakyReLU(0.2), Linear(500,1))

        self.mlp = Sequential(Linear(in_channels*2, hidden_channels),
                              ReLU(),
                              Linear(hidden_channels, hidden_channels),
                              ReLU(),
                              Linear(hidden_channels, latent_channels))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index

        # define interaction tensor; every pair contains features from input and
        # output node together with
        #out = torch.cat([x[row], x[col], edge_attr], dim=1)
        out = torch.cat([x[row], x[col]], dim=1)
        #print("node pre", x.shape, out.shape)

        # take interaction feature tensor and embedd it into another tensor
        #out = self.node_mlp_1(out)
        out = self.mlp(out)
        #print("node mlp", out.shape)

        # compute the mean,sum and max of each embed feature tensor for each node
        out1 = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out3 = scatter_max(out, col, dim=0, dim_size=x.size(0))[0]
        out4 = scatter_min(out, col, dim=0, dim_size=x.size(0))[0]

        # every node contains a feature tensor with the pooling of the messages from
        # neighbors, its own state, and a global feature
        out = torch.cat([x, out1, out3, out4, u[batch]], dim=1)
        #print("node post", out.shape)

        #return self.node_mlp_2(out)
        return out

# Global model for the MetaLayer
class GlobalModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels):
        super(GlobalModel, self).__init__()
        #self.global_mlp = Seq(Linear(5, 500), LeakyReLU(0.2), Linear(500,500),LeakyReLU(0.2), Linear(500,2))

        self.global_mlp = Sequential(Linear((in_channels+latent_channels*3+2)*3+2, hidden_channels),
                              ReLU(),
                              Linear(hidden_channels, hidden_channels),
                              ReLU(),
                              Linear(hidden_channels, latent_channels))

        print("we",(in_channels+latent_channels*3+2), (in_channels+latent_channels*3+2)*3+2)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out1 = scatter_mean(x, batch, dim=0)
        out3 = scatter_max(x, batch, dim=0)[0]
        out4 = scatter_min(x, batch, dim=0)[0]
        out = torch.cat([u, out1, out3, out4], dim=1)
        #print("global pre",out.shape, x.shape, u.shape)
        out = self.global_mlp(out)
        #print("global post",out.shape)
        return out


#--------------------------------------------
# General Graph Neural Network architecture
#--------------------------------------------
class ModelGNN(torch.nn.Module):
    def __init__(self, use_model, node_features, n_layers, k_nn, hidden_channels=300, latent_channels=100, loop=False):
        super(ModelGNN, self).__init__()

        # Graph layers
        layers = []
        in_channels = node_features
        for i in range(n_layers):

            # Choose the model
            if use_model=="DeepSet":
                lay = Sequential(
                    Linear(in_channels, hidden_channels),
                	ReLU(),
                	Linear(hidden_channels, hidden_channels),
                	ReLU(),
                	Linear(hidden_channels, latent_channels))

            elif use_model=="GCN":
                lay = GCNConv(in_channels, latent_channels)

            elif use_model=="PointNet":
                lay = PointNetLayer(in_channels, hidden_channels, latent_channels)

            elif use_model=="EdgeNet":
                lay = EdgeLayer(in_channels, hidden_channels, latent_channels)
                #lay = EdgeConv(Sequential(Linear(2*in_channels, hidden_channels),ReLU(),Linear(hidden_channels, hidden_channels),ReLU(),Linear(hidden_channels, latent_channels)))  # Using the pytorch-geometric implementation, same result

            elif use_model=="EdgePoint":
                lay = EdgePointLayer(in_channels, hidden_channels, latent_channels)

            elif use_model=="MetaNet":
                if use_model=="MetaNet" and i==2:   in_channels = 610
                #lay = MetaLayer(node_model=NodeModel(in_channels, hidden_channels, latent_channels), global_model=GlobalModel(in_channels, hidden_channels, latent_channels))
                lay = MetaLayer(node_model=NodeModel(in_channels, hidden_channels, latent_channels))

            else:
                print("Model not known...")

            layers.append(lay)
            in_channels = latent_channels
            if use_model=="MetaNet":    in_channels = (node_features+latent_channels*3+2)


        self.layers = ModuleList(layers)

        lin_in = latent_channels*3+2
        if use_model=="MetaNet":    lin_in = (in_channels +latent_channels*3 +2)*3 + 2
        if use_model=="MetaNet" and n_layers==3:    lin_in = 2738
        self.lin = Sequential(Linear(lin_in, latent_channels),
                              ReLU(),
                              Linear(latent_channels, latent_channels),
                              ReLU(),
                              Linear(latent_channels, 2))

        self.k_nn = k_nn
        self.pooled = 0.
        self.h = 0.
        self.loop = loop
        if use_model=="PointNet" or use_model=="GCN":    self.loop = True
        self.namemodel = use_model

    def forward(self, data):

        x, pos, batch, u = data.x, data.pos, data.batch, data.u

        # Get edges using positions by computing the kNNs or the neighbors within a radius
        #edge_index = knn_graph(pos, k=self.k_nn, batch=batch, loop=self.loop)
        edge_index = radius_graph(pos, r=self.k_nn, batch=batch, loop=self.loop)

        # Start message passing
        for layer in self.layers:
            if self.namemodel=="DeepSet":
                x = layer(x)
            elif self.namemodel=="PointNet":
                x = layer(x=x, pos=pos, edge_index=edge_index)
            elif self.namemodel=="MetaNet":
                x, dumb, u = layer(x, edge_index, None, u, batch)
            else:
                x = layer(x=x, edge_index=edge_index)
            self.h = x
            x = x.relu()


        # Mix different global pooling layers
        addpool = global_add_pool(x, batch) # [num_examples, hidden_channels]
        meanpool = global_mean_pool(x, batch)
        maxpool = global_max_pool(x, batch)
        #self.pooled = torch.cat([addpool, meanpool, maxpool], dim=1)
        self.pooled = torch.cat([addpool, meanpool, maxpool, u], dim=1)

        # Final linear layer
        return self.lin(self.pooled)

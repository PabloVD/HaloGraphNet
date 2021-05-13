import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_cluster import knn_graph
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric import nn
from torch_cluster import knn_graph, radius_graph



# PointNet layer
class PointNetLayer(MessagePassing):
    def __init__(self, in_channels, mid_channels, out_channels):
        # Message passing with "max" aggregation.
        super(PointNetLayer, self).__init__('max')

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=2).
        self.mlp = Sequential(Linear(in_channels+3, mid_channels),
                              ReLU(),
                              Linear(mid_channels, mid_channels),
                              ReLU(),
                              Linear(mid_channels, out_channels))

        self.messages = 0.
        self.input = 0.


    def forward(self, x, pos, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_j, pos_j, pos_i):
        # x_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]

        input = pos_j - pos_i  # Compute spatial relation.

        if x_j is not None:
            # In the first layer, we may not have any hidden node features,
            # so we only combine them in case they are present.
            input = torch.cat([x_j, input], dim=-1)

        self.input = input
        self.messages = self.mlp(input)

        return self.messages


# Edge convolution layer
class EdgeConv(MessagePassing):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(EdgeConv, self).__init__(aggr='max') #  "Max" aggregation.
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

        input = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]

        self.input = input
        self.messages = self.mlp(input)

        return self.messages


# GeneralGNN
class ModelGNN(torch.nn.Module):
    def __init__(self, use_model, node_features, n_layers, k_nn, hidden_channels=300, latent_channels=100, loop=True):
        super(ModelGNN, self).__init__()

        # Graph layers
        layers = []
        in_channels = node_features
        for i in range(n_layers):
            # Choose the model
            if use_model=="FCN":
                lay = Sequential(
                    Linear(in_channels, hidden_channels),
                	ReLU(),
                	Linear(hidden_channels, hidden_channels),
                	ReLU(),
                	Linear(hidden_channels, latent_channels))
            elif use_model=="GCN":
                lay = nn.GCNConv(in_channels, latent_channels)
            elif use_model=="PointNet":
                lay = PointNetLayer(in_channels, hidden_channels, latent_channels)
                #self.layer2 = PointNetLayer(hidden_channels, latent_channels)
            elif use_model=="EdgeNet":
                lay = EdgeConv(in_channels, hidden_channels, latent_channels)
                #self.layer2 = EdgeConv(hidden_channels, latent_channels)
            elif use_model=="PPFNet":
                mlp1 = Sequential(Linear(in_channels+3, hidden_channels),
                                      ReLU(),
                                      Linear(hidden_channels, hidden_channels),
                                      ReLU(),
                                      Linear(hidden_channels, latent_channels))
                lay = nn.PPFConv( mlp1 )
            else:
                print("Model not known...")
            layers.append(lay)
            in_channels = latent_channels
            """if use_model=="FCN":
                self.layer1 = Sequential(
                    Linear(node_features, hidden_channels),
                	ReLU(),
                	Linear(hidden_channels, hidden_channels),
                	ReLU(),
                	Linear(hidden_channels, latent_channels))
            elif use_model=="GCN":
                self.layer1 = nn.GCNConv(node_features, latent_channels)
            elif use_model=="PointNet":
                self.layer1 = PointNetLayer(node_features, hidden_channels, latent_channels)
                #self.layer2 = PointNetLayer(hidden_channels, latent_channels)
            elif use_model=="EdgeNet":
                self.layer1 = EdgeConv(node_features, hidden_channels, latent_channels)
                #self.layer2 = EdgeConv(hidden_channels, latent_channels)
            elif use_model=="PPFNet":
                mlp1 = Sequential(Linear(node_features+3, hidden_channels),
                                      ReLU(),
                                      Linear(hidden_channels, hidden_channels),
                                      ReLU(),
                                      Linear(hidden_channels, latent_channels))
                self.layer1 = nn.PPFConv( mlp1 )
            else:
                print("Model not known...")"""
        #self.layer1 = Sequential(*layers)
        self.layers = layers

        self.lin = Sequential(Linear(latent_channels, latent_channels),
                              ReLU(),
                              Linear(latent_channels, latent_channels),
                              ReLU(),
                              Linear(latent_channels, 1))
        self.k_nn = k_nn
        self.pooled = 0.
        self.h = 0.
        self.loop = loop
        if use_model=="EdgeNet":    self.loop = False
        self.namemodel = use_model

    def forward(self, x, batch):

        # Get edges by computing the kNN graph using positions
        pos = x[:,:3]
        edge_index = knn_graph(pos, k=self.k_nn, batch=batch, loop=self.loop)
        #edge_index = radius_graph(pos, r=1., batch=batch, loop=self.loop)

        # Start bipartite message passing
        for layer in self.layers:
            if self.namemodel=="FCN":
                x = layer(x)
            elif self.namemodel=="PointNet":
                x = layer(x=x, pos=pos, edge_index=edge_index)
            else:
                x = layer(x=x, edge_index=edge_index)
            self.h = x
            x = x.relu()
        #h = self.layer2(h=h, pos=pos, edge_index=edge_index)

        # Mean global Pooling.
        self.pooled = global_mean_pool(x, batch)  # [num_examples, hidden_channels]

        # Final linear layer
        return self.lin(self.pooled)

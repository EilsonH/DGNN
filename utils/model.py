import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.graph import Graph

class Model(nn.Module):

        '''
        Directed Graph Neural Networks

        Input:
            in_channels (int): input channels of the netowrk
            num_classes (int): number of classes of the final classification
            graph_args (dict): arguments for graph building (see graph.py)

        Data:
            vs (N, C, T, Nv, P): input joint sequence
            es (N, C, T, Ne, P): input bone sequence

        Output:
            vs', es' with the same sizes
        '''

    def __init__(self, in_channels, num_classes, graph_args):
        super().__init__()

        self.graph = Graph(**graph_args)
        self.adj_source = torch.tensor(self.graph.adj_source, dtype = torch.float32, requires_grad = False)
        self.adj_target = torch.tensor(self.graph.adj_target, dtype = torch.float32, requires_grad = False)

        self.data_bn_vs = torch.BatchNorm1d(in_channels * self.graph.num_v)
        self.data_bn_es = torch.BatchNorm1d(in_channels * self.graph.num_e)

        self.dgnn = nn.ModuleList([
            DGNN(in_channels, 64, residual = False),
            DGNN(64, 64),
            DGNN(64, 64),
            DGNN(64, 64),
            DGNN(64, 128),
            DGNN(128, 128),
            DGNN(128, 128),
            DGNN(128, 256),
            DGNN(256, 256),
            DGNN(256, 256),
        ])

        self.fc = nn.Linear(256 * 2, 10)

    def forward(self, vs, es):
        N, C, T, V, M = vs.size()
        E = es.size()[3]

        vs = vs.permute(0, 4, 3, 1, 2).contiguous().view(N * M, V * C, T)
        es = es.permute(0, 4, 3, 1, 2).contiguous().view(N * M, E * C, T)

        vs = self.data_bn_vs(vs)
        es = self.data_bn_es(es)

        vs = vs.view(N * M, V, C, T).permute(0, 2, 3, 1)
        es = es.view(N * M, E, C, T).permute(0, 2, 3, 1)

        for dgnn in self.dgnn:
            vs, es = dgnn(self, vs, es, self.adj_source, self.adj_target)

        vs = vs.view(N, M, C, -1).mean(3).mean(1)
        es = es.view(N, M, C, -1).mean(3).mean(1)

        info = F.softmax(self.fc(torch.cat([vs, es], dim = 1)), dim = 1)

        return info

class BiConv2d(nn.Module):

    '''
    A 2d convolution that takes two input and performs convolutions respectively
    '''

    def __init__(self, in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0):
        super().__init__()

        self.conv_v = nn.Conv2d(in_channels, out_channels, (kernel_size, 1), (stride, 1), (padding, 0))
        self.conv_e = nn.Conv2d(in_channels, out_channels, (kernel_size, 1), (stride, 1), (padding, 0))

        self.bn_v = nn.BatchNorm2d(out_channels)
        self.bn_e = nn.BatchNorm2d(out_channels)

    def forward(self, vs, es):
        vs_prime = self.bn_v(self.conv_v(vs))
        es_prime = self.bn_e(self.conv_e(es))

        return vs_prime, es_prime

class DGNN(nn.Module):

    '''
    A single Directed Graph Neural Network Block

    Input:
        in_channels (int): input channels of the netowrk
        out_channels (int): output channels of the network

    Data:
        vs (N, C, T, Nv): input joint sequence
        es (N, C, T, Ne): input bone sequence
        adj_source (Nv, Ne): the adjacency matrix of the incoming edges
        adj_target (Nv, Ne): the adjacency matrix of the outgoing edges

    Output:
        vs', es' with the same sizes
    '''

    def __init__(self, in_channels, out_channels, t_kernel = 9, stride = 1, residual = False):
        super().__init__()

        self.fc_v = nn.Sequential(
            nn.Linear(in_channels * 3, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True))
        self.fc_e = nn.Sequential(
            nn.Linear(in_channels * 3, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True))

        self.tcn = BiConv2d(out_channels, out_channels, t_kernel, stride, (t_kernel -1) // 2)

        self.relu = nn.ReLU(inplace = True)

        if not residual:
            self.residual = lambda es, vs: (0, 0)

        elif in_channels == out_channels:
            self.residual = lambda es, vs: (es, vs)

        else:
            self.residual = BiConv2d(in_channels, out_channels)

    def forward(self, vs, es, adj_source, adj_target):
        res_vs, res_es = self.residual(vs, es)

        N, C, T= vs.size()[0: 3]
        new_in_edges = torch.matmul(vs, adj_source)
        new_out_edges = torch.matmul(vs, adj_target)
        new_source_vertices = torch.matmul(es, torch.t(adj_source))
        new_target_vertices = torch.matmul(es, torch.t(adj_target))


        input_hv = torch.cat([vs, new_source_vertices, new_target_vertices], dim = 1).permute(0, 2, 3, 1)
        input_he = torch.cat([es, new_in_edges, new_out_edges], dim = 1).permute(0, 2, 3, 1)
        vs_prime, es_prime = self.tcn(self.fc_v(input_hv).permute(0, 3, 1, 2),
            self.fc_e(input_he).permute(0, 3, 1, 2))

        return self.relu(vs_prime + res_vs), self.relu(es_prime + res_es)

if __name__ = '__main__':
    print("Debugging...")
    model = Model(in_channels = 3, num_class = 400, graph_args = {'layout':'openpose', 'strategy':'spatial'})

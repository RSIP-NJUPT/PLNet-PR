import math
import torch 
import torch.nn as nn
import torch.nn.functional as F



class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features,init='xavier', bias=True, ):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.weight.retain_grad()
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
            self.bias.retain_grad()
        else:
            self.register_parameter('bias', None)

        if init == 'uniform':
            #print("| Uniform Initialization")
            self.reset_parameters_uniform()
        elif init == 'xavier':
            #print("| Xavier Initialization")
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            #print("| Kaiming Initialization")
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        
class GPRLM(nn.Module):
    def  __init__(self, in_channels=256, roi_feat_size=7, fc_out_channels=518, dropout=0.5, n_shift=4,n_anchors=512, init='xavier'):
        super(GPRLM, self).__init__()
        
        self.gc1 = GraphConvolution(fc_out_channels, fc_out_channels,init=init)
        self.gc2 = GraphConvolution(fc_out_channels, fc_out_channels, init=init)
        self.dropout = dropout

        self.down_fc2 = nn.Linear(fc_out_channels, fc_out_channels)

        self.down_fc1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=fc_out_channels, 
                groups=fc_out_channels, kernel_size=roi_feat_size),
            nn.ReLU(),
            nn.BatchNorm2d(fc_out_channels),
            )
        self.up_fc = nn.Linear(fc_out_channels, in_channels * (roi_feat_size//2) * (roi_feat_size//2))
        
        nhid = fc_out_channels
        self.graph_fc1 = nn.Linear(nhid, nhid)
        self.graph_fc2 = nn.Linear(nhid, nhid)
    
        self.n_shift = n_shift
        self.shift_fc1 = nn.Linear(nhid, nhid//2)
        self.shift_fc2 = nn.Linear(nhid//2, nhid)
        self.shift_ch_idx = [0]

        for i in range(n_shift+1):
            start_c = i * (nhid//2)//n_shift + nhid//2
            self.shift_ch_idx.append(start_c)



    
    def feat2graph(self, bbox_feat):
        # adj = adj #+ overlap
        qx = self.graph_fc1(bbox_feat)
        kx = self.graph_fc2(bbox_feat)

        dot_mat = qx.matmul(kx.transpose(-1, -2))
        adj = F.normalize(dot_mat, p=2, dim=-1)
        return adj
    
    def forward(self, bbox_feats, rois, overlaps):
        # bbox_feats = bbox_feats.mean([2,3],keepdim=True)
        batch_idx = rois[:, 0].unique()
        bbox_feat_lst = []
        for batch in batch_idx:
            batch_mask = (rois[:, 0] == batch).detach()
            overlap = overlaps[batch_mask, :][:, batch_mask]
            
            bbox_feat = bbox_feats[batch_mask, :]
            N,C,H,W = bbox_feat.shape
            ori_bbox_feat = bbox_feat.flatten(1)
            bbox_feat = F.relu(self.down_fc1(bbox_feat).flatten(1))
            bbox_feat = self.down_fc2(bbox_feat)

            # shift
            _, shift_idx = torch.topk(overlap, self.n_shift+1, dim=1, largest=True)
            shift_feat_lst = []
            for i in range(self.n_shift+1):
                start_c = self.shift_ch_idx[i]
                end_c = self.shift_ch_idx[i+1]
                shift_feat = torch.index_select(bbox_feat, dim=0, index=shift_idx[:, i])[:, start_c:end_c]

                shift_feat_lst.append(shift_feat)
            shift_feat = torch.concat(shift_feat_lst,dim=1)
            shift_feat = F.relu(self.shift_fc1(shift_feat))
            shift_feat = self.shift_fc2(shift_feat)

            res_bbox_feat = bbox_feat

            bbox_feat = shift_feat  + bbox_feat

            adj = self.feat2graph(bbox_feat)
            bbox_feat = F.relu(self.gc1(bbox_feat, adj))
            bbox_feat = F.dropout(bbox_feat, self.dropout, training=self.training)
            bbox_feat = self.gc2(bbox_feat, adj) + res_bbox_feat
            #bbox_feat = self.up_fc(bbox_feat) + ori_bbox_feat
            bbox_feat = self.up_fc(bbox_feat).reshape([N,C, (H//2), (W//2)])
            bbox_feat = F.interpolate(bbox_feat, size=[H, W], mode='bilinear').flatten(1)
            bbox_feat = bbox_feat + ori_bbox_feat

            bbox_feat_lst.append(bbox_feat.reshape(N, C, H, W))
        bbox_feat_lst = torch.concat(bbox_feat_lst, dim=0)
        return bbox_feat_lst


        

if __name__=='__main__':
    print('test well')
    N = 5
    D = 256*40*40
    bs = 2
    size = [(bs, 256*4*4, 10, 10), (bs, 256*2*2, 20, 20), (bs, 256, 40, 40),
             (bs, 256//2//2, 80, 80), (bs, 256//4//4, 160, 160)]
    x = [torch.randn(s, device='cuda') for s in size]

    


    # x = torch.randn([N, D], device='cuda')

    # 
    # net = GCN(nfeat=D, nhid=2048,in_channels=256, dropout=0.01)
    # print(net(x).shape)

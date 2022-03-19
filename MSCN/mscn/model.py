import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Define model architecture

# original MSCN
class SetConv(nn.Module):
    def __init__(self, sample_feats, predicate_feats, join_feats, hid_units):
        super(SetConv, self).__init__()
        self.sample_mlp1 = nn.Linear(sample_feats, hid_units)
        self.sample_mlp2 = nn.Linear(hid_units, hid_units)
        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
        self.join_mlp1 = nn.Linear(join_feats, hid_units)
        self.join_mlp2 = nn.Linear(hid_units, hid_units)
        self.out_mlp1 = nn.Linear(hid_units * 3, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, samples, predicates, joins, sample_mask, predicate_mask, join_mask):
        # samples has shape [batch_size x num_joins+1 x sample_feats]
        # predicates has shape [batch_size x num_predicates x predicate_feats]
        # joins has shape [batch_size x num_joins x join_feats]

        hid_sample = F.relu(self.sample_mlp1(samples))
        hid_sample = F.relu(self.sample_mlp2(hid_sample))
        hid_sample = hid_sample * sample_mask  # Mask
        hid_sample = torch.sum(hid_sample, dim=1, keepdim=False)
        sample_norm = sample_mask.sum(1, keepdim=False)
        hid_sample = hid_sample / sample_norm  # Calculate average only over non-masked parts

        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
        hid_predicate = hid_predicate * predicate_mask
        hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
        predicate_norm = predicate_mask.sum(1, keepdim=False)
        hid_predicate = hid_predicate / predicate_norm

        hid_join = F.relu(self.join_mlp1(joins))
        hid_join = F.relu(self.join_mlp2(hid_join))
        hid_join = hid_join * join_mask
        hid_join = torch.sum(hid_join, dim=1, keepdim=False)
        join_norm = join_mask.sum(1, keepdim=False)
        hid_join = hid_join / join_norm

        hid = torch.cat((hid_sample, hid_predicate, hid_join), 1)
        hid = F.relu(self.out_mlp1(hid))
        out = torch.sigmoid(self.out_mlp2(hid))
        return out
class MSCN(nn.Module):
    def __init__(self, sample_feats, predicate_feats, join_feats, hid_units):
        super(MSCN, self).__init__()
        self.sample_mlp1 = nn.Linear(sample_feats, hid_units)
        self.sample_mlp2 = nn.Linear(hid_units, hid_units)
        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
        self.join_mlp1 = nn.Linear(join_feats, hid_units)
        self.join_mlp2 = nn.Linear(hid_units, hid_units)
        self.out_mlp1 = nn.Linear(hid_units * 3, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, samples, predicates, joins, sample_mask, predicate_mask, join_mask):
        # samples has shape [batch_size x num_joins+1 x sample_feats]
        # predicates has shape [batch_size x num_predicates x predicate_feats]
        # joins has shape [batch_size x num_joins x join_feats]

        hid_sample = F.relu(self.sample_mlp1(samples))
        hid_sample = F.relu(self.sample_mlp2(hid_sample))
        hid_sample = hid_sample * sample_mask  # Mask
        hid_sample = torch.sum(hid_sample, dim=1, keepdim=False)
        sample_norm = sample_mask.sum(1, keepdim=False)
        hid_sample = hid_sample / sample_norm  # Calculate average only over non-masked parts

        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
        hid_predicate = hid_predicate * predicate_mask
        hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
        predicate_norm = predicate_mask.sum(1, keepdim=False)
        hid_predicate = hid_predicate / predicate_norm

        hid_join = F.relu(self.join_mlp1(joins))
        hid_join = F.relu(self.join_mlp2(hid_join))
        hid_join = hid_join * join_mask
        hid_join = torch.sum(hid_join, dim=1, keepdim=False)
        join_norm = join_mask.sum(1, keepdim=False)
        hid_join = hid_join / join_norm

        hid = torch.cat((hid_sample, hid_predicate, hid_join), 1)
        hid = F.relu(self.out_mlp1(hid))
        out = torch.sigmoid(self.out_mlp2(hid))
        return out

# FCN + Pool
class MSCN_FCN(nn.Module):
    def __init__(self, feature_sizes, hid_units):
        super(MSCN_FCN, self).__init__()

        # input_dim = sum
        num_tables = len(feature_sizes)
        
        self.num_tables = num_tables
        # self.feature_sizes = feature_sizes
        self.feature_starts = list()
        self.feature_end = list()
        self.hid_units = hid_units
        self.fcns =  nn.ModuleList()
        start = 0
        for i in range(0, num_tables):
            self.feature_starts.append(start)
            start += feature_sizes[i]
            self.fcns.append(
                nn.Sequential(
                    nn.Linear(feature_sizes[i], hid_units),
                    nn.ReLU(),
                    nn.Linear(hid_units,hid_units),
                    nn.ReLU()
                )
            )
        self.feature_starts.append(start)
        self.out_mlp1 = nn.Linear(hid_units, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, features, masks):
        # samples has shape [batch_size x num_joins+1 x sample_feats]
        # predicates has shape [batch_size x num_predicates x predicate_feats]
        # joins has shape [batch_size x num_joins x join_feats]
        
        batch = features.shape[0]
        fcn_output = torch.zeros((batch, self.hid_units), device=features.device)
        
        feature_start = 0
        for i in range(0, self.num_tables):
            features_i = features[:, self.feature_starts[i] : self.feature_starts[i+1]]
            mask_i = masks[:,i].unsqueeze(1)
            output = torch.where(mask_i == 1, self.fcns[i](features_i), torch.zeros_like(fcn_output))
            fcn_output += output
        
        fcn_norm = masks.sum(1, keepdim=False).unsqueeze(1)
        fcn_output = fcn_output / fcn_norm

        hid = F.relu(self.out_mlp1(fcn_output))
        out = torch.sigmoid(self.out_mlp2(hid))
        return out

# FCN with fixed size operands (for datasets with numeric columns only)
class FCN(nn.Module):
    def __init__(self, table_feats, sample_feats, num_columns, predicate_feats, join_feats, hid_units):
        super(FCN, self).__init__()
       
        input_dim = table_feats + table_feats * sample_feats + num_columns * predicate_feats + join_feats
        
        self.mlp1 = nn.Linear(input_dim, hid_units)
        self.mlp2 = nn.Linear(hid_units, hid_units)
        self.mlp3 = nn.Linear(hid_units, 1)


    def forward(self, tables, samples, predicates, joins):
       
        input = torch.cat((tables, samples,predicates, joins), 1)

        hid = F.relu(self.mlp1(input))
        hid = F.relu(self.mlp2(hid))
        out = torch.sigmoid(self.mlp3(hid))

        return out

# FCN with variable size operands (for datasets with numeric and string type columns)
class FCN_var(nn.Module):
    def __init__(self, table_feats, sample_feats, predicate_feats, join_feats, hid_units):
        super(FCN_var, self).__init__()
       
        input_dim = table_feats + table_feats * sample_feats + predicate_feats + join_feats
        
        self.mlp1 = nn.Linear(input_dim, hid_units)
        self.mlp2 = nn.Linear(hid_units, hid_units)
        self.mlp3 = nn.Linear(hid_units, 1)


    def forward(self, tables, samples, predicates, joins):
       
        input = torch.cat((tables, samples, predicates, joins), 1)

        hid = F.relu(self.mlp1(input))
        hid = F.relu(self.mlp2(hid))
        out = torch.sigmoid(self.mlp3(hid))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F



def get_activation(s_act):
    if s_act == 'relu':
        return nn.ReLU(inplace=True)
    elif s_act == 'sigmoid':
        return nn.Sigmoid()
    elif s_act == 'softplus':
        return nn.Softplus()
    elif s_act == 'linear':
        return None
    elif s_act == 'tanh':
        return nn.Tanh()
    elif s_act == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif s_act == 'softmax':
        return nn.Softmax(dim=1)
    elif s_act == 'swish':
        return nn.SiLU(inplace=True)
    else:
        raise ValueError(f'Unexpected activation: {s_act}')


def get_activation_F(x, s_act):
    """functional version of get_activation"""
    if s_act == 'relu':
        return F.relu(x, inplace=True)
    elif s_act == 'sigmoid':
        return F.sigmoid(x)
    elif s_act == 'softplus':
        return F.softplus(x)
    elif s_act == 'linear':
        return x
    elif s_act == 'tanh':
        return F.tanh(x)
    elif s_act == 'leakyrelu':
        return F.leaky_relu(x, 0.2, inplace=True)
    elif s_act == 'softmax':
        return F.softmax(x, dim=1)
    elif s_act == 'spherical':
        return x / x.norm(dim=1, p=2, keepdim=True)
    elif s_act == 'sphericalV2':
        return x / x.norm(dim=1, p=2, keepdim=True)
    elif s_act == 'swish':
        return F.silu(x)
    else:
        raise ValueError(f'Unexpected activation: {s_act}')


class FCNet(nn.Module):
    """fully-connected network"""
    def __init__(self,
            in_dim,
            out_dim,
            l_hidden=(50,),
            activation='sigmoid',
            out_activation='linear',
            use_spectral_norm=False,
            flatten_input=False,
            batch_norm=False,
            out_batch_norm=False,
            learn_out_scale=False,
            bias=True):
        super().__init__()
        l_neurons = tuple(l_hidden) + (out_dim,)
        if isinstance(activation, str):
            activation = (activation,) * len(l_hidden)
        activation = tuple(activation) + (out_activation,)

        l_layer = []
        prev_dim = in_dim
        for i_layer, (n_hidden, act) in enumerate(zip(l_neurons, activation)):
            if use_spectral_norm and i_layer < len(l_neurons) - 1:  # don't apply SN to the last layer
                l_layer.append(P.spectral_norm(nn.Linear(prev_dim, n_hidden)))
            else:
                l_layer.append(nn.Linear(prev_dim, n_hidden, bias=bias))
            if batch_norm:
                if out_batch_norm:
                    l_layer.append(nn.BatchNorm1d(num_features=n_hidden))
                else:
                    if i_layer < len(l_neurons) - 1:  # don't apply BN to the last layer
                        l_layer.append(nn.BatchNorm1d(num_features=n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        # add learnable scaling operation at the end
        if learn_out_scale:
            l_layer.append(nn.Linear(1, 1, bias=True))

        self.net = nn.Sequential(*l_layer)
        self.in_dim = in_dim
        self.out_shape = (out_dim,)
        self.flatten_input = flatten_input

    def forward(self, x):
        if self.flatten_input and len(x.shape) == 4:
            x = x.view(len(x), -1)
        return self.net(x)
    
class Flow_Module(nn.Module):
    def __init__(self,intermediate_num=1000,feature_num=768,flow_num=2,complex=False, spherical=True):
        super().__init__()
        self.spherical = spherical
        if complex:
            self.flows = nn.ModuleList([nn.Sequential(nn.Linear(feature_num,intermediate_num),
                                          nn.GELU(),nn.Linear(intermediate_num,intermediate_num),
                                          nn.GELU(),nn.Linear(intermediate_num,feature_num)) for i in range(flow_num)])
        else:
            self.flows = nn.ModuleList([nn.Sequential(nn.Linear(feature_num,intermediate_num),nn.GELU(),nn.Linear(intermediate_num,feature_num)) for i in range(flow_num)])

    def forward(self,z):
        z_before = z
        g_z_cum_norm = 0

        for flow in self.flows:
            res = flow(z_before)
            g_z_cum_norm += res.norm(dim=1, p=2, keepdim=True).sum(dim=1).mean() 
            z_before = res+z_before
            if self.spherical:
                z_before = z_before / z_before.norm(dim=1, p=2, keepdim=True)
        return z_before, g_z_cum_norm
        
        
class MLP(nn.Module):
    def __init__(self,intermediate_num=100,feature_num=768,num_hidden=2):
        super().__init__()
        
        self.flows = nn.ModuleList([nn.Sequential(nn.Linear(feature_num,intermediate_num),nn.GELU())] + [nn.Sequential(nn.Linear(intermediate_num,intermediate_num),nn.GELU()) for i in range(num_hidden)] + [nn.Linear(intermediate_num,feature_num)])    
                                       
    def forward(self,z):
        z_before = z
        g_z_cum_norm = 0

        for flow in self.flows:
            res = z_before
            z_before = flow(z_before)
            if(res.shape==z_before.shape):
                z_before = z_before + res
        return z_before, torch.tensor(g_z_cum_norm)
    
class MLP_onedimoutput(nn.Module):
    def __init__(self,intermediate_num=100,feature_num=768,num_hidden=2,**kwargs):
        super().__init__()
        
        self.flows = nn.ModuleList([nn.Sequential(nn.Linear(feature_num,intermediate_num),nn.GELU())] + [nn.Sequential(nn.Linear(intermediate_num,intermediate_num),nn.GELU()) for i in range(num_hidden)] + [nn.Linear(intermediate_num,1)])    
                                       
    def forward(self,z):
        for flow in self.flows:
            z = flow(z)
            
        return z, torch.tensor(0, dtype=torch.float32)
        
        # z_before = z
        # g_z_cum_norm = 0

        # for flow in self.flows:
        #     res = z_before
        #     z_before = flow(z_before)
        #     if(res.shape==z_before.shape):
        #         z_before = z_before + res
        # return z_before, torch.tensor(g_z_cum_norm)

"""
Reconfiguration Flow
"""
import torch
import torch.nn as nn
from torch.autograd import grad
import sys
from models.modules import FCNet


class FlowNet(nn.Module):
    """
    network architecture for multiple flows.
    each flow v(x) is an MLP (FCNet): x -> x + v(x)
    """
    def __init__(self, input_dim, n_flow, l_hidden, activation='relu'):
        super().__init__()
        self.flows = nn.ModuleList([FCNet(input_dim, 
                                          input_dim, l_hidden, activation=activation, out_activation='linear') for _ in range(n_flow)])
        self.input_dim = input_dim
        self.n_flow = n_flow

    def forward(self, x):
        for flow in self.flows:
            x = x + flow(x)
        return x

    def forward_train(self, x):
        """
        additionally returns the list of flows
        additional output can be used for regularization
        """
        l_flow = []
        for flow in self.flows:
            v = flow(x)
            x = x + v
            l_flow.append(v)
        return x, l_flow


class LangevinSampler:
    """Langevin Monte Carlo Sampler"""
    def __init__(self, n_step, stepsize, noise_std, spherical=False, normalize=False):
        self.n_step = n_step
        self.stepsize = stepsize
        self.noise_std = noise_std
        self.spherical = spherical 
        self.normalize = normalize

    def sample(self, x_init, energy_fn):
        x = x_init.clone()
        x.requires_grad = True
        for i_step in range(self.n_step):
            energy = energy_fn(x)
            g = grad(energy.sum(), x, create_graph=True)[0]
            if self.normalize:
                g = g / g.norm(p=2,dim=1,keepdim=True)
            x = x - self.stepsize * g + torch.randn_like(x) * self.noise_std
            if self.spherical:
                x = x / x.norm(p=2,dim=1,keepdim=True)
        return x.detach()


class ReconfFlow(nn.Module):
    def __init__(self, net, NS, proj_b, reg_norm=None):
        """
        Assume the following two quantities are computed separately:
        NS: null space vectors (data_dim x subspace_dim)
        proj_b: projection center
        """
        super().__init__()
        self.net = net
        self.register_buffer('proj_b', proj_b.clone())
        self.register_buffer('NS', NS.clone())
        self.reg_norm = reg_norm

    def forward(self, x):
        z = self.net(x)
        return self.proj_error(z)

    def predict(self, x):
        return self(x)
    
    def proj_error(self, z):
        perp = (z - self.proj_b) @ self.NS
        proj_error = (perp ** 2).sum(dim=1)
        return proj_error

    def train_step(self, x, opt):
        opt.zero_grad()
        z, l_flow = self.net.forward_train(x)
        subspace_loss = self.proj_error(z).mean()
        loss = subspace_loss
        d_result = {'subspace_loss': subspace_loss.item()}
        
        vel = (torch.stack(l_flow) ** 2).sum(dim=-1).mean()
        d_result['vel'] = vel.item()
        if self.reg_norm is not None:
            loss += self.reg_norm * vel

        d_result['loss'] = loss.item()
        loss.backward()
        opt.step()
        return d_result


class ReconfFlowEBM(ReconfFlow):
    """
    ReconfFlow with energy-based training
    For now, we use Contrastive Divergence (CD) for EBM training
    """
    def __init__(self, net, sampler, NS, proj_b, reg_norm=None, gamma=1., neg_initial_mode='cd'):
        super().__init__(net=net, NS=NS, proj_b=proj_b, reg_norm=reg_norm)
        self.sampler = sampler
        self.gamma = gamma
        self.neg_initial_mode = neg_initial_mode
        assert neg_initial_mode in {'cd', 'projected'}

    def init_neg(self, x):
        """
        get initial points for negative samples
        """
        if self.neg_initial_mode == 'cd':
            return x
        elif self.neg_initial_mode == 'projected':
            return self.net(x)
 
    def train_step(self, x, opt):
        opt.zero_grad()
        z, l_flow = self.net.forward_train(x)
        pos_E = self.proj_error(z)

        # negative sampling
        x0 = self.init_neg(x).detach()
        xn = self.sampler.sample(x0, self)
        zn, l_flow_n = self.net.forward_train(xn.detach())
        neg_E = self.proj_error(zn)

        loss = pos_E.mean() - neg_E.mean()
        d_result = {'subspace_loss': pos_E.mean(),
                    'ediff': loss.item(),
                    'x': x.detach().cpu(),
                     'xn': xn.detach().cpu()}
        
        if self.reg_norm is not None:
            vel = (torch.stack(l_flow) ** 2).sum(dim=-1).mean()
            loss += self.reg_norm * vel
            d_result['vel'] = vel.item()

            veln = (torch.stack(l_flow_n) ** 2).sum(dim=-1).mean()
            loss += self.reg_norm * veln
            d_result['veln'] = veln.item()

        loss += self.gamma * (neg_E ** 2).mean()
        d_result['loss'] = loss.item()
        
        loss.backward()
        opt.step()
        return d_result


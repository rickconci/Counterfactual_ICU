import torch
from torch import Tensor, nn

import numpy as np


class MLPSimple(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, depth, activations=None, dropout_p=None, final_activation=None):
        super().__init__()
        print(f"Initializing MLPSimple: input_dim={input_dim}, output_dim={output_dim}, hidden_dim={hidden_dim}, depth={depth}")
        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())

        # Define the output layer with an optional final activation
        if final_activation is not None:
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                final_activation
            )
        else:
            self.output_layer = nn.Sequential(nn.Linear(hidden_dim, output_dim))
        
        if activations is None:
            activations = [nn.ReLU() for _ in range(depth)]
        
        if dropout_p is None:
            dropout_p = [0. for _ in range(depth)]
        
        assert len(activations) == depth, "Mismatch between depth and number of provided activations."
        
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout_p[i]), activations[i])
            for i in range(depth)
        ])

    def forward(self, x):
        x = self.input_layer(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.output_layer(x)
        return x
    

CV_params = {"r_tpr_mod": 0.,
            "f_hr_max": 3.0,
            "f_hr_min": 2.0 / 3.0,
            "r_tpr_max": 2.134,
            "r_tpr_min": 0.5335,
            "sv_mod": 0.001,
            "ca": 4.0,
            "cv": 111.0,

            # dS/dt parameters
            "k_width": 0.1838,
            "p_aset": 70,
            "tau": 20,
            "p_0lv": 2.03,
            "r_valve": 0.0025,
            "k_elv": 0.066,
            "v_ed0": 7.14,
            "T_sys": 4. / 15.,
            "cprsw_max": 103.8,
            "cprsw_min": 25.9,
            
            "pa_divisor": 100,
            "pv_divisor": 100,
            "s_divisor":1,
            "sv_divisor":100
}


CV_params_max_min_2STD = {'max_pa': torch.tensor(130.9138),
                'min_pa': torch.tensor(82.4030),
                'max_pv': torch.tensor(60.1224),
                'min_pv': torch.tensor(20.9298),
                'max_s': torch.tensor(0.2204),
                'min_s': torch.tensor(0.0722),
                'max_sv': torch.tensor(100.7716),
                'min_sv': torch.tensor(89.1963),
}

CV_params_max_min_2_5STD = {'max_pa': torch.tensor(136.97),
                'min_pa': torch.tensor(76.3392),
                'max_pv': torch.tensor(65.0215),
                'min_pv': torch.tensor(16.0307),
                'max_s': torch.tensor(0.2390),
                'min_s': torch.tensor(0.0537),
                'max_sv': torch.tensor(102.2185),
                'min_sv': torch.tensor(87.7494),
}



CV_params_prior_mu = {
    'pa': torch.tensor(89.2),
    'pv': torch.tensor(50.3),
    's': torch.tensor(0.037),
    'sv': torch.tensor(88.6),
    "sv_mod": 0.001,
    "ca": 4.0,
    "cv": 111.0,
    "k_width": 0.1838,
    "p_aset": 70,
    "tau": 20,
    "p_0lv": 2.03,
    "r_valve": 0.0025,
    "k_elv": 0.066,
    "v_ed0": 7.14,
    "T_sys": 4.0 / 15.0,
    "hr": 1.83333,
    "tpr": 1.33375,
    "cprsw": 64.85,
   
}

CV_params_prior_sigma = {
   'pa': torch.tensor(4.9),
   'pv': torch.tensor(10.2),
   's': torch.tensor(0.024),
   'sv': torch.tensor(1.8)
}


def process_input(input_list):
    # Check if the input list is exactly [0, 1]
    if input_list == [0, 1]:
        return [0, 1]
    # Check if the list contains exactly one element
    elif len(input_list) == 1:
        return [0]
    # Optionally handle other cases
    else:
        return "Unexpected input"  
    

def str2bool(v):
    """Used for boolean arguments in argparse; avoiding `store_true` and `store_false`."""
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')


def _stable_division(a, b, epsilon=1e-7):
    b = torch.where(b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * ((b>0).float()*2-1))
    return a / b


def interpolate_colors(color1, color2, num_colors, rgb=False):
    colors = []
    for t in np.linspace(0, 1, num_colors):
        r = int(color1[0] + (color2[0] - color1[0]) * t)
        g = int(color1[1] + (color2[1] - color1[1]) * t)
        b = int(color1[2] + (color2[2] - color1[2]) * t)
        if rgb:
            colors.append(f'rgb({r}, {g}, {b})')
        else:
            colors.append((r, g, b))
    return colors


class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0, start = 0):
            self._iters = max(1, iters)
            self._val = 0 # maxval / self._iters
            self._maxval = maxval
            self._start = start
            self.current_iter = 0
    
    def step(self):
        if self.current_iter>self._iters:
            self._val = min(self._maxval, self._val + self._maxval / self._iters)
        self.current_iter += 1

    @property
    def val(self):
        return self._val
    



def ensure_scalar(value):
    """Converts a single-element tensor or array to a Python scalar if necessary."""
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value
    elif isinstance(value, np.ndarray):
        return value.item() if value.size == 1 else value
    else:
        return value






def gaussian_nll_loss(input, target, var, *, full=False, eps=1e-6, reduction='mean'):
    r"""Gaussian negative log likelihood loss.
    See :class:`~torch.nn.GaussianNLLLoss` for details.
    Args:
        input: expectation of the Gaussian distribution.
        target: sample from the Gaussian distribution.
        var: tensor of positive variance(s), one for each of the expectations
            in the input (heteroscedastic), or a single one (homoscedastic).
        full: ``True``/``False`` (bool), include the constant term in the loss
            calculation. Default: ``False``.
        eps: value added to var, for stability. Default: 1e-6.
        reduction: specifies the reduction to apply to the output:
            `'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output is the average of all batch member losses,
            ``'sum'``: the output is the sum of all batch member losses.
            Default: ``'mean'``.
    """
    if not torch.jit.is_scripting():
        tens_ops = (input, target, var)
        if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
            return handle_torch_function(
                gaussian_nll_loss, tens_ops, input, target, var, full=full, eps=eps, reduction=reduction)

    # Inputs and targets much have same shape
    #input = input.view(input.size(0), -1)
    #target = target.view(target.size(0), -1)
    if input.size() != target.size():
        raise ValueError("input and target must have same size")

    # Second dim of var must match that of input or be equal to 1
    #var = var.view(input.size(0), -1)
    if var.size() != input.size():
        raise ValueError("var is of incorrect size")

    # Check validity of reduction mode
    if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if torch.any(var < 0):
        raise ValueError("var has negative entry/entries")

    # Clamp for stability
    var = var.clone()
    with torch.no_grad():
        var.clamp_(min=eps)

    # Calculate loss (without constant)
    #loss = 0.5 * (torch.log(var) + (input - target)**2 / var).view(input.size(0), -1).sum(dim=1)
    loss = 0.5 * (torch.log(var) + (input - target)**2 / var)

    # Add constant to loss term if required
    if full:
        D = input.size(1)
        loss = loss + 0.5 * D * math.log(2 * math.pi)

    # Apply reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

class GaussianNLLLoss(torch.nn.modules.loss._Loss):
    r"""Gaussian negative log likelihood loss.

    The targets are treated as samples from Gaussian distributions with
    expectations and variances predicted by the neural network. For a
    ``target`` tensor modelled as having Gaussian distribution with a tensor
    of expectations ``input`` and a tensor of positive variances ``var`` the loss is:

    .. math::
        \text{loss} = \frac{1}{2}\left(\log\left(\text{max}\left(\text{var},
        \ \text{eps}\right)\right) + \frac{\left(\text{input} - \text{target}\right)^2}
        {\text{max}\left(\text{var}, \ \text{eps}\right)}\right) + \text{const.}

    where :attr:`eps` is used for stability. By default, the constant term of
    the loss function is omitted unless :attr:`full` is ``True``. If ``var`` is not the same
    size as ``input`` (due to a homoscedastic assumption), it must either have a final dimension
    of 1 or have one fewer dimension (with all other sizes being the same) for correct broadcasting.

    Args:
        full (bool, optional): include the constant term in the loss
            calculation. Default: ``False``.
        eps (float, optional): value used to clamp ``var`` (see note below), for
            stability. Default: 1e-6.
        reduction (string, optional): specifies the reduction to apply to the
            utput:``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
            will be applied, ``'mean'``: the output is the average of all batch
            member losses, ``'sum'``: the output is the sum of all batch member
            losses. Default: ``'mean'``.

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input, or same shape as the input
          but with one dimension equal to 1 (to allow for broadcasting)
        - Var: :math:`(N, *)`, same shape as the input, or same shape as the input but
          with one dimension equal to 1, or same shape as the input but with one fewer
          dimension (to allow for broadcasting)
        - Output: scalar if :attr:`reduction` is ``'mean'`` (default) or
          ``'sum'``. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same
          shape as the input

    Examples::
        >>> loss = nn.GaussianNLLLoss()
        >>> input = torch.randn(5, 2, requires_grad=True)
        >>> target = torch.randn(5, 2)
        >>> var = torch.ones(5, 2, requires_grad=True) #heteroscedastic
        >>> output = loss(input, target, var)
        >>> output.backward()

        >>> loss = nn.GaussianNLLLoss()
        >>> input = torch.randn(5, 2, requires_grad=True)
        >>> target = torch.randn(5, 2)
        >>> var = torch.ones(5, 1, requires_grad=True) #homoscedastic
        >>> output = loss(input, target, var)
        >>> output.backward()

    Note:
        The clamping of ``var`` is ignored with respect to autograd, and so the
        gradients are unaffected by it.

    Reference:
        Nix, D. A. and Weigend, A. S., "Estimating the mean and variance of the
        target probability distribution", Proceedings of 1994 IEEE International
        Conference on Neural Networks (ICNN'94), Orlando, FL, USA, 1994, pp. 55-60
        vol.1, doi: 10.1109/ICNN.1994.374138.
    """
    __constants__ = ['full', 'eps', 'reduction']
    full: bool
    eps: float

    def __init__(self, *, full: bool = False, eps: float = 1e-6, reduction: str = 'mean') -> None:
        super(GaussianNLLLoss, self).__init__(None, None, reduction)
        self.full = full
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor, var: Tensor) -> Tensor:
        return gaussian_nll_loss(input, target, var, full=self.full, eps=self.eps, reduction=self.reduction)




def print_model_details(model, indent=0):
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(' ' * indent, f'{name} | {module.__class__.__name__} | {num_params} params')
        if list(module.children()):
            print_model_details(module, indent + 4)

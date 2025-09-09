import random

import numpy as np
import torch
from torch import Tensor, nn


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")  # CUDA GPU
    elif (
        torch.backends.mps.is_available()
    ):  # Check for MPS availability (specific to PyTorch 1.12+)
        return torch.device("mps")  # MPS-supported GPU on Apple Silicon
    else:
        return torch.device("cpu")  # Default to CPU


device = get_device()


def select_tensor_by_index_list_advanced(tensor, index_list):
    # Convert index list to a tensor of type long
    indices = torch.tensor(index_list, dtype=torch.long, device=tensor.device)

    # Use index_select to gather the specified indices
    selected_tensor = torch.index_select(tensor, -1, indices)

    return selected_tensor


def manual_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def scale_unnormalised_experts(expert_vals):
    assert expert_vals.shape[1] == len(
        CV_params_divisors
    ), "expert_vals does not match the expected number of features."
    # Keys must be in the same order as dimensions in expert_dims tensor
    keys = ["pa", "pv", "s", "sv"]
    scaled_vals = [
        expert_vals[:, i] * CV_params_divisors[keys[i]] for i in range(len(keys))
    ]
    scaled_tensor = torch.stack(scaled_vals, dim=1)

    if len(keys) < expert_vals.shape[1]:
        remaining_vals = expert_vals[:, len(keys) :]
        remaining_vals = (
            remaining_vals
            if len(remaining_vals.shape) == 2
            else remaining_vals.unsqueeze(-1)
        )
        scaled_experts = torch.cat([scaled_tensor, remaining_vals], dim=1)
    else:
        scaled_experts = scaled_tensor

    return scaled_experts


def normalize_latent_output(latent_out):
    keys = [
        "pa",
        "pv",
        "s",
        "sv",
        "r_tpr_mod",
        "f_hr_max",
        "f_hr_min",
        "r_tpr_max",
        "r_tpr_min",
        "ca",
        "cv",
        "k_width",
        "p_aset",
        "tau",
    ]
    for i, key in enumerate(keys):
        latent_out[:, :, :, i] = (
            latent_out[:, :, :, i] - CV_params_prior_mu[key]
        ) / CV_params_prior_sigma[key]

    return latent_out


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def normalise_expert_data(y):
    # If first 2 dims are NN outputs, we should skip them
    nn_output_dims = 2  # Number of NN output dimensions

    keys = [
        "pa",
        "pv",
        "s",
        "sv",
        "r_tpr_mod",
        "f_hr_max",
        "f_hr_min",
        "r_tpr_max",
        "r_tpr_min",
        "ca",
        "cv",
        "k_width",
        "p_aset",
        "tau",
    ]

    normalized_tensors = []

    # First, keep the NN outputs as-is (or normalize them differently if needed)
    for i in range(nn_output_dims):
        normalized_tensors.append(
            y[..., i].unsqueeze(-1)
        )  # ← CHANGE: y[:, i] to y[..., i]

    # Then normalize the CV parameters starting from the correct index
    for i, key in enumerate(keys):
        # Offset by nn_output_dims to get the correct dimension
        actual_idx = i + nn_output_dims
        if actual_idx < y.shape[-1]:  # Safety check
            normalized_tensor = (
                (y[..., actual_idx] - CV_params_prior_mu[key])
                / CV_params_prior_sigma[key]
            ).unsqueeze(-1)  # ← CHANGE: y[:, actual_idx] to y[..., actual_idx]
            normalized_tensors.append(normalized_tensor)

    SDE_NN_norm_inputs = torch.cat(normalized_tensors, dim=-1)

    return SDE_NN_norm_inputs


def get_last_valid_timestep_fast(X_mask):
    has_any_data = X_mask.sum(dim=-1) > 0  # [batch, time]

    # Create position indices [0, 1, 2, ..., time-1]
    positions = torch.arange(X_mask.shape[1], device=X_mask.device).unsqueeze(0)

    # Mask invalid positions as -1, keep valid positions
    masked_positions = torch.where(has_any_data, positions, -1)

    # Find the maximum position (last valid timestep) for each batch
    last_valid_indices, _ = masked_positions.max(dim=1)

    # Convert to lengths (+1) and handle no-data cases
    lengths = torch.clamp(last_valid_indices + 1, min=0)

    return lengths


def sigmoid_scale(z0, use_2_5std_encoder_minmax):
    # print('unnormalising the data and scaling it back to its appropriate values ')
    batch_size = z0.shape[0]

    # Splitting the input tensor into individual variables, one for each column
    p_a, p_v, s_reflex, sv = z0[:, 0], z0[:, 1], z0[:, 2], z0[:, 3]

    # Selecting the appropriate dictionary based on the condition
    params_dict = (
        CV_params_max_min_2_5STD
        if use_2_5std_encoder_minmax
        else CV_params_max_min_2STD
    )

    # Assigning values from the chosen dictionary
    pa_max = params_dict["max_pa"]
    pa_min = params_dict["min_pa"]
    pv_max = params_dict["max_pv"]
    pv_min = params_dict["min_pv"]
    s_max = params_dict["max_s"]
    s_min = params_dict["min_s"]
    sv_max = params_dict["max_sv"]
    sv_min = params_dict["min_sv"]

    # print('CV_params', params_dict)

    # Applying sigmoid and scaling to each dimension
    p_a = torch.sigmoid(p_a).unsqueeze(1) * (pa_max - pa_min) + pa_min
    p_v = torch.sigmoid(p_v).unsqueeze(1) * (pv_max - pv_min) + pv_min
    s_reflex = torch.sigmoid(s_reflex).unsqueeze(1) * (s_max - s_min) + s_min
    sv = torch.sigmoid(sv).unsqueeze(1) * (sv_max - sv_min) + sv_min

    # Concatenating along the second dimension (columns)
    scaled_out = torch.cat([p_a, p_v, s_reflex, sv], dim=1)
    # print('scaled_out shape:', scaled_out.shape, scaled_out[:3, :])

    return scaled_out


class MLPSimple(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        depth,
        activations=None,
        dropout_p=None,
        final_activation=None,
        use_batch_norm=False,
        debug=False,
    ):
        super().__init__()
        self.debug = debug
        if self.debug:
            print(
                f"[DEBUG] MLPSimple __init__: input_dim={input_dim}, output_dim={output_dim}, hidden_dim={hidden_dim}, depth={depth}, use_batch_norm={use_batch_norm}"
            )

        # Avoid LazyLinear for DDP compatibility: require explicit input_dim
        if input_dim is None:
            raise ValueError("MLPSimple requires explicit input_dim to avoid LazyLinear with DDP.")
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        # Define the output layer with an optional final activation
        if final_activation is not None:
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dim, output_dim), final_activation
            )
        else:
            self.output_layer = nn.Sequential(nn.Linear(hidden_dim, output_dim))

        # Default activation function to ReLU if not specified
        if activations is None:
            activations = [nn.ReLU() for _ in range(depth)]

        # Default dropout to 0 if not specified
        if dropout_p is None:
            dropout_p = [0.0 for _ in range(depth)]

        assert (
            len(activations) == depth
        ), "Mismatch between depth and number of provided activations."

        # Building layers with optional batch normalization
        self.layers = nn.ModuleList()
        for i in range(depth):
            layer_components = [nn.Linear(hidden_dim, hidden_dim)]
            if use_batch_norm:
                layer_components.append(nn.BatchNorm1d(hidden_dim))
            layer_components.append(nn.Dropout(dropout_p[i]))
            layer_components.append(activations[i])
            self.layers.append(nn.Sequential(*layer_components))

    def forward(self, x):
        x_in = self.input_layer(x)
        x_hidden = x_in
        for i, layer in enumerate(self.layers):
            x_hidden = layer(x_hidden)
        x_out = self.output_layer(x_hidden)
        return x_out


CV_params = {
    "r_tpr_mod": 0.0,
    "f_hr_max": 3.0,
    "f_hr_min": 2.0 / 3.0,
    "r_tpr_max": 2.134,
    "r_tpr_min": 0.5335,
    "ca": 4.0,
    "cv": 111.0,
    # dS/dt parameters
    "k_width": 0.1838,
    "p_aset": 70,
    "tau": 20,
}

CV_params_divisors = {
    "pa": 100,
    "pv": 100,
    "s": 1,
    "sv": 100,
    "r_tpr_mod": 1,
    "f_hr_max": 10,
    "f_hr_min": 1,
    "r_tpr_max": 10,
    "r_tpr_min": 1,
    "ca": 10,
    "cv": 100,
    "k_width": 1,
    "p_aset": 10,
    "tau": 10,
}


CV_params_max_min_2STD = {
    "max_pa": torch.tensor(130.9138),
    "min_pa": torch.tensor(82.4030),
    "max_pv": torch.tensor(60.1224),
    "min_pv": torch.tensor(20.9298),
    "max_s": torch.tensor(0.2204),
    "min_s": torch.tensor(0.0722),
    "max_sv": torch.tensor(100.7716),
    "min_sv": torch.tensor(89.1963),
}

CV_params_max_min_2_5STD = {
    "max_pa": torch.tensor(136.97),
    "min_pa": torch.tensor(76.3392),
    "max_pv": torch.tensor(65.0215),
    "min_pv": torch.tensor(16.0307),
    "max_s": torch.tensor(0.2390),
    "min_s": torch.tensor(0.0537),
    "max_sv": torch.tensor(102.2185),
    "min_sv": torch.tensor(87.7494),
}


CV_params_prior_mu = {
    "pa": torch.tensor(89.2),
    "pv": torch.tensor(50.3),
    "s": torch.tensor(0.037),
    "sv": torch.tensor(88.6),
    "r_tpr_mod": torch.tensor(0.0),
    "f_hr_max": torch.tensor(3.0),
    "f_hr_min": torch.tensor(2.0 / 3.0),
    "r_tpr_max": torch.tensor(2.134),
    "r_tpr_min": torch.tensor(0.5335),
    "ca": torch.tensor(4.0),
    "cv": torch.tensor(111.0),
    "k_width": torch.tensor(0.1838),
    "p_aset": torch.tensor(70),
    "tau": torch.tensor(20),
}

CV_params_prior_sigma = {
    "pa": torch.tensor(4.9),
    "pv": torch.tensor(10.2),
    "s": torch.tensor(0.024),
    "sv": torch.tensor(1.8),
    "r_tpr_mod": torch.tensor(1 / 20),
    "f_hr_max": torch.tensor(3.0 / 20),
    "f_hr_min": torch.tensor((2.0 / 3.0) / 20),
    "r_tpr_max": torch.tensor(2.134 / 20),
    "r_tpr_min": torch.tensor(0.5335 / 20),
    "ca": torch.tensor(4.0 / 20),
    "cv": torch.tensor(111.0 / 20),
    "k_width": torch.tensor(0.1838 / 20),
    "p_aset": torch.tensor(70 / 20),
    "tau": torch.tensor(20 / 20),
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
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def _stable_division(a, b, epsilon=1e-7):
    b = torch.where(
        b.abs().detach() > epsilon,
        b,
        torch.full_like(b, fill_value=epsilon) * ((b > 0).float() * 2 - 1),
    )
    return a / b


def interpolate_colors(color1, color2, num_colors, rgb=False):
    colors = []
    for t in np.linspace(0, 1, num_colors):
        r = int(color1[0] + (color2[0] - color1[0]) * t)
        g = int(color1[1] + (color2[1] - color1[1]) * t)
        b = int(color1[2] + (color2[2] - color1[2]) * t)
        if rgb:
            colors.append(f"rgb({r}, {g}, {b})")
        else:
            colors.append((r, g, b))
    return colors


class LinearScheduler(object):
    def __init__(self, iters, startval=1.0, endval=0, start=0):
        self._iters = max(1, iters)
        self._val = startval
        self._endval = endval
        self._start = start  # Starting iteration for decrement
        self.current_iter = 0
        self._decrement = (startval - endval) / self._iters

    def step(self):
        # Only start decrementing once the current iteration reaches the start iteration
        if (
            self.current_iter >= self._start
            and self.current_iter < self._iters + self._start
        ):
            self._val -= self._decrement
        self._val = max(
            self._endval, self._val
        )  # Ensure that val does not go below endval
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


def gaussian_nll_loss(input, target, var, *, full=False, eps=1e-6, reduction="mean"):
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
        if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(
            tens_ops
        ):
            return handle_torch_function(
                gaussian_nll_loss,
                tens_ops,
                input,
                target,
                var,
                full=full,
                eps=eps,
                reduction=reduction,
            )

    # Inputs and targets much have same shape
    # input = input.view(input.size(0), -1)
    # target = target.view(target.size(0), -1)
    if input.size() != target.size():
        raise ValueError("input and target must have same size")

    # Second dim of var must match that of input or be equal to 1
    # var = var.view(input.size(0), -1)
    if var.size() != input.size():
        raise ValueError("var is of incorrect size")

    # Check validity of reduction mode
    if reduction != "none" and reduction != "mean" and reduction != "sum":
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if torch.any(var < 0):
        raise ValueError("var has negative entry/entries")

    # Clamp for stability
    var = var.clone()
    with torch.no_grad():
        var.clamp_(min=eps)

    # Calculate loss (without constant)
    # loss = 0.5 * (torch.log(var) + (input - target)**2 / var).view(input.size(0), -1).sum(dim=1)
    loss = 0.5 * (torch.log(var) + (input - target) ** 2 / var)

    # Add constant to loss term if required
    if full:
        D = input.size(1)
        loss = loss + 0.5 * D * math.log(2 * math.pi)

    # Apply reduction
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
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

    __constants__ = ["full", "eps", "reduction"]
    full: bool
    eps: float

    def __init__(
        self, *, full: bool = False, eps: float = 1e-6, reduction: str = "mean"
    ) -> None:
        super(GaussianNLLLoss, self).__init__(None, None, reduction)
        self.full = full
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor, var: Tensor) -> Tensor:
        return gaussian_nll_loss(
            input, target, var, full=self.full, eps=self.eps, reduction=self.reduction
        )


def activate_auto_debug_mode(model_instance, location: str, tensor_name: str, tensor_value=None):
    """Activate global debug mode and log comprehensive state information."""
    # Set debug flags
    model_instance.debug = True
    
    print("\n" + "="*80)
    print("🚨 AUTO-DEBUG MODE ACTIVATED 🚨")
    print("="*80)
    print(f"Location: {location}")
    print(f"Tensor: {tensor_name}")
    print(f"Global step: {getattr(model_instance, 'global_step', 'unknown')}")
    print(f"Current epoch: {getattr(model_instance, 'current_epoch', 'unknown')}")
    print("="*80)
    
    # Log tensor statistics if provided
    if tensor_value is not None:
        try:
            print(f"Tensor shape: {tensor_value.shape}")
            print(f"Tensor dtype: {tensor_value.dtype}")
            print(f"Tensor device: {tensor_value.device}")
            print(f"NaN count: {torch.isnan(tensor_value).sum().item()}")
            print(f"Inf count: {torch.isinf(tensor_value).sum().item()}")
            print(f"Min value: {tensor_value.min().item()}")
            print(f"Max value: {tensor_value.max().item()}")
            print(f"Mean value: {tensor_value.mean().item()}")
            print(f"Std value: {tensor_value.std().item()}")
        except Exception as e:
            print(f"Could not log tensor stats: {e}")
    
    # Log model state
    try:
        print("\nModel State:")
        print(f"  use_encoder: {model_instance.use_encoder}")
        print(f"  normalise_for_SDENN: {model_instance.normalise_for_SDENN}")
        print(f"  controller_type: {model_instance.controller_type}")
        print(f"  SDE_control_weighting: {model_instance.SDE_control_weighting}")
        print(f"  learning_rate: {model_instance.learning_rate}")
        print(f"  SDEnet_out_dims: {model_instance.SDEnet_out_dims}")
        print(f"  expert_latent_dims: {model_instance.expert_latent_dims}")
        print(f"  num_samples: {model_instance.num_samples}")
        print(f"  integration_step_size: {model_instance.integration_step_size}")
        print(f"  integration_method: {model_instance.integration_method}")
    except Exception as e:
        print(f"Could not log model state: {e}")
    
    # Log parameter statistics
    try:
        print("\nParameter Statistics:")
        total_params = 0
        for name, param in model_instance.named_parameters():
            if param is not None:
                total_params += param.numel()
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"  ❌ {name}: {param.shape} - CONTAINS NaN/Inf!")
                else:
                    print(f"  ✅ {name}: {param.shape} - OK")
        print(f"  Total parameters: {total_params:,}")
    except Exception as e:
        print(f"Could not log parameter stats: {e}")
    
    print("="*80 + "\n")
    
    # Set breakpoint for debugging
    import pdb; pdb.set_trace()


def check_for_nan_inf(tensor, name: str, location: str, model_instance=None, raise_error: bool = True):
    """
    Comprehensive NaN/Inf detection with detailed logging.
    
    Args:
        tensor: Tensor to check
        name: Name of the tensor for logging
        location: Location where the check is performed
        model_instance: Model instance for context
        raise_error: Whether to raise an error on NaN/Inf detection
    """
    if tensor is None:
        return False
        
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    
    if has_nan or has_inf:
        print(f"\n🚨 NaN/Inf detected in {name} at {location}")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Device: {tensor.device}")
        print(f"  NaN count: {torch.isnan(tensor).sum().item()}")
        print(f"  Inf count: {torch.isinf(tensor).sum().item()}")
        
        if has_nan:
            nan_indices = torch.isnan(tensor).nonzero()
            print(f"  NaN indices (first 10): {nan_indices[:10]}")
            
        if has_inf:
            inf_indices = torch.isinf(tensor).nonzero()
            print(f"  Inf indices (first 10): {inf_indices[:10]}")
            
        if model_instance is not None:
            activate_auto_debug_mode(model_instance, location, name, tensor)
            
        if raise_error:
            raise RuntimeError(f"NaN/Inf detected in {name} at {location}")
            
    return has_nan or has_inf


def fail_on_nan_inf(param, name: str, location: str, model_instance=None):
    """
    Fail fast on NaN/Inf values with comprehensive debugging.
    
    Args:
        param: Parameter tensor to check
        name: Name of the parameter
        location: Location where check occurs
        model_instance: Model instance for context
    """
    if param is None:
        return
        
    has_nan = torch.isnan(param).any()
    has_inf = torch.isinf(param).any()
    
    if has_nan or has_inf:
        print(f"[ERROR] NaN/Inf weights in {name}! FAILING FAST - NO SANITIZATION!")
        
        if model_instance is not None:
            activate_auto_debug_mode(model_instance, location, name, param)
            
        raise RuntimeError(f"NaN/Inf detected in {name} at {location}. Failing fast for debugging.")


def print_model_details(model, indent=0):
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(
            " " * indent, f"{name} | {module.__class__.__name__} | {num_params} params"
        )
        if list(module.children()):
            print_model_details(module, indent + 4)

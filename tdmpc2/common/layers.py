import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import from_modules
from copy import deepcopy
from .downsample import conv3x3, ResidualBlock, DownSample

class Ensemble(nn.Module):
	"""
	Vectorized ensemble of modules.
	"""

	def __init__(self, modules, **kwargs):
		super().__init__()
		# combine_state_for_ensemble causes graph breaks
		self.modules_list = modules
		self.params = from_modules(*modules, as_module=True)
		with self.params[0].data.to("meta").to_module(modules[0]):
			self.module = deepcopy(modules[0])
		self._repr = str(modules)

	def _call(self, params, *args, **kwargs):
		with params.to_module(self.module):
			return self.module(*args, **kwargs)

	def forward(self, *args, **kwargs):
		return torch.vmap(self._call, (0, None), randomness="different")(self.params, *args, **kwargs)

	def __repr__(self):
		return 'Vectorized ' + self._repr

class ShiftAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, pad=3):
		super().__init__()
		self.pad = pad
		self.padding = tuple([self.pad] * 4)

	def forward(self, x):
		x = x.float()
		n, _, h, w = x.size()
		assert h == w
		x = F.pad(x, self.padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class PixelPreprocess(nn.Module):
	"""
	Normalizes pixel observations to [-0.5, 0.5].
	"""

	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.).sub(0.5)


class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.dim = cfg.simnorm_dim

	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)

	def __repr__(self):
		return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
	"""
	Linear layer with LayerNorm, activation, and optionally dropout.
	"""

	def __init__(self, *args, dropout=0., act=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		if act is None:
			act = nn.Mish(inplace=False)
		self.act = act
		self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

	def forward(self, x):
		x = super().forward(x)
		if self.dropout:
			x = self.dropout(x)
		return self.act(self.ln(x))

	def __repr__(self):
		repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
		return f"NormedLinear(in_features={self.in_features}, "\
			f"out_features={self.out_features}, "\
			f"bias={self.bias is not None}{repr_dropout}, "\
			f"act={self.act.__class__.__name__})"

class ResidualBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout: float = 0., act = None, dtype = torch.float32):

        super().__init__()
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.dropout = dropout
        if act is None:
            act = nn.Mish(inplace=False)

        self.fc1 = NormedLinear(hidden_dim, 4 * hidden_dim, dropout=dropout, act=act)
        self.fc2 = NormedLinear(4 * hidden_dim, hidden_dim, dropout=dropout, act=act)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        residual = x
        out = self.fc1(x)
        out = self.fc2(out)
        return residual + out

    def __repr__(self):
        return (f"{self.__class__.__name__}(hidden_dim={self.hidden_dim}, "
                f"\n fc1={self.fc1},\n fc2={self.fc2})")
	
def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.):
	"""
	Basic building block of TD-MPC2.
	MLP with LayerNorm, Mish activations, and optionally dropout.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
	mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
	return nn.Sequential(*mlp)

def res_mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.) -> nn.Sequential:

    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()

    # ResidualBlock, 1 residual block for 2 layers
    for i in range(0, len(dims) - 2, 2):
        mlp.append(ResidualBlock(hidden_dim=dims[i], dropout=dropout))
    
    # 输出层：使用 NormedLinear 或标准 Linear
    if act:
        mlp.append(NormedLinear(dims[0], dims[-1], act=act))
    else:
        mlp.append(nn.Linear(dims[0], dims[-1]))
    
    return nn.Sequential(*mlp)

def conv(in_shape, num_channels, act=None):
	"""
	Basic convolutional encoder for TD-MPC2 with raw image observations.
	4 layers of convolution with ReLU activations, followed by a linear layer.
	"""
	assert in_shape[-1] == 64 # assumes rgb observations to be 64x64
	layers = [
		ShiftAug(), PixelPreprocess(),
		nn.Conv2d(in_shape[0], num_channels, 7, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=1), nn.Flatten()]
	if act:
		layers.append(act)
	return nn.Sequential(*layers)

def conv_atari(in_shape, num_channels, act=None):
	"""
	Basic convolutional encoder for TD-MPC2 with raw image observations.
	4 layers of convolution with ReLU activations, followed by a linear layer.
	"""
	layers = [
		PixelPreprocess(),
		nn.Conv2d(in_shape, num_channels, 7, stride=2, padding=3), nn.ReLU(inplace=False),#hard code for grayscale
		nn.Conv2d(num_channels, num_channels, 5, stride=2, padding=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=2, padding=1), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3,stride=3, padding=1), nn.Flatten()]

	if act:
		layers.append(act)
	return nn.Sequential(*layers)

def conv_atari_downsample(in_shape, num_channels, reduced_channels=16 , act=None):
	"""
	Basic convolutional encoder for TD-MPC2 with raw image observations.
	4 layers of convolution with ReLU activations, followed by a linear layer.
	"""
	layers = [
		PixelPreprocess(),
		DownSample(in_shape, num_channels),
		ResidualBlock(num_channels, num_channels),
		nn.Flatten(),
		nn.Linear(num_channels * 6 * 6, 512), 
		nn.ReLU(inplace=False)
		]
	if act:
		layers.append(act)
	return nn.Sequential(*layers)

def enc(cfg, out={}):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	"""
	for k in cfg.obs_shape.keys():
		if k == 'state':
			if not cfg.simba:
				out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
			else:
				post_layer = res_mlp(cfg.enc_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
				post_layer.insert(0, NormedLinear(cfg.obs_shape[k][0] + cfg.task_dim, cfg.enc_dim, act=nn.Mish(inplace=False)))
				out[k] = post_layer
		elif k == 'rgb':
			if cfg.task_platform != 'atari':
				out[k] = conv(cfg.obs_shape[k], cfg.num_channels, act=SimNorm(cfg))
			else:
				if cfg.downsample:
					out[k] = conv_atari_downsample(4 if cfg.gray_scale else 12, cfg.num_channels, act=SimNorm(cfg))
				else:
					out[k] = conv_atari(4 if cfg.gray_scale else 12, cfg.num_channels, act=SimNorm(cfg))
		else:
			raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
	return nn.ModuleDict(out)
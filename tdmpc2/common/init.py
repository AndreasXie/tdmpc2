import torch.nn as nn
import torch
from tensordict import TensorDict

def weight_init(m):
	"""Custom weight initialization for TD-MPC2."""
	if isinstance(m, nn.Linear):
		nn.init.kaiming_normal_(m.weight)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)
	elif isinstance(m, nn.Embedding):
		nn.init.uniform_(m.weight, -0.02, 0.02)
	elif isinstance(m, nn.ParameterList):
		for i,p in enumerate(m):
			if p.dim() == 3: # Linear
				nn.init.kaiming_normal_(m.weight)
				nn.init.constant_(m[i+1], 0) # Bias
	elif isinstance(m, TensorDict):
		reset_parameters(m)

# Reset function for weight initialization
def _weight_init(m):
    if isinstance(m, torch.Tensor):
        nn.init.kaiming_normal_(m.weight)

# Function to reset parameters
def reset_parameters(parameters):
    if 'weight' in parameters:
        _weight_init(parameters['weight'])
    if 'bias' in parameters:
        nn.init.constant_(parameters['bias'], 0)
    if 'ln' in parameters:
        ln_params = parameters['ln']
        if 'weight' in ln_params:
            _weight_init(ln_params['weight'])
        if 'bias' in ln_params:
            nn.init.constant_(ln_params['bias'], 0)

def zero_(params):
	"""Initialize parameters to zero."""
	for p in params:
		p.data.fill_(0)

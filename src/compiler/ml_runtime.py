import torch
import torch.nn as nn

class FourierDense(nn.Module):
    def __init__(self, in_features, out_features, activation=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softmax":
            self.activation = nn.Softmax(dim=-1)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = None
    
    @property
    def shape(self):
        return self.linear.weight.shape

    def forward(self, x):
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class FourierSequential(nn.Module):
    def __init__(self, layers_data):
        super().__init__()
        self.raw_layers = layers_data
        self.built_layers = nn.ModuleList()
        self.initialized = False
    
    @property
    def shape(self):
        # For a model, we return the internal weights or informative string
        # if not built yet.
        if not self.initialized: return "uninitialized"
        # Just return a summary for now
        return [l.shape if hasattr(l, 'shape') else 'unknown' for l in self.built_layers]

    def _build(self, input_data):
        # Convert to tensor if not already
        input_data = _to_tensor(input_data)
        
        # Ensure 2D (batch, features)
        if input_data.dim() == 1:
            input_data = input_data.unsqueeze(0)
            
        current_size = input_data.shape[-1]
        for layer_item in self.raw_layers:
            if callable(layer_item) and not isinstance(layer_item, nn.Module):
                built_layer = layer_item(current_size)
            else:
                built_layer = layer_item
            
            self.built_layers.append(built_layer)
            if hasattr(built_layer, 'linear'):
                current_size = built_layer.linear.out_features
            elif isinstance(built_layer, nn.Linear):
                current_size = built_layer.out_features
            elif hasattr(built_layer, 'out_features'):
                current_size = built_layer.out_features
        self.initialized = True

    def _recursive_to_tensor(self, data):
        if isinstance(data, torch.Tensor): return data
        if isinstance(data, (list, tuple)):
            try:
                converted = [self._recursive_to_tensor(x) for x in data]
                if any(isinstance(x, torch.Tensor) for x in converted):
                    return torch.stack(converted)
            except: pass
        # Use dynamic dtype inference
        try: return torch.as_tensor(data)
        except: return data

    def forward(self, x):
        # Robust tensor conversion
        if not isinstance(x, torch.Tensor):
            x = self._recursive_to_tensor(x)
            
        # Fallback for nested lists that _to_tensor might have missed
        if not isinstance(x, torch.Tensor):
            try:
                x = torch.as_tensor(x, dtype=torch.float32)
            except:
                pass
        
        # Automatic batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Device management
        device = "cpu"
        params = list(self.parameters())
        if params:
            device = params[0].device
        x = x.to(device)

        if not self.initialized:
            self._build(x)
            
        for layer in self.built_layers:
            x = layer(x)
        return x

def Dense(out_features, activation=None, in_features=None):
    if in_features is None:
        return lambda in_feat: FourierDense(in_feat, out_features, activation)
    return FourierDense(in_features, out_features, activation)

def Sequential(layers):
    return FourierSequential(layers)

class FourierCompiledModel:
    """
    Robust wrapper for torch.compile that handles lazy compilation errors.
    If the native compiler is missing, it falls back to the interpreted model.
    """
    def __init__(self, original_model):
        self.original_model = original_model
        self.failed = False
        try:
            self.compiled = torch.compile(original_model)
        except:
            self.failed = True

    def __call__(self, *args, **kwargs):
        if not self.failed:
            try:
                # First call triggers lazy compilation in Inductor
                return self.compiled(*args, **kwargs)
            except Exception as e:
                print(f"Fourier Warning: Runtime compilation failed ({type(e).__name__}). Falling back to interpreted mode.")
                self.failed = True
        return self.original_model(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def __getattr__(self, name):
        # Proxy all other calls to the original model (e.g., parameters, to, cuda)
        return getattr(self.original_model, name)

def _to_grad(data):
    """Marks a tensor to require gradients."""
    tensor = _to_tensor(data)
    if tensor.is_floating_point():
        tensor.requires_grad = True
    return tensor

def _to_tensor(data):
    """Internal helper to convert nested lists/NumPy to PyTorch tensors."""
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, (list, tuple)):
        # Handle empty lists
        if not data: return torch.tensor([], dtype=torch.float32)
        # Try converting directly, let PyTorch infer the type (Float or Complex)
        try:
            return torch.as_tensor(data)
        except:
            # Fallback for complex nesting
            return torch.stack([_to_tensor(x) for x in data])
    return torch.as_tensor(data)

def compile_model(model):
    """
    Fourier Native Compilation Hook.
    Returns a robust wrapper that manages the transition to native code.
    """
    if hasattr(torch, 'compile'):
        return FourierCompiledModel(model)
    return model

def random_vector(size):
    return torch.randn(size)

def random_matrix(rows, cols):
    return torch.randn(rows, cols)

# --- Standard Library: Matrix Operations ---

def transpose(data):
    data = _to_tensor(data)
    return data.t()

def inverse(data):
    data = _to_tensor(data)
    return torch.inverse(data)

def dot_product(a, b):
    a = _to_tensor(a)
    b = _to_tensor(b)
    return torch.dot(a.flatten(), b.flatten())

# --- Standard Library: Machine Learning ---

def relu(data):
    data = _to_tensor(data)
    return torch.relu(data)

def softmax(data, dim=-1):
    data = _to_tensor(data)
    return torch.softmax(data, dim=dim)

def convolution(input, kernel, padding=0, stride=1):
    input = _to_tensor(input)
    kernel = _to_tensor(kernel)
    # Basic 2D convolution assumption for now
    if input.dim() == 2: input = input.unsqueeze(0).unsqueeze(0)
    if kernel.dim() == 2: kernel = kernel.unsqueeze(0).unsqueeze(0)
    return torch.nn.functional.conv2d(input, kernel, padding=padding, stride=stride)

def pooling(input, kernel_size=2):
    input = _to_tensor(input)
    if input.dim() == 2: input = input.unsqueeze(0).unsqueeze(0)
    return torch.nn.functional.max_pool2d(input, kernel_size=kernel_size)

# --- Standard Library: Signal Processing ---

def fft(data):
    data = _to_tensor(data)
    return torch.fft.fft(data)

def ifft(data):
    data = _to_tensor(data)
    return torch.fft.ifft(data)

# --- Standard Library: Sorting ---

def sort(data, descending=False):
    data = _to_tensor(data)
    sorted_values, _ = torch.sort(data, descending=descending)
    return sorted_values

def quicksort(data):
    return sort(data)

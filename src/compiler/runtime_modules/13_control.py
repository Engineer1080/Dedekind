import torch

class Block:
    """Base class for differentiable simulation blocks."""
    def __init__(self):
        self.inputs = []
        self._output = torch.tensor(0.0, dtype=torch.float64)
        self.is_stateful = False

    def set_input(self, inp):
        self.inputs = [inp]
        return self

    def set_inputs(self, inps):
        self.inputs = list(inps)
        return self

    def _eval_input(self, inp, memo, dt):
        if isinstance(inp, Block):
            return inp.eval(memo, dt)
        # Convert non-block inputs (floats/tensors) to double tensor
        return _to_tensor(inp).double()

    def eval(self, memo, dt):
        if self in memo:
            return memo[self]
        if self.is_stateful:
            memo[self] = self._output
            return self._output

        out = self._compute_output(memo, dt)
        memo[self] = out
        return out

    def _compute_output(self, memo, dt):
        raise NotImplementedError("Subclasses must implement _compute_output")

    def reset(self):
        pass


class ConstantBlock(Block):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def _compute_output(self, memo, dt):
        return _to_tensor(self.value).double()


class GainBlock(Block):
    def __init__(self, inp=None, K=1.0):
        super().__init__()
        self.K = K
        if inp is not None:
            self.set_input(inp)

    def _compute_output(self, memo, dt):
        u = self._eval_input(self.inputs[0], memo, dt)
        return _to_tensor(self.K).double() * u


class SumBlock(Block):
    def __init__(self, inputs=None, signs=None):
        super().__init__()
        self.signs = signs
        if inputs is not None:
            self.set_inputs(inputs)

    def _compute_output(self, memo, dt):
        out = torch.tensor(0.0, dtype=torch.float64, device=self.inputs[0]._output.device if (self.inputs and isinstance(self.inputs[0], Block)) else None)
        for i, inp in enumerate(self.inputs):
            val = self._eval_input(inp, memo, dt)
            sign = 1.0
            if self.signs is not None and i < len(self.signs):
                sign = float(self.signs[i])
            out = out + sign * val
        return out


class ProductBlock(Block):
    def __init__(self, inputs=None):
        super().__init__()
        if inputs is not None:
            self.set_inputs(inputs)

    def _compute_output(self, memo, dt):
        out = torch.tensor(1.0, dtype=torch.float64)
        for inp in self.inputs:
            val = self._eval_input(inp, memo, dt)
            out = out * val
        return out


class SaturationBlock(Block):
    def __init__(self, inp=None, min_val=-1.0, max_val=1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        if inp is not None:
            self.set_input(inp)

    def _compute_output(self, memo, dt):
        u = self._eval_input(self.inputs[0], memo, dt)
        return torch.clamp(u, self.min_val, self.max_val)


class IntegratorBlock(Block):
    def __init__(self, inp=None, x0=0.0):
        super().__init__()
        self.is_stateful = True
        self.x0 = x0
        if inp is not None:
            self.set_input(inp)
        self.reset()

    def reset(self):
        self.state = _to_tensor(self.x0).double().clone()
        self._output = self.state

    def update_state(self, dt, memo):
        u = self._eval_input(self.inputs[0], memo, dt)
        self.state = self.state + u * dt
        self._output = self.state


class PIDBlock(Block):
    def __init__(self, inp=None, Kp=1.0, Ki=0.0, Kd=0.0, min_val=None, max_val=None):
        super().__init__()
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.min_val = min_val
        self.max_val = max_val
        if inp is not None:
            self.set_input(inp)
        self.reset()

    def reset(self):
        self.integral = torch.tensor(0.0, dtype=torch.float64)
        self.prev_err = torch.tensor(0.0, dtype=torch.float64)
        self.has_prev = False

    def _compute_output(self, memo, dt):
        err = self._eval_input(self.inputs[0], memo, dt)
        p_term = _to_tensor(self.Kp).double() * err
        i_term = _to_tensor(self.Ki).double() * self.integral
        if self.has_prev:
            d_term = _to_tensor(self.Kd).double() * (err - self.prev_err) / dt
        else:
            d_term = torch.tensor(0.0, dtype=torch.float64, device=err.device)
        u = p_term + i_term + d_term
        if self.min_val is not None and self.max_val is not None:
            u = torch.clamp(u, self.min_val, self.max_val)
        return u

    def update_state(self, dt, memo):
        err = self._eval_input(self.inputs[0], memo, dt)
        self.integral = self.integral + err * dt
        self.prev_err = err
        self.has_prev = True


def tf2ss(num, den):
    # Ensure inputs are normalized
    den_coeffs = [_to_tensor(a).double() for a in den]
    num_coeffs = [_to_tensor(b).double() for b in num]
    
    a_lead = den_coeffs[0]
    den_coeffs = [a / a_lead for a in den_coeffs]
    num_coeffs = [b / a_lead for b in num_coeffs]
    
    n = len(den_coeffs) - 1
    # Pad num with zeros if shorter than den
    if len(num_coeffs) < len(den_coeffs):
        num_coeffs = [torch.tensor(0.0, dtype=torch.float64, device=a_lead.device)] * (len(den_coeffs) - len(num_coeffs)) + num_coeffs
        
    # CCF realization: A is (n, n), B is (n, 1), C is (1, n), D is (1, 1)
    if n == 0:
        A = torch.zeros((0, 0), dtype=torch.float64, device=a_lead.device)
        B = torch.zeros((0,), dtype=torch.float64, device=a_lead.device)
        C = torch.zeros((0,), dtype=torch.float64, device=a_lead.device)
        D = num_coeffs[0]
        return A, B, C, D

    A = [[torch.tensor(0.0, dtype=torch.float64, device=a_lead.device) for _ in range(n)] for _ in range(n)]
    for i in range(n - 1):
        A[i][i+1] = torch.tensor(1.0, dtype=torch.float64, device=a_lead.device)
    for j in range(n):
        A[n-1][j] = -den_coeffs[len(den_coeffs) - 1 - j]
        
    B = [torch.tensor(0.0, dtype=torch.float64, device=a_lead.device) for _ in range(n)]
    B[n-1] = torch.tensor(1.0, dtype=torch.float64, device=a_lead.device)
    
    b_n = num_coeffs[0]
    C = []
    for j in range(n):
        b_j = num_coeffs[len(num_coeffs) - 1 - j]
        a_j = den_coeffs[len(den_coeffs) - 1 - j]
        C.append(b_j - b_n * a_j)
        
    D = b_n
    
    A_t = torch.stack([torch.stack(row) for row in A])
    B_t = torch.stack(B)
    C_t = torch.stack(C)
    D_t = _to_tensor(D).double()
    return A_t, B_t, C_t, D_t


class StateSpaceBlock(Block):
    def __init__(self, inp=None, A=None, B=None, C=None, D=None):
        super().__init__()
        self.is_stateful = True
        self.A = _to_tensor(A).double()
        self.B = _to_tensor(B).double()
        self.C = _to_tensor(C).double()
        self.D = _to_tensor(D).double() if D is not None else torch.zeros((C.shape[0], B.shape[1] if B.ndim > 1 else 1), dtype=torch.float64)
        if inp is not None:
            self.set_input(inp)
        self.reset()

    def reset(self):
        n = self.A.shape[0]
        self.state = torch.zeros((n,), dtype=torch.float64, device=self.A.device)
        self._output = self.C @ self.state

    def update_state(self, dt, memo):
        u = self._eval_input(self.inputs[0], memo, dt)
        dx = self.A @ self.state + self.B * u
        self.state = self.state + dx * dt
        self._output = self.C @ self.state


class TransferFunctionBlock(Block):
    def __init__(self, inp=None, num=[1.0], den=[1.0, 1.0]):
        super().__init__()
        self.is_stateful = True
        self.num = num
        self.den = den
        if inp is not None:
            self.set_input(inp)
        self.reset()

    def reset(self):
        self.A, self.B, self.C, self.D = tf2ss(self.num, self.den)
        n = self.A.shape[0]
        self.state = torch.zeros((n,), dtype=torch.float64, device=self.A.device)
        self._output = self.C @ self.state

    def update_state(self, dt, memo):
        u = self._eval_input(self.inputs[0], memo, dt)
        dx = self.A @ self.state + self.B * u
        self.state = self.state + dx * dt
        self._output = self.C @ self.state


class BlockDiagram:
    def __init__(self):
        self.blocks = []

    def add_block(self, block):
        if block not in self.blocks:
            self.blocks.append(block)
        return block

    def simulate(self, t_max, dt):
        t_t = _to_tensor(t_max).double()
        dt_t = _to_tensor(dt).double()
        steps = int(torch.round(t_t / dt_t).item())

        # Reset states of all blocks
        for block in self.blocks:
            block.reset()

        history = {block: [] for block in self.blocks}
        time_history = []

        t = 0.0
        for _ in range(steps):
            memo = {}
            # Evaluate all blocks recursively
            for block in self.blocks:
                block.eval(memo, dt_t)

            # Record outputs
            for block in self.blocks:
                history[block].append(memo[block])
            time_history.append(t)

            # Update states
            for block in self.blocks:
                if hasattr(block, 'update_state'):
                    block.update_state(dt_t, memo)

            t += dt_t.item()

        # Convert history arrays to stacked PyTorch tensors
        res = {}
        for block in self.blocks:
            res[block] = torch.stack(history[block])
        res["t"] = torch.tensor(time_history, dtype=torch.float64)
        return res


# Helper factories for DDK binding
def block_diagram_impl():
    return BlockDiagram()

def constant_block_impl(value):
    return ConstantBlock(value)

def gain_block_impl(inp, K):
    return GainBlock(inp, K)

def sum_block_impl(inputs, signs=None):
    return SumBlock(inputs, signs)

def product_block_impl(inputs):
    return ProductBlock(inputs)

def saturation_block_impl(inp, min_val, max_val):
    return SaturationBlock(inp, min_val, max_val)

def integrator_block_impl(inp, x0=0.0):
    return IntegratorBlock(inp, x0)

def pid_block_impl(inp, Kp, Ki, Kd):
    return PIDBlock(inp, Kp, Ki, Kd)

def pid_block_saturated_impl(inp, Kp, Ki, Kd, min_val, max_val):
    return PIDBlock(inp, Kp, Ki, Kd, min_val, max_val)

def state_space_block_impl(inp, A, B, C, D=None):
    return StateSpaceBlock(inp, A, B, C, D)

def transfer_function_block_impl(inp, num, den):
    return TransferFunctionBlock(inp, num, den)

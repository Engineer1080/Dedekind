# --- Digital Signal Processing (DSP) & Z-Transformation ---

import torch

def _to_double_tensor(v):
    t = _to_tensor(v)
    return t.double()

def fir_filter_impl(x, b):
    """
    Filtert das Eingangssignal `x` mit den FIR-Koeffizienten `b` (Feedforward).
    Vollständig differenzierbar bezüglich `b` und `x`.
    """
    x_t = _to_double_tensor(x)
    b_t = _to_double_tensor(b)
    
    is_1d = (x_t.ndim == 1)
    if is_1d:
        x_t = x_t.unsqueeze(0).unsqueeze(0)  # (1, 1, L)
    elif x_t.ndim == 2:
        x_t = x_t.unsqueeze(1)  # (N, 1, L)
        
    M = b_t.shape[0]
    import torch.nn.functional as F
    x_padded = F.pad(x_t, (M - 1, 0), mode='constant', value=0.0)
    
    # We flip b_t for correlation filtering (since PyTorch conv1d is correlation)
    b_flipped = torch.flip(b_t, dims=[0]).view(1, 1, M)
    
    y = F.conv1d(x_padded, b_flipped)
    if is_1d:
        return y.squeeze(0).squeeze(0)
    return y.squeeze(1)

def iir_filter_impl(x, b, a):
    """
    Realisiert ein allgemeines IIR-Filter mit Feedforward-Koeffizienten `b`
    und Feedback-Koeffizienten `a` (mit standardisiertem a[0] = 1.0).
    Vollständig differenzierbar bezüglich `b`, `a` und `x`.
    """
    x_t = _to_double_tensor(x)
    b_t = _to_double_tensor(b)
    a_t = _to_double_tensor(a)
    
    # Normalize by a[0] if it's not 1.0
    if a_t[0] != 1.0:
        norm = a_t[0]
        b_t = b_t / norm
        a_t = a_t / norm
        
    M = b_t.shape[0]
    N = a_t.shape[0]
    L = x_t.shape[0]
    
    y_list = []
    for n in range(L):
        val = b_t[0] * x_t[n]
        for k in range(1, M):
            if n - k >= 0:
                val = val + b_t[k] * x_t[n - k]
        for j in range(1, N):
            if n - j >= 0:
                val = val - a_t[j] * y_list[n - j]
        y_list.append(val)
        
    return torch.stack(y_list)

def _get_val_hz(v):
    if isinstance(v, Quantity):
        return _convert_to_base(v.value, v.unit, "frequency")
    return float(v)

def biquad_lowpass_impl(fc, Q, fs=1.0):
    """
    Berechnet biquadratische 2nd-Order Lowpass Filterkoeffizienten b, a.
    Grenzfrequenz fc und Güte Q können Tensors oder Zahlen sein.
    Abtastrate fs kann ein Quantity oder float sein.
    Filterdesign ist vollständig differenzierbar!
    """
    fs_val = _get_val_hz(fs)
    
    if isinstance(fc, Quantity):
        fc_t = _to_double_tensor(_convert_to_base(fc.value, fc.unit, "frequency"))
    else:
        fc_t = _to_double_tensor(fc)
        
    Q_t = _to_double_tensor(Q)
    
    w0 = 2.0 * 3.141592653589793 * fc_t / fs_val
    alpha = torch.sin(w0) / (2.0 * Q_t)
    cos_w0 = torch.cos(w0)
    
    b0 = (1.0 - cos_w0) / 2.0
    b1 = 1.0 - cos_w0
    b2 = (1.0 - cos_w0) / 2.0
    
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    
    b = torch.stack([b0, b1, b2]) / a0
    a = torch.stack([torch.ones_like(a0), a1 / a0, a2 / a0])
    return b, a

def biquad_highpass_impl(fc, Q, fs=1.0):
    """
    Berechnet biquadratische 2nd-Order Highpass Filterkoeffizienten b, a.
    Grenzfrequenz fc und Güte Q können Tensors oder Zahlen sein.
    Filterdesign ist vollständig differenzierbar!
    """
    fs_val = _get_val_hz(fs)
    
    if isinstance(fc, Quantity):
        fc_t = _to_double_tensor(_convert_to_base(fc.value, fc.unit, "frequency"))
    else:
        fc_t = _to_double_tensor(fc)
    Q_t = _to_double_tensor(Q)
    
    w0 = 2.0 * 3.141592653589793 * fc_t / fs_val
    alpha = torch.sin(w0) / (2.0 * Q_t)
    cos_w0 = torch.cos(w0)
    
    b0 = (1.0 + cos_w0) / 2.0
    b1 = -(1.0 + cos_w0)
    b2 = (1.0 + cos_w0) / 2.0
    
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    
    b = torch.stack([b0, b1, b2]) / a0
    a = torch.stack([torch.ones_like(a0), a1 / a0, a2 / a0])
    return b, a

def biquad_bandpass_impl(fc, Q, fs=1.0):
    """
    Berechnet biquadratische 2nd-Order Bandpass Filterkoeffizienten b, a.
    Grenzfrequenz fc und Güte Q können Tensors oder Zahlen sein.
    Filterdesign ist vollständig differenzierbar!
    """
    fs_val = _get_val_hz(fs)
    
    if isinstance(fc, Quantity):
        fc_t = _to_double_tensor(_convert_to_base(fc.value, fc.unit, "frequency"))
    else:
        fc_t = _to_double_tensor(fc)
    Q_t = _to_double_tensor(Q)
    
    w0 = 2.0 * 3.141592653589793 * fc_t / fs_val
    alpha = torch.sin(w0) / (2.0 * Q_t)
    cos_w0 = torch.cos(w0)
    
    b0 = alpha
    b1 = torch.zeros_like(w0)
    b2 = -alpha
    
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    
    b = torch.stack([b0, b1, b2]) / a0
    a = torch.stack([torch.ones_like(a0), a1 / a0, a2 / a0])
    return b, a

def freqz_impl(b, a, worN=512):
    """
    Berechnet die komplexe Frequenzantwort H(jw) voll vektorisiert an worN Punkten.
    Vollständig differenzierbar bezüglich b und a.
    """
    b_t = _to_double_tensor(b)
    a_t = _to_double_tensor(a)
    
    omega = torch.linspace(0.0, 3.141592653589793, int(worN), dtype=b_t.dtype, device=b_t.device)
    j_unit = torch.complex(torch.tensor(0.0, dtype=b_t.dtype), torch.tensor(1.0, dtype=b_t.dtype))
    
    num = torch.zeros(int(worN), dtype=torch.complex128, device=b_t.device)
    for k, bk in enumerate(b_t):
        num = num + bk * torch.exp(-j_unit * k * omega)
        
    den = torch.zeros(int(worN), dtype=torch.complex128, device=a_t.device)
    for k, ak in enumerate(a_t):
        den = den + ak * torch.exp(-j_unit * k * omega)
        
    h = num / den
    return omega, h

def butter_impl(order, Wn, btype='low', fs=None):
    """
    Wrapper für klassisches Butterworth-Filterdesign (SciPy).
    Gibt die b und a Koeffizienten als PyTorch float64 Tensors zurück.
    """
    import scipy.signal as sig  # type: ignore[import-untyped]
    
    if isinstance(Wn, Quantity):
        Wn_val = _convert_to_base(Wn.value, Wn.unit, "frequency")
    elif hasattr(Wn, "tolist"):
        Wn_val = Wn.tolist()
    else:
        Wn_val = Wn
        
    fs_val = _get_val_hz(fs) if fs is not None else None
    
    b, a = sig.butter(int(order), Wn_val, btype=btype, fs=fs_val)
    return torch.tensor(b, dtype=torch.float64), torch.tensor(a, dtype=torch.float64)

def cheby1_impl(order, rp, Wn, btype='low', fs=None):
    """
    Wrapper für klassisches Chebyshev Type I Filterdesign (SciPy).
    Gibt die b und a Koeffizienten als PyTorch float64 Tensors zurück.
    """
    import scipy.signal as sig  # type: ignore[import-untyped]
    
    if isinstance(Wn, Quantity):
        Wn_val = _convert_to_base(Wn.value, Wn.unit, "frequency")
    elif hasattr(Wn, "tolist"):
        Wn_val = Wn.tolist()
    else:
        Wn_val = Wn
        
    fs_val = _get_val_hz(fs) if fs is not None else None
    
    b, a = sig.cheby1(int(order), float(rp), Wn_val, btype=btype, fs=fs_val)
    return torch.tensor(b, dtype=torch.float64), torch.tensor(a, dtype=torch.float64)

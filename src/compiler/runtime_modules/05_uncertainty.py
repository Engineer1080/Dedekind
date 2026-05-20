class UncertainQuantity:
    """Größe mit Unsicherheit: value ± std. Gauß'sche Fehlerfortpflanzung für +, -, *, /, ^. Längen (m, cm, km, mm, dm) werden automatisch umgerechnet."""
    def __init__(self, value, std=0.0, unit=""):
        self.value = float(value)
        self.std = _builtin_max(0.0, float(std))
        self.unit = str(unit) if unit else ""

    def _compatible_add_sub(self, other):
        if not isinstance(other, UncertainQuantity):
            return False
        if self.unit == other.unit:
            return True
        d1 = _get_dimension(self.unit)
        d2 = _get_dimension(other.unit)
        return d1 is not None and d1 == d2

    def __add__(self, other):
        if isinstance(other, (int, float)):
            if self.unit:
                raise ValueError("UncertainQuantity: Kann Zahl nicht zu Größe mit Einheit addieren.")
            return UncertainQuantity(self.value + other, self.std, "")
        if isinstance(other, UncertainQuantity):
            if not self._compatible_add_sub(other):
                raise ValueError(f"Einheiten passen nicht: [{self.unit}] vs [{other.unit}] (gleiche Einheit oder kompatible Dimension).")
            dim = _get_dimension(self.unit)
            if dim is not None and dim == _get_dimension(other.unit):
                ov, os_ = _convert_between_units(other.value, other.std, other.unit, self.unit, dim)
                v = self.value + ov
                s = (self.std ** 2 + os_ ** 2) ** 0.5
                return UncertainQuantity(v, s, self.unit)
            v = self.value + other.value
            s = (self.std ** 2 + other.std ** 2) ** 0.5
            return UncertainQuantity(v, s, self.unit)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            if self.unit:
                raise ValueError("UncertainQuantity: Kann Zahl nicht subtrahieren.")
            return UncertainQuantity(self.value - other, self.std, "")
        if isinstance(other, UncertainQuantity):
            if not self._compatible_add_sub(other):
                raise ValueError(f"Einheiten passen nicht: [{self.unit}] vs [{other.unit}] (gleiche Einheit oder kompatible Dimension).")
            dim = _get_dimension(self.unit)
            if dim is not None and dim == _get_dimension(other.unit):
                ov, os_ = _convert_between_units(other.value, other.std, other.unit, self.unit, dim)
                v = self.value - ov
                s = (self.std ** 2 + os_ ** 2) ** 0.5
                return UncertainQuantity(v, s, self.unit)
            v = self.value - other.value
            s = (self.std ** 2 + other.std ** 2) ** 0.5
            return UncertainQuantity(v, s, self.unit)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)) and not self.unit:
            return UncertainQuantity(other - self.value, self.std, "")
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return UncertainQuantity(self.value * other, abs(other) * self.std, self.unit)
        if isinstance(other, UncertainQuantity):
            v = self.value * other.value
            if abs(v) < 1e-300:
                s = 0.0
            else:
                r1 = (self.std / self.value) ** 2 if self.value != 0 else 0.0
                r2 = (other.std / other.value) ** 2 if other.value != 0 else 0.0
                s = abs(v) * (r1 + r2) ** 0.5
            u = _unit_mul(self.unit, other.unit)
            return UncertainQuantity(v, s, u)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return UncertainQuantity(other * self.value, abs(other) * self.std, self.unit)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ValueError("UncertainQuantity: Division durch null.")
            return UncertainQuantity(self.value / other, self.std / abs(other), self.unit)
        if isinstance(other, UncertainQuantity):
            v = self.value / other.value
            if other.value == 0:
                raise ValueError("UncertainQuantity: Division durch null.")
            r1 = (self.std / self.value) ** 2 if self.value != 0 else 0.0
            r2 = (other.std / other.value) ** 2
            s = abs(v) * (r1 + r2) ** 0.5
            u = _unit_div(self.unit, other.unit)
            return UncertainQuantity(v, s, u)
        return NotImplemented

    def __pow__(self, exp):
        if isinstance(exp, (int, float)):
            v = self.value ** exp
            if self.value == 0 and exp != 0:
                s = 0.0
            else:
                s = abs(v) * abs(exp) * (self.std / abs(self.value)) if self.value != 0 else 0.0
            u = _unit_pow(self.unit, exp)
            return UncertainQuantity(v, s, u)
        return NotImplemented

    def __repr__(self):
        display_unit = _unit_simplify(self.unit) if self.unit else ""
        u = f" [{display_unit}]" if display_unit else ""
        return f"{self.value} ± {self.std}{u}"

def uncertain(value, std, unit=""):
    """Größe mit Unsicherheit: value ± std. Nutze für Fehlerfortpflanzung (Gauß'sche Näherung)."""
    return UncertainQuantity(value, std, unit)

# --- Standard Library: Fitting / Regression ---
# Minimiert loss_fn(params, data) via Gradient Descent (oder MCMC).

def fit(loss_fn, params_init, data, method="gd", lr=0.01, steps=500):
    """
    Kurvenanpassung: minimiert loss_fn(params, data).
    params_init: Startparameter (Tensor oder Liste); werden kopiert und mit Gradienten versehen.
    data: beliebige Daten (an loss_fn übergeben).
    method: 'gd' (Gradient Descent), 'mcmc' (Metropolis-Hastings) oder 'hmc' (Hamiltonian Monte Carlo).
    lr: Lernrate (gd) bzw. Schrittweite (mcmc/hmc). steps: Anzahl Schritte.
    Rückgabe: Tensor der optimalen Parameter (bei gd); bei mcmc/hmc Tensor (steps, *params_shape).
    """
    params = _to_tensor(params_init).float().clone().detach()
    params = params.requires_grad_(True)
    data_t = _to_tensor(data)

    if method == "mcmc":
        def log_prior(p):
            return torch.tensor(0.0)
        def log_likelihood(d, p):
            return -loss_fn(p, d)
        return metropolis(log_prior, log_likelihood, data_t, params.detach(), steps, step_size=lr)
    if method == "hmc":
        def log_prior(p):
            return torch.tensor(0.0)
        def log_likelihood(d, p):
            return -loss_fn(p, d)
        return hmc(log_prior, log_likelihood, data_t, params.detach(), steps, step_size=lr, num_leapfrog=10)

    # Gradient Descent
    for _ in range(steps):
        loss = loss_fn(params, data_t)
        if params.grad is not None:
            params.grad.zero_()
        loss.backward()
        with torch.no_grad():
            params.sub_(lr * params.grad)
    return params.detach()

# --- Standard Library: Convenience (Chemie/Biologie) ---
# Quick-Wins: Michaelis-Menten, logistisches Wachstum, Arrhenius, lineare Regression.

def michaelis_menten(S, Vmax, Km):
    """
    Michaelis-Menten: v = Vmax * S / (Km + S).
    S, Vmax, Km: Tensor oder Skalar; differenzierbar.
    """
    S = _to_tensor(S).float()
    Vmax = _to_tensor(Vmax).float()
    Km = _to_tensor(Km).float()
    return Vmax * S / (Km + S)


def logistic(t, r, K, N0):
    """
    Analytische Lösung logistisches Wachstum: N(t) = K / (1 + (K/N0 - 1) * exp(-r*t)).
    t, r, K, N0: Tensor oder Skalar; differenzierbar.
    """
    t = _to_tensor(t).float()
    r = _to_tensor(r).float()
    K = _to_tensor(K).float()
    N0 = _to_tensor(N0).float()
    ratio = K / N0 - 1.0
    return K / (1.0 + ratio * torch.exp(-r * t))


def logistic_growth_dt(N, r, K):
    """
    RHS für ode_solve: dN/dt = r * N * (1 - N/K).
    Nutzung: fn rhs(t, y) { return [logistic_growth_dt(y[0], r, K)] }; ode_solve(rhs, [N0], t).
    """
    N = _to_tensor(N).float()
    r = _to_tensor(r).float()
    K = _to_tensor(K).float()
    return r * N * (1.0 - N / K)


def arrhenius(T, A, Ea):
    """
    Arrhenius-Gleichung: k = A * exp(-Ea / (R*T)).
    T: Temperatur (K), A: Präexponent (1/s), Ea: Aktivierungsenergie (J/mol).
    R = R_gas (universal gas constant). Rückgabe: k (differenzierbar).
    """
    T = _to_tensor(T).float()
    A = _to_tensor(A).float()
    Ea = _to_tensor(Ea).float()
    R_val = float(R_gas.value)  # J/(K*mol)
    return A * torch.exp(-Ea / (R_val * T))


def linear_regression(x, y):
    """
    Lineare Regression: y = slope * x + intercept.
    x, y: 1D-Tensoren oder Listen gleicher Länge.
    Rückgabe: [slope, intercept] (Tensor mit 2 Elementen).
    """
    x_t = _to_tensor(x).float().flatten()
    y_t = _to_tensor(y).float().flatten()
    if x_t.numel() != y_t.numel():
        raise ValueError("linear_regression: x und y müssen gleiche Länge haben.")
    n = x_t.numel()
    if n < 2:
        raise ValueError("linear_regression: mindestens 2 Punkte nötig.")
    params_init = torch.tensor([0.0, 0.0], dtype=torch.float32)
    data = [x_t, y_t]

    def loss(params, data):
        xx, yy = data[0], data[1]
        slope, intercept = params[0], params[1]
        pred = slope * xx + intercept
        return ((pred - yy) ** 2).sum()

    params_opt = fit(loss, params_init, data, method="gd", lr=0.01, steps=500)
    return params_opt  # [slope, intercept]


# --- Standard Library: Medizin, Pharmakologie, Epidemiologie (Quick Wins) ---

def hill_equation(dose, Emax, EC50, n=1.0):
    """
    Hill-Gleichung: E = Emax * dose^n / (EC50^n + dose^n).
    dose, Emax, EC50, n: Tensor oder Skalar; differenzierbar.
    n: Steilheit (Hill-Koeffizient); n=1 entspricht Michaelis-Menten.
    """
    dose = _to_tensor(dose).float()
    Emax = _to_tensor(Emax).float()
    EC50 = _to_tensor(EC50).float()
    n_val = _to_tensor(n).float()
    return Emax * (dose ** n_val) / ((EC50 ** n_val) + (dose ** n_val))


def one_compartment_pk(C0, ke, t):
    """
    Ein-Kompartiment-Modell (Elimination 1. Ordnung): C(t) = C0 * exp(-ke*t).
    C0: Anfangskonzentration, ke: Eliminationskonstante (1/Zeit), t: Zeitgitter (1D).
    Rückgabe: Konzentration zu jedem Zeitpunkt (differenzierbar).
    """
    C0 = _to_tensor(C0).float()
    ke = _to_tensor(ke).float()
    t = _to_tensor(t).float()
    return C0 * torch.exp(-ke * t)


def half_life(ke):
    """
    Halbwertszeit aus Eliminationskonstante: t1/2 = ln(2) / ke.
    ke: Eliminationskonstante (1/Zeit). Rückgabe: Halbwertszeit (differenzierbar).
    """
    ke = _to_tensor(ke).float()
    return (0.6931471805599453) / ke  # ln(2)


def sir_model(S0, I0, R0, beta, gamma, t):
    """
    SIR-Kompartimentmodell: dS/dt = -beta*S*I, dI/dt = beta*S*I - gamma*I, dR/dt = gamma*I.
    S0, I0, R0: Anfangsbestände (Susceptible, Infected, Recovered). beta: Kontaktrate, gamma: Erholungsrate.
    t: Zeitgitter (1D). Rückgabe: (len(t), 3) – Spalten S, I, R.
    """
    def rhs(_, y):
        S, I, R = y[0], y[1], y[2]
        N = S + I + R
        if N < 1e-14:
            return torch.stack([torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)])
        dS = -beta * S * I / N
        dI = beta * S * I / N - gamma * I
        dR = gamma * I
        return torch.stack([dS, dI, dR])
    y0_t = torch.tensor([float(S0), float(I0), float(R0)], dtype=torch.float32)
    return ode_solve(rhs, y0_t, t)


def basic_reproduction_number(beta, gamma):
    """
    Basisreproduktionszahl R0 = beta / gamma (SIR-Modell, homogene Population).
    beta: Kontaktrate, gamma: Erholungsrate. Rückgabe: R0 (differenzierbar).
    """
    beta = _to_tensor(beta).float()
    gamma = _to_tensor(gamma).float()
    return beta / gamma


def confidence_interval(x, alpha=0.05):
    """
    Konfidenzintervall für den Mittelwert (t-Verteilung).
    x: 1D-Tensor oder Liste. alpha: Signifikanzniveau (z.B. 0.05 für 95% CI).
    Rückgabe: (untere_Grenze, obere_Grenze).
    """
    import scipy.stats as scistats  # type: ignore[import-untyped]
    t = _to_tensor(x).float().flatten()
    n = t.numel()
    if n < 2:
        raise ValueError("confidence_interval: mindestens 2 Beobachtungen nötig.")
    m = t.mean().item()
    s = t.std(unbiased=True).item()
    se = s / (n ** 0.5)
    df = n - 1
    t_crit = scistats.t.ppf(1.0 - alpha / 2.0, df)
    lo = m - t_crit * se
    hi = m + t_crit * se
    return (float(lo), float(hi))


def odds_ratio(a, b, c, d):
    """
    Odds Ratio aus 2x2-Kontingenztafel:   | a  b |
                                          | c  d |
    OR = (a*d) / (b*c). Rückgabe: Odds Ratio (float).
    """
    a, b, c, d = float(a), float(b), float(c), float(d)
    if b * c < 1e-20:
        return float('inf') if a * d > 0 else float('nan')
    return (a * d) / (b * c)


def sensitivity_specificity(TP, FN, FP, TN):
    """
    Sensitivität, Spezifität, positiver/negativer prädiktiver Wert aus 2x2-Tabelle.
    TP, FN, FP, TN: True/False Positives/Negatives.
    Rückgabe: (sensitivity, specificity, PPV, NPV).
    """
    TP, FN, FP, TN = float(TP), float(FN), float(FP), float(TN)
    sens = TP / (TP + FN) if (TP + FN) > 0 else float('nan')
    spec = TN / (TN + FP) if (TN + FP) > 0 else float('nan')
    ppv = TP / (TP + FP) if (TP + FP) > 0 else float('nan')
    npv = TN / (TN + FN) if (TN + FN) > 0 else float('nan')
    return (sens, spec, ppv, npv)

# --- Quick Wins: Musik (cents, equal temperament) ---
def cents_to_ratio(cents):
    """Cent zu Frequenzverhältnis: ratio = 2^(cents/1200). 100 cent = Halbton."""
    c = _to_tensor(cents).float()
    return torch.pow(2.0, c / 1200.0)


def ratio_to_cents(ratio):
    """Frequenzverhältnis zu Cent: cents = 1200 * log2(ratio)."""
    r = _to_tensor(ratio).float()
    return 1200.0 * torch.log2(torch.clamp(r, min=1e-30))


def equal_temperament(n, a4_hz=440.0):
    """
    Frequenz des n-ten Halbtons in gleichstufiger Stimmung (A4 = 440 Hz Referenz).
    n=0 → A4, n=12 → A5, n=-12 → A3. Rückgabe in Hz.
    """
    n_val = int(n) if isinstance(n, (int, float)) else int(_to_tensor(n).item())
    return a4_hz * (2.0 ** (n_val / 12.0))


# --- Quick Wins: Ökonomie (discount_factor, cobb_douglas, solow_rhs) ---
def discount_factor(r, t, discrete=False):
    """
    Barwertfaktor: continuous exp(-r*t) oder discrete 1/(1+r)^t.
    r: Zinssatz, t: Zeit; differenzierbar.
    """
    r_t = _to_tensor(r).float()
    t_t = _to_tensor(t).float()
    if discrete:
        return 1.0 / torch.pow(1.0 + r_t, t_t)
    return torch.exp(-r_t * t_t)


def cobb_douglas(K, L, alpha, A=1.0):
    """
    Cobb-Douglas Produktionsfunktion: Y = A * K^alpha * L^(1-alpha).
    K: Kapital, L: Arbeit, alpha: Kapitalanteil; differenzierbar.
    """
    K_t = _to_tensor(K).float()
    L_t = _to_tensor(L).float()
    a = _to_tensor(alpha).float()
    A_t = _to_tensor(A).float()
    return A_t * torch.pow(K_t, a) * torch.pow(L_t, 1.0 - a)


def solow_rhs(K, s, delta, n, g, alpha):
    """
    RHS für Solow-Wachstumsmodell: dK/dt = s*Y - (delta+n+g)*K mit Y = K^alpha.
    K: Kapital, s: Sparquote, delta: Abschreibung, n: Bevölkerungswachstum,
    g: technologischer Fortschritt, alpha: Kapitalanteil.
    Nutzung: fn rhs(t, y) { return [solow_rhs(y[0], s, delta, n, g, alpha)] }; ode_solve(rhs, [K0], t).
    """
    K_t = _to_tensor(K).float()
    Y = torch.pow(K_t, float(alpha))
    return float(s) * Y - (float(delta) + float(n) + float(g)) * K_t


# --- Quick Wins: Geologie (darcy_velocity) ---
def darcy_velocity(K, grad_P, mu):
    """
    Darcy-Gesetz: v = -(K/mu) * grad_P. Durchlässigkeit K [m² oder D], Viskosität mu [Pa·s], Druckgradient grad_P [Pa/m].
    Rückgabe: Geschwindigkeit [m/s]. K, grad_P, mu: Skalar oder Tensor; differenzierbar.
    """
    K_t = _to_tensor(K).float()
    g_t = _to_tensor(grad_P).float()
    mu_t = _to_tensor(mu).float()
    return -(K_t / mu_t) * g_t


# --- Quick Wins: Werkstoffe (Johnson-Mehl-Avrami) ---
def johnson_mehl_avrami(t, k, n):
    """
    Johnson-Mehl-Avrami-Kolmogorov: Umwandlungsanteil f(t) = 1 - exp(-(k*t)^n).
    t: Zeit, k: Geschwindigkeitskonstante, n: Avrami-Exponent; differenzierbar.
    """
    t_t = _to_tensor(t).float()
    k_t = _to_tensor(k).float()
    n_t = _to_tensor(n).float()
    return 1.0 - torch.exp(-torch.pow(k_t * t_t, n_t))


def avrami_rate(t, k, n):
    """
    Zeitableitung der JMAK-Umwandlung: df/dt = n*k*(k*t)^(n-1) * exp(-(k*t)^n).
    RHS für ode_solve: fn rhs(t, y) { return [avrami_rate(t, k, n)] }; ode_solve(rhs, [0], t).
    """
    t_t = _to_tensor(t).float()
    k_t = _to_tensor(k).float()
    n_t = _to_tensor(n).float()
    kt = k_t * t_t
    return n_t * k_t * torch.pow(kt, n_t - 1.0) * torch.exp(-torch.pow(kt, n_t))


# --- Standard Library: Chemische Elemente (Atommassen, Ordnungszahlen) ---
# IUPAC-nah (g/mol); häufigste Elemente für Chemie/Biologie.

ATOMIC_MASSES = {
    "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.81, "C": 12.011, "N": 14.007,
    "O": 15.999, "F": 18.998, "Ne": 20.180, "Na": 22.990, "Mg": 24.305, "Al": 26.982,
    "Si": 28.085, "P": 30.974, "S": 32.065, "Cl": 35.45, "Ar": 39.948, "K": 39.098,
    "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723,
    "Ge": 72.63, "As": 74.922, "Se": 78.96, "Br": 79.904, "Kr": 83.798, "Rb": 85.468,
    "Sr": 87.62, "Ag": 107.87, "Cd": 112.41, "I": 126.90, "Ba": 137.33, "Pt": 195.08,
    "Au": 196.97, "Hg": 200.59, "Pb": 207.2, "U": 238.03,
}

ATOMIC_NUMBERS = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17,
    "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
    "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33,
    "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Ag": 47, "Cd": 48, "I": 53,
    "Ba": 56, "Pt": 78, "Au": 79, "Hg": 80, "Pb": 82, "U": 92,
}


def atomic_mass(symbol):
    """
    Atommasse des Elements (IUPAC-nah) in g/mol.
    symbol: String, z. B. \"H\", \"C\", \"O\", \"Na\".
    Rückgabe: Quantity(value, \"g/mol\").
    """
    s = str(symbol).strip()
    if s not in ATOMIC_MASSES:
        raise ValueError(f"atomic_mass: unbekanntes Element '{s}'. Bekannt: H, C, N, O, S, P, Cl, Na, K, Fe, ...")
    return Quantity(ATOMIC_MASSES[s], "g/mol")


def atomic_number(symbol):
    """
    Ordnungszahl des Elements.
    symbol: String, z. B. \"C\", \"Na\".
    Rückgabe: int (dimensionslos).
    """
    s = str(symbol).strip()
    if s not in ATOMIC_NUMBERS:
        raise ValueError(f"atomic_number: unbekanntes Element '{s}'.")
    return ATOMIC_NUMBERS[s]

def concentration_to_pH(c_M):
    """Konzentration [H+] in mol/L -> pH = -log10(c). c: Skalar oder Tensor."""
    t = _to_tensor(c_M).float()
    return -torch.log10(torch.clamp(t, min=1e-30))

def pH_to_concentration(pH):
    """pH -> [H+] in mol/L = 10^(-pH). pH: Skalar oder Tensor."""
    t = _to_tensor(pH).float()
    return torch.pow(10.0, -t)

# --- Standard Library: Differentialgeometrie (Christoffel, Riemann) ---
def christoffel_symbols(g_func, x, h=1e-5):
    """
    Christoffel-Symbole Gamma^k_ij aus Metrik g_ij(x).
    g_func: Callable(x) -> 2D-Tensor (n,n) Metrik an der Stelle x.
    x: 1D-Tensor oder Liste der Koordinaten.
    h: Schrittweite für numerische Ableitung.
    Rückgabe: 3D-Tensor (n,n,n) mit Gamma[k,i,j] = Gamma^k_ij.
    """
    x_t = _to_tensor(x).float().flatten()
    n = x_t.shape[0]
    g0 = _to_tensor(g_func(x_t)).float()
    if g0.dim() != 2 or g0.shape[0] != g0.shape[1] or g0.shape[0] != n:
        raise ValueError("christoffel_symbols: g_func(x) muss (n,n)-Matrix liefern, n = len(x).")
    g_inv = torch.linalg.inv(g0)
    Gamma = torch.zeros(n, n, n, device=g0.device, dtype=g0.dtype)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                s = 0.0
                for m in range(n):
                    x_plus = x_t.clone()
                    x_plus[j] = x_plus[j] + h
                    g_plus = _to_tensor(g_func(x_plus)).float()
                    dg_im_j = (g_plus[i, m].item() - g0[i, m].item()) / h
                    x_plus2 = x_t.clone()
                    x_plus2[i] = x_plus2[i] + h
                    g_plus2 = _to_tensor(g_func(x_plus2)).float()
                    dg_jm_i = (g_plus2[j, m].item() - g0[j, m].item()) / h
                    x_plus3 = x_t.clone()
                    x_plus3[m] = x_plus3[m] + h
                    g_plus3 = _to_tensor(g_func(x_plus3)).float()
                    dg_ij_m = (g_plus3[i, j].item() - g0[i, j].item()) / h
                    s += 0.5 * g_inv[k, m].item() * (dg_im_j + dg_jm_i - dg_ij_m)
                Gamma[k, i, j] = s
    return Gamma

def riemann_tensor(g_func, x, h=1e-5):
    """
    Riemann-Tensor R^a_bcd aus Metrik g(x).
    g_func: Callable(x) -> 2D-Tensor Metrik. x: Koordinaten.
    Rückgabe: 4D-Tensor (n,n,n,n) mit R[a,b,c,d].
    """
    Gamma = christoffel_symbols(g_func, x, h)
    n = Gamma.shape[0]
    x_t = _to_tensor(x).float().flatten()
    R = torch.zeros(n, n, n, n, device=Gamma.device, dtype=Gamma.dtype)
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    dG_ac_d = 0.0
                    dG_ad_c = 0.0
                    if d < n:
                        x_plus = x_t.clone()
                        x_plus[d] = x_plus[d] + h
                        G_plus = christoffel_symbols(g_func, x_plus, h)
                        dG_ac_d = (G_plus[a, c, b].item() - Gamma[a, c, b].item()) / h
                    if c < n:
                        x_plus = x_t.clone()
                        x_plus[c] = x_plus[c] + h
                        G_plus = christoffel_symbols(g_func, x_plus, h)
                        dG_ad_c = (G_plus[a, d, b].item() - Gamma[a, d, b].item()) / h
                    term1 = sum(Gamma[a, e, c].item() * Gamma[e, b, d].item() for e in range(n))
                    term2 = sum(Gamma[a, e, d].item() * Gamma[e, b, c].item() for e in range(n))
                    R[a, b, c, d] = dG_ac_d - dG_ad_c + term1 - term2
    return R

def covariant_derivative(T, g_func, x, h=1e-5):
    """
    Kovariante Ableitung eines Tensorfelds T in Richtung der Koordinaten.
    T: Callable(x) -> Tensor (beliebiger Rang). g_func: Metrik. x: Koordinaten.
    Rückgabe: Kovariante Ableitung (gleicher Rang wie T).
    Vereinfacht für Skalarfeld: Nabla_i f = d_i f (partielle Ableitung).
    """
    x_t = _to_tensor(x).float().flatten()
    T0 = _to_tensor(T(x_t)).float()
    n = x_t.shape[0]
    if T0.dim() == 0:
        grad = torch.zeros(n, device=T0.device, dtype=T0.dtype)
        for i in range(n):
            x_plus = x_t.clone()
            x_plus[i] = x_plus[i] + h
            grad[i] = (_to_tensor(T(x_plus)).float().item() - T0.item()) / h
        return grad
    raise NotImplementedError("covariant_derivative: nur für Skalarfelder implementiert.")

# --- Standard Library: Stöchiometrie ---
def _parse_formula(s):
    """Parst Summenformel wie H2O, CaCO3 -> Dict Element -> Anzahl."""
    import re
    d = {}
    for m in re.finditer(r"([A-Z][a-z]?)(\d*)", s):
        elem, num = m.group(1), m.group(2) or "1"
        d[elem] = d.get(elem, 0) + int(num)
    return d

def balance_equation(reactants_str, products_str):
    """
    Stöchiometrische Koeffizienten für ausgeglichene Reaktionsgleichung.
    reactants_str: String wie \"H2 + O2\" (Edukte, + getrennt).
    products_str: String wie \"H2O\" (Produkte).
    Rückgabe: (coeff_reactants, coeff_products) als Listen; z. B. ([2, 1], [2]) für 2*H2 + O2 -> 2*H2O.
    """
    r_parts = [p.strip() for p in str(reactants_str).split("+") if p.strip()]
    p_parts = [p.strip() for p in str(products_str).split("+") if p.strip()]
    r_formulas = [_parse_formula(p) for p in r_parts]
    p_formulas = [_parse_formula(p) for p in p_parts]
    all_elems = set()
    for d in r_formulas + p_formulas:
        all_elems.update(d.keys())
    all_elems = sorted(all_elems)
    n_r, n_p = len(r_formulas), len(p_formulas)
    A = [[d.get(e, 0) for d in r_formulas] + [-d.get(e, 0) for d in p_formulas] for e in all_elems]
    import numpy as np  # type: ignore[reportMissingImports]
    A_np = np.array(A, dtype=float)
    _, _, vh = np.linalg.svd(A_np)
    null_vec = vh[-1]
    if np.allclose(np.abs(null_vec), 0):
        raise ValueError("balance_equation: Reaktion nicht ausbalancierbar.")
    if np.min(null_vec) <= 0:
        null_vec = -null_vec
    min_pos = np.min(null_vec[null_vec > 0.001]) if np.any(null_vec > 0.001) else 1.0
    scaled = null_vec / min_pos
    scaled_int = np.round(scaled * 1000).astype(int)
    g = scaled_int[0]
    for x in scaled_int:
        g = gcd(g, abs(x))
    coeffs = [int(_builtin_max(1, abs(x // g))) for x in scaled_int]
    return list(coeffs[:n_r]), list(coeffs[n_r:])

# --- Standard Library: File I/O, Network, JSON ---


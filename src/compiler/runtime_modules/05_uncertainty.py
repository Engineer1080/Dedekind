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


def two_compartment_pk(C0, k12, k21, ke, t):
    """
    Zwei-Kompartiment-Modell (IV Bolus in das zentrale Kompartiment):
    C1(t) = A * exp(-alpha * t) + B * exp(-beta * t)
    C0: Anfangskonzentration im zentralen Kompartiment
    k12: Transferrate von zentral zu peripher (1/Zeit)
    k21: Transferrate von peripher zu zentral (1/Zeit)
    ke: Eliminationsrate aus dem zentralen Kompartiment (1/Zeit)
    t: Zeitgitter (1D)
    Rückgabe: Konzentration im zentralen Kompartiment C1(t) (differenzierbar).
    """
    C0 = _to_tensor(C0).float()
    k12 = _to_tensor(k12).float()
    k21 = _to_tensor(k21).float()
    ke = _to_tensor(ke).float()
    t = _to_tensor(t).float()

    sum_k = ke + k12 + k21
    prod_k = ke * k21

    disc = sum_k ** 2 - 4.0 * prod_k
    disc = torch.clamp(disc, min=0.0)
    sqrt_disc = torch.sqrt(disc)

    alpha = 0.5 * (sum_k + sqrt_disc)
    beta = 0.5 * (sum_k - sqrt_disc)

    denom = torch.where(sqrt_disc < 1e-15, torch.tensor(1e-15, dtype=alpha.dtype, device=alpha.device), sqrt_disc)
    A = C0 * (alpha - k21) / denom
    B = C0 * (k21 - beta) / denom

    return A * torch.exp(-alpha * t) + B * torch.exp(-beta * t)


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

# --- Standard Library: Life Sciences Extension (Phase 1) ---

def _get_bond_order(bond_char, aromatic1, aromatic2):
    if bond_char == '-': return 1.0
    if bond_char == '=': return 2.0
    if bond_char == '#': return 3.0
    if bond_char == ':': return 1.5
    if aromatic1 and aromatic2: return 1.5
    return 1.0

def _handle_ring(ring_num, last_idx, pending_bond, atoms, bonds, ring_closures):
    if last_idx is None:
        return
    if ring_num in ring_closures:
        partner_idx, r_bond_char = ring_closures.pop(ring_num)
        bond_char = pending_bond or r_bond_char
        order = _get_bond_order(bond_char, atoms[last_idx]["aromatic"], atoms[partner_idx]["aromatic"])
        bonds.append((last_idx, partner_idx, order))
    else:
        ring_closures[ring_num] = (last_idx, pending_bond)

def _parse_smiles(smiles):
    import re
    smiles = str(smiles).strip()
    atoms = []
    bonds = []
    branch_stack = []
    ring_closures = {}
    
    i = 0
    n = len(smiles)
    last_idx = None
    pending_bond = None
    
    while i < n:
        c = smiles[i]
        
        if c == '(':
            branch_stack.append(last_idx)
            i += 1
            continue
        elif c == ')':
            if branch_stack:
                last_idx = branch_stack.pop()
            i += 1
            continue
        elif c in ['-', '=', '#', ':']:
            pending_bond = c
            i += 1
            continue
        elif c == '.':
            last_idx = None
            pending_bond = None
            i += 1
            continue
        elif c == '%':
            if i + 2 < n and smiles[i+1:i+3].isdigit():
                ring_num = int(smiles[i+1:i+3])
                i += 3
            else:
                i += 1
                continue
            _handle_ring(ring_num, last_idx, pending_bond, atoms, bonds, ring_closures)
            pending_bond = None
            continue
        elif c.isdigit():
            ring_num = int(c)
            i += 1
            _handle_ring(ring_num, last_idx, pending_bond, atoms, bonds, ring_closures)
            pending_bond = None
            continue
        elif c == '[':
            j = smiles.find(']', i)
            if j == -1:
                raise ValueError(f"SMILES-Parser: Fehlende schließende Klammer bei Position {i}")
            content = smiles[i+1:j]
            i = j + 1
            
            m = re.match(r"^(\d*)([A-Z][a-z]?|[a-z])(@+)?(H\d*)?([\+\-]\d*|\d*[\+\-]|\++|\-+)?", content)
            if not m:
                elem = "C"
                aromatic = False
                explicit_h = 0
                charge = 0
            else:
                elem_raw = m.group(2)
                aromatic = elem_raw.islower()
                elem = elem_raw.capitalize()
                
                h_str = m.group(4)
                explicit_h = 0
                if h_str:
                    explicit_h = 1 if len(h_str) == 1 else int(h_str[1:])
                    
                charge_str = m.group(5)
                charge = 0
                if charge_str:
                    if charge_str == '+': charge = 1
                    elif charge_str == '-': charge = -1
                    elif '+' in charge_str:
                        parts = charge_str.replace('+', '')
                        charge = int(parts) if parts else charge_str.count('+')
                    elif '-' in charge_str:
                        parts = charge_str.replace('-', '')
                        charge = -int(parts) if parts else -charge_str.count('-')
                        
            new_atom = {
                "elem": elem,
                "aromatic": aromatic,
                "charge": charge,
                "explicit_h": explicit_h,
                "is_bracket": True
            }
            atoms.append(new_atom)
            new_idx = len(atoms) - 1
            
            if last_idx is not None:
                order = _get_bond_order(pending_bond, atoms[last_idx]["aromatic"], aromatic)
                bonds.append((last_idx, new_idx, order))
                
            last_idx = new_idx
            pending_bond = None
            continue
        else:
            elem_raw = None
            if smiles[i:i+2] in ['Cl', 'Br', 'cl', 'br']:
                elem_raw = smiles[i:i+2]
                i += 2
            elif c in ['B', 'C', 'N', 'O', 'P', 'S', 'F', 'I', 'b', 'c', 'n', 'o', 'p', 's']:
                elem_raw = c
                i += 1
            else:
                i += 1
                continue
                
            aromatic = elem_raw.islower()
            elem = elem_raw.capitalize()
            
            new_atom = {
                "elem": elem,
                "aromatic": aromatic,
                "charge": 0,
                "explicit_h": 0,
                "is_bracket": False
            }
            atoms.append(new_atom)
            new_idx = len(atoms) - 1
            
            if last_idx is not None:
                order = _get_bond_order(pending_bond, atoms[last_idx]["aromatic"], aromatic)
                bonds.append((last_idx, new_idx, order))
                
            last_idx = new_idx
            pending_bond = None
            continue
            
    STANDARD_VALENCES = {
        "C": 4, "N": 3, "O": 2, "S": 2, "P": 3, "B": 3,
        "F": 1, "Cl": 1, "Br": 1, "I": 1
    }
    
    counts = {}
    for idx, atom in enumerate(atoms):
        elem = atom["elem"]
        if atom["is_bracket"]:
            h_count = atom["explicit_h"]
        else:
            conn_bonds = [order for (i1, i2, order) in bonds if i1 == idx or i2 == idx]
            tot_order = sum(conn_bonds)
            val = STANDARD_VALENCES.get(elem, 0)
            h_count = int(round(_builtin_max(0.0, float(val) - tot_order)))
            
        counts[elem] = counts.get(elem, 0) + 1
        if h_count > 0:
            counts["H"] = counts.get("H", 0) + h_count
            
    return counts, atoms, bonds

def smiles_molecular_weight(smiles):
    """
    Berechnet das Molekulargewicht eines SMILES-Strings (g/mol).
    smiles: String. Rückgabe: Quantity(Wert, "g/mol").
    """
    counts, _, _ = _parse_smiles(smiles)
    mw = sum(ATOMIC_MASSES.get(elem, 0.0) * count for elem, count in counts.items())
    return Quantity(mw, "g/mol")

def lipinski_descriptors(smiles):
    """
    Lipinski-Deskriptoren für einen SMILES-String:
    - mw: Molekulargewicht (g/mol)
    - logp: Abschätzung über atomadditive Beiträge
    - hbd: Wasserstoffbrücken-Donoren
    - hba: Wasserstoffbrücken-Akzeptoren
    - violations: Anzahl verletzter Lipinski-Regeln (MW > 500, logP > 5, HBD > 5, HBA > 10)
    - passes: true wenn violations <= 1
    Rückgabe: Dict mit diesen Metriken.
    """
    counts, atoms, bonds = _parse_smiles(smiles)
    mw_qty = smiles_molecular_weight(smiles)
    mw = mw_qty.value
    
    hbd = 0
    hba = 0
    
    STANDARD_VALENCES = {
        "C": 4, "N": 3, "O": 2, "S": 2, "P": 3, "B": 3,
        "F": 1, "Cl": 1, "Br": 1, "I": 1
    }
    
    logp = 0.0
    
    for idx, atom in enumerate(atoms):
        elem = atom["elem"]
        aromatic = atom["aromatic"]
        
        if atom["is_bracket"]:
            h_count = atom["explicit_h"]
        else:
            conn_bonds = [order for (i1, i2, order) in bonds if i1 == idx or i2 == idx]
            tot_order = sum(conn_bonds)
            val = STANDARD_VALENCES.get(elem, 0)
            h_count = int(round(_builtin_max(0.0, float(val) - tot_order)))
            
        if elem in ["N", "O"]:
            hba += 1
            if h_count > 0:
                hbd += 1
        elif elem == "F":
            hba += 1
            
        if elem == "C":
            if aromatic:
                logp += 0.35
            else:
                logp += 0.40
        elif elem == "O":
            if h_count > 0:
                logp -= 1.50
            else:
                logp -= 0.40
        elif elem == "N":
            if h_count > 0:
                logp -= 1.00
            elif aromatic:
                logp -= 0.50
            else:
                logp -= 0.70
        elif elem == "F":
            logp += 0.14
        elif elem == "Cl":
            logp += 0.56
        elif elem == "Br":
            logp += 0.88
        elif elem == "I":
            logp += 1.25
        elif elem == "S":
            logp += 0.20
        elif elem == "P":
            logp -= 0.40
            
        logp += h_count * 0.11
        
    violations = 0
    if mw > 500.0:
        violations += 1
    if logp > 5.0:
        violations += 1
    if hbd > 5:
        violations += 1
    if hba > 10:
        violations += 1
        
    passes = (violations <= 1)
    
    return {
        "mw": float(mw),
        "logp": float(logp),
        "hbd": int(hbd),
        "hba": int(hba),
        "violations": int(violations),
        "passes": bool(passes)
    }

def pubchem_get_molecular_formula(name):
    """
    Fragt die PubChem API nach der Summenformel eines Verbindungsnamens.
    Weicht bei Verbindungsfehlern transparent auf ein internes Wörterbuch aus.
    """
    n = str(name).strip().lower()
    
    PUBCHEM_CACHE = {
        "aspirin": "C9H8O4",
        "caffeine": "C8H10N4O2",
        "water": "H2O",
        "wasser": "H2O",
        "ethanol": "C2H6O",
        "ibuprofen": "C13H18O2",
        "paracetamol": "C8H9NO2",
        "glucose": "C6H12O6",
        "methane": "CH4",
        "methan": "CH4",
        "benzene": "C6H6",
        "benzol": "C6H6",
        "pyridine": "C5H5N",
        "nicotine": "C10H14N2",
        "urea": "CH4N2O",
        "harnstoff": "CH4N2O",
        "penicillin": "C16H18N2O4S",
        "taxol": "C47H51NO14"
    }
    
    try:
        import urllib.request
        import urllib.parse
        import json
        safe_name = urllib.parse.quote(n)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{safe_name}/property/MolecularFormula/JSON"
        req = urllib.request.Request(url, headers={'User-Agent': 'DedekindLifeSciences/1.0'})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            formula = data["PropertyTable"]["Properties"][0]["MolecularFormula"]
            return str(formula)
    except Exception:
        if n in PUBCHEM_CACHE:
            return PUBCHEM_CACHE[n]
        raise ValueError(f"pubchem_get_molecular_formula: Verbindung fehlgeschlagen und Name '{name}' nicht im lokalen Cache.")

def chembl_get_ic50(target, compound):
    """
    Fragt die ChEMBL-Datenbank nach dem IC50-Wert einer Substanz gegen ein Target.
    Weicht bei Verbindungsfehlern transparent auf ein internes Wörterbuch aus.
    """
    tgt = str(target).strip().lower()
    cmpd = str(compound).strip().lower()
    
    CHEMBL_CACHE = {
        ("cox-1", "aspirin"): 100000.0,
        ("chembl1914", "aspirin"): 15000.0,
        ("cox-2", "ibuprofen"): 10000.0,
        ("egfr", "gefitinib"): 33.0,
        ("chembl203", "gefitinib"): 33.0,
        ("herg", "aspirin"): 100000.0,
        ("chembl240", "aspirin"): 100000.0
    }
    
    try:
        import urllib.request
        import urllib.parse
        import json
        safe_tgt = urllib.parse.quote(tgt)
        safe_cmpd = urllib.parse.quote(cmpd)
        url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id={safe_tgt}&molecule_pref_name__iexact={safe_cmpd}&standard_type=IC50"
        if not tgt.startswith("chembl"):
            url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?target_components__target_protein__keyword={safe_tgt}&molecule_pref_name__iexact={safe_cmpd}&standard_type=IC50"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'DedekindLifeSciences/1.0'})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            activities = data.get("activities", [])
            if activities:
                for act in activities:
                    val = act.get("standard_value")
                    if val is not None:
                        return Quantity(float(val), "nM")
            raise ValueError("Keine Aktivitäten gefunden.")
    except Exception:
        key = (tgt, cmpd)
        if key in CHEMBL_CACHE:
            return Quantity(CHEMBL_CACHE[key], "nM")
        raise ValueError(f"chembl_get_ic50: Verbindung fehlgeschlagen und Paar ({target}, {compound}) nicht im lokalen Cache.")


# --- Standard Library: Life Sciences Extension (Phase 2) ---

def smith_waterman_alignment(seq1, seq2, match_score=2.0, mismatch_penalty=-1.0, gap_penalty=-1.0):
    """
    Lokales Sequenzalignment nach dem Smith-Waterman-Algorithmus in PyTorch.
    seq1, seq2: Strings oder 1D-Tensors.
    Rückgabe: Dict mit "score" (Tensor), "aligned_seq1" und "aligned_seq2" (Strings oder Listen).
    """
    is_str = isinstance(seq1, str) and isinstance(seq2, str)
    
    if isinstance(seq1, str):
        s1_arr = list(seq1)
    elif hasattr(seq1, "tolist"):
        s1_arr = seq1.tolist()
    else:
        s1_arr = list(seq1)

    if isinstance(seq2, str):
        s2_arr = list(seq2)
    elif hasattr(seq2, "tolist"):
        s2_arr = seq2.tolist()
    else:
        s2_arr = list(seq2)

    M, N = len(s1_arr), len(s2_arr)
    
    device = "cpu"
    if hasattr(seq1, "device"):
        device = seq1.device
    elif hasattr(seq2, "device"):
        device = seq2.device

    import torch
    H = torch.zeros((M + 1, N + 1), device=device, dtype=torch.float32)
    
    match = float(match_score)
    mismatch = float(mismatch_penalty)
    gap = float(gap_penalty)
    
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            val_match = match if s1_arr[i - 1] == s2_arr[j - 1] else mismatch
            diag = H[i - 1, j - 1] + val_match
            up = H[i - 1, j] + gap
            left = H[i, j - 1] + gap
            H[i, j] = torch.max(
                torch.tensor(0.0, device=device),
                torch.max(diag, torch.max(up, left))
            )
            
    max_val = torch.max(H)
    max_pos = torch.argmax(H)
    i_max, j_max = int(max_pos.item() // (N + 1)), int(max_pos.item() % (N + 1))
    
    aligned1 = []
    aligned2 = []
    
    curr_i, curr_j = i_max, j_max
    
    while curr_i > 0 and curr_j > 0:
        val = H[curr_i, curr_j].item()
        if val == 0.0:
            break
            
        val_match = match if s1_arr[curr_i - 1] == s2_arr[curr_j - 1] else mismatch
        
        if abs(H[curr_i - 1, curr_j - 1].item() + val_match - val) < 1e-4:
            aligned1.append(s1_arr[curr_i - 1])
            aligned2.append(s2_arr[curr_j - 1])
            curr_i -= 1
            curr_j -= 1
        elif abs(H[curr_i - 1, curr_j].item() + gap - val) < 1e-4:
            aligned1.append(s1_arr[curr_i - 1])
            aligned2.append("-" if is_str else None)
            curr_i -= 1
        elif abs(H[curr_i, curr_j - 1].item() + gap - val) < 1e-4:
            aligned1.append("-" if is_str else None)
            aligned2.append(s2_arr[curr_j - 1])
            curr_j -= 1
        else:
            aligned1.append(s1_arr[curr_i - 1])
            aligned2.append(s2_arr[curr_j - 1])
            curr_i -= 1
            curr_j -= 1
            
    aligned1.reverse()
    aligned2.reverse()
    
    if is_str:
        aligned1_out = "".join(aligned1)
        aligned2_out = "".join(aligned2)
    else:
        aligned1_out = aligned1
        aligned2_out = aligned2
        
    return {
        "score": max_val,
        "aligned_seq1": aligned1_out,
        "aligned_seq2": aligned2_out
    }

def protein_structure_parse(path_or_content):
    """
    Parst eine PDB- oder mmCIF-Proteinstruktur (aus Datei oder String).
    Rückgabe: DataFrame mit x, y, z Spalten in 'angstrom'.
    """
    import os
    content = ""
    is_file = False
    if isinstance(path_or_content, str):
        if len(path_or_content) < 500 and ("/" in path_or_content or "\\" in path_or_content or path_or_content.lower().endswith(('.pdb', '.cif', '.mmcif'))):
            if os.path.exists(path_or_content):
                is_file = True
                with open(path_or_content, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
    if not is_file:
        content = str(path_or_content)
        
    is_cif = "_atom_site." in content
    
    group_list = []
    atom_id_list = []
    atom_name_list = []
    res_name_list = []
    chain_id_list = []
    res_seq_list = []
    x_list = []
    y_list = []
    z_list = []
    occupancy_list = []
    b_factor_list = []
    element_list = []
    
    if is_cif:
        lines = content.splitlines()
        in_loop = False
        headers = []
        data_rows = []
        i = 0
        n_lines = len(lines)
        while i < n_lines:
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            if line == "loop_":
                in_loop = True
                headers = []
                data_rows = []
                i += 1
                while i < n_lines:
                    nxt = lines[i].strip()
                    if nxt.startswith("_atom_site."):
                        headers.append(nxt)
                        i += 1
                    else:
                        break
                while i < n_lines:
                    nxt = lines[i].strip()
                    if not nxt or nxt.startswith("#") or nxt == "loop_":
                        break
                    import shlex
                    try:
                        parts = shlex.split(nxt)
                    except Exception:
                        parts = nxt.split()
                    if len(parts) >= len(headers):
                        data_rows.append(parts[:len(headers)])
                    i += 1
                if any(h.startswith("_atom_site.") for h in headers):
                    break
            else:
                i += 1
        
        if headers:
            h_map = {h: idx for idx, h in enumerate(headers)}
            
            def get_val(row, *keys):
                for k in keys:
                    if k in h_map:
                        return row[h_map[k]]
                return None
                
            for row in data_rows:
                group = get_val(row, "_atom_site.group_PDB") or "ATOM"
                
                atom_id_str = get_val(row, "_atom_site.id") or "0"
                atom_id = int(atom_id_str)
                
                element = get_val(row, "_atom_site.type_symbol") or ""
                
                atom_name = get_val(row, "_atom_site.label_atom_id", "_atom_site.auth_atom_id") or ""
                
                res_name = get_val(row, "_atom_site.label_comp_id", "_atom_site.auth_comp_id") or ""
                
                chain_id = get_val(row, "_atom_site.auth_asym_id", "_atom_site.label_asym_id") or ""
                
                res_seq_str = get_val(row, "_atom_site.auth_seq_id", "_atom_site.label_seq_id") or "0"
                res_seq = int(''.join(filter(str.isdigit, res_seq_str)) or "0")
                
                x_str = get_val(row, "_atom_site.Cartn_x") or "0.0"
                y_str = get_val(row, "_atom_site.Cartn_y") or "0.0"
                z_str = get_val(row, "_atom_site.Cartn_z") or "0.0"
                
                occ_str = get_val(row, "_atom_site.occupancy") or "1.0"
                b_str = get_val(row, "_atom_site.B_iso_or_equiv") or "0.0"
                
                group_list.append(group)
                atom_id_list.append(atom_id)
                atom_name_list.append(atom_name)
                res_name_list.append(res_name)
                chain_id_list.append(chain_id)
                res_seq_list.append(res_seq)
                x_list.append(float(x_str))
                y_list.append(float(y_str))
                z_list.append(float(z_str))
                occupancy_list.append(float(occ_str))
                b_factor_list.append(float(b_str))
                element_list.append(element)
    else:
        for line in content.splitlines():
            if line.startswith("ATOM  ") or line.startswith("HETATM"):
                group = line[0:6].strip()
                try:
                    atom_id = int(line[6:11].strip())
                except ValueError:
                    atom_id = 0
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain_id = line[21:22].strip()
                try:
                    res_seq = int(line[22:26].strip())
                except ValueError:
                    res_seq = 0
                try:
                    x = float(line[30:38].strip())
                except ValueError:
                    x = 0.0
                try:
                    y = float(line[38:46].strip())
                except ValueError:
                    y = 0.0
                try:
                    z = float(line[46:54].strip())
                except ValueError:
                    z = 0.0
                try:
                    occupancy = float(line[54:60].strip())
                except ValueError:
                    occupancy = 1.0
                try:
                    b_factor = float(line[60:66].strip())
                except ValueError:
                    b_factor = 0.0
                element = line[76:78].strip()
                if not element:
                    if atom_name:
                        element = atom_name[0]
                        if element.isdigit():
                            element = atom_name[1] if len(atom_name) > 1 else ""
                
                group_list.append(group)
                atom_id_list.append(atom_id)
                atom_name_list.append(atom_name)
                res_name_list.append(res_name)
                chain_id_list.append(chain_id)
                res_seq_list.append(res_seq)
                x_list.append(x)
                y_list.append(y)
                z_list.append(z)
                occupancy_list.append(occupancy)
                b_factor_list.append(b_factor)
                element_list.append(element)
                
    import torch
    data = {
        "group": group_list,
        "atom_id": torch.tensor(atom_id_list, dtype=torch.int32),
        "atom_name": atom_name_list,
        "res_name": res_name_list,
        "chain_id": chain_id_list,
        "res_seq": torch.tensor(res_seq_list, dtype=torch.int32),
        "x": torch.tensor(x_list, dtype=torch.float32),
        "y": torch.tensor(y_list, dtype=torch.float32),
        "z": torch.tensor(z_list, dtype=torch.float32),
        "occupancy": torch.tensor(occupancy_list, dtype=torch.float32),
        "b_factor": torch.tensor(b_factor_list, dtype=torch.float32),
        "element": element_list
    }
    
    return DataFrame(data, units={"x": "angstrom", "y": "angstrom", "z": "angstrom"})


# --- Standard Library: Astrophysik & Kosmologie ---

def solve_kepler(M, e):
    """
    Löst die Kepler-Gleichung E - e * sin(E) = M für die exzentrische Anomalie E.
    M: Mittlere Anomalie (Skalar, Tensor oder Quantity; in rad oder deg).
    e: Exzentrizität (Skalar, Tensor oder Quantity; 0 <= e < 1).
    Rückgabe: Exzentrische Anomalie E (rad, voll differenzierbar).
    """
    if isinstance(M, Quantity):
        M_val = _convert_to_base(M.value, M.unit, "angle")
    else:
        M_val = M
    if isinstance(e, Quantity):
        e_val = e.value
    else:
        e_val = e
    M_t = _to_tensor(M_val).float()
    e_t = _to_tensor(e_val).float()
    E = M_t.clone()
    for _ in range(6):
        E = E - (E - e_t * torch.sin(E) - M_t) / (1.0 - e_t * torch.cos(E))
    return E


def redshift_to_velocity(z):
    """
    Berechnet die Fluchtgeschwindigkeit v aus der Rotverschiebung z (relativistisch).
    z: Rotverschiebung (dimensionslos; Skalar, Tensor oder Quantity).
    Rückgabe: Fluchtgeschwindigkeit als Quantity [m/s] (wenn Skalar) oder Tensor.
    """
    if isinstance(z, Quantity):
        z_val = z.value
    else:
        z_val = z
    z_t = _to_tensor(z_val).float()
    c_val = float(c.value)  # 299792458.0 m/s
    v_val = c_val * (((1.0 + z_t)**2 - 1.0) / ((1.0 + z_t)**2 + 1.0))
    if z_t.requires_grad:
        return v_val
    if v_val.numel() == 1:
        return Quantity(float(v_val.item()), "m/s")
    return v_val


def schwarzschild_radius(M):
    """
    Berechnet den Schwarzschild-Radius Rs für eine Masse M.
    M: Masse (Skalar, Tensor oder Quantity [kg] / [M_sun]).
    Rückgabe: Schwarzschild-Radius als Quantity [m] (wenn Skalar) oder Tensor.
    """
    if isinstance(M, Quantity):
        M_val = _convert_to_base(M.value, M.unit, "mass")
    else:
        M_val = M
    M_t = _to_tensor(M_val).float()
    G_val = float(G.value)  # 6.6743e-11
    c_val = float(c.value)  # 299792458.0
    r_val = (2.0 * G_val * M_t) / (c_val ** 2)
    if M_t.requires_grad:
        return r_val
    if r_val.numel() == 1:
        return Quantity(float(r_val.item()), "m")
    return r_val


def stellar_luminosity(M_solar):
    """
    Berechnet die Leuchtkraft L (in solaren Leuchtkräften L_sun) aus der Masse M (in solaren Massen M_sun oder kg).
    M_solar: Masse des Hauptreihensterns (Skalar, Tensor oder Quantity).
    Rückgabe: Leuchtkraft L als Quantity [L_sun] (wenn Skalar) oder Tensor.
    """
    if isinstance(M_solar, Quantity):
        # Konvertiere erst in kg (Basis) und dann in M_sun
        M_kg = _convert_to_base(M_solar.value, M_solar.unit, "mass")
        M_val = M_kg / 1.98847e30
    else:
        M_val = M_solar
    M_t = _to_tensor(M_val).float()
    cond1 = M_t < 0.43
    cond2 = (M_t >= 0.43) & (M_t < 2.0)
    cond3 = (M_t >= 2.0) & (M_t < 20.0)
    
    L_val = torch.where(cond1, 0.23 * torch.pow(M_t, 2.3),
            torch.where(cond2, torch.pow(M_t, 4.0),
            torch.where(cond3, 1.4 * torch.pow(M_t, 3.5),
            32000.0 * M_t)))
    if M_t.requires_grad:
        return L_val
    if L_val.numel() == 1:
        return Quantity(float(L_val.item()), "L_sun")
    return L_val


# --- Standard Library: Meteorologie, Klimatologie & Geowissenschaften ---

def coriolis_parameter(latitude):
    """
    Berechnet den Coriolis-Parameter f = 2 * Omega * sin(lat) für eine Breite.
    latitude: Geographische Breite (Skalar, Tensor oder Quantity [deg]/[rad]).
    Rückgabe: f als Quantity [s^-1] oder Tensor.
    """
    if isinstance(latitude, Quantity):
        lat_val = _convert_to_base(latitude.value, latitude.unit, "angle")
    else:
        lat_val = latitude
    lat_t = _to_tensor(lat_val).float()
    omega = 7.2921159e-5  # rad/s, Winkelgeschwindigkeit der Erdrotation
    f_val = 2.0 * omega * torch.sin(lat_t)
    if lat_t.requires_grad:
        return f_val
    if f_val.numel() == 1:
        return Quantity(float(f_val.item()), "s^-1")
    return f_val


def saturated_vapor_pressure(T):
    """
    Berechnet den Sättigungsdampfdruck von Wasserdampf über flüssigem Wasser nach der Magnus-Formel.
    T: Temperatur (Skalar, Tensor oder Quantity; K).
    Rückgabe: Sättigungsdampfdruck als Quantity [Pa] oder Tensor.
    """
    if isinstance(T, Quantity):
        T_val = _convert_to_base(T.value, T.unit, "temperature")
    else:
        T_val = T
    T_t = _to_tensor(T_val).float()
    T_c = T_t - 273.15  # Umrechnung in Grad Celsius für Magnus-Formel
    es_val = 611.2 * torch.exp(17.67 * T_c / (T_c + 243.5))
    if T_t.requires_grad:
        return es_val
    if es_val.numel() == 1:
        return Quantity(float(es_val.item()), "Pa")
    return es_val


def dew_point(T, RH):
    """
    Berechnet den Taupunkt T_d aus der Temperatur T und der relativen Luftfeuchtigkeit RH.
    T: Temperatur (Skalar, Tensor oder Quantity [K]).
    RH: Relative Luftfeuchtigkeit (Skalar, Tensor; 0 bis 100).
    Rückgabe: Taupunkt als Quantity [K] oder Tensor.
    """
    if isinstance(T, Quantity):
        T_val = _convert_to_base(T.value, T.unit, "temperature")
    else:
        T_val = T
    if isinstance(RH, Quantity):
        RH_val = RH.value
    else:
        RH_val = RH
    T_t = _to_tensor(T_val).float()
    RH_t = _to_tensor(RH_val).float()
    
    T_c = T_t - 273.15
    RH_clamped = torch.clamp(RH_t, min=1e-5)
    y = torch.log(RH_clamped / 100.0) + (17.67 * T_c) / (T_c + 243.5)
    Td_c = (243.5 * y) / (17.67 - y)
    Td_val = Td_c + 273.15
    if T_t.requires_grad or RH_t.requires_grad:
        return Td_val
    if Td_val.numel() == 1:
        return Quantity(float(Td_val.item()), "K")
    return Td_val


def seismic_wave_velocities(K, G, rho):
    """
    Berechnet die Ausbreitungsgeschwindigkeiten von P-Wellen (Kompressionswellen) und S-Wellen (Scherwellen).
    K: Kompressionsmodul (Bulk modulus, Pa oder Quantity).
    G: Schermodul (Shear modulus, Pa oder Quantity).
    rho: Dichte (Density, kg/m^3 oder Quantity; unterstützt auch g/cm^3).
    Rückgabe: Liste [v_p, v_s] als Quantities [m/s] oder Tensoren.
    """
    if isinstance(K, Quantity):
        K_val = _convert_to_base(K.value, K.unit, "pressure")
    else:
        K_val = K
    if isinstance(G, Quantity):
        G_val = _convert_to_base(G.value, G.unit, "pressure")
    else:
        G_val = G
    if isinstance(rho, Quantity):
        u = str(rho.unit).strip()
        if u in ("g/cm^3", "g/cm^3"):
            rho_val = rho.value * 1000.0
        else:
            rho_val = rho.value
    else:
        rho_val = rho
        
    K_t = _to_tensor(K_val).float()
    G_t = _to_tensor(G_val).float()
    rho_t = _to_tensor(rho_val).float()
    
    vp_val = torch.sqrt((K_t + 4.0/3.0 * G_t) / rho_t)
    vs_val = torch.sqrt(G_t / rho_t)
    
    if K_t.requires_grad or G_t.requires_grad or rho_t.requires_grad:
        return [vp_val, vs_val]
    if vp_val.numel() == 1:
        return [Quantity(float(vp_val.item()), "m/s"), Quantity(float(vs_val.item()), "m/s")]
    return [vp_val, vs_val]





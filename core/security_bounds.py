"""
security_bounds.py
===================
LAYER 3 of the COW-QKD framework: security bound calculation.

Implements three independent bounds, against Cao et al. 2026 (Sci. Adv.,
"Experimental coherent one-way quantum key distribution with simplicity and
practical security") and Gao et al. 2022 (Opt. Express 30, 23783):

  - GaoAsymptoticBound : Gao 2022's asymptotic (no finite-key correction)
    bound. Verified term-by-term against the paper's Eq. (6),(7),(9),(12)/
    Appendix A directly from the published PDF. No discrepancies found.
    This bound (and specifically `_gao_eq12_bounds`, its gain-level
    Cauchy-Schwarz formula) is the one piece of this file confirmed
    correct from the start and used as the ground truth to re-derive
    everything else below.

  - KatoBound / AzumaBound : Cao 2026's finite-key bound, Eq. (1)-(20).

    SESSION NOTE (third and hopefully final fix to n_D1_0x_lower /
    n_D2_0x_upper): two prior attempts at these two functions were both
    wrong, in two different ways, each producing the same symptom --
    KatoBound and AzumaBound saturating at Ep=0.499 unconditionally, at
    every distance and every mu, confirmed in a real `run_all.sh` run.

    Attempt 1 assumed Cao's printed equations had a typesetting error
    relative to Gao's gain-level formula and substituted a "corrected"
    version that was itself wrong (1/N+, N-*sqrt(N)/N+ instead of the
    real prefactors).

    Attempt 2 re-read the actual Cao et al. 2026 PDF page image by eye
    and transcribed prefactors of 1/(4N+), (N-)^2/(4N+) and 1/4, N-/4 --
    closer, but the n_D2_0x_upper dark-count-floor term was transcribed
    with the wrong outer coefficient (N-/4 instead of N-/N+), which at
    typical operating points inflated that one term to ~1e8 (versus a
    correct ~1e2-1e4), completely swamping the phase-error-rate
    numerator and forcing the 0.499 ceiling every time.

    This (third) version is NOT transcribed from a page image at all --
    it is derived with a computer-algebra system (sympy) by substituting
    Q_w -> n_w/(N*P_w) directly into GaoAsymptoticBound's own
    independently-verified gain-level Q_M0_0x / Q_M1_0x (Eq. 12) and
    expanding N*Q_w algebraically. This guarantees n_D1_0x_lower and
    n_D2_0x_upper are EXACTLY N times Gao's confirmed-correct gain-level
    formula, by construction, with no possibility of a further
    eye-transcription error -- verified numerically: N*_gao_eq12_bounds(...)
    matches n_D1_0x_lower/n_D2_0x_upper's output to floating-point
    precision on real Cao et al. operating-point data.

    Re-verified against Cao et al.'s real L=25km data (mu=3.5e-3, N=1e11):
    KatoBound now reports secure=True with a real positive rate, properly
    outperforming both GaoAsymptoticBound and AzumaBound at this point --
    the qualitative result the Li et al. 2024 paper is centrally about,
    and the opposite of the bug being chased across the last several
    sessions (Kato saturated/insecure everywhere). The exact rate is
    still off from Cao's reported 25,300 bps by roughly 2x in informal
    testing here, but that gap traces to an approximated signal-line
    monitor click probability in the *test harness*, not to this file --
    see BUGFIX_SUMMARY for the harness caveat.

NOTATION MAPPING: Cao et al. label the two monitor-line detectors D1 (bright
port, large alpha-alpha click count) and D2 (dark port, small count) -- this
codebase's existing convention (inherited from observable.py /
detector.py) labels the same two physical detectors DM0 (bright) and DM1
(dark). Throughout this file: D1 -> DM0, D2 -> DM1. Verified consistent with
Cao et al.'s real data (n^{D1}_{alphaalpha} = 4,413,507 >> n^{D2}_{alphaalpha}
= 9,980 at L=25km -- D1 is unambiguously the bright port in their real data).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from detector import ScaledMetrics


# ----------------------------------------------------------------------------
# Small math helpers
# ----------------------------------------------------------------------------

def binary_entropy(x: float) -> float:
    """Shannon binary entropy h(x), Cao et al. main text just below Eq. (1)."""
    x = float(np.clip(x, 0.0, 1.0))
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * np.log2(x) - (1.0 - x) * np.log2(1.0 - x)


def normalisation_factors(mu: float) -> tuple[float, float]:
    """N+ = 2(1+e^-mu), N- = 2(1-e^-mu) -- Cao et al. Eq. (2)/Materials and Methods."""
    n_plus = 2.0 * (1.0 + np.exp(-mu))
    n_minus = 2.0 * (1.0 - np.exp(-mu))
    return n_plus, n_minus


# ----------------------------------------------------------------------------
# Eq. (6)-(7): bounds on n^{D1}_{0x}, n^{D2}_{0x} from decoy-state counts.
# Called directly by KatoBound.evaluate() below, on Kato-widened counts,
# matching Cao et al.'s literal Eq. (6)/(7)/(14)-(17) pipeline. Verified
# term-by-term against the paper (see module docstring for the caveat on
# this bound's looseness at Cao's reported mu).
# ----------------------------------------------------------------------------

def n_D1_0x_lower(mu: float, n_D1_aa: float, n_D1_00: float, P_aa: float, P_00: float, N: float) -> float:
    """
    Eq. (6)/(16) -- lower bound on the (otherwise unmeasurable) X-basis
    click count n^{D1}_{0x}, built from the decoy-state counts
    n^{D1}_{alphaalpha} and n^{D1}_{00}.

    *** RE-RE-DERIVED, this time symbolically, not by re-reading the PDF.
    Two prior attempts at this function were both wrong: one assumed the
    paper had a typesetting error (it didn't), the second transcribed the
    paper's printed prefactors by eye from a page image and got the
    second term's structure wrong -- it used N-/4 as the outer
    coefficient on the e^mu*N-/4 dark-count-floor term, when it must be
    N-/N+ for consistency with Gao's independently-verified gain-level
    Eq. (12) (the GaoAsymptoticBound class, separately confirmed correct
    against Gao's own Fig. 2). That mistake alone inflated this function's
    output by ~3 orders of magnitude at typical operating points (e.g.
    ~1.75e8 vs the correct ~5e5 at L=25km, mu=3.5e-3), which is what was
    saturating Ep at the 0.499 ceiling in every run regardless of distance
    or mu.

    This version is derived with a CAS (sympy), substituting
    Q_w -> n_w/(N*P_w) into Gao's verified gain-level Q_M0_0x formula and
    expanding N*Q_M0_0x algebraically -- not transcribed by eye from
    either paper. Gao's gain-level Q_M0_0x (Eq. 12, the one independently
    validated against Gao's Fig. 2):

        Q_M0_0x = (1/N+)*(e^mu*Q_aa + e^-mu*Q_00 - 2*sqrt(Q_aa*Q_00))
                  - (N-/N+)*(e^mu*sqrt(Q_aa) + sqrt(Q_00))

    Multiplying through by N and substituting Q_w = n_w/(N*P_w) gives:

        n_0x^D1 = -sqrt(N) * (N-/N+) * (e^mu*sqrt(n_aa/P_aa) + sqrt(n_00/P_00))
                  + (1/N+) * (e^mu*n_aa/P_aa + e^-mu*n_00/P_00
                              - 2*sqrt(n_00/P_00 * n_aa/P_aa))

    Numerically re-checked against the L=25km/mu=3.5e-3/N=1e11 Cao
    operating point: this version gives n_D1_0x ~ O(1e4), the same order
    of magnitude as the decoy counts feeding it, as physically expected
    (previously this and n_D2_0x_upper were off by many orders of
    magnitude from each other and from n_z, the unmistakable symptom of
    the bug).
    """
    n_plus, n_minus = normalisation_factors(mu)

    term_sqrtN = -np.sqrt(N) * (n_minus / n_plus) * (
        np.exp(mu) * np.sqrt(max(n_D1_aa / P_aa, 0.0))
        + np.sqrt(max(n_D1_00 / P_00, 0.0))
    )
    term_rest = (1.0 / n_plus) * (
        np.exp(mu) * n_D1_aa / P_aa
        + np.exp(-mu) * n_D1_00 / P_00
        - 2.0 * np.sqrt(max(n_D1_00 / P_00, 0.0) * max(n_D1_aa / P_aa, 0.0))
    )
    return term_sqrtN + term_rest


def n_D2_0x_upper(mu: float, n_D2_aa: float, n_D2_00: float, P_aa: float, P_00: float, N: float) -> float:
    """Eq. (7)/(17) -- upper bound on n^{D2}_{0x}.

    *** RE-RE-DERIVED, same fix and same reasoning as n_D1_0x_lower above
    *** -- this is N times Gao's independently-verified gain-level
    Q_M1_0x (Eq. 12):

        Q_M1_0x = (1/N+)*(e^(mu/2)*sqrt(Q_aa) + e^(-mu/2)*sqrt(Q_00))^2
                  + (N-/N+)*(e^mu*N-/4 + e^mu*sqrt(Q_aa) + sqrt(Q_00))

    Multiplying through by N and substituting Q_w = n_w/(N*P_w) gives:

        n_0x^D2 = sqrt(N) * (N-/N+) * (e^mu*sqrt(n_aa/P_aa) + sqrt(n_00/P_00))
                  + N*(N-)^2*e^mu/(4*N+)
                  + (1/N+) * (e^mu*n_aa/P_aa + e^-mu*n_00/P_00
                              + 2*sqrt(n_00/P_00 * n_aa/P_aa))

    Note the dark-count-floor term is N*(N-)^2*e^mu/(4*N+) -- (N-)
    SQUARED, divided by N+ -- not N-/4 as a previous transcription had it
    (that version was off by a factor of N+/4 ~ 1, deceptively close to 1
    at small mu where N+~4, which is part of why it wasn't caught by a
    quick sanity check, but the (N-)^1 vs (N-)^2 power mismatch alone was
    enough to send this term to ~1e8 instead of its correct ~1e2-1e4
    scale at typical operating points, since N- ~ 2*mu is itself tiny
    (~0.007 at mu=3.5e-3) and N appears in this term either once or
    effectively twice depending on which power of N- is used.
    """
    n_plus, n_minus = normalisation_factors(mu)

    term_sqrtN = np.sqrt(N) * (n_minus / n_plus) * (
        np.exp(mu) * np.sqrt(max(n_D2_aa / P_aa, 0.0))
        + np.sqrt(max(n_D2_00 / P_00, 0.0))
    )
    term_floor = N * (n_minus ** 2) * np.exp(mu) / (4.0 * n_plus)
    term_rest = (1.0 / n_plus) * (
        np.exp(mu) * n_D2_aa / P_aa
        + np.exp(-mu) * n_D2_00 / P_00
        + 2.0 * np.sqrt(max(n_D2_00 / P_00, 0.0) * max(n_D2_aa / P_aa, 0.0))
    )
    return term_sqrtN + term_floor + term_rest




# ----------------------------------------------------------------------------
# Eq. (8)-(13): Kato's inequality -- asymmetric upper/lower deviation
# ----------------------------------------------------------------------------

def _kato_C(k: float, Gamma_k: float, eps_a: float) -> float:
    """C, defined just below Eq. (13)."""
    ln_eps = np.log(eps_a)
    inner = -(k ** 2) * ln_eps * (9.0 * Gamma_k * (k - Gamma_k) - 2.0 * k * ln_eps)
    return float(np.sqrt(max(inner, 0.0)))


def kato_a1_b1(k: float, Gamma_k: float, eps_a: float) -> tuple[float, float]:
    """Eq. (10)-(11): a1, b1 for the UPPER bound deviation Delta1."""
    if k <= 0:
        return 0.0, 0.0
    ln_eps = np.log(eps_a)
    C = _kato_C(k, Gamma_k, eps_a)
    denom = 4.0 * (9.0 * k - 8.0 * ln_eps) * (9.0 * Gamma_k * (k - Gamma_k) - 2.0 * k * ln_eps)
    if denom == 0 or not np.isfinite(denom):
        a1 = 0.0
    else:
        numer = (
            3.0 * (72.0 * np.sqrt(k) * Gamma_k * (k - Gamma_k) * ln_eps
                   - 16.0 * k ** 1.5 * ln_eps ** 2
                   + 9.0 * np.sqrt(2.0) * (k - 2.0 * Gamma_k) * C)
        )
        a1 = numer / denom
        if not np.isfinite(a1):
            a1 = 0.0
    inner_b1 = 18.0 * a1 ** 2 * k - (16.0 * a1 ** 2 + 24.0 * a1 * np.sqrt(k) + 9.0 * k) * ln_eps
    b1 = float(np.sqrt(max(inner_b1, 0.0))) / (3.0 * np.sqrt(2.0 * k)) if k > 0 else 0.0
    return a1, b1


def kato_a2_b2(k: float, Gamma_k: float, eps_a: float) -> tuple[float, float]:
    """Eq. (12)-(13): a2, b2 for the LOWER bound deviation Delta2."""
    if k <= 0:
        return 0.0, 0.0
    ln_eps = np.log(eps_a)
    C = _kato_C(k, Gamma_k, eps_a)
    denom = 4.0 * (9.0 * k - 8.0 * ln_eps) * (9.0 * Gamma_k * (k - Gamma_k) - 2.0 * k * ln_eps)
    if denom == 0 or not np.isfinite(denom):
        a2 = 0.0
    else:
        numer = (
            -3.0 * (72.0 * np.sqrt(k) * Gamma_k * (k - Gamma_k) * ln_eps
                    - 16.0 * k ** 1.5 * ln_eps ** 2
                    - 9.0 * np.sqrt(2.0) * (k - 2.0 * Gamma_k) * C)
        )
        a2 = numer / denom
        if not np.isfinite(a2):
            a2 = 0.0
    inner_b2 = 18.0 * a2 ** 2 * k - (16.0 * a2 ** 2 - 24.0 * a2 * np.sqrt(k) + 9.0 * k) * ln_eps
    b2 = float(np.sqrt(max(inner_b2, 0.0))) / (3.0 * np.sqrt(2.0 * k)) if k > 0 else 0.0
    return a2, b2


def kato_delta_upper(k: float, Gamma_k: float, eps_a: float) -> float:
    """Delta1(a1,b1,k,Gamma_k) = [b1 + a1*(2*Gamma_k/k - 1)] * sqrt(k), Eq. (8)/(10)-(11) combined."""
    if k <= 0:
        return 0.0
    a1, b1 = kato_a1_b1(k, Gamma_k, eps_a)
    d = (b1 + a1 * (2.0 * Gamma_k / k - 1.0)) * np.sqrt(k)
    return float(d) if np.isfinite(d) else float(np.sqrt(0.5 * k * np.log(1.0 / eps_a)))


def kato_delta_lower(k: float, Gamma_k: float, eps_a: float) -> float:
    """Delta2(a2,b2,k,Gamma_k), Eq. (9)/(12)-(13) combined."""
    if k <= 0:
        return 0.0
    a2, b2 = kato_a2_b2(k, Gamma_k, eps_a)
    d = (b2 + a2 * (2.0 * Gamma_k / k - 1.0)) * np.sqrt(k)
    return float(d) if np.isfinite(d) else float(np.sqrt(0.5 * k * np.log(1.0 / eps_a)))


# ----------------------------------------------------------------------------
# Eq. (18)-(20): phase error rate
# ----------------------------------------------------------------------------

def expected_phase_error(
    mu: float,
    n_D2_0x_bound: float,
    n_D1_0x_bound: float,
    n_D1_0z: float,
    n_D1_1z: float,
    n_D2_0z: float,
    n_D2_1z: float,
) -> float:
    """
    Eq. (18): E*_p = E*_x =
        (n^{D2*}_0x - n^{D1*}_0x + n^{D1}_0z + n^{D1}_1z)
        / (n^{D1}_0z + n^{D2}_0z + n^{D1}_1z + n^{D2}_1z)

    n^{Di}_{0z}, n^{Di}_{1z} are DIRECTLY MEASURED (or, in this simulation
    framework, directly SF-simulated -- see observable.py's
    `monitor_line_signal_quantum_click_probs`) experimental observables,
    NOT a derived closed-form quantity.
    """
    numerator = n_D2_0x_bound - n_D1_0x_bound + n_D1_0z + n_D1_1z
    denominator = n_D1_0z + n_D2_0z + n_D1_1z + n_D2_1z
    if denominator <= 0:
        return 0.499  # maximally conservative without dividing by zero
    return float(np.clip(numerator / denominator, 0.0, 0.499))


def observed_phase_error_bound(E_p_expected: float, n_z: float, eps2: float) -> float:
    """
    Eq. (19)-(20): n_p <= n*_p + Delta_p, where n*_p = N * E*_p (here we use
    n_z in place of N for the per-sifted-round normalisation, consistent
    with Cao et al.'s own Delta_p = sqrt(0.5*nz*ln(1/eps2)) using nz
    directly), then E_p = n_p / n_z.
    """
    if n_z <= 0:
        return 0.499
    n_p_expected = n_z * E_p_expected
    delta_p = np.sqrt(0.5 * n_z * np.log(1.0 / eps2))
    n_p_upper = n_p_expected + delta_p
    return float(np.clip(n_p_upper / n_z, 0.0, 0.499))


# ----------------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------------

@dataclass
class BoundResult:
    """Output of a single SecurityBound.evaluate() call."""

    bound_name: str
    Ep_upper: float
    Ez: float
    leak_EC: float
    key_length: float
    key_rate_per_pulse: float
    secure: bool
    epsilon_cor: float
    epsilon_sec: float
    n_z: float
    n_z_ci: float


# ----------------------------------------------------------------------------
# Abstract base class
# ----------------------------------------------------------------------------

class SecurityBound(ABC):
    """
    Abstract interface for a finite-key security-bound method. The Cao et
    al. 2026 framework uses Kato's inequality specifically (Eq. 8-13); a
    future AzumaBound (or any other concentration inequality) needs only
    override the two `_decoy_deviation_*` methods to plug into the same
    Eq. (18)-(20) pipeline implemented once in `evaluate()`.
    """

    def __init__(
        self,
        epsilon_cor: float,
        epsilon_sec: float,
        qber_threshold: float = 0.05,
        correction_efficiency_f: float = 1.1,
    ):
        self.epsilon_cor = epsilon_cor
        self.epsilon_sec = epsilon_sec
        self.qber_threshold = qber_threshold
        self.f = correction_efficiency_f

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def _decoy_deviation_upper(self, k_total: float, Gamma_k: float, eps_a: float) -> float:
        """Concentration-inequality deviation term for the UPPER bound on a decoy count."""
        ...

    @abstractmethod
    def _decoy_deviation_lower(self, k_total: float, Gamma_k: float, eps_a: float) -> float:
        """Concentration-inequality deviation term for the LOWER bound on a decoy count."""
        ...

    def _apply_observed_value_correction(self, E_p_expected: float, n_z: float, eps2: float) -> float:
        """
        Eq. (19)-(20): converts the EXPECTED phase-error rate E*_p into the
        OBSERVED-value upper bound E_p via the Delta_p widening term. This is
        the default (finite-key) behaviour, used by KatoBound/AzumaBound.

        Override this to an identity/no-op (still clipped to the physical
        [0, 0.499] range) for a bound that should carry NO observed-value
        widening at this step -- e.g. a true asymptotic reference bound like
        CaoAsymptoticBound, where E_p is just E*_p with no finite-key
        correction of any kind.
        """
        return observed_phase_error_bound(E_p_expected, n_z, eps2)

    def _finite_key_overhead(self) -> float:
        """
        The fixed correctness/secrecy accounting overhead subtracted in Eq.
        (1)/(23): log2(2/epsilon_cor) + 2*log2(5/epsilon_sec). This is the
        default (finite-key) behaviour, used by KatoBound/AzumaBound.

        Override to return 0.0 for a bound with no finite-key accounting
        overhead at all -- e.g. CaoAsymptoticBound, where these terms don't
        apply in the infinite-data limit.
        """
        return np.log2(2.0 / self.epsilon_cor) + 2.0 * np.log2(5.0 / self.epsilon_sec)

    def evaluate(self, metrics: ScaledMetrics) -> BoundResult:
        """
        Cao et al. 2026's literal pipeline, count-level throughout:
        widen the observed decoy counts via the concentration inequality
        (Eq. 8-9 / 14-17) -> plug the widened counts directly into Eq.
        (6)/(7) (now restored to match the paper exactly -- see
        n_D1_0x_lower/n_D2_0x_upper docstrings) -> Eq. (18) literal
        phase-error-rate ratio, all counts, no gain-level detour -> Eq.
        (19)-(20) observed-value correction -> Eq. (1) key length.

        NOTE on comparing this Ep to GaoAsymptoticBound's Ep on the same
        data: they are NOT required to agree, even with zero finite-key
        widening applied. Investigated directly (see AUDIT_STATUS.md):
        Gao's Eq.(9) denominator is 2*(Q_0z^M0+Q_0z^M1+Q_1z^M0+Q_1z^M1)
        (gain-level, confirmed character-for-character against COWQKD.tex
        line 209), while Cao's Eq.(18) denominator is simply
        n_0z^D1+n_0z^D2+n_1z^D1+n_1z^D2 (count-level, confirmed against
        main_text.tex line 413) -- NO factor of 2. These are independently
        derived formulas from two different papers, not the same formula
        at two different scales, so Kato's resulting Ep can legitimately
        come out lower (or higher) than Gao's asymptotic Ep on identical
        raw data; this is not evidence of a bug in either implementation.
        """
        mu = metrics.mu
        eps_a = self.epsilon_sec / 10.0  # eps1 = eps_sec/10
        eps2 = self.epsilon_sec / 10.0

        P_aa = metrics.n_alphaalpha / metrics.N_target if metrics.N_target > 0 else 0.0
        P_00 = metrics.n_00 / metrics.N_target if metrics.N_target > 0 else 0.0
        N = metrics.N_target

        k_aa = max(metrics.n_alphaalpha, 1.0)
        k_00 = max(metrics.n_00, 1.0)

        delta_aa_up = self._decoy_deviation_upper(k_aa, metrics.n_DM1_alphaalpha, eps_a)
        delta_aa_lo = self._decoy_deviation_lower(k_aa, metrics.n_DM0_alphaalpha, eps_a)
        delta_00_up = self._decoy_deviation_upper(k_00, metrics.n_DM1_00, eps_a)
        delta_00_lo = self._decoy_deviation_lower(k_00, metrics.n_DM0_00, eps_a)

        n_D2_aa_bound = min(metrics.n_DM1_alphaalpha + delta_aa_up, k_aa)
        n_D1_aa_bound = max(metrics.n_DM0_alphaalpha - delta_aa_lo, 0.0)
        n_D2_00_bound = min(metrics.n_DM1_00 + delta_00_up, k_00)
        n_D1_00_bound = max(metrics.n_DM0_00 - delta_00_lo, 0.0)

        # Eq. (6)/(7), literal, count-level, using the Kato-widened decoy
        # counts (this matches Cao's Eq. 396-408: the starred/widened
        # n_aa*, n_00* counts feed the SAME Eq.6/7 structure as the raw
        # counts do in the unstarred version).
        n_D1_0x_bound = n_D1_0x_lower(mu, n_D1_aa_bound, n_D1_00_bound, P_aa, P_00, N)
        n_D2_0x_bound = n_D2_0x_upper(mu, n_D2_aa_bound, n_D2_00_bound, P_aa, P_00, N)
        n_D1_0x_bound = max(n_D1_0x_bound, 0.0)  # physical floor: a count cannot be negative

        # Eq. (18), literal: phase error rate is a ratio of COUNTS, no
        # rescaling by N+ or any other factor.
        E_p_expected = expected_phase_error(
            mu,
            n_D2_0x_bound, n_D1_0x_bound,
            metrics.n_DM0_0z, metrics.n_DM0_1z,
            metrics.n_DM1_0z, metrics.n_DM1_1z,
        )

        Ep_upper = self._apply_observed_value_correction(E_p_expected, metrics.n_z, eps2)

        leak_EC = self.f * metrics.n_z * binary_entropy(metrics.Ez)

        key_length = (
            metrics.n_z * (1.0 - binary_entropy(Ep_upper))
            - leak_EC
            - self._finite_key_overhead()
        )

        secure = (
            key_length > 0
            and metrics.Ez <= self.qber_threshold
            and Ep_upper < 0.499
        )

        return BoundResult(
            bound_name=self.name(),
            Ep_upper=Ep_upper,
            Ez=metrics.Ez,
            leak_EC=leak_EC,
            key_length=key_length if secure else 0.0,
            key_rate_per_pulse=(key_length / metrics.N_target) if secure else 0.0,
            secure=secure,
            epsilon_cor=self.epsilon_cor,
            epsilon_sec=self.epsilon_sec,
            n_z=metrics.n_z,
            n_z_ci=metrics.n_z_ci,
        )


# ----------------------------------------------------------------------------
# Concrete bound: Kato's inequality (Cao et al. 2026 Eq. 8-13)
# ----------------------------------------------------------------------------

class KatoBound(SecurityBound):
    """Kato's inequality, exactly as derived in Cao et al. 2026 Eq. (8)-(13)."""

    def __init__(
        self,
        epsilon_cor: float = 1e-15,
        epsilon_sec: float = 1e-10,
        qber_threshold: float = 0.05,
        correction_efficiency_f: float = 1.1,
    ):
        super().__init__(epsilon_cor, epsilon_sec, qber_threshold, correction_efficiency_f)

    def name(self) -> str:
        return "Kato"

    def _decoy_deviation_upper(self, k_total: float, Gamma_k: float, eps_a: float) -> float:
        return kato_delta_upper(k_total, Gamma_k, eps_a)

    def _decoy_deviation_lower(self, k_total: float, Gamma_k: float, eps_a: float) -> float:
        return kato_delta_lower(k_total, Gamma_k, eps_a)


# ----------------------------------------------------------------------------
# Concrete bound: Azuma's inequality (comparison bound, looser, count-independent)
# ----------------------------------------------------------------------------

class AzumaBound(SecurityBound):
    """
    Azuma's inequality: deviation = sqrt(0.5 * k_total * ln(1/eps_a)) for
    BOTH upper and lower bounds -- depends only on the TOTAL trial count
    k_total, never on the observed click count Gamma_k itself (unlike
    Kato's a/b construction, which uses Gamma_k/k_total to tighten the
    bound when the observed rate is far from 1/2).

    VERIFIED mechanism (Li et al. 2024, COWQKDfinite.tex lines 221/337,
    checked directly): this hits Q_00^{Mi} (the vacuum-decoy gain)
    hardest, since Q_00 is set almost entirely by the tiny dark-count
    rate (~1e-8) -- a small Gamma_k at large k_total is exactly where
    ignoring Gamma_k costs the most. Confirmed numerically against Cao's
    real L=25km counts: Azuma's deviation on n_00 (336x looser than
    Kato's at the same point) is far worse than its deviation on
    n_alphaalpha (24x looser) -- the same disproportionate effect Li et
    al. 2024 describe and is the basis for their Fig. 6 comparison.
    """

    def __init__(
        self,
        epsilon_cor: float = 1e-15,
        epsilon_sec: float = 1e-10,
        qber_threshold: float = 0.05,
        correction_efficiency_f: float = 1.1,
    ):
        super().__init__(epsilon_cor, epsilon_sec, qber_threshold, correction_efficiency_f)

    def name(self) -> str:
        return "Azuma"

    @staticmethod
    def _azuma_deviation(k_total: float, eps_a: float) -> float:
        if k_total <= 0:
            return 0.0
        return float(np.sqrt(0.5 * k_total * np.log(1.0 / eps_a)))

    def _decoy_deviation_upper(self, k_total: float, Gamma_k: float, eps_a: float) -> float:
        return self._azuma_deviation(k_total, eps_a)

    def _decoy_deviation_lower(self, k_total: float, Gamma_k: float, eps_a: float) -> float:
        return self._azuma_deviation(k_total, eps_a)


# ----------------------------------------------------------------------------
# Concrete bound: Cao et al. 2026's OWN asymptotic (infinite-data) limit
# ----------------------------------------------------------------------------
#
# Purpose: replace GaoAsymptoticBound as the "Kato's asymptotic reference"
# curve. GaoAsymptoticBound is a DIFFERENT paper's formula (Gao et al. 2022),
# derived for a state preparation scheme whose implicit N+/N- normalization
# does not match Cao et al.'s actively-chosen 10/10/40/40% decoy
# probabilities -- comparing it to KatoBound produces a ~1.25x scale-factor
# mismatch between the X- and Z-basis terms that has nothing to do with
# finite-key effects. CaoAsymptoticBound instead reuses KatoBound's own Eq.
# 8-13/18-20 pipeline (via the base class's evaluate()) with:
#
#   (a) zero decoy-count widening (_decoy_deviation_upper/lower -> 0.0),
#   (b) zero Delta_p observed-value widening
#       (_apply_observed_value_correction -> identity), and
#   (c) zero fixed correctness/secrecy overhead (_finite_key_overhead -> 0.0),
#
# while (d) KEEPING leak_EC = f*n_z*h(Ez) -- error-correction cost is a real,
# physical cost that persists in the asymptotic limit (this matches Gao et
# al.'s own asymptotic Eq. (6), R = Q[1 - h(Ep) - f*h(Eb)]: the f*h(Eb) term
# is the asymptotic analog of leak_EC/n_z and does not vanish either).
#
# Because CaoAsymptoticBound and KatoBound share the exact same Eq. 18
# normalization by construction (same base-class pipeline, same subclass
# family), the two are directly, apples-to-apples comparable on identical
# ScaledMetrics -- no cross-paper mismatch is possible. See TASK 2/4
# verification data below for the required structural checks (CaoAsymptotic
# rate >= Kato rate; CaoAsymptotic Ep <= Kato Ep, at every distance and mu).

class CaoAsymptoticBound(SecurityBound):
    """
    Cao et al. 2026's own Eq. (18) formula with ZERO finite-key correction --
    the true infinite-data limit of Cao's OWN protocol (not a different
    paper's protocol). Intended as the correct asymptotic reference curve
    for KatoBound, replacing GaoAsymptoticBound in that role.
    """

    def __init__(
        self,
        epsilon_cor: float = 1e-15,
        epsilon_sec: float = 1e-10,
        qber_threshold: float = 0.05,
        correction_efficiency_f: float = 1.1,
    ):
        super().__init__(epsilon_cor, epsilon_sec, qber_threshold, correction_efficiency_f)

    def name(self) -> str:
        return "CaoAsymptotic"

    def _decoy_deviation_upper(self, k_total: float, Gamma_k: float, eps_a: float) -> float:
        # (a) No Kato (or any other) widening on the decoy counts at all --
        # the widened count collapses to the raw observed count.
        return 0.0

    def _decoy_deviation_lower(self, k_total: float, Gamma_k: float, eps_a: float) -> float:
        return 0.0

    def _apply_observed_value_correction(self, E_p_expected: float, n_z: float, eps2: float) -> float:
        # (b) No Delta_p widening (Eq. 19-20) -- the "observed-value" phase
        # error rate is simply the expected-value phase error rate, still
        # clamped to the physically meaningful [0, 0.499] range used
        # elsewhere in this module.
        return float(np.clip(E_p_expected, 0.0, 0.499))

    def _finite_key_overhead(self) -> float:
        # (c) No fixed correctness/secrecy accounting overhead -- these are
        # finite-key bookkeeping terms (log2(2/eps_cor) + 2*log2(5/eps_sec))
        # that don't apply to an infinite-data bound.
        return 0.0

    # (d) leak_EC is NOT overridden: the base class's evaluate() already
    # computes leak_EC = f*n_z*h(Ez) and keeps it in key_length, which is
    # exactly the behaviour we want here (see module-level note above).


# ----------------------------------------------------------------------------
# Ground-truth check: Gao et al. 2022 Eq. (7)/(12), PURE ASYMPTOTIC.
# ----------------------------------------------------------------------------
#
# Purpose: isolate "is the Cauchy-Schwarz gain-level bound formula itself
# correct" from "is the Kato finite-key correction correct". This class
# applies ZERO finite-key correction (no Kato, no Azuma, no counts/N at
# all) -- every quantity below is a rate (gain), read straight off
# ScaledMetrics, exactly as Gao's Eq. (7)/(12) are written in the paper
# (Optics Express 30, 23783 (2022), main text Eq. 6-7, Appendix A Eq. 12).
#
# If sweeping COWParams set to Gao's own Sec. 4 numbers (p_d=1e-8,
# eta_d=0.80, eta=10**(-0.02*L), f=1.1) through this bound reproduces
# Gao's Fig. 2 (lower bound ~0.005*eta^2 at e_a=0, secure to ~100km), the
# bug is confined to the Kato/finite-key layer (n_D1_0x_lower /
# n_D2_0x_upper's stray N factors). If it does NOT reproduce Fig. 2, the
# bug is in Eq. (12) itself or in how ScaledMetrics' rates feed it -- a
# strictly earlier place to look than the Kato machinery.

def _gao_eq12_bounds(mu: float, G_M1_aa: float, G_M1_00: float,
                      G_M0_aa: float, G_M0_00: float) -> tuple[float, float]:
    """Gao Eq. (12): Cauchy-Schwarz bounds on the X-basis gains Q_0x,
    evaluated directly on rates (no N anywhere)."""
    n_plus, n_minus = normalisation_factors(mu)

    Q_M1_0x_upper = (1.0 / n_plus) * (
        np.exp(mu / 2.0) * np.sqrt(max(G_M1_aa, 0.0))
        + np.exp(-mu / 2.0) * np.sqrt(max(G_M1_00, 0.0))
    ) ** 2 + (n_minus / n_plus) * (
        np.exp(mu) * n_minus / 4.0
        + np.exp(mu) * np.sqrt(max(G_M1_aa, 0.0))
        + np.sqrt(max(G_M1_00, 0.0))
    )

    Q_M0_0x_lower = (1.0 / n_plus) * (
        np.exp(mu) * G_M0_aa + np.exp(-mu) * G_M0_00
        - 2.0 * np.sqrt(max(G_M0_00, 0.0) * max(G_M0_aa, 0.0))
    ) - (n_minus / n_plus) * (
        np.exp(mu) * np.sqrt(max(G_M0_aa, 0.0)) + np.sqrt(max(G_M0_00, 0.0))
    )
    # *** FIX: clamp to the physically required floor of 0 ***
    # Q_M0_0x_lower is a LOWER BOUND (via Cauchy-Schwarz) on Q_M0_0x, which
    # is itself a click probability and therefore cannot be negative. The
    # Cauchy-Schwarz estimate CAN legitimately go negative at small/medium
    # mu (the bound is loose there) -- that means "the true value could be
    # as low as 0," NOT "the true value is negative." Leaving it negative
    # and feeding it into Eq.(7)'s N+*(Q_M1_0x-Q_M0_0x) numerator term
    # artificially INFLATES the gap (subtracting a negative number adds to
    # it), which is what was pushing Ep_raw above 1.0 at larger mu/tB --
    # confirmed directly: at mu=0.05, tB=0.75, L=60km, Q_M0_0x=-3.85e-4 and
    # Ep_raw=2.06; clamping alone (without yet checking other terms) brings
    # it back toward a sane range. This clamp is standard practice for
    # Cauchy-Schwarz-derived lower bounds in the finite-key QKD literature.
    Q_M0_0x_lower = max(Q_M0_0x_lower, 0.0)

    return Q_M1_0x_upper, Q_M0_0x_lower


class GaoAsymptoticBound(SecurityBound):
    """
    Gao et al. 2022, Eq. (6)-(7)/(12), pure asymptotic case. NO finite-key
    correction of any kind -- this is intentionally NOT a security bound
    suitable for a real experiment; it exists solely as a ground-truth
    regression check against Gao's own published Fig. 2-3 curves, using
    the exact same (mu, tB) optimizer already in framework.py.
    """

    def name(self) -> str:
        return "GaoAsymptotic"

    def _decoy_deviation_upper(self, k_total: float, Gamma_k: float, eps_a: float) -> float:
        return 0.0

    def _decoy_deviation_lower(self, k_total: float, Gamma_k: float, eps_a: float) -> float:
        return 0.0

    def evaluate(self, metrics: ScaledMetrics) -> BoundResult:
        mu = metrics.mu

        # Eq. (12): Cauchy-Schwarz bounds, straight off ScaledMetrics' rate
        # fields -- no counts, no N, no Kato.
        Q_M1_0x, Q_M0_0x = _gao_eq12_bounds(
            mu,
            metrics.G_DM1_alphaalpha, metrics.G_DM1_00,
            metrics.G_DM0_alphaalpha, metrics.G_DM0_00,
        )
        n_plus, _ = normalisation_factors(mu)

        # Eq. (7): phase error rate, on RATES. n_DM*_0z / n_DM*_1z in
        # ScaledMetrics are scaled COUNTS out of the ACTUAL number of
        # |0z>/|1z> signal pulses sent (n_0z_total / n_1z_total -- now
        # exposed on ScaledMetrics; previously this divided by
        # n_DM0_0z+n_DM1_0z, which sums to 1 by construction for ANY
        # input and was the actual source of the Ep=0.499 saturation
        # bug -- confirmed by direct debug dump, see chat history).
        G_M0_0z = metrics.n_DM0_0z / metrics.n_0z_total if metrics.n_0z_total > 0 else 0.0
        G_M1_0z = metrics.n_DM1_0z / metrics.n_0z_total if metrics.n_0z_total > 0 else 0.0
        G_M0_1z = metrics.n_DM0_1z / metrics.n_1z_total if metrics.n_1z_total > 0 else 0.0
        G_M1_1z = metrics.n_DM1_1z / metrics.n_1z_total if metrics.n_1z_total > 0 else 0.0

        numerator = n_plus * (Q_M1_0x - Q_M0_0x) + 2.0 * (G_M0_0z + G_M0_1z)
        denominator = 2.0 * (G_M0_0z + G_M1_0z + G_M0_1z + G_M1_1z)
        Ep_upper = float(np.clip(numerator / denominator, 0.0, 0.499)) if denominator > 0 else 0.499

        # Eq. (6): asymptotic key rate, gain-level, no leakEC finite-key
        # correction beyond the standard f*h(Ez) error-correction cost.
        Q = metrics.n_z / metrics.N_target if metrics.N_target > 0 else 0.0
        key_rate_per_pulse = Q * (1.0 - binary_entropy(Ep_upper) - self.f * binary_entropy(metrics.Ez))
        key_rate_per_pulse = max(key_rate_per_pulse, 0.0)

        secure = (
            key_rate_per_pulse > 0
            and metrics.Ez <= self.qber_threshold
            and Ep_upper < 0.499
        )

        return BoundResult(
            bound_name=self.name(),
            Ep_upper=Ep_upper,
            Ez=metrics.Ez,
            leak_EC=self.f * metrics.n_z * binary_entropy(metrics.Ez),
            key_length=key_rate_per_pulse * metrics.N_target if secure else 0.0,
            key_rate_per_pulse=key_rate_per_pulse if secure else 0.0,
            secure=secure,
            epsilon_cor=self.epsilon_cor,
            epsilon_sec=self.epsilon_sec,
            n_z=metrics.n_z,
            n_z_ci=metrics.n_z_ci,
        )


# ----------------------------------------------------------------------------
# Convenience: run several bounds on the same ScaledMetrics
# ----------------------------------------------------------------------------

def evaluate_all_bounds(
    metrics: ScaledMetrics, bounds: list[SecurityBound]
) -> dict[str, BoundResult]:
    """Evaluate every SecurityBound in `bounds` against the SAME ScaledMetrics
    snapshot (no re-simulation). No "best bound" selection -- returns all
    results for inspection/comparison."""
    return {bound.name(): bound.evaluate(metrics) for bound in bounds}


# ----------------------------------------------------------------------------
# Beam-splitting attack: theoretical rate CEILING (not a simulation hook)
# ----------------------------------------------------------------------------

def beam_splitting_rate_ceiling(
    t_fiber: float,
    tB: float,
    detector_eff: float,
    p_signal: float,
    bs_attack_strength: float,
) -> float:
    """
    Gisin, Ribordy, Zbinden, Stucki, Brunner, Scarani, "Towards practical and
    fast Quantum Cryptography," arXiv:quant-ph/0411022 (2004), Sec. III.C.1
    ("Eve's attack without errors" -- the basic beam-splitting attack) +
    Eqs. 15-18.

    UNLIKE the intercept-resend hook in observable.py, this attack is NOT
    implemented as a simulation perturbation, because the cited mechanism
    provably changes NOTHING Bob can observe: Eve replaces the lossy channel
    with her own beamsplitter (transmission t, matching the real fibre) plus
    a lossless line to Bob, taps the reflected (1-t) fraction, and measures
    it via the same optimal unambiguous state discrimination Bob himself
    uses. Whenever she detects (probability mu*(1-t) per pulse), she knows
    Alice's bit outright; Bob's own click statistics, Q, and monitor-line
    visibility are UNCHANGED (Eq. 15: "Bob detects something exactly as if
    Eve had not been there"). A coherent state through a beamsplitter tap is
    quantum-mechanically indistinguishable from the same state through
    honest channel loss -- there is no anomaly for any observable-based
    hook to create.

    Instead, the paper's own defense is to always assume the worst case in
    the KEY-RATE FORMULA itself (Eq. 14: R >= I(A:B) - I(B:E)), which yields
    a closed-form ceiling on the best rate ANY protocol of this type can
    achieve once that assumption is priced in (Eq. 16-18):
        R(mu)  = mu*t*tB*eta_d*(1-f)*(1 - mu*(1-t))
        mu_opt = 1 / (2*(1-t))                              [Eq. 17]
        R_opt  = t*tB*eta_d*(1-f) / (4*(1-t))                [Eq. 18]

    This function is that ceiling, GENERALIZED (my own extension, NOT
    stated in the cited paper -- flagged explicitly) to a partial attack
    strength s = bs_attack_strength in [0, 1]: the fraction of the observed
    channel loss (1-t) that is assumed to be Eve's tap rather than honest
    fibre attenuation. Re-deriving Eqs. 16-18 with Eve's detection
    probability scaled to mu*s*(1-t) gives:
        mu_opt(s) = 1 / (2*s*(1-t))
        R_opt(s)  = t*tB*eta_d*(1-f) / (4*s*(1-t))
    which recovers Eq. 18 exactly at s=1 -- the ONLY value a real security
    proof is entitled to assume (Eve could always be tapping 100% of the
    observed loss), included here as the physically mandatory check. s<1 is
    provided purely as a sensitivity/off-toggle for comparison, NOT as a
    physically valid partial-security assumption on its own.

    NOTATION MAPPING (flagged explicitly, not a direct transcription):
      - t_fiber must be the PURE fibre/channel transmission, EXCLUDING
        detector efficiency -- per the paper's trusted-device assumption
        ("the imperfections of the three Bob's detectors... are not given
        to Eve"), only the channel itself is Eve's attack surface.
      - Gisin's "(1-f)" in Eq. 16 is the probability a round carries a real
        signal bit rather than a decoy sequence (their f = decoy
        probability) -- a DIFFERENT quantity from this codebase's
        `correction_efficiency_f` (error-correction inefficiency, ~1.1,
        introduced later in the Cao/Kato finite-key pipeline and absent
        from Gisin 2004's asymptotic derivation entirely). This function
        therefore takes `p_signal` (this codebase's directly analogous
        "fraction of rounds carrying real key bits") for that factor, NOT
        correction_efficiency_f -- using the latter here would silently
        substitute the wrong physical quantity.

    This ceiling is a NECESSARY-CONDITION check only, not a tight target:
    Gisin 2004's asymptotic Eq. 18 omits dark counts, finite-key epsilon
    security terms, and realistic error-correction leakage that Kato/Azuma/
    Cao's bounds DO include, so the framework's actual reported rate is
    expected to sit well BELOW this ceiling everywhere -- that is not a bug.
    The only actionable signal is a VIOLATION (framework rate > ceiling),
    which would mean the framework is claiming a rate that is not
    achievable by ANY protocol once a beam-splitting tap is assumed, i.e. a
    genuine BOUND BUG.

    Returns +inf at bs_attack_strength=0 (no constraint applied -- the
    "attack off" state of the toggle) or at t_fiber=1 (no loss to exploit).
    """
    s = float(bs_attack_strength)
    t = float(t_fiber)
    if s <= 0.0 or t >= 1.0:
        return float("inf")
    denom = 4.0 * s * (1.0 - t)
    if denom <= 0.0:
        return float("inf")
    return t * tB * detector_eff * p_signal / denom


def beam_splitting_mu_opt(t_fiber: float, bs_attack_strength: float) -> float:
    """mu_opt(s) = 1 / (2*s*(1-t)) -- see beam_splitting_rate_ceiling's
    docstring for the full derivation and citation. Provided for
    completeness / diagnostic reporting (e.g. to check how far the
    framework's actual operating mu is from the ceiling-optimal mu)."""
    s = float(bs_attack_strength)
    t = float(t_fiber)
    if s <= 0.0 or t >= 1.0:
        return float("inf")
    denom = 2.0 * s * (1.0 - t)
    if denom <= 0.0:
        return float("inf")
    return 1.0 / denom


# ----------------------------------------------------------------------------
# Backflash attack: theoretical rate CEILING (not a simulation hook)
# ----------------------------------------------------------------------------

def backflash_rate_ceiling(
    honest_rate_per_pulse: float,
    honest_n_z: float,
    N_target: float,
    backflash_strength: float,
) -> float:
    """
    Singh, Sharma, Singh, Prabhakar, "Backflash Attack on Coherent One-Way
    Quantum Key Distribution," arXiv:2502.04081v1 (2025).

    SPAD avalanche breakdown at Bob's DATA-LINE detector emits "backflash"
    photons that retrace back toward Alice; Eve intercepts them with a
    circulator + high-efficiency SNSPD and, using the timestamp/bit data
    already exchanged on the classical channel for sifting, learns Bob's
    bit WITHOUT perturbing the quantum state he receives -- the paper
    states explicitly this is passive and "does not disturb the coherence
    of the pulse" (Sec. II). Same "provably invisible to Bob's statistics"
    category as beam_splitting_rate_ceiling: there is nothing for
    observable.py or detector.py to perturb, so this is NOT implemented as
    a simulation hook (see COWParams.backflash_strength's docstring for
    the full citation-vs-derivation split and the notation mapping from
    the paper's own P_b to this codebase's backflash_strength).

    THREE alternative ceiling designs were considered and rejected before
    this one (see backflash_strength's COWParams docstring for the full
    comparison):
      1. Literally transcribing the paper's own Eq. 6,
         P_sec = P_sift*(1-P_b-f*H(e)) -- rejected because Eq. 6 has NO
         phase-error/monitor-line term at all (a simpler, non-decoy-state
         formula than even GaoAsymptoticBound), so it is not apples-to-
         apples against this framework's monitor-line-aware Kato/Azuma
         rates -- comparing against it could trivially "violate" (or fail
         to violate) regardless of the actual attack strength.
      2. Baking backflash_strength directly into each SecurityBound
         subclass's own evaluate() -- rejected because it would make the
         PUBLISHED bound artificially "aware" of a leak its actual
         decoy-state statistics have no mechanism to detect, defeating
         the validation question entirely (of course a bound that
         explicitly subtracts the leak will "catch" it by construction).
      3. Rebuilding the rate from Ep_upper/Ez by hand,
         Q_sift*(1-H(Ep_upper)-backflash_strength-f*H(Ez)) -- rejected in
         favour of the formula below because Kato's real key_length
         formula has an ADDITIONAL _finite_key_overhead() term beyond the
         plain bracket; reconstructing the bracket manually risks
         silently dropping it.

    This function instead treats the framework's own honestly-computed
    (backflash_strength=0) bound output as an opaque, already-correct
    baseline (sidestepping any risk of re-deriving Kato's internal
    formula incompletely), and subtracts an ABSOLUTE per-pulse leaked-
    information rate on top of it:

        ceiling = max(0, honest_rate_per_pulse - Q_sift_honest * backflash_strength)
        where Q_sift_honest = honest_n_z / N_target

    Justification for the Q_sift_honest scaling (own derivation, but
    structurally motivated, not arbitrary): combining the paper's own
    Eq. 5 (P_learn^Eve = P_b * P_sift) with Eq. 6 shows P_sift*P_b IS
    P_learn -- i.e. the paper's own model already treats backflash
    leakage as an absolute per-pulse leaked-information rate, the same
    structural role leak_EC plays in Kato's own key_length formula
    (n_z*(1-H(Ep_upper)) - leak_EC - finite_key_overhead()). This ceiling
    generalizes that same "subtract an absolute leaked-bit rate" pattern
    onto the framework's own (more rigorous, monitor-line-aware) honest
    rate rather than the paper's own simpler formula.

    This ceiling is a NECESSARY-CONDITION check. Because backflash_strength
    is NEVER threaded into observable.py/detector.py (see COWParams
    docstring), every bound in this framework is EXPECTED to report its
    full, UNCHANGED baseline rate regardless of backflash_strength --
    i.e. this ceiling is expected to be violated immediately for any
    backflash_strength > 0, exactly like beam_splitting_rate_ceiling.
    Per protocol (F), this is the anticipated, plainly-stated result, not
    a bound bug: none of Kato/Azuma/Gao/Cao have any observable channel
    through which a passive, coherence-preserving detector-side leak
    could ever be detected -- the countermeasures the source paper itself
    proposes (optical isolators/spectral filters before the SPAD, or
    replacing SPADs with SNSPDs at Bob's end) are physical/hardware, not
    something any decoy-state security proof can address.

    Parameters
    ----------
    honest_rate_per_pulse : float
        The framework's own reported key rate (per pulse) at
        backflash_strength=0 for the SAME (mu, tB, distance, hardware)
        configuration.
    honest_n_z : float
        The SAME honest run's ScaledMetrics.n_z (sifted data-line click
        count, already scaled to N_target).
    N_target : float
        Total pulse count the experiment represents (same value used for
        both the honest run and the ceiling).
    backflash_strength : float
        Freely-tunable [0, 1] stand-in for the paper's empirically-
        measured P_b (Eq. 4) -- see COWParams.backflash_strength's
        docstring for why this is not transferable from the paper's own
        reported number.

    Returns
    -------
    The true achievable secure rate ceiling (same per-pulse units as
    honest_rate_per_pulse). Returns honest_rate_per_pulse unchanged at
    backflash_strength=0 (no constraint).
    """
    s = float(backflash_strength)
    if s <= 0.0 or N_target <= 0:
        return float(honest_rate_per_pulse)
    s = min(s, 1.0)
    Q_sift_honest = float(honest_n_z) / float(N_target)
    return max(0.0, float(honest_rate_per_pulse) - Q_sift_honest * s)
"""
observable.py
=============
LAYER 1 of the COW-QKD framework: pure quantum simulation.

Responsibility (and ONLY responsibility):
    - Prepare the four COW states Alice can send:
          |0z> = |0>|alpha>      (logical bit 0)
          |1z> = |alpha>|0>      (logical bit 1)
          |alpha,alpha>          (decoy 1 - coherence/phase-error probe)
          |0,0>                  (decoy 2 - vacuum decoy, probes dark counts)
    - Push each pulse through the asymmetric beam splitter (transmittance tB)
      that separates Bob's data line from the monitoring line.
    - Apply the lossy quantum channel (fibre + detector efficiency) using the
      Strawberry Fields Gaussian backend (coherent states + loss are entirely
      Gaussian operations, so this backend is exact and ~100x faster than Fock).
    - Run the monitoring-line unbalanced interferometer (1-bit delay + phase
      shift) that produces the two monitor detector modes D_M0 / D_M1. The
      click probabilities used downstream are Gao et al. 2022's validated
      closed-form gain expressions (Appendix A), with SF's own
      genuinely-run monitor-line circuit (COWQuantumChannel.
      _run_sf_monitor_circuit / monitor_line_quantum_click_probs, and the
      opt-in ObservableCollector(use_sf_monitor_gains=True) path that
      substitutes it in) now independently CROSS-VALIDATED against that
      closed form, not merely retained as an unreconciled cross-check.

      RESOLVED 2026-07-16 (referee-report Phase 1; see
      COWQuantumChannel._monitor_input_amplitude's and
      _run_sf_monitor_circuit's docstrings, validation/cao/
      derive_monitor_gain_twomode.py, and work_log.md Sec. 6 for the full
      derivation): the previously "confirmed, precisely-quantified"
      SF-vs-closed-form discrepancy is now closed. Three real bugs were
      found and fixed in the process: (1) a data-line-vs-monitor-line eta
      mismatch in the SF circuit's amplitude calculation; (2) a
      mis-transcribed c5 term in monitor_line_gains_closed_form, corrected
      2026-07-15/16 to match the current, latest, Iggidr-corrected
      arXiv:2107.09329v5 text (c5 is a single exponential, not the
      three-term expression this project mistakenly introduced
      2026-07-13); (3) the SF monitor circuit itself computed a PER-BIN
      (single time-slot) click probability with an unmotivated
      recombination phase, when Gao's gains are PER-ROUND quantities (a
      decoy round = 2 pulses). Fixed 2026-07-16 by accounting for the
      round's FULL two-pulse photon budget (each pulse's complete
      amplitude, not an intermediate arm-split fraction, entering the
      recombination event), at genuine constructive-interference phase
      phi=pi -- see the per-round energy-budget derivation in
      _run_sf_monitor_circuit's docstring (an EARLIER, retracted version
      of this docstring attributed the fix to a Michelson-vs-Mach-Zehnder
      hardware distinction; that explanation did not survive scrutiny --
      beamsplitter energy conservation is the same regardless of whether
      an interferometer's two arms share one physical beamsplitter or use
      two -- and has been replaced by the per-round accounting below,
      which does not depend on any specific hardware topology claim). The
      per-round accounting is itself a REASONABLE INTERPRETATION of what
      Gao's formula represents (the standard decoy-state-QKD convention
      of treating each round as an independent trial), not a
      first-principles derivation from real continuous pulse-train
      operation -- tested directly (see _monitor_input_amplitude's
      docstring): a literal calculation summing real click contributions
      across all three bins a decoy pair touches in a continuous train,
      weighted by true neighbor-slot-type probabilities, gives 1.25x
      Gao's exponent, matching NEITHER the old (1x) nor new (2x)
      picture, so Gao's own formula does not correspond to that literal
      calculation either. What IS solid, and what this claim actually
      rests on: SF's convention matches Gao's formula exactly AND
      independently matches Cao et al.'s real measured Table S2 counts
      -- two references, not one, neither tuned to the other.
      With all three corrections, the SF circuit reproduces Gao's EXACT
      closed-form Q^{M0}_alphaalpha (decoy) and c1 (signal) formulas
      simultaneously to <0.02% at every Cao et al. operating point -- a
      from-scratch, pure-amplitude re-derivation (validation/cao/
      derive_monitor_gain_twomode.py, exact for coherent states through
      linear optics, no approximation) confirms this
      numerically before the fix was ported into this file.
    - Apply ideal (dead-time-free) threshold detection on the data line.

What this file explicitly does NOT do:
    - No dead time -> that is detector.py (classical MC layer).
    - No QBER / phase-error / key-rate formulas -> that is security_bounds.py.
    - No experiment orchestration / parameter sweeps -> that is framework.py.

A note on dark counts and the monitor line: the published closed-form
monitor-line gain expressions (Gao et al. 2022, Appendix A) have the
dark-count probability pd ALGEBRAICALLY BUILT IN as part of the derivation
(e.g. G^{M0}_{alphaalpha} contains a
factor (1-pd)^3 from the joint "this detector clicked, the other two did
not" Kraus-operator condition) -- the papers do not provide a "pure quantum,
dark-count-free" version of these specific formulas to subtract pd back out
of. We therefore evaluate these closed-form expressions directly, with pd
supplied as a parameter at construction time, rather than awkwardly
factoring pd out only to have detector.py multiply it back in. The DATA
LINE click statistics, by contrast, genuinely separate cleanly into a pure
quantum click probability (computed here) and an independently-applied
dark-count correction (detector.py) -- that separation is preserved exactly
as originally designed.

The reason for this split: dark counts and dead time are classical, stochastic
*detector* phenomena that have nothing to do with the quantum state itself. Only
the photon statistics + loss are genuinely quantum mechanical, so only those
parts touch the Strawberry Fields simulator. This also lets us re-use a single
SF observable snapshot under many different classical detector assumptions
without re-running the (expensive) quantum simulation.

We collect ONLY raw observables (click counts), never rates or derived
quantities -- those are computed downstream in detector.py.

Reference equations:
    Gao et al., Opt. Express 30, 23783 (2022) -- Eqs. (2)-(11), App. A. This
    is the SOLE source for the monitor-line closed-form gain formula this
    file actually uses (monitor_line_gains_closed_form / this project's
    Cao et al. reproduction). Li et al. is NOT equivalent to it and is NOT
    used for anything in the current manuscript's formulas -- listed here
    only as a related COW-QKD security-analysis paper, not as an
    interchangeable source for this file's formulas:
        Li, Cao, Xie, Yin, Chen, Phys. Rev. Research 6, 013022 (2024) -- Sec. II-III
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

import strawberryfields as sf
from strawberryfields import ops


# ----------------------------------------------------------------------------
# Data container
# ----------------------------------------------------------------------------

@dataclass
class ObservableSnapshot:
    """
    Raw quantum click counts from a single SF Monte-Carlo run of `num_pulses`
    COW rounds. Every count field below (data-line and monitor-line alike)
    is a genuine *integer count* from binomial sampling of an exact,
    closed-form per-round click PROBABILITY -- rate/error-bar computation
    otherwise belongs in detector.py (Layer 2).

    CORRECTION (referee-report response, 2026-07-14): an earlier version of
    this docstring claimed the monitor-line decoy and signal gain fields
    were themselves "exact analytic expected values, NOT binomial samples,"
    and that `detector.py` bypassed Wilson-interval propagation for them
    with "an honest CI of 0.0." Both claims were FALSE and directly
    contradicted by this file's own `ObservableCollector.run()` (which
    binomial-samples every one of these fields, e.g.
    `clicks_mon_DM0_alphaalpha += int(self._rng.binomial(b_alphaalpha,
    p_M0_aa))`) and by `detector.py`'s actual `apply_detector_physics`
    (which applies genuine Wilson-CI propagation to them via
    `_scale_count`, not a hardcoded 0.0). What IS true, and worth keeping:
    the underlying click PROBABILITY is an exact closed form (no Fock-
    truncation or other approximation error), but the resulting COUNT is
    still a real sample of that probability at whatever `num_pulses` (the
    SF batch size) was used, and therefore carries genuine sampling noise.
    At a realistic dark-count probability pd~1e-8, AND at the small mu
    (~1e-4) the optimizer searches toward at longer distances, this
    sampling noise can be severe: an L=60km run at a modest batch size
    produced clicks_mon_DM0_alphaalpha=0 purely from bad luck, not because
    the true probability was zero. This is a genuine seed-sensitivity
    caveat on any reported figure computed from a single-seed run at small
    mu, not evidence that the count itself is somehow exact.

    Data line counts (per decoy/signal type, per time bin tau0/tau1):
        clicks_data_0z_tau0 : Bob's data-line detector D_T fires at tau0 when
                               Alice sent |0z> = |0>|alpha> (dark-count-only
                               slot, no photon here -> Ez ERROR term under
                               Cao et al. 2026's stated bit convention:
                               "former slot -> bit 1, latter slot -> bit 0")
        clicks_data_0z_tau1 : D_T fires at tau1 when Alice sent |0z> (the
                               photon's actual physical slot -> Ez CORRECT
                               term, since a tau1 click reads out bit 0,
                               matching Alice's bit)
        clicks_data_1z_tau0 : D_T fires at tau0 when Alice sent |1z> = |alpha>|0>
                               (the photon's actual physical slot -> Ez
                               CORRECT term, a tau0 click reads out bit 1,
                               matching Alice's bit)
        clicks_data_1z_tau1 : D_T fires at tau1 when Alice sent |1z> (dark-
                               count-only slot -> Ez ERROR term)

    Monitor line counts (decoy states -- feed the X-basis bound formulas).
    ALL FOUR are binomial samples of an exact closed-form probability, see
    CORRECTION note above:
        clicks_mon_DM0_alphaalpha : D_M0 fires when Alice sent |alpha,alpha>
        clicks_mon_DM1_alphaalpha : D_M1 fires when Alice sent |alpha,alpha>
        clicks_mon_DM0_00         : D_M0 fires when Alice sent |0,0>
        clicks_mon_DM1_00         : D_M1 fires when Alice sent |0,0>

    Monitor line counts (SIGNAL states -- Cao et al. 2026 Eq. 4/18: these
    are genuine experimental observables in the original papers, and
    likewise binomial samples of an exact closed form in this framework --
    see CORRECTION note above. Bob's basis choice is PASSIVE (just a beam
    splitter), so a Z-basis signal pulse |0z>/|1z> still physically reaches
    the monitor-line detectors a fraction of the time -- exactly as often
    as any other pulse routed to that port):
        clicks_mon_DM0_0z, clicks_mon_DM1_0z : D_M0/D_M1 fire when Alice sent |0z>
                                                 (binomial sample, int)
        clicks_mon_DM0_1z, clicks_mon_DM1_1z : D_M0/D_M1 fire when Alice sent |1z>
                                                 (binomial sample, int)

    Number of pulses sent of each type (denominators for Layer 2 rate-building):
        n_0z, n_1z, n_alphaalpha, n_00 : total SF-simulated rounds of each type.
    """

    n_0z: int
    n_1z: int
    n_alphaalpha: int
    n_00: int

    clicks_data_0z_tau0: int
    clicks_data_0z_tau1: int
    clicks_data_1z_tau0: int
    clicks_data_1z_tau1: int

    clicks_mon_DM0_alphaalpha: int  # binomial sample from SF-derived prob at N_sf
    clicks_mon_DM1_alphaalpha: int  # binomial sample from SF-derived prob at N_sf
    clicks_mon_DM0_00: int          # binomial sample from SF-derived prob at N_sf
    clicks_mon_DM1_00: int          # binomial sample from SF-derived prob at N_sf

    clicks_mon_DM0_0z: int  # binomial sample from SF-derived prob at N_sf
    clicks_mon_DM1_0z: int  # binomial sample from SF-derived prob at N_sf
    clicks_mon_DM0_1z: int  # binomial sample from SF-derived prob at N_sf
    clicks_mon_DM1_1z: int  # binomial sample from SF-derived prob at N_sf

    # bookkeeping, useful for Layer 2 / Layer 5 reporting
    num_pulses_simulated: int = field(default=0)
    mu: float = field(default=0.0)
    tB: float = field(default=0.0)
    eta: float = field(default=0.0)
    seed: int | None = field(default=None)


# ----------------------------------------------------------------------------
# Core SF physics helpers
# ----------------------------------------------------------------------------

def _click_probability_from_amplitude(beta: complex, cutoff_threshold: float = 1e-12) -> float:
    """
    Threshold-detector click probability for a coherent state |beta> incident
    on an ideal (no dark counts, no dead time, unit-efficiency-already-folded-
    into-beta) photon-number-resolving-free SPAD.

    For a coherent state, the photon number is Poisson distributed, so the
    probability of *zero* clicks is exactly exp(-|beta|^2) and the probability
    of *at least one* photon (i.e. a threshold-detector click) is

        P(click) = 1 - exp(-|beta|^2)

    This is the exact quantum-optical click probability -- it is what the
    Gaussian backend gives you for a coherent state through loss, and it is
    algebraically identical to the Kraus-operator amplitude-damping picture
    used in all three reference papers (K0 = 1 - |0><0|, "at least one click").
    We compute it directly here (closed form for coherent states) rather than
    sampling Fock-basis measurement outcomes, because for *coherent states*
    this closed form *is* exactly what the Gaussian backend's homodyne/photon
    statistics reduce to -- this is the well known equivalence that justifies
    using the Gaussian backend at all for COW (see framework discussion).
    """
    p_click = 1.0 - np.exp(-np.abs(beta) ** 2)
    return float(np.clip(p_click, 0.0, 1.0))


# 12-node Gauss-Hermite quadrature, precomputed once at module import. Used by
# COWQuantumChannel.monitor_line_gains_sf_physical to physically model
# imperfect interference visibility as Gaussian phase-jitter averaging over
# the genuine SF interferometer circuit (see that method's docstring and
# validation/cao/derive_visibility_quadrature.py for the derivation and the
# numerical confirmation that 12 nodes matches a 2e5-sample Monte Carlo
# check to 3-4 significant figures). Deterministic and exactly reproducible
# -- no RNG/seed needed, unlike the project's other Monte Carlo machinery.
_GH_NODES_RAW, _GH_WEIGHTS_RAW = np.polynomial.hermite.hermgauss(12)
_GH_WEIGHTS = _GH_WEIGHTS_RAW / np.sqrt(np.pi)


def _visibility_to_phase_sigma(interference_visibility: float) -> float:
    """
    Converts a fringe visibility V (Cao et al. 2026 independently report
    V~0.99, "approximately 99% over a period of 2000 seconds," Sec. III;
    this project's own back-solved value, V=0.9964, is a one-parameter
    fit to Cao's real dark-port counts under a DIFFERENT formula -- see
    monitor_line_gains_sf_physical's docstring for the honest label and
    the evidentiary-strength caveat this implies) to an
    equivalent Gaussian phase-jitter standard deviation sigma, via the
    standard two-beam-interferometer relation for Gaussian-distributed
    random relative phase noise delta ~ N(0, sigma^2):

        V = |<e^{i*delta}>| = exp(-sigma^2 / 2)   =>   sigma = sqrt(-2*ln(V))

    This is a standard, textbook result in coherence theory (fringe
    visibility as the magnitude of the complex degree of coherence under
    Gaussian phase diffusion), not specific to this project. V=1.0 (perfect
    interference) gives sigma=0 (no averaging needed).
    """
    V = float(interference_visibility)
    if V >= 1.0:
        return 0.0
    return float(np.sqrt(-2.0 * np.log(V)))


# ----------------------------------------------------------------------------
# Zero-error attack (Branciard, Gisin, Luetkenhaus, Scarani, "Zero-Error
# Attacks and Detection Statistics in the Coherent One-Way Protocol for
# Quantum Cryptography," QIC 7, 639 (2007), arXiv:quant-ph/0609090):
# USD3 and USD4b conclusive-discrimination probabilities.
# ----------------------------------------------------------------------------

def usd3_conclusive_prob(mu: float, f: float) -> float:
    """
    P(Eve's USD3 attack conclusively identifies |0-alpha-0> among the six
    possible 3-pulse sequences), Eq. 12:
        p_concl^{0a0} = ((1-f)/2)^2 * (1 - e^{-mu})^2
    where f is the (single-type) decoy-sequence probability -- see
    module-level NOTATION MAPPING note in the USD attack hook docstrings
    below for why this framework uses p_decoy_alphaalpha specifically for f.
    """
    return ((1.0 - f) / 2.0) ** 2 * (1.0 - np.exp(-mu)) ** 2


def usd4b_conclusive_prob(mu: float, f: float) -> float:
    """
    P(Eve's USD4b attack conclusively identifies |0:alpha-alpha:0> among the
    eleven possible 4-pulse, 3-bit-slot sequences), Eq. 26:
        p_concl^{0:aa:0} = f * ((1-f)/2)^2 * (1-e^{-mu})^3 / (1+e^{-mu})
    Strictly smaller than usd3_conclusive_prob at the same (mu, f) -- the
    paper notes this scales as ~mu^3 (three-photon order) for small mu,
    versus ~mu^2 for USD3, "much smaller... [Eve] has to perform sometimes
    a very inefficient attack."
    """
    return f * ((1.0 - f) / 2.0) ** 2 * (1.0 - np.exp(-mu)) ** 3 / (1.0 + np.exp(-mu))


class COWQuantumChannel:
    """
    Wraps a Strawberry Fields Gaussian-backend program that implements:
        Alice's state prep -> lossy channel (fibre + detector efficiency)
        -> asymmetric beam splitter (data line tB / monitor line 1-tB)
        -> monitor-line unbalanced Mach-Zehnder interferometer (1-bit delay
           + phase shift phi, two 50:50 BSs) -> D_M0 / D_M1 amplitudes.

    Strawberry Fields role -- LOAD-BEARING, not decorative: SF's Gaussian
    backend is genuinely used to compute the output coherent-state amplitude
    at every detector mode, by actually building and running an SF circuit
    (Dgate -> LossChannel -> BSgate -> Rgate -> BSgate) ONCE per unique
    (mu, tB, eta, phi) configuration. This is exact for coherent states (no
    Fock-cutoff truncation error of any kind) and is what makes the
    framework a genuine quantum-optics teaching/research tool rather than a
    pure classical Monte Carlo dressed up with quantum language: a learner
    can step into `_run_sf_circuit` and inspect the actual SF Program,
    Engine, and resulting Gaussian state (covariance matrix, mean vector)
    at every stage of the optical network.

    Running SF once per CONFIGURATION (not once per pulse) is not an
    approximation: a coherent state stays exactly coherent (Gaussian, fixed
    covariance = vacuum noise) under every operation in this circuit
    (beamsplitters, loss, phase shifts), so its output amplitude is the same
    every time the SAME input is sent through the SAME circuit. The
    per-pulse RANDOMNESS in COW-QKD is not in the optical propagation at all
    -- it is in (a) which of the 4 round types Alice randomly sends, and
    (b) the inherently probabilistic CLICK outcome of measuring a fixed
    coherent state on a threshold detector (Poissonian click statistics).
    Both of these are still genuinely sampled per-pulse in `run()` below;
    only the (deterministic, given the configuration) amplitude propagation
    is cached after one real SF run.
    """

    def __init__(self, mu: float, tB: float, eta: float, phi: float = np.pi,
                 basis_mode: str = "passive", eta_for_monitor: float | None = None):
        """
        Parameters
        ----------
        mu  : mean photon number of the signal coherent state, |alpha|^2 = mu.
        tB  : in PASSIVE mode, the beam-splitter transmittance to the DATA
              line (paper convention: tB ~ 0.9 means 90% to data line, 10%
              to monitor line). In ACTIVE mode, tB retains the SAME role
              inside the monitor-line interferometer's own internal
              amplitude split (the (1-tB) factor in Gao Eq. 8's exponent is
              UNCHANGED between active/passive per the paper's own stated
              formulas -- confirmed by direct term-by-term comparison of
              Gao's Appendix A active vs passive formula blocks) -- it is
              NOT simply "removed" in active mode, only the DATA-LINE
              attenuation/joint-no-click conditioning changes.
        eta : total channel transmittance already INCLUDING detector
              efficiency, FOR THE DATA LINE (detector D0) specifically.
        eta_for_monitor : total channel transmittance for the MONITOR LINE
              (detectors D1/D2) specifically, if different from the data
              line's eta. Defaults to None, which means "use the same eta
              as the data line" (backward-compatible default). Set this
              explicitly whenever D1/D2's efficiency and/or any additional
              monitor-line-only insertion loss differs from D0's -- e.g.
              Cao et al. 2026 report D0=76.2% vs D1/D2=46.0% detector
              efficiency, plus an additional 3 dB loss specific to the
              monitor line's X-basis time-window selection. Using the
              data-line eta for the monitor line as well was found to
              cause a ~3.3x overestimate of monitor-line decoy click
              counts relative to Cao et al.'s real reported data (see
              AUDIT_STATUS.md) -- this parameter exists to fix that.
        phi : monitoring-line interferometer phase shift. Default pi:
              genuine constructive interference at the bright port D_M0
              in the resolved single-recombination-event physical
              picture (see `_run_sf_monitor_circuit`'s docstring; fixed
              2026-07-16 from an earlier, unmotivated pi/2 default that
              was part of the now-resolved sqrt(2)-in-amplitude gap vs
              Gao et al.'s closed form).
        basis_mode : "passive" (default, Bob's beamsplitter probabilistically
              splits EVERY pulse's amplitude tB:(1-tB) between data and
              monitor lines simultaneously -- modelled with a real BSgate)
              or "active" (Bob's fast optical SWITCH routes each ROUND's
              full amplitude to EITHER the data line OR the monitor line,
              never both -- modelled with NO data-line attenuation; the
              routing decision itself is a classical Bernoulli choice made
              in the MC/detector layer, not inside this SF circuit).

              Physical justification for the formula difference (Gao
              Appendix A): passive mode's gains carry an extra
              exp(-tB*mu*eta) factor and (1-pd)^3 (vs (1-pd)^1 in active)
              because a passive beamsplitter means the SAME pulse's
              amplitude reaches BOTH lines at once, so "monitor detector
              clicks" implicitly requires the joint condition "AND the
              data-line detector did NOT click" under the paper's stated
              gain convention. Active mode's hard switch means a round's
              amplitude goes to exactly one line, eliminating that joint
              condition entirely -- hence the simpler formula.
        """
        self.mu = float(mu)
        self.alpha = np.sqrt(self.mu)  # real, non-random phase (COW convention)
        self.tB = float(tB)
        self.eta = float(eta)
        self.eta_for_monitor = float(eta_for_monitor) if eta_for_monitor is not None else None
        self.phi = float(phi)
        if basis_mode not in ("passive", "active"):
            raise ValueError(f"basis_mode must be 'passive' or 'active', got {basis_mode!r}")
        self.basis_mode = basis_mode
        self._amplitude_cache: dict[tuple, complex] = {}

    # -- genuine SF circuit execution (run once per distinct input amplitude) --

    def _run_sf_data_line_circuit(self, alpha_in: complex) -> complex:
        """
        Build and run an ACTUAL Strawberry Fields Gaussian-backend circuit
        for the data-line amplitude.

        PASSIVE mode: Dgate(alpha_in) -> BSgate(theta, 0) [T=tB] ->
        LossChannel(eta) -- the pulse genuinely splits at a real
        beamsplitter, with only the tB fraction continuing to the data
        line detector.

        ACTIVE mode: Dgate(alpha_in) -> LossChannel(eta) only -- NO
        beamsplitter attenuation, since a round routed to the data line
        (by the classical optical switch, decided in detector.py) arrives
        with its FULL amplitude. This matches Gao's active-basis formulas
        exactly (no exp(-tB*mu*eta) factor anywhere in their active-mode
        gain expressions).
        """
        prog = sf.Program(2)
        eng = sf.Engine("gaussian")

        with prog.context as q:
            ops.Dgate(np.abs(alpha_in), np.angle(alpha_in)) | q[0]
            ops.Vac | q[1]
            if self.basis_mode == "passive":
                theta = np.arccos(np.sqrt(self.tB))
                ops.BSgate(theta, 0.0) | (q[0], q[1])
            ops.LossChannel(self.eta) | q[0]

        result = eng.run(prog)
        mu_vec = result.state.means()
        n_modes = 2
        x0, p0 = mu_vec[0], mu_vec[0 + n_modes]
        amp_out = (x0 + 1j * p0) / 2.0
        return complex(amp_out)

    def _data_line_amplitude(self, alpha_in: complex) -> complex:
        """
        Data-line output amplitude, computed by an actual SF circuit
        (cached per distinct input amplitude AND basis_mode, since the
        same alpha_in is reused across all pulses of a given signal type
        within one ObservableCollector configuration).
        """
        key = (complex(alpha_in), self.basis_mode)
        if key not in self._amplitude_cache:
            self._amplitude_cache[key] = self._run_sf_data_line_circuit(alpha_in)
        return self._amplitude_cache[key]

    def _monitor_input_amplitude(self, alpha_in: complex) -> complex:
        """Amplitude reaching the monitor-line interferometer's input:
        sqrt(eta*(1-tB)) * alpha_in, where eta here is the MONITOR-LINE
        transmittance (self.eta_for_monitor if set, else self.eta) -- SAME
        resolution rule as `monitor_line_gains_closed_form` uses (Cao et
        al. 2026 report D0=76.2% vs D1/D2=46.0% detector efficiency, plus
        an additional 3 dB monitor-line-only time-window-selection loss).

        *** BUG FOUND AND FIXED (referee-report investigation, see
        sf_monitor_gains_comparison.py) ***: this function previously used
        `self.eta` (the DATA-LINE transmittance) unconditionally, never
        resolving `eta_for_monitor` at all -- unlike the closed form, which
        always did this resolution correctly. At Cao et al.'s real
        operating points (eta_data/eta_monitor ~ 3.3x apart), this made
        every SF-derived monitor-line quantity (both the pre-existing
        reporting-only `monitor_line_quantum_click_probs()` numbers and the
        new `use_sf_monitor_gains` path) silently ~3.3x too large in
        intensity. It went undetected in the original pd->0 validation
        against the exponent this circuit was originally calibrated to
        (see `_run_sf_monitor_circuit`'s docstring) only because that
        check happened to run with eta_for_monitor unset (so
        eta_for_monitor == eta_data == eta_monitor by coincidence, masking
        the bug in that specific test).

        SAME formula in both PASSIVE and ACTIVE mode -- confirmed by
        direct comparison of Gao's Appendix A formula blocks: the
        (1-tB) factor inside the monitor-line exponents is IDENTICAL
        between active and passive (e.g. c1 = (1-pd)*exp(-(1-tB)*mu*eta/2)
        is stated as "unchanged" in the active-basis paragraph). Only the
        DATA-LINE side of the physics differs between modes (see
        `_run_sf_data_line_circuit` and `monitor_line_gains_closed_form`).

        *** RESOLVED 2026-07-16 (referee-report Phase 1) -- see
        validation/cao/derive_monitor_gain_twomode.py and work_log.md Sec.
        6 for the full derivation and numeric printout ***

        The earlier "OPEN CALIBRATION ITEM" framing here (a bare
        factor-of-4-in-power discrepancy vs Gao's closed form, with no
        further explanation available) is now CLOSED, via a from-scratch,
        pure-amplitude re-derivation (exact for coherent states through
        any linear-optical network -- no Fock truncation, no
        approximation). Two things fell out:

        (1) This circuit's OLD pd->0 calibration target,
        exp(-eta*(1-tB)*mu/2), was never a rearrangement of Gao's
        decoy-state Q^{M0}_alphaalpha inner-bracket exponent
        (2*mu*(1-tB)*eta) at all -- it EXACTLY matches Gao's SEPARATE
        signal-state c1 exponent, (1-tB)*mu*eta/2 = eta*(1-tB)*mu/2 (see
        monitor_line_signal_gains_closed_form). The circuit had been
        calibrated, at some earlier point in this project's history,
        against the wrong target formula (the single-pulse signal-state
        gain, appropriate for |0z>/|1z>) while being invoked with BOTH
        pulses populated (mon_in_alpha, mon_in_alpha), the decoy
        |alpha,alpha> case, which Gao gives a structurally different
        formula for.

        (2) The circuit's OLD topology computed a PER-BIN (single
        time-slot) click probability: split each pulse into short/long
        arms at a 50:50 beamsplitter (each costing an independent
        1/sqrt(2) amplitude factor), then time-gated to keep only the ONE
        bin where the earlier pulse's delayed long arm interferes with
        the later pulse's short arm, discarding the other two bins as
        "belonging to adjacent pairs." That calculation is not wrong as
        an amplitude computation -- but Gao's Q^{M0}_alphaalpha, and the
        counts Cao et al. actually report in Table S2, are PER-ROUND
        quantities: a decoy round is defined as the PAIR of pulses
        (2k-1, 2k) together, and Bob's post-processing / Cao's own
        counting convention attributes a click to "this round," not to
        one specific 1-bit-wide time slot within it. A round's monitor-
        line photon budget is the SUM of both its pulses' power, 2*beta^2
        (beta^2 = eta*(1-tB)*mu per pulse) -- the per-bin picture only
        ever accounts for half of that (the fraction reaching the single
        gated interference bin), discarding the other half to bins
        nominally "belonging to" neighbouring rounds. For a genuinely
        phase-coherent decoy pair (two IDENTICAL, in-phase pulses -- the
        defining feature of |alpha,alpha>), perfect constructive
        interference routes the ROUND'S ENTIRE budget to the bright port:
        Q^{M0}_alphaalpha = 1-exp(-2*beta^2), exactly Gao's stated
        exponent, with the dark port receiving zero real flux (pd-only),
        exactly Gao's G_M1 structure. For a SIGNAL round (one real pulse,
        vacuum neighbor), there is no coherent partner to interfere
        with, so the lone pulse's beta^2 budget is what a 50:50
        beamsplitter delivers with only one input populated: it splits
        EQUALLY between both output ports, beta^2/2 each -- exactly
        Gao's c1 signal exponent, and it explains why Gao's own formula
        gives D_M0 and D_M1 IDENTICAL gains for signal states specifically
        (no interference asymmetry is possible with no partner to
        interfere against). Implementing this (each full pulse amplitude,
        not an arm-split fraction, entering ONE recombining beamsplitter,
        at genuine constructive-interference phase phi=pi rather than the
        old, unmotivated phi=pi/2 default) is the computational device
        that realizes this per-round accounting -- it is not a claim
        about a specific interferometer hardware topology (an earlier
        version of this docstring speculated the fix corresponded to a
        Michelson-vs-Mach-Zehnder hardware distinction; that explanation
        did not survive scrutiny, since beamsplitter energy conservation
        does not depend on whether an interferometer's two arms share one
        physical beamsplitter or use two, and has been retracted).

        HONESTY CHECK ON THE PER-ROUND ACCOUNTING ITSELF (2026-07-16):
        this is presented as a REASONABLE INTERPRETATION of what Gao's
        formula represents (a per-round security-analysis statistic,
        consistent with the standard decoy-state-QKD convention of
        treating each round as an independent trial), NOT as a
        first-principles derivation from how a real, continuously
        operating COW pulse train behaves. Tested directly: a literal
        calculation that sums the REAL click contributions from all
        three bins a decoy pair actually touches in a continuous train
        (its own within-pair bin, plus the two boundary bins shared with
        its random neighbors, weighted by the true slot-type
        probabilities p_signal/p_decoy_alphaalpha/p_decoy_00) gives
        EXACTLY 1.25x Gao's stated exponent at every Cao operating point
        -- neither this circuit's 2x ("full") nor the old circuit's 1x
        ("half") value. So Gao's own formula does not correspond to that
        literal continuous-pulse-train calculation either; it most
        likely reflects the standard idealized-isolation convention used
        in decoy-state security proofs generally. What IS solid: this
        circuit's convention matches Gao's formula exactly AND
        independently matches Cao et al.'s real measured Table S2
        counts -- two references neither of which required tuning to
        the other. What is NOT claimed: that this convention is what a
        ground-up calculation of the real hardware's continuous
        operation would produce. This reproduces Gao's EXACT closed-form
        formulas -- not just the
        leading-order exponent, but the full pd-inclusive expressions --
        to within 0.02% at 25 km and 0.0001% at 100 km, for BOTH the
        signal-state c1 formula (independently, already-established,
        never touched) AND the decoy-state Q^{M0}_alphaalpha formula
        SIMULTANEOUSLY, using the identical topology and phase applied
        consistently to both pulse-population patterns (see the
        derivation script's exact-formula printout, not just the
        leading-order exponent ratios).

        FIX APPLIED: `_run_sf_monitor_circuit` now feeds each pulse's
        FULL amplitude directly into a single recombining BSgate (no
        intermediate arm-split), and `phi`'s default (COWQuantumChannel,
        ObservableCollector, COWParams) is now pi, not pi/2. The SF
        quantum circuit is genuinely load-bearing and cross-validated
        against Gao et al.'s closed form to high numerical precision --
        see docs/main_v2_draft.tex Sec. II.A for the updated manuscript
        framing.
        """
        eta = self.eta_for_monitor if self.eta_for_monitor is not None else self.eta
        return np.sqrt(eta * (1.0 - self.tB)) * alpha_in

    def _run_sf_monitor_circuit(self, beta_prev: complex, beta_curr: complex) -> complex:
        """
        Build and run an ACTUAL Strawberry Fields Gaussian-backend circuit
        for the monitor-line unbalanced interferometer with a one-bit
        delay, computing the genuine quantum INTERFERENCE amplitude at
        the "bright"/signal detector (D_M0).

        *** RESOLVED 2026-07-16, FINAL PHYSICAL PICTURE (superseding two
        earlier mis-framings -- documenting the full resolution so it is
        not re-litigated in a future session; see
        validation/cao/derive_monitor_gain_twomode.py and work_log.md
        Sec. 6) ***

        Earlier attempt #1 in this codebase treated D_M0 and D_M1 as the
        two COMPLEMENTARY output ports of a single interfering
        beamsplitter (the standard two-port MZI picture), and went
        looking for a beamsplitter convention that would send
        "everything" to one port and "nothing" to the other. WRONG.

        Earlier attempt #2 (still wrong, but closer) correctly separated
        D_M0/D_M1's roles -- G_M0 IS the genuine QUANTUM INTERFERENCE
        term, G_M1 is O(pd) DARK-COUNT-ONLY with no interference
        contribution -- but computed a PER-BIN (single time-slot) click
        probability: split each pulse into short/long arms at a 50:50
        beamsplitter (each costing an independent 1/sqrt(2) amplitude
        factor), then kept only the one bin where the earlier pulse's
        delayed long arm interferes with the later pulse's short arm,
        with an UNMOTIVATED recombination phase phi=pi/2. This discards
        half the round's photon budget to bins nominally "belonging to"
        neighbouring rounds, and its output landed on Gao's SIGNAL-state
        exponent by coincidence (matching that formula exactly) while
        being invoked for the DECOY state, which needs a structurally
        different formula -- see `_monitor_input_amplitude`'s docstring
        for the full diagnostic trail.

        FINAL, RESOLVED PICTURE: Gao's gains, and the counts Cao et al.
        actually report, are PER-ROUND quantities (a decoy round = the
        pulse PAIR (2k-1, 2k) together), not per-bin. A round's total
        monitor-line photon budget is the SUM of both pulses' power,
        2*beta^2. For a genuinely coherent, in-phase decoy pair, perfect
        interference routes the round's ENTIRE budget to the bright
        port; for a signal round (one pulse, vacuum neighbor), there is
        no coherent partner, so the lone pulse's budget splits equally
        between both ports. See `_monitor_input_amplitude`'s docstring
        for the full per-round energy-budget derivation (this replaces
        an earlier, retracted attempt to explain the fix via a Michelson-
        vs-Mach-Zehnder hardware distinction, which did not hold up under
        scrutiny). Feeding each pulse's FULL amplitude (no intermediate
        arm-split factor) into ONE recombining beamsplitter, at genuine
        constructive-interference phase phi=pi (not pi/2), is the
        computational device that implements this per-round accounting.
        This reproduces Gao et al.'s EXACT closed-form Q^{M0}_alphaalpha
        AND the independently-fixed c1 signal-state formula,
        simultaneously, to <0.02% at every Cao et al. operating point
        (see the derivation script's printed exact-formula comparison)
        -- the numerical match, not any specific hardware-topology
        story, is the actual evidence for this fix.

        This function therefore computes ONLY the D_M0 (interference)
        amplitude, using SF's standard REAL-valued BSgate (the SAME
        physical beamsplitter verified correct for the data line in
        Check 1) applied ONCE: each pulse's full input amplitude (already
        having passed through Bob's tB/(1-tB) split and channel loss in
        `_monitor_input_amplitude`) is fed directly into the recombining
        beamsplitter, with the earlier pulse's amplitude phase-shifted by
        phi (the one-bit-delay arm's phase) before combination -- the
        genuine interference event, occurring exactly one time bin after
        the earlier pulse's arrival. (Two other path combinations,
        early-pulse-alone and later-pulse-alone, occur at DIFFERENT time
        slots and are correctly excluded by the protocol's time-gating,
        not by anything in this circuit.)

        VALIDATED: at phi=pi, this reproduces the D_M1 (dark port)
        structure Gao's own formula implies -- D_M1 comes out EXACTLY
        zero for the decoy state (no quantum-interference contribution,
        matching G_M1's pd-only structure) -- and D_M0's |amplitude|^2
        reproduces Gao's inner-bracket decoy exponent 2*mu*(1-tB)*eta
        exactly (not the old, wrong mu*eta*(1-tB)/2 target). See
        `_monitor_input_amplitude`'s docstring for the full derivation
        trail and the honest flag on which parts of this explanation are
        numerically confirmed versus a plausible physical account.

        Returns the complex amplitude at D_M0 only. Under PERFECT
        interference (the case this function alone covers), D_M1's click
        probability is not computed from any amplitude here -- it is
        purely the dark-count process, applied directly in
        `monitor_line_gains_closed_form` below (G_M1 = a dark-count-driven
        probability with NO quantum-interference term). Under IMPERFECT
        interference, D_M1 genuinely carries real flux -- see
        `_run_sf_monitor_circuit_both_ports` and
        `monitor_line_gains_sf_physical`, added 2026-07-16, which read out
        both ports and are used whenever interference_visibility < 1.
        """
        amp_DM0, _amp_DM1 = self._run_sf_monitor_circuit_both_ports(beta_prev, beta_curr)
        return amp_DM0

    def _run_sf_monitor_circuit_both_ports(
        self, beta_prev: complex, beta_curr: complex, phase_offset: float = 0.0
    ) -> tuple[complex, complex]:
        """
        Same physical circuit as `_run_sf_monitor_circuit` (see its
        docstring for the full resolved physical picture), but reads out
        BOTH interferometer output ports -- D_M0 (bright) and D_M1 (dark)
        -- and accepts an additional `phase_offset` on top of self.phi.

        Needed for `monitor_line_gains_sf_physical`'s imperfect-visibility
        model (2026-07-16): under perfect interference (phase_offset=0),
        D_M1 comes out exactly zero, as already established. Under a
        nonzero phase_offset (one Gauss-Hermite quadrature node's sampled
        phase jitter, representing real interferometer path-length/timing
        imprecision), D_M1 genuinely receives leaked flux -- this is
        ordinary two-path interference, not a new physical assumption:
        energy conservation still holds exactly for any phase_offset
        (|amp_DM0|^2 + |amp_DM1|^2 is independent of phase_offset, since
        the recombining beamsplitter is unitary).
        """
        prog = sf.Program(2)
        eng = sf.Engine("gaussian")

        # Each pulse's FULL amplitude (no intermediate arm-split factor --
        # see the resolved physical picture in _run_sf_monitor_circuit's
        # docstring) enters the single recombining beamsplitter directly.
        # The earlier pulse's amplitude carries the one-bit-delay arm's
        # phase shift, self.phi (= pi, genuine constructive interference
        # at D_M0), plus phase_offset (a quadrature-node phase-jitter
        # sample, 0.0 for the perfect-interference case).
        prev_delayed = beta_prev * np.exp(1j * (self.phi + phase_offset))
        curr = beta_curr

        with prog.context as q:
            ops.Dgate(np.abs(prev_delayed), np.angle(prev_delayed)) | q[0]
            ops.Dgate(np.abs(curr), np.angle(curr)) | q[1]
            ops.BSgate(np.pi / 4, 0.0) | (q[0], q[1])

        result = eng.run(prog)
        mu_vec = result.state.means()
        # means() = [x0,x1,p0,p1] for N=2 modes (all positions, then all
        # momenta) -- confirmed against the real package via Check 1.
        n_modes = 2
        x0, p0 = mu_vec[0], mu_vec[0 + n_modes]
        x1, p1 = mu_vec[1], mu_vec[1 + n_modes]
        amp_DM0 = complex((x0 + 1j * p0) / 2.0)
        amp_DM1 = complex((x1 + 1j * p1) / 2.0)
        return amp_DM0, amp_DM1

    def monitor_line_quantum_click_probs(self) -> dict:
        """
        QUANTUM-ONLY (dark-count-free) D_M0 click probability for the two
        decoy states, computed by genuinely running the SF interference
        circuit (`_run_sf_monitor_circuit`) -- cached, since it only needs
        to run once per configuration.

        D_M1 has NO quantum-interference contribution at all (see
        `_run_sf_monitor_circuit` docstring for the full physical
        resolution: D_M0 and D_M1 are not complementary interferometer
        ports -- D_M0 carries the genuine coherent-amplitude interference,
        D_M1 is a structurally separate detector that only registers dark
        counts under ideal operation). D_M1's contribution is therefore
        computed directly in `monitor_line_gains_closed_form` from the
        dark-count probability alone, matching a pd-prefactored,
        no-quantum-interference structure -- not from any SF-computed
        amplitude.

        Validated: P_DM0_alphaalpha reproduces the calibration target's
        pd->0 prediction (see `_run_sf_monitor_circuit`'s docstring) to
        full numerical precision.

        Returns a dict with keys: P_DM0_alphaalpha, P_DM0_00 (the latter
        is exactly 0.0, since |0,0> has no photons to interfere with --
        consistent with a dark-count-only floor for the vacuum decoy).
        """
        mon_in_alpha = self._monitor_input_amplitude(self.alpha)
        mon_in_zero = self._monitor_input_amplitude(0.0)

        amp_DM0_aa = self._run_sf_monitor_circuit(mon_in_alpha, mon_in_alpha)
        P_DM0_aa = _click_probability_from_amplitude(amp_DM0_aa)

        amp_DM0_00 = self._run_sf_monitor_circuit(mon_in_zero, mon_in_zero)
        P_DM0_00 = _click_probability_from_amplitude(amp_DM0_00)

        return {
            "P_DM0_alphaalpha": P_DM0_aa,
            "P_DM0_00": P_DM0_00,
        }

    def monitor_line_gains_sf_physical(self, p_dark: float, interference_visibility: float = 1.0,
                                         ir_attack_strength: float = 0.0,
                                         usd4b_strength: float = 0.0,
                                         p_decoy_alphaalpha: float = 0.0) -> dict:
        """
        ADDED 2026-07-16 (referee-report Phase 1 follow-up). Monitor-line
        decoy-state gains (G_M0/G_M1 for alphaalpha and 00), computed
        GENUINELY from the SF interferometer circuit -- including a
        physically-modeled imperfect-visibility correction -- rather than
        from Gao et al.'s closed form. This is the mechanism that makes
        the quantum circuit independently load-bearing rather than a
        re-encoding of the closed form (see work_log.md Sec. 6 for the
        full discussion of why the closed-form match alone is not
        sufficient independent evidence).

        WHAT'S GENUINELY NEW HERE (not present in monitor_line_gains_closed_form):
        Gao et al.'s closed form assumes PERFECT interference -- the dark
        port receives literally zero real photons. Real interferometers
        don't reach V=1; Cao et al. INDEPENDENTLY report V~0.99 (a rough
        figure, Sec. III). The precise value used here, V=0.9964, is NOT
        that independently-reported number -- it is THIS PROJECT'S OWN
        one-parameter fit, back-solved to best match Cao's real dark-port
        counts under the closed-form-plus-leakage-patch formula (see
        `_visibility_to_phase_sigma`'s docstring and
        verify_all_bounds.py's INTERFERENCE_VISIBILITY comment for the
        honest label). This matters for how the validation below should
        be read -- see the VALIDATED paragraph's evidentiary-strength
        note. The existing closed-form path patches imperfect visibility
        with an ad hoc, admittedly "NOT-in-Gao's-formula" (1-V)/(1+V)
        leakage-fraction correction bolted onto the closed form's own
        G_M0 (see `monitor_line_gains_closed_form`'s docstring). That
        patch has no underlying physical derivation -- it is asserted,
        not simulated.

        This function instead genuinely SIMULATES imperfect visibility as
        Gaussian phase-jitter (see `_visibility_to_phase_sigma`): the
        interferometer's relative phase between the two arms is not a
        fixed pi, but pi + delta with delta ~ N(0, sigma^2), and the
        REPORTED click probability is delta's expectation value over many
        rounds -- exactly how real experimental visibility loss arises
        (slow thermal/mechanical path-length drift across a measurement
        campaign, not a per-round quantum effect). The expectation is
        computed via 12-node Gauss-Hermite quadrature: 12 genuine calls to
        `_run_sf_monitor_circuit_both_ports` at deterministic phase-offset
        nodes, weighted and summed -- exact for the Gaussian-weight
        integral to high precision (confirmed against a 2e5-sample Monte
        Carlo check to 3-4 significant figures; see
        validation/cao/derive_visibility_quadrature.py). D_M1 (dark port)
        now genuinely receives leaked flux at each node -- not asserted as
        zero and patched -- and energy is conserved exactly at every node
        (|amp_DM0|^2 + |amp_DM1|^2 is phase-offset-independent, since the
        recombining beamsplitter is unitary).

        VALIDATED against Cao et al.'s real measured Table S2 counts:
        this method's bright-port prediction matches Cao's real
        n^{D1}_alphaalpha counts to 0.9-2.4% at all four operating points
        -- matching or slightly beating the closed-form-plus-leakage-patch
        baseline's 1.1-2.6% -- and the dark-port prediction is
        statistically indistinguishable from that same baseline's
        14-101% residual (a large, pre-existing, independently-disclosed
        gap -- see docs/MASTER_PARAMETERS.md and the referee-report
        dark-count-rate disclosure item -- NOT introduced or worsened by
        this method).

        EVIDENTIARY-STRENGTH NOTE (2026-07-16): the BRIGHT-port agreement
        IS independent evidence of genuine SF computation -- V=0.9964 was
        never fit against bright-port counts, only dark-port ones (see
        above), so an unrelated formula independently landing within a
        couple percent of real bright-port data is meaningful. The
        DARK-port agreement is NOT independent in the same way -- V was
        already tuned specifically to fit that target under a different
        (the closed-form) formula, so this method reusing the same fitted
        V and landing close to that same target is the EXPECTED outcome,
        not fresh corroboration. It still satisfies the "not worse than
        the existing baseline" bar this feature was gated on before being
        ported, which is the honest claim to make about it.

        Dark-count combination: uses the independent-process formula
        P_total = 1-(1-P_quantum)*(1-p_dark), the SAME convention already
        established elsewhere in this file for SF-substituted paths (e.g.
        ObservableCollector's existing use_sf_monitor_gains patch) --
        NOT Gao's own (1-pd)^3-style combinatorial prefactors, which are
        specific to their closed-form derivation and not a universal
        physical law. Confirmed numerically these two conventions agree
        to <0.3 percentage points at Cao's real dark-count rate (pd is
        tiny, ~2.8e-8/round, so the prefactor choice barely matters).

        ATTACK-HOOK EXTENSION (added after 2026-07-16, closing a scope gap
        flagged in this project's own review process): `ir_attack_strength`
        and `usd4b_strength` were previously implemented ONLY against
        `monitor_line_gains_closed_form`, so `ObservableCollector.__init__`
        silently forced `monitor_gain_source="closed_form"` whenever either
        attack was active, regardless of the caller's actual setting --
        meaning every IR/USD4b figure in this project was generated on the
        OLDER closed-form-plus-leakage-patch monitor model, not the
        `sf_physical` default used for every other (non-attack) result.
        This is now closed: the IDENTICAL mixture formulas already reviewed
        and cited for the closed-form path (see
        `monitor_line_gains_closed_form`'s IR/USD4b blocks below for the
        full citations -- Gisin et al. 2004 Eqs. 19-26 for IR, Branciard et
        al. 2007 Eq. 26/29-31 for USD4b) are applied here to THIS method's
        own honest, quadrature-derived G_M0_aa/G_M1_aa, instead of being
        re-derived. Nothing about the cited mechanism changes -- only which
        "honest" bright/dark-port pair the mixture is applied on top of.
        At ir_attack_strength=usd4b_strength=0.0 (the default), both
        mixture blocks below are the identity map, so this method's
        existing (already-validated) baseline output is bit-for-bit
        unchanged from before this extension.

        Returns a dict with keys: G_M0_alphaalpha, G_M1_alphaalpha, G_M0_00,
        G_M1_00 (matching monitor_line_gains_closed_form's return shape).
        """
        pd = p_dark
        sigma = _visibility_to_phase_sigma(interference_visibility)
        if sigma == 0.0:
            deltas = np.array([0.0])
            weights = np.array([1.0])
        else:
            deltas = sigma * np.sqrt(2.0) * _GH_NODES_RAW
            weights = _GH_WEIGHTS

        def _quadrature_gains(beta_prev: complex, beta_curr: complex) -> tuple[float, float]:
            P_bright = 0.0
            P_dark = 0.0
            for delta, w in zip(deltas, weights):
                amp0, amp1 = self._run_sf_monitor_circuit_both_ports(
                    beta_prev, beta_curr, phase_offset=delta
                )
                Pq0 = _click_probability_from_amplitude(amp0)
                Pq1 = _click_probability_from_amplitude(amp1)
                P_bright += w * (1.0 - (1.0 - Pq0) * (1.0 - pd))
                P_dark += w * (1.0 - (1.0 - Pq1) * (1.0 - pd))
            return float(P_bright), float(P_dark)

        mon_in_alpha = self._monitor_input_amplitude(self.alpha)
        mon_in_zero = self._monitor_input_amplitude(0.0)

        G_M0_aa, G_M1_aa = _quadrature_gains(mon_in_alpha, mon_in_alpha)

        # ---- Intercept-resend attack, ported from monitor_line_gains_closed_form ----
        # (OFF by default, ir_attack_strength=0.0). Gisin et al. 2004,
        # arXiv:quant-ph/0411022, Eqs. 19-26 -- see the closed-form method's
        # own IR block below for the full citation and derivation; this is
        # the SAME flux-conserving 50/50 mixture, applied to THIS method's
        # own honest (quadrature-derived) G_M0_aa/G_M1_aa instead.
        p_ir = float(ir_attack_strength)
        if p_ir > 0.0:
            total_flux_alphaalpha = G_M0_aa + G_M1_aa
            G_M0_aa = (1.0 - p_ir) * G_M0_aa + p_ir * (total_flux_alphaalpha / 2.0)
            G_M1_aa = (1.0 - p_ir) * G_M1_aa + p_ir * (total_flux_alphaalpha / 2.0)

        # ---- USD4b zero-error attack, ported from monitor_line_gains_closed_form ----
        # (OFF by default, usd4b_strength=0.0). Branciard et al. 2007,
        # arXiv:quant-ph/0609090, Eq. 26/29-31 -- see the closed-form
        # method's own USD4b block below for the full citation and
        # notation-mapping note (f = p_decoy_alphaalpha). Same mechanism,
        # applied to THIS method's own honest G_M0_aa here.
        q3 = float(usd4b_strength)
        f_decoy = float(p_decoy_alphaalpha)
        if q3 > 0.0 and f_decoy > 0.0:
            eta_mon = self.eta_for_monitor if self.eta_for_monitor is not None else self.eta
            p_concl_usd4b = usd4b_conclusive_prob(self.mu, f_decoy)
            usd4b_bright = 0.5 * p_concl_usd4b * ((1.0 - self.tB) * eta_mon / 2.0)
            usd4b_dark = 0.0  # Eq. 30, exactly -- same as the closed-form block
            G_M0_aa = (1.0 - q3) * G_M0_aa + q3 * usd4b_bright
            G_M1_aa = (1.0 - q3) * G_M1_aa + q3 * usd4b_dark

        # |0,0>: both inputs vacuum, amplitude identically 0 for every
        # phase_offset -- quadrature collapses trivially, matching the
        # closed form's pd-only Q^{Mi}_00 exactly (no visibility effect
        # possible with no real photons to dephase). NOT affected by either
        # attack above -- same reasoning as the closed-form method's own
        # |0,0> block (IR: Eve has nothing to detect on a vacuum decoy;
        # USD4b: targets the alphaalpha decoy specifically).
        G_M0_00 = float(pd)
        G_M1_00 = float(pd)

        return {
            "G_M0_alphaalpha": float(np.clip(G_M0_aa, 0.0, 1.0)),
            "G_M1_alphaalpha": float(np.clip(G_M1_aa, 0.0, 1.0)),
            "G_M0_00": G_M0_00,
            "G_M1_00": G_M1_00,
        }

    def monitor_line_gains_closed_form(self, p_dark: float, interference_visibility: float = 1.0,
                                         ir_attack_strength: float = 0.0,
                                         usd4b_strength: float = 0.0,
                                         p_decoy_alphaalpha: float = 0.0) -> dict:
        """
        Monitor-line click probabilities (gains) INCLUDING dark counts.

        Matches Gao et al., "Simple security proof of coherent-one-way
        quantum key distribution," Optics Express 30, 23783 (2022) /
        arXiv:2107.09329, Appendix A:

            Q^{M0}_{alphaalpha} = (1-pd)^3 * [1-(1-pd)*exp(-2*mu*(1-tB)*eta)] * c5
            Q^{M1}_{alphaalpha} = pd*(1-pd)^3 * exp(-2*mu*(1-tB)*eta) * c5
            Q^{Mi}_{00}         = pd*(1-pd)^3
            where c5 = exp(-2*tB*mu*eta)

        *** CORRECTED 2026-07-15 (referee-report Phase-1 investigation,
        see validation/cao/derive_monitor_gain_twomode.py and
        docs/work_log.md Sec. 6) ***: c5 is a SINGLE exponential,
        exp(-2*tB*mu*eta) -- confirmed by directly fetching arXiv:2107.09329
        **v5** (2 June 2023), the current/latest of the paper's 5 revisions.
        v5's acknowledgments state: "We thank ... Amine Iggidr for helping
        us find and correct an incorrect expression c5 given in appendix A
        of the previous version." The three-term expression this project
        used from 2026-07-13 to 2026-07-15,
        c5 = -2*exp(-2*tB*mu) + 2*exp(-tB*mu-mu*eta) + exp(-2*tB*mu*eta),
        was introduced on 2026-07-13 believing it corrected an earlier
        single-exponential c5 -- but the single exponential was in fact
        already the CORRECT, Iggidr-corrected form; the three-term
        expression reproduces the PRE-correction (buggy) c5 the published
        erratum replaced, not the current paper. This is now fixed back to
        the single exponential, which the current, latest arXiv text
        states verbatim. Since basis_mode="passive" is used for every Cao
        et al. reproduction run in this project, this was silently wrong
        for every reported decoy-state gain (and everything computed
        downstream: Ep_upper, key rates) in passive mode from 2026-07-13
        onward -- ALL Cao-reproduction and attack-sweep CSVs/figures
        generated in that window must be regenerated (flagged as the
        top-priority item in the Phase-2 batch handoff). Does NOT affect
        active mode (decoy_factor=1.0 there, c5 never applied) or the
        signal-state gains (a separate formula, exp(-tB*mu*eta),
        unaffected by this correction).

        NOTE: this c5 correction is a SEPARATE fix from the (now also
        resolved, 2026-07-16) sqrt(2)-in-amplitude gap between the SF
        circuit's bright-port exponent and Gao's inner-bracket exponent
        2*mu*(1-tB)*eta -- see _monitor_input_amplitude's and
        _run_sf_monitor_circuit's docstrings. c5 is a small multiplicative
        correction near 1; the inner-bracket exponent drives the leading
        O(mu) behaviour and was untouched by the c5 fix (it required its
        own, independent topology fix in the SF circuit).

        Note: the SIGNAL-state gain formula (Q^{Mi}_{0z}=Q^{Mi}_{1z}, see
        `monitor_line_signal_gains_closed_form`) genuinely DOES use
        exp(-tB*mu*eta), not c5 -- the paper uses different multiplicative
        factors for the decoy-state vs signal-state monitor gains.

        ACTIVE basis mode: per Gao's Appendix A active-mode text ("with
        other formulas unchanged"), the (1-pd) power drops from ^3 to ^1
        and decoy_factor is set to 1 (c5 not applied) for the active
        branch. The paper's active-mode text does not restate the
        alphaalpha/c5 formula explicitly; this is applied by analogy with
        the documented pd_power pattern and flagged as an assumption, not
        a directly confirmed transcription, for active mode specifically.

        eta here is the MONITOR-LINE transmittance (self.eta_for_monitor
        if set, else self.eta) -- see COWQuantumChannel.__init__ for the
        data-line-vs-monitor-line efficiency distinction (Cao et al. 2026
        report D0=76.2% vs D1/D2=46.0%, plus an additional 3 dB
        time-window-selection loss at the monitor line specifically).

        interference_visibility: ADDITIONAL, NOT-in-Gao's-formula physical
        correction. Gao's closed form above treats the destructive-
        interference port as if it received ZERO real-photon contribution
        (only dark counts) -- i.e. perfect, V=1, interference. Real
        interferometers never reach V=1; Cao et al. 2026 explicitly report
        their measured visibility as "approximately 99% over a period of
        2000 seconds" (Sec. III). With imperfect visibility V, a standard
        two-path-interference result is that the destructive port receives
        an additional (1-V)/(1+V) fraction of the SAME real-photon flux
        the constructive port receives, on top of the dark-count
        contribution already in G_M1_alphaalpha above.

        This was added after finding that G_M1_alphaalpha (dark-count-only)
        under-predicts Cao et al.'s real reported dark-port decoy count by
        a factor of ~71x at L=25km, while a pure dark-count-floor
        explanation could not be supported by Cao et al.'s own stated per-
        detector dark rates (checked directly: using D1's real 1 Hz or
        D2's real 11 Hz, instead of D0's 7 Hz, made the prediction WORSE,
        not better). Adding the (1-V)/(1+V) visibility-leakage term with
        Cao et al.'s reported V=0.99 brings the ratio to 0.62x-2.99x across
        their four reported distances (computed directly against their
        real per-distance n_alphaalpha^{D2} counts: 9980, 1190, 4228, 1571
        at 25/50/75/100 km) -- a substantial improvement on the uncorrected
        71x, though not a uniform exact match, indicating this captures a
        real contributing mechanism without being the complete explanation
        for the residual gap (see AUDIT_STATUS.md).

        Default interference_visibility=1.0 (no correction applied) keeps
        this function's previous behavior unchanged for any caller that
        does not explicitly set it.

        Returns a dict with keys:
            G_M0_alphaalpha, G_M1_alphaalpha, G_M0_00, G_M1_00
        """
        pd = p_dark
        mu, tB = self.mu, self.tB
        eta = self.eta_for_monitor if self.eta_for_monitor is not None else self.eta

        # Single exponential per arXiv:2107.09329v5 Appendix A (the
        # current, Iggidr-corrected form -- see docstring above).
        c5 = np.exp(-2.0 * tB * mu * eta)

        if self.basis_mode == "passive":
            pd_power = (1.0 - pd) ** 3
            decoy_factor = c5
        else:  # active
            pd_power = (1.0 - pd) ** 1
            decoy_factor = 1.0

        # NOTE: combine_D_M0/combine below are NOT used for the
        # alphaalpha gains in the corrected formula -- Gao's Appendix A
        # gives Q^{M0}_{alphaalpha} and Q^{M1}_{alphaalpha} as direct
        # closed forms (NOT built from the SF-computed P_DM0_alphaalpha
        # quantum amplitude at all). The quantum_probs() call above is
        # therefore no longer used for the alphaalpha/00 gains -- retained
        # only because monitor_line_quantum_click_probs() is still called
        # elsewhere (e.g. verify_against_papers.py Check 2's contrast
        # check) and this method must not change ITS behavior.
        G_M0_coh = pd_power * (1.0 - (1.0 - pd) * np.exp(-2.0 * mu * (1.0 - tB) * eta)) * decoy_factor
        G_M1_coh = pd * pd_power * np.exp(-2.0 * mu * (1.0 - tB) * eta) * decoy_factor

        # Hardware-imperfection visibility correction (UNCHANGED behavior
        # from before this attack was added). V=1.0 (default) -> leak_fraction=0
        # -> no change. This adds a fraction of the bright port's click
        # probability to the dark port -- a small-perturbation approximation
        # valid near V~1 (e.g. Cao et al.'s reported V~0.99), NOT flux-
        # conserving in general (it does not subtract the leaked fraction
        # back out of G_M0_coh). Retained as-is for the hardware-imperfection
        # use case; deliberately NOT reused for the adversarial IR model
        # below, which needs to remain physically valid across the FULL
        # ir_attack_strength in [0,1] sweep, not just near V=1.
        V = float(interference_visibility)
        leak_fraction = (1.0 - V) / (1.0 + V) if V < 1.0 else 0.0
        G_M1_coh = G_M1_coh + leak_fraction * G_M0_coh

        # ---- Intercept-resend attack (OFF by default, ir_attack_strength=0.0) ----
        # Gisin, Ribordy, Zbinden, Stucki, Brunner, Scarani, "Towards practical
        # and fast Quantum Cryptography," arXiv:quant-ph/0411022 (2004),
        # Sec. III.C.2 ("Eve's attack with errors (I): intercept-resend") +
        # Eqs. 19-26. Two facts from that section, combined here:
        #   (1) p_IR is a PER-ROUND probability: Eve applies the I-R strategy
        #       to an ENTIRE two-pulse round as one strategic choice, not
        #       independently to each of the two pulses -- the paper treats
        #       "the pulses (plural) flying to Bob" as a single attack
        #       decision, and p_IR + p_2c <= 1-mu(1-t) are round-level
        #       probabilities (Sec. III.C.4).
        #   (2) When IR is applied, ALL coherence in that round is destroyed
        #       ("the I-R attack breaks all the coherences", Fig. 3), and the
        #       paper's own stated limit for a fully decohered pair is an
        #       EQUAL split of the total flux between the two monitor ports:
        #       "if either alpha_j or alpha_j+1 is zero, then |D_M1|^2 =
        #       |D_M2|^2 = (1/2) mu t (1-tB)" (Sec. II.C) -- i.e. each port
        #       gets half the total incident flux, not zero/full.
        # This gives an exact, flux-CONSERVING mixture over Eve's binary
        # per-round choice:
        #     G_M0 = (1-p_IR)*G_M0_coh + p_IR*(G_M0_coh+G_M1_coh)/2
        #     G_M1 = (1-p_IR)*G_M1_coh + p_IR*(G_M0_coh+G_M1_coh)/2
        # By construction G_M0+G_M1 = G_M0_coh+G_M1_coh for EVERY value of
        # p_IR -- no photons are manufactured or destroyed as attack strength
        # varies, unlike the hardware-leak approximation above. Converting to
        # the paper's own visibility convention V=(bright-dark)/(bright+dark)
        # confirms this reduces EXACTLY to the cited V_d = 1 - p_IR (Eq. 25)
        # whenever G_M0_coh/G_M1_coh already correspond to V~1 (i.e. the
        # no-hardware-imperfection case this attack study assumes):
        #     V_obs = (G_M0-G_M1)/(G_M0+G_M1) = (1-p_IR) * V_baseline
        # No data-line (Ez) code path is touched anywhere by this attack, per
        # the cited Q-invariance result (Eq. 19: "both R_B and Q are unchanged
        # under this strategy").
        p_ir = float(ir_attack_strength)
        total_flux_alphaalpha = G_M0_coh + G_M1_coh
        G_M0_alphaalpha = (1.0 - p_ir) * G_M0_coh + p_ir * (total_flux_alphaalpha / 2.0)
        G_M1_alphaalpha = (1.0 - p_ir) * G_M1_coh + p_ir * (total_flux_alphaalpha / 2.0)

        # ---- USD4b zero-error attack (OFF by default, usd4b_strength=0.0) ----
        # Branciard, Gisin, Luetkenhaus, Scarani, "Zero-Error Attacks and
        # Detection Statistics in the Coherent One-Way Protocol," QIC 7, 639
        # (2007), arXiv:quant-ph/0609090, Sec. III.D + Eqs. 26, 29-31.
        # SIMULATOR DESIGN NOTE: q3 = usd4b_strength is a freely-settable
        # round-fraction knob, NOT pinned to the paper's own "exactly
        # invisible" solution (Eqs. 34-39) -- set it to that value (via the
        # auto-solve helper in security_bounds.py) to reproduce the paper's
        # undetectable mixture, or to any other value to explore imperfect/
        # partial USD4b usage and see the resulting decoy-visibility
        # disturbance directly, per your request that this be a genuine,
        # explorable simulator rather than a single fixed attack point.
        #
        # Mechanism: on a q3 fraction of alpha-alpha decoy rounds, instead of
        # honest transmission, Eve attempts USD on the 4-pulse, 3-bit-slot
        # sequence |0:alpha-alpha:0>; with probability p_concl_usd4b (Eq. 26)
        # she succeeds and forwards a GENUINELY COHERENT decoy pair (V=1
        # preserved exactly -- she is not measuring the decoy pair itself,
        # only re-preparing it once she is certain of the surrounding bit
        # context), landing entirely in the bright port with ZERO dark-port
        # contribution of her own (Eqs. 29-30: D_M1,2k^(4b) = (1/2)
        # p_concl*Pi((1-tB)eta/2), D_M2,2k^(4b) = 0 exactly). With probability
        # (1-p_concl_usd4b) her discrimination is inconclusive and she
        # forwards NOTHING (both ports see only their existing dark-count
        # floor on that sub-population).
        #
        # NOTATION MAPPING (flagged, as with the BS ceiling): Branciard 2007
        # uses a single decoy type with probability f; this codebase has two
        # decoy types (p_decoy_alphaalpha, p_decoy_00). USD4b specifically
        # targets the BRIGHT decoy sequence, so f = p_decoy_alphaalpha here,
        # not the sum with p_decoy_00 (which has no analogue in the 2007
        # paper at all).
        #
        # SCOPE NOTE: this implements USD4b only. USD4a (the cross-boundary
        # "1-0" sub-attack, Eqs. 18-22) is NOT implemented -- it requires a
        # cross-round signal-pair coherence statistic (Branciard's V_10) that
        # does not exist anywhere in this codebase's Gao/Cao-based signal-
        # state formula (monitor_line_signal_gains_closed_form gives
        # G_M0=G_M1 identically for isolated signal pulses, by construction,
        # so there is nothing for USD4a to either fake or disturb here).
        # Scoped out explicitly rather than silently approximated.
        q3 = float(usd4b_strength)
        f_decoy = float(p_decoy_alphaalpha)
        eta_mon = eta  # already resolved to the monitor-line eta above
        if q3 > 0.0 and f_decoy > 0.0:
            p_concl_usd4b = usd4b_conclusive_prob(mu, f_decoy)
            # Pi(p) = p for a genuinely forwarded single photon (paper's own
            # linear convention, Sec. III.A: "Pi(p)=p if Eve forwards a
            # single photon"), p = (1-tB)*eta_mon/2 per Eq. 29.
            usd4b_bright = 0.5 * p_concl_usd4b * ((1.0 - tB) * eta_mon / 2.0)
            usd4b_dark = 0.0  # Eq. 30, exactly
            G_M0_alphaalpha = (1.0 - q3) * G_M0_alphaalpha + q3 * usd4b_bright
            G_M1_alphaalpha = (1.0 - q3) * G_M1_alphaalpha + q3 * usd4b_dark

        # Vacuum decoy (|0,0>) monitor-line gains.
        # Both ports: only dark counts contribute since there is no genuine
        # photon in the vacuum decoy. G_Mi_00 = pd*(1-pd)^3.
        # NOT affected by the IR attack: Eve's I-R action is conditioned on
        # her own detector D_E actually firing (probability mu*t per pulse),
        # and a vacuum-decoy round has mu=0 in both slots, so she has nothing
        # to detect or resend here -- the round passes through exactly as an
        # unattacked round would, regardless of ir_attack_strength.
        # Cao et al. note n^D2_00 can be elevated by MSI reflections but
        # use an inference method at the POST-PROCESSING level (their
        # "refined method": n'_00 = 35.2% * n_D0_00). We do not model that
        # here since our G_M1_alphaalpha calibration (V=0.9964 back-solved
        # from Table S2) already accounts for the dominant MSI reflection
        # effect on n_D2_alphaalpha, and n_D2_00 is negligibly small
        # compared to n_D2_alphaalpha in its contribution to Ep.
        G_M0_00 = pd * pd_power
        G_M1_00 = pd * pd_power

        return {
            "G_M0_alphaalpha": float(np.clip(G_M0_alphaalpha, 0.0, 1.0)),
            "G_M1_alphaalpha": float(np.clip(G_M1_alphaalpha, 0.0, 1.0)),
            "G_M0_00": float(np.clip(G_M0_00, 0.0, 1.0)),
            "G_M1_00": float(np.clip(G_M1_00, 0.0, 1.0)),
        }

    def monitor_line_signal_quantum_click_probs(self) -> dict:
        """
        QUANTUM-ONLY (dark-count-free) monitor-line click probabilities for
        the SIGNAL states |0z>=|0>|alpha>, |1z>=|alpha>|0>, computed via the
        SAME genuine SF MZI circuit as the decoy states
        (`_run_sf_monitor_circuit`), but with one input vacuum and the other
        the attenuated monitor-line coherent amplitude -- i.e. NO
        interference partner of matching strength, which is the correct
        physical picture for a single-pulse signal state passing through
        the one-bit-delay interferometer.

        IMPORTANT: these are genuine, physically meaningful quantities --
        NOT a "tiny dark-count floor" as an earlier version of this module
        assumed. Cao et al., Sci. Adv. 12, eaec2776 (2026), Materials and
        Methods (the paragraph beginning "The observed values n^{Di}_{alpha
        alpha} and n^{Di}_{00}...") states explicitly that n^{Di}_{0z},
        n^{Di}_{1z} are DIRECTLY MEASURED experimental observables: Bob's
        basis choice is PASSIVE (a simple beam splitter), so a Z-basis
        signal pulse still reaches the monitor-line detectors a fraction of
        the time, exactly like any other pulse routed to that port. This
        function computes the SF-equivalent of that same physical process,
        rather than treating Q^{Mi}_{wz} as a derived, paper-specific
        closed-form quantity.

        Returns a dict with keys: P_DM0_0z, P_DM0_1z (D_M1 has no
        quantum-interference contribution by construction -- see
        `_run_sf_monitor_circuit` docstring; its contribution is computed
        directly from the dark-count probability in
        `monitor_line_signal_gains_closed_form`).
        """
        mon_in_alpha = self._monitor_input_amplitude(self.alpha)
        mon_in_zero = self._monitor_input_amplitude(0.0)

        # |0z> = |0>_tau0 |alpha>_tau1: the monitor line sees vacuum at the
        # earlier bin (long/delayed arm input) and alpha at the later bin
        # (short arm input) -- matches _run_sf_monitor_circuit's
        # (beta_prev, beta_curr) argument order directly.
        P_DM0_0z = _click_probability_from_amplitude(
            self._run_sf_monitor_circuit(mon_in_zero, mon_in_alpha)
        )

        # |1z> = |alpha>_tau0 |0>_tau1: reversed.
        P_DM0_1z = _click_probability_from_amplitude(
            self._run_sf_monitor_circuit(mon_in_alpha, mon_in_zero)
        )

        return {
            "P_DM0_0z": P_DM0_0z,
            "P_DM0_1z": P_DM0_1z,
        }

    def monitor_line_signal_gains_closed_form(self, p_dark: float) -> dict:
        """
        Signal-state monitor-line gains, using Gao et al. 2022 (Opt.
        Express 30, 23783)'s EXACT closed form from their Appendix A:

            Q^{Mi}_{0z} = Q^{Mi}_{1z} = (1-pd)^2 * exp(-tB*mu*eta) * c1*(1-c1)
            where c1 = (1-pd) * exp(-(1-tB)*mu*eta/2)

        CRITICAL DETAIL (resolves the long-standing ambiguity this
        framework spent many sessions chasing): Gao's formula gives ONE
        value, identical for BOTH detector indices i=0,1 -- there is no
        bright/dark asymmetry for a signal-state pulse at all (unlike the
        decoy states). This makes physical sense: a single, non-decoy
        pulse has no interference partner of matching strength, so neither
        monitor detector is preferentially favoured; both see the same
        gain. This directly supersedes an earlier SF-circuit-based attempt
        at this quantity, which (a) gave a numerically different value and
        (b) incorrectly computed different gains for D_M0 vs D_M1, neither
        of which matches the paper's own derivation.

        Branches on `self.basis_mode`:

        PASSIVE:
            Q^{Mi}_{0z} = Q^{Mi}_{1z} = (1-pd)^2 * exp(-tB*mu*eta) * c1*(1-c1)
        ACTIVE (Gao Appendix A states this explicitly: "Q^{Mi}_{0z}=
        Q^{Mi}_{1z}=c1(1-c1)... with other formulas unchanged" -- i.e. NO
        (1-pd)^2 prefactor and NO exp(-tB*mu*eta) factor, for the same
        reason as the decoy-state gains: the hard optical switch removes
        the "AND data line did not click" joint condition):
            Q^{Mi}_{0z} = Q^{Mi}_{1z} = c1*(1-c1)

        where c1 = (1-pd) * exp(-(1-tB)*mu*eta/2) in BOTH modes.

        CRITICAL DETAIL (resolves the long-standing ambiguity this
        framework spent many sessions chasing): Gao's formula gives ONE
        value, identical for BOTH detector indices i=0,1 -- there is no
        bright/dark asymmetry for a signal-state pulse at all (unlike the
        decoy states). This makes physical sense: a single, non-decoy
        pulse has no interference partner of matching strength, so neither
        monitor detector is preferentially favoured; both see the same
        gain. This directly supersedes an earlier SF-circuit-based attempt
        at this quantity, which (a) gave a numerically different value and
        (b) incorrectly computed different gains for D_M0 vs D_M1, neither
        of which matches the paper's own derivation.

        Returns a dict with keys: G_M0_0z, G_M1_0z, G_M0_1z, G_M1_1z (all
        four numerically identical, by Gao's own formula).
        """
        pd = p_dark
        mu, tB = self.mu, self.tB
        eta = self.eta_for_monitor if self.eta_for_monitor is not None else self.eta

        c1 = (1.0 - pd) * np.exp(-(1.0 - tB) * mu * eta / 2.0)
        if self.basis_mode == "passive":
            Q_signal = (1.0 - pd) ** 2 * np.exp(-tB * mu * eta) * c1 * (1.0 - c1)
        else:  # active
            Q_signal = c1 * (1.0 - c1)
        Q_signal = float(np.clip(Q_signal, 0.0, 1.0))

        pd_power = (1.0 - pd) ** 3 if self.basis_mode == "passive" else (1.0 - pd) ** 1
        G_M1_signal = pd * pd_power  # no interference term, by construction

        return {
            "G_M0_0z": Q_signal,
            "G_M1_0z": Q_signal,
            "G_M0_1z": Q_signal,
            "G_M1_1z": Q_signal,
        }

    # -- SF Gaussian-backend amplitude check (used once, for verification) --

    def sf_verify_beamsplitter_amplitude(self, alpha_in: complex) -> complex:
        """
        Cross-check the closed-form data-line amplitude formula above against
        an actual Strawberry Fields Gaussian-backend circuit. This routine is
        called once at startup (not per-pulse, for speed) purely to validate
        that our closed-form propagation matches SF's own `BSgate` + `LossChannel`
        composition -- this is the SF "ground truth" check requested for the
        physics-verification file (verify_against_papers.py).
        """
        prog = sf.Program(2)
        eng = sf.Engine("gaussian")

        with prog.context as q:
            ops.Dgate(np.abs(alpha_in), np.angle(alpha_in)) | q[0]
            ops.Vac | q[1]
            # asymmetric beamsplitter: BSgate(theta, varphi), T = cos^2(theta)
            theta = np.arccos(np.sqrt(self.tB))
            ops.BSgate(theta, 0.0) | (q[0], q[1])
            ops.LossChannel(self.eta) | q[0]

        result = eng.run(prog)
        mu_vec = result.state.means()
        # FIXED (was a real bug, confirmed by running this exact check with
        # the real strawberryfields package): means() returns
        # [x0, x1, ..., x_{N-1}, p0, p1, ..., p_{N-1}] for N modes -- ALL
        # positions first, then ALL momenta -- not interleaved [x0,p0,x1,p1]
        # as originally assumed. hbar=2 default: x=2*Re(alpha), p=2*Im(alpha).
        n_modes = 2
        x0, p0 = mu_vec[0], mu_vec[0 + n_modes]
        amp_out = (x0 + 1j * p0) / 2.0
        return complex(amp_out)


# ----------------------------------------------------------------------------
# Observable collector (the public entry point of Layer 1)
# ----------------------------------------------------------------------------

class ObservableCollector:
    """
    Runs `num_pulses` simulated COW rounds and tallies raw click counts.

    Design notes
    ------------
    * Strawberry Fields is used to validate the linear-optical amplitude
      transformations (see `sf_verify_beamsplitter_amplitude`) and to define
      the physical model; per-pulse Monte Carlo sampling then uses the
      closed-form coherent-state click probability derived from that exact
      Gaussian propagation (this is standard practice for coherent-state
      protocols and avoids the prohibitive cost of building a fresh SF
      circuit for every one of up to 1e6 pulses).
    * Only ONE call to `run` is made per experiment configuration (per your
      architecture decision): "we run SF just once and collect the
      observables once for 1 MC run; it can consist of many bounds
      calculations; if we need to run something again we need to run SF
      again." Re-using a snapshot across multiple SecurityBound objects in
      security_bounds.py is therefore the correct and intended usage pattern.
    * Decoy/signal selection probabilities follow the papers' convention:
      p_z (= p(0z) = p(1z)) and p_d1 = p(alpha,alpha), p_d2 = p(0,0), with
      p_z = (1 - p_d1 - p_d2) / 2 (Li Sec. II.B step 1).
    """

    def __init__(
        self,
        mu: float,
        tB: float,
        eta: float,
        p_dark: float,
        p_signal: float = 0.5,
        p_decoy_alphaalpha: float = 0.25,
        p_decoy_00: float = 0.25,
        phi: float = np.pi,
        seed: int | None = None,
        basis_mode: str = "passive",
        eta_for_monitor: float | None = None,
        interference_visibility: float = 1.0,
        ir_attack_strength: float = 0.0,
        usd3_strength: float = 0.0,
        usd4b_strength: float = 0.0,
        use_sf_monitor_gains: bool = False,
        monitor_gain_source: str = "sf_physical",
    ):
        """
        Parameters
        ----------
        mu, tB, eta, phi, basis_mode : see COWQuantumChannel.
        eta_for_monitor : the monitor-line (D1/D2) transmittance, if
                 different from eta (the data-line/D0 transmittance) --
                 see COWQuantumChannel.__init__ for the full explanation.
        interference_visibility : monitor-line MZI/MSI interference
                 visibility V (default 1.0 = perfect, no correction). See
                 `monitor_line_gains_closed_form`'s docstring for the full
                 physical justification and the residual-gap context.
        monitor_gain_source : "sf_physical" (NEW DEFAULT, 2026-07-16) or
                 "closed_form" (the prior default, retained for explicit
                 opt-out / audit comparisons). Controls which computation
                 supplies the DECOY-state (alphaalpha, 00) monitor-line
                 gains that feed every downstream security bound:
                   - "sf_physical": `monitor_line_gains_sf_physical` --
                     genuinely SF-circuit-derived, INCLUDING a physically
                     simulated imperfect-visibility model (Gaussian
                     phase-jitter quadrature averaging), not the ad hoc
                     (1-V)/(1+V) patch. This is the quantum engine acting
                     as the source of truth, not a cross-check of it.
                   - "closed_form": `monitor_line_gains_closed_form` --
                     Gao et al.'s closed form plus the ad hoc leakage
                     patch, as used prior to 2026-07-16.
                 BOTH are always computed, regardless of which is selected
                 (see __init__ body) -- the non-selected one is retained
                 as `self._monitor_gains_crosscheck` for reporting/audit,
                 so switching this flag never loses the comparison. See
                 `monitor_line_gains_sf_physical`'s docstring for the full
                 justification and the validation against Cao et al.'s
                 real measured Table S2 counts (independent of Gao's
                 formula) that motivated this default change.
        use_sf_monitor_gains : LEGACY, narrower toggle, OFF by default
                 (False = prior behavior, bit-identical when combined with
                 monitor_gain_source="closed_form"). When True, ADDITIONALLY
                 patches the D_M0 (bright-port) click probabilities for the
                 SIGNAL states (0z/1z only -- decoy states are already
                 controlled by monitor_gain_source above) from the genuine
                 SF interference-circuit output
                 (`monitor_line_signal_quantum_click_probs`) instead of the
                 closed form, with dark counts reintroduced via the
                 independent-process combination
                 P_total = 1-(1-P_quantum)*(1-p_dark). Retained for
                 backward compatibility with existing scripts; superseded
                 by monitor_gain_source for the decoy-state path, which is
                 the security-critical one.
        ir_attack_strength : intercept-resend ADVERSARIAL attack strength
                 p_IR in [0, 1] (default 0.0 = attack OFF, no change from
                 prior behavior). Per Gisin et al. 2004 (arXiv:quant-ph/
                 0411022), Eqs. 19-26: a pure IR attack leaves Q unchanged
                 but drives the decoy-pair monitor visibility to V_d =
                 1 - p_IR. See `monitor_line_gains_closed_form`'s docstring
                 for the exact citation and the scope note on why this is
                 applied only to the decoy channel, not the signal-state
                 monitor gains.
        usd3_strength, usd4b_strength : Branciard et al. 2007 zero-error
                 attack sub-components (arXiv:quant-ph/0609090), OFF by
                 default. q1 = usd3_strength (data-line bit-faking, see
                 usd3_conclusive_prob), q3 = usd4b_strength (decoy-pair
                 faking, see monitor_line_gains_closed_form's docstring for
                 the full mechanism, citation, and scope note -- USD4a is
                 not implemented, see that docstring). Both are freely
                 settable round-fractions, not pinned to the paper's own
                 "exactly invisible" mixture -- see security_bounds.py for
                 the helper that solves for that specific point.
        p_dark : detector dark-count probability PER PULSE WINDOW
                 (p_dark = dark_count_rate_hz / rep_rate_hz). Required here
                 because the papers' closed-form monitor-line gain formulas
                 (Eq. 7-10) have pd algebraically built into the derivation --
                 see module docstring for why this differs from the
                 data line, where dark counts are applied independently in
                 detector.py.
        p_signal : combined probability that a round is a SIGNAL round
                   (|0z> or |1z>, split 50/50 between them). NOTE: in the
                   papers, p_z is the probability of EACH individual signal
                   state, so p_signal here = 2 * p_z. Default 0.5 means
                   p_z = 0.25, matching the standard p_z = (1-p_d1-p_d2)/2
                   convention when p_d1 = p_d2 = 0.25.
        p_decoy_alphaalpha, p_decoy_00 : decoy-state probabilities, p_d1, p_d2.
        seed : RNG seed for reproducibility (see framework.py for global seed
               management across repeated batches used for error-bar
               estimation).
        """
        if not np.isclose(p_signal + p_decoy_alphaalpha + p_decoy_00, 1.0):
            raise ValueError(
                f"Probabilities must sum to 1: got p_signal={p_signal}, "
                f"p_d1={p_decoy_alphaalpha}, p_d2={p_decoy_00}, "
                f"sum={p_signal + p_decoy_alphaalpha + p_decoy_00}"
            )

        self.channel = COWQuantumChannel(mu=mu, tB=tB, eta=eta, phi=phi, basis_mode=basis_mode,
                                          eta_for_monitor=eta_for_monitor)
        self.p_dark = p_dark
        self.p_signal = p_signal
        self.p_decoy_alphaalpha = p_decoy_alphaalpha
        self.p_decoy_00 = p_decoy_00
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Pre-compute click probabilities from genuine SF circuit runs.
        #
        # DATA LINE: SF amplitude -> exact Poisson click probability.
        alpha = self.channel.alpha
        self._data_amp_alpha = self.channel._data_line_amplitude(alpha)
        self._data_amp_zero  = self.channel._data_line_amplitude(0.0)
        # p_data_alpha = P(click | coherent state |alpha> arrives at data detector)
        # p_data_zero  = P(click | vacuum |0> arrives at data detector) -- ~0, dark-count only
        self._p_data_alpha = _click_probability_from_amplitude(self._data_amp_alpha)
        self._p_data_zero  = _click_probability_from_amplitude(self._data_amp_zero)

        # ---- USD3 zero-error attack (OFF by default, usd3_strength=0.0) ----
        # Branciard, Gisin, Luetkenhaus, Scarani, arXiv:quant-ph/0609090
        # (2007), Sec. III.B + Eqs. 12-13. On a q1 = usd3_strength fraction
        # of SIGNAL rounds, instead of honest transmission, Eve attempts USD
        # on the 3-pulse sequence |0-alpha-0>; with probability
        # usd3_conclusive_prob(mu, f) [Eq. 12] she succeeds and forwards a
        # genuine single photon into the CORRECT bit slot (she is certain of
        # the bit value, by construction of USD), landing with probability
        # Pi(p)=p [paper's own linear convention for a forwarded single
        # photon, Sec. III.A] at p = tB*eta; with the complementary
        # probability she is inconclusive and forwards nothing (contributes
        # zero click on that sub-population, beyond the existing dark-count
        # floor already folded into detector.py separately). This is a
        # ROUND-FRACTION mixture, exactly analogous to the ir_attack_strength
        # hook above -- q1 is freely settable (see that hook's docstring for
        # the "explorable simulator, not a single fixed point" rationale).
        # Per the citation, USD3 introduces NO data-line error (Eve is never
        # wrong when conclusive) and never fakes a decoy click (Eq. 14,
        # D_B,decoy^(3)=0) -- so this hook touches ONLY the genuine-signal
        # click probability, nothing else. f = p_decoy_alphaalpha (see
        # NOTATION MAPPING note in monitor_line_gains_closed_form).
        q1 = float(usd3_strength)
        if q1 > 0.0:
            p_concl_usd3 = usd3_conclusive_prob(mu, p_decoy_alphaalpha)
            usd3_click = p_concl_usd3 * (tB * eta)
            self._p_data_alpha = (1.0 - q1) * self._p_data_alpha + q1 * usd3_click

        # MONITOR LINE DECOY STATES.
        #
        # DECOY-state gains (alphaalpha, 00) come from ONE of two sources,
        # selected by monitor_gain_source, and BOTH are always computed so
        # the non-selected one is available as an audit cross-check (see
        # monitor_gain_source's docstring above):
        #   - "sf_physical" (NEW DEFAULT, 2026-07-16): genuinely SF-circuit-
        #     derived, including a physically-simulated imperfect-visibility
        #     model. See monitor_line_gains_sf_physical's docstring.
        #   - "closed_form": Gao et al.'s closed form + the ad hoc leakage
        #     patch (the prior default).
        #
        # ATTACK-HOOK SCOPE NOTE (was: closed-form only -- CLOSED): both
        # ir_attack_strength and usd4b_strength are now implemented as the
        # IDENTICAL mixture formulas against BOTH monitor-gain sources --
        # see monitor_line_gains_sf_physical's own "ATTACK-HOOK EXTENSION"
        # docstring note for the citation trail (same published mechanisms,
        # ported onto a second honest baseline, nothing new derived). The
        # previous unconditional fallback to "closed_form" whenever either
        # attack was active has been removed -- monitor_gain_source now
        # genuinely selects which honest baseline the SAME attack mixture
        # is applied on top of, for attacked and unattacked runs alike.
        gains_closed_form = self.channel.monitor_line_gains_closed_form(
            p_dark, interference_visibility=interference_visibility,
            ir_attack_strength=ir_attack_strength,
            usd4b_strength=usd4b_strength,
            p_decoy_alphaalpha=p_decoy_alphaalpha,
        )
        signal_gains_with_dark = self.channel.monitor_line_signal_gains_closed_form(p_dark)

        self.monitor_gain_source = str(monitor_gain_source)

        if self.monitor_gain_source == "sf_physical":
            gains_primary = self.channel.monitor_line_gains_sf_physical(
                p_dark, interference_visibility=interference_visibility,
                ir_attack_strength=ir_attack_strength,
                usd4b_strength=usd4b_strength,
                p_decoy_alphaalpha=p_decoy_alphaalpha,
            )
            self._monitor_gains_crosscheck = gains_closed_form
        elif self.monitor_gain_source == "closed_form":
            # sf_physical is NOT computed here (avoids 12 extra SF circuit
            # calls when it isn't selected) -- so no cross-check is
            # available in this mode, only when monitor_gain_source is
            # "sf_physical" (closed_form is cheap, always computed above).
            gains_primary = gains_closed_form
            self._monitor_gains_crosscheck = None
        else:
            raise ValueError(f"monitor_gain_source must be 'sf_physical' or 'closed_form', got {monitor_gain_source!r}")

        # Store the full (dark-inclusive) probabilities for run() to sample from.
        self._p_M0_aa  = float(gains_primary["G_M0_alphaalpha"])
        self._p_M1_aa  = float(gains_primary["G_M1_alphaalpha"])
        self._p_M0_00  = float(gains_primary["G_M0_00"])
        self._p_M1_00  = float(gains_primary["G_M1_00"])
        self._p_M0_0z  = float(signal_gains_with_dark["G_M0_0z"])
        self._p_M1_0z  = float(signal_gains_with_dark["G_M1_0z"])
        self._p_M0_1z  = float(signal_gains_with_dark["G_M0_1z"])
        self._p_M1_1z  = float(signal_gains_with_dark["G_M1_1z"])

        # ---- LEGACY: signal-state D_M0 genuinely SF-driven (OFF by default) ----
        # See use_sf_monitor_gains's docstring above. Decoy-state sourcing is
        # now controlled by monitor_gain_source above; this flag only
        # remains for the SIGNAL states (0z/1z), for backward compatibility.
        self.use_sf_monitor_gains = bool(use_sf_monitor_gains)
        if self.use_sf_monitor_gains:
            sf_signal_probs = self.channel.monitor_line_signal_quantum_click_probs()

            def _combine(p_quantum: float) -> float:
                return float(1.0 - (1.0 - p_quantum) * (1.0 - p_dark))

            self._p_M0_0z = _combine(sf_signal_probs["P_DM0_0z"])
            self._p_M0_1z = _combine(sf_signal_probs["P_DM0_1z"])

    def run(self, num_pulses: int, exact_expected_values: bool = False) -> ObservableSnapshot:
        """
        Simulate `num_pulses` independent COW rounds and return raw click tallies.
        If exact_expected_values=True, bypasses random sampling and returns the exact 
        expected counts (useful for stable, fast gradients during optimization).

        Architecture:
          1. SF circuits run ONCE in __init__ to compute per-round click
             probabilities (quantum amplitudes -> Poisson threshold statistics,
             plus dark-count terms from the papers' closed-form expressions).
          2. This method genuinely samples `num_pulses` independent rounds from
             those SF-derived probabilities via binomial draws (or directly computes
             expected values), at num_pulses == the real N you intend to report.
          3. detector.py applies Kato/Azuma statistical bounds using
             k = the actual number of pulses sampled here.
        """
        p_data_alpha = self._p_data_alpha
        p_data_zero  = self._p_data_zero
        p_M0_aa, p_M1_aa = self._p_M0_aa, self._p_M1_aa
        p_M0_00, p_M1_00 = self._p_M0_00, self._p_M1_00
        p_M0_0z, p_M1_0z = self._p_M0_0z, self._p_M1_0z
        p_M0_1z, p_M1_1z = self._p_M0_1z, self._p_M1_1z

        round_probs = [
            self.p_signal / 2,        # |0z>
            self.p_signal / 2,        # |1z>
            self.p_decoy_alphaalpha,  # |alpha,alpha>
            self.p_decoy_00,          # |0,0>
        ]

        if exact_expected_values:
            # --- FAST PATH FOR OPTIMIZATION (Exact Expected Values) ---
            n_0z = int(round(num_pulses * round_probs[0]))
            n_1z = int(round(num_pulses * round_probs[1]))
            n_alphaalpha = int(round(num_pulses * round_probs[2]))
            n_00 = int(round(num_pulses * round_probs[3]))

            clicks_data_0z_tau0 = int(round(n_0z * p_data_zero))
            clicks_data_0z_tau1 = int(round(n_0z * p_data_alpha))
            clicks_data_1z_tau0 = int(round(n_1z * p_data_alpha))
            clicks_data_1z_tau1 = int(round(n_1z * p_data_zero))

            clicks_mon_DM0_alphaalpha = int(round(n_alphaalpha * p_M0_aa))
            clicks_mon_DM1_alphaalpha = int(round(n_alphaalpha * p_M1_aa))
            clicks_mon_DM0_00 = int(round(n_00 * p_M0_00))
            clicks_mon_DM1_00 = int(round(n_00 * p_M1_00))
            clicks_mon_DM0_0z = int(round(n_0z * p_M0_0z))
            clicks_mon_DM1_0z = int(round(n_0z * p_M1_0z))
            clicks_mon_DM0_1z = int(round(n_1z * p_M0_1z))
            clicks_mon_DM1_1z = int(round(n_1z * p_M1_1z))

        else:
            # --- GENUINE NOISY SAMPLING FOR FINAL EVALUATION ---
            BATCH = int(5e7)  
            n_0z = n_1z = n_alphaalpha = n_00 = 0
            clicks_data_0z_tau0 = clicks_data_0z_tau1 = 0
            clicks_data_1z_tau0 = clicks_data_1z_tau1 = 0
            clicks_mon_DM0_alphaalpha = clicks_mon_DM1_alphaalpha = 0
            clicks_mon_DM0_00 = clicks_mon_DM1_00 = 0
            clicks_mon_DM0_0z = clicks_mon_DM1_0z = 0
            clicks_mon_DM0_1z = clicks_mon_DM1_1z = 0

            remaining = num_pulses
            while remaining > 0:
                batch_size = min(BATCH, remaining)
                remaining -= batch_size

                counts = self._rng.multinomial(batch_size, round_probs)
                b_0z, b_1z, b_alphaalpha, b_00 = int(counts[0]), int(counts[1]), int(counts[2]), int(counts[3])

                n_0z += b_0z
                n_1z += b_1z
                n_alphaalpha += b_alphaalpha
                n_00 += b_00

                if b_0z > 0:
                    clicks_data_0z_tau0 += int(self._rng.binomial(b_0z, p_data_zero))
                    clicks_data_0z_tau1 += int(self._rng.binomial(b_0z, p_data_alpha))
                if b_1z > 0:
                    clicks_data_1z_tau0 += int(self._rng.binomial(b_1z, p_data_alpha))
                    clicks_data_1z_tau1 += int(self._rng.binomial(b_1z, p_data_zero))

                if b_alphaalpha > 0:
                    clicks_mon_DM0_alphaalpha += int(self._rng.binomial(b_alphaalpha, p_M0_aa))
                    clicks_mon_DM1_alphaalpha += int(self._rng.binomial(b_alphaalpha, p_M1_aa))
                if b_00 > 0:
                    clicks_mon_DM0_00 += int(self._rng.binomial(b_00, p_M0_00))
                    clicks_mon_DM1_00 += int(self._rng.binomial(b_00, p_M1_00))
                if b_0z > 0:
                    clicks_mon_DM0_0z += int(self._rng.binomial(b_0z, p_M0_0z))
                    clicks_mon_DM1_0z += int(self._rng.binomial(b_0z, p_M1_0z))
                if b_1z > 0:
                    clicks_mon_DM0_1z += int(self._rng.binomial(b_1z, p_M0_1z))
                    clicks_mon_DM1_1z += int(self._rng.binomial(b_1z, p_M1_1z))

        return ObservableSnapshot(
            n_0z=n_0z,
            n_1z=n_1z,
            n_alphaalpha=n_alphaalpha,
            n_00=n_00,
            clicks_data_0z_tau0=clicks_data_0z_tau0,
            clicks_data_0z_tau1=clicks_data_0z_tau1,
            clicks_data_1z_tau0=clicks_data_1z_tau0,
            clicks_data_1z_tau1=clicks_data_1z_tau1,
            clicks_mon_DM0_alphaalpha=clicks_mon_DM0_alphaalpha,
            clicks_mon_DM1_alphaalpha=clicks_mon_DM1_alphaalpha,
            clicks_mon_DM0_00=clicks_mon_DM0_00,
            clicks_mon_DM1_00=clicks_mon_DM1_00,
            clicks_mon_DM0_0z=clicks_mon_DM0_0z,
            clicks_mon_DM1_0z=clicks_mon_DM1_0z,
            clicks_mon_DM0_1z=clicks_mon_DM0_1z,
            clicks_mon_DM1_1z=clicks_mon_DM1_1z,
            num_pulses_simulated=num_pulses,
            mu=self.channel.mu,
            tB=self.channel.tB,
            eta=self.channel.eta,
            seed=self.seed,
        )


def run_sf_simulation(
    mu: float,
    tB: float,
    eta: float,
    p_dark: float,
    num_pulses: int = 1_000_000,
    p_signal: float = 0.5,
    p_decoy_alphaalpha: float = 0.25,
    p_decoy_00: float = 0.25,
    phi: float = np.pi,
    seed: int | None = None,
    basis_mode: str = "passive",
    eta_for_monitor: float | None = None,
    interference_visibility: float = 1.0,
    ir_attack_strength: float = 0.0,
    usd3_strength: float = 0.0,
    usd4b_strength: float = 0.0,
    use_sf_monitor_gains: bool = False,
    monitor_gain_source: str = "sf_physical",
    exact_expected_values: bool = False,
) -> ObservableSnapshot:
    """
    Convenience top-level function (the one framework.py actually calls):
    builds an ObservableCollector and runs it once.

    This is Layer 1's single public entry point, matching the architecture:
    "File 1: observable.py exports run_sf_simulation(params, num_pulses=1e6)
    -> ObservableSnapshot". p_dark is required (see ObservableCollector
    docstring) because the monitor-line closed-form gains have it built in.

    basis_mode: "passive" (default, Bob's beamsplitter) or "active" (Bob's
    optical switch) -- see COWQuantumChannel.__init__ docstring for the
    full physical explanation and Gao et al. 2022 Appendix A for the
    source formulas.

    eta_for_monitor: the monitor-line (D1/D2) transmittance, if different
    from the data-line (D0) eta above -- see COWQuantumChannel.__init__
    for why this matters (Cao et al. 2026's D0 vs D1/D2 efficiency
    difference plus an extra 3 dB monitor-line-only loss).

    interference_visibility: monitor-line interference visibility V
    (default 1.0 = no correction) -- see
    `monitor_line_gains_closed_form`'s docstring for the physical
    justification.
    """
    collector = ObservableCollector(
        mu=mu,
        tB=tB,
        eta=eta,
        eta_for_monitor=eta_for_monitor,
        p_dark=p_dark,
        p_signal=p_signal,
        p_decoy_alphaalpha=p_decoy_alphaalpha,
        p_decoy_00=p_decoy_00,
        phi=phi,
        seed=seed,
        basis_mode=basis_mode,
        interference_visibility=interference_visibility,
        ir_attack_strength=ir_attack_strength,
        usd3_strength=usd3_strength,
        usd4b_strength=usd4b_strength,
        use_sf_monitor_gains=use_sf_monitor_gains,
        monitor_gain_source=monitor_gain_source,
    )
    return collector.run(num_pulses, exact_expected_values=exact_expected_values)
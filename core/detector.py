"""
detector.py
===========
LAYER 2 of the COW-QKD framework: classical detector physics + MC scaling.

Responsibility (and ONLY responsibility):
    - Take the raw quantum click counts from observable.py (ObservableSnapshot,
      collected from a finite number of SF-simulated pulses, typically 1e6)
      and apply CLASSICAL, NON-QUANTUM detector effects:
          * dark counts (Poissonian false clicks, independent per detector)
          * dead time (a detector that just fired cannot fire again for a
            fixed blind interval -- modelled as a simple thinning factor
            on the *effective* click rate, since explicit per-pulse dead-time
            simulation is unnecessary once we already work at the rate level)
    - Combine signal-line gains into the standard COW bit-error-rate Ez.
    - Combine monitor-line decoy gains into the four canonical phase-error
      input quantities G^{M0}_{alpha,alpha}, G^{M1}_{alpha,alpha},
      G^{M0}_{00}, G^{M1}_{00} (in the papers' "Q" or "G" notation).
    - Carry binomial/Wilson confidence intervals through to every downstream
      quantity in security_bounds.py.

    A note on N_target vs N_sf (the actual pulses sampled in observable.py):
    for a statistically VALID security bound, these two should be equal --
    Kato's/Azuma's k parameter is the number of genuine independent trials,
    so the count fed in must come from that many actual draws, not from a
    smaller batch linearly rescaled. The scaling logic below (`_scale_count`,
    `_scale_rate`) is kept as a no-op-when-equal safety net (it multiplies
    by N_target/N_sf, which is 1.0 when the caller did the right thing) so
    that diagnostic code paths (e.g. framework.py's check_stationarity,
    which deliberately samples a smaller batch to compare against a known
    target) still work, but every validation script in validation/ now
    calls run_experiment with num_sf_pulses == N_target.

What this file explicitly does NOT do:
    - No quantum simulation (that's observable.py).
    - No phase-error-bound concentration inequalities, no key-rate formulas
      (that's security_bounds.py -- Kato vs Azuma live there, not here).
    - No experiment orchestration (that's framework.py).

Why dark counts/dead time are classical and belong here, not in observable.py:
    A dark count is a spontaneous detector event uncorrelated with any
    incoming photon -- it is a property of the *detector electronics*
    (thermal/tunnelling effects in the SPAD), not of the optical field.
    Dead time is similarly a purely electronic blind-time effect. Neither
    requires propagating a quantum state through Strawberry Fields.

    IMPORTANT ARCHITECTURE NOTE (monitor line vs data line): for the DATA
    LINE, dark counts are applied here in Layer 2, independently combined
    with the dark-count-free quantum click probability from observable.py
    (see `apply_dark_counts` below). For the MONITOR LINE, however, Gao et
    al. 2022's own closed-form gain formula (Appendix A -- the sole source
    used here) has the dark-count probability pd ALGEBRAICALLY BUILT IN to
    the derivation itself (the (1-pd)^3 prefactors
    etc. are not a simple independent-event correction layered on top of a
    separate "ideal" gain -- they emerge from the joint multi-detector click
    statistics of the interferometer). Consequently `observable.py` already
    requires p_dark as an input and returns DARK-COUNT-INCLUSIVE monitor-line
    click counts. This file must NOT re-apply `apply_dark_counts` to the
    monitor-line counts a second time -- doing so would double-count the
    dark-count contribution. The monitor-line section below therefore only
    performs MC-scaling (Wilson CIs + extrapolation to N_target), not a
    second dark-count correction.

Reference equations:
    Gao et al., Opt. Express 30, 23783 (2022):
        Eq. (4)              -- bit error rate E_z definition
        App. A               -- gain formulas with dark-count probability p_d
                                built in (see note above), including the
                                monitor-line gains (the sole source used here)
    Li, Cao, Xie, Yin, Chen, Phys. Rev. Research 6, 013022 (2024):
        Sec. II.B step 4     -- click counts n^{Mi}_{0a}, n^{Mi}_{a0},
                                n^{Mi}_{aa}, n^{Mi}_{00} definitions
        Eq. (9)              -- E_x formula (the un-bounded phase error rate,
                                computed directly here; the *bounded* version
                                E_p with Kato/Azuma fluctuations is
                                security_bounds.py's job)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import stats

from observable import ObservableSnapshot


# ----------------------------------------------------------------------------
# Data container
# ----------------------------------------------------------------------------

@dataclass
class ScaledMetrics:
    """
    Detector-corrected, MC-scaled observables ready for security-bound
    calculation in Layer 3. All "n_*" fields are scaled to the FULL
    experiment size N_target (e.g. 1e8), while *_ci fields carry the
    propagated 95% binomial confidence interval (absolute count units,
    symmetric for convenience: report as value +/- ci).

    Click counts (data line), scaled to N_target, dark-count corrected:
        n_z              : total sifted clicks on data line, n_z
                            = sum of all 4 G^{tau_j}_{wz} gains * N
        n_z_ci            : propagated 95% CI on n_z

    Bit error rate (Z basis), from data line:
        Ez                : QBER, Gao Eq. (4)
        Ez_ci             : 95% CI on Ez (binomial, from the *unscaled* SF
                            sample -- scaling N does not shrink relative
                            statistical uncertainty of a *rate*, only of
                            absolute counts; see note in `_scale_rate`)

    Monitor-line decoy click counts, scaled to N_target, dark-count corrected:
        n_DM0_alphaalpha, n_DM1_alphaalpha : clicks on D_M0 / D_M1 when decoy
                            |alpha,alpha> was sent (Gao et al.'s
                            Q^{M0}_{alphaalpha} etc.)
        n_DM0_00, n_DM1_00 : clicks on D_M0 / D_M1 when vacuum decoy |0,0> was
                            sent -- ideally dark-count-only.
        n_alphaalpha, n_00 : total number of each decoy type sent, scaled to
                            N_target (denominators for gain/rate calculations)

    Monitor-line SIGNAL-state click counts, scaled to N_target, dark-count
    corrected (Cao et al. 2026 notation n^{Di}_{0z}/n^{Di}_{1z} -- genuine
    experimental/simulated observables, NOT a derived dark-count floor; see
    security_bounds.py module docstring for the full explanation):
        n_DM0_0z, n_DM1_0z : clicks on D_M0/D_M1 when Alice sent |0z>
        n_DM0_1z, n_DM1_1z : clicks on D_M0/D_M1 when Alice sent |1z>

    All of the above _ci fields are absolute-count 95% confidence intervals
    propagated from the finite SF sample via binomial statistics, then scaled
    linearly to N_target (valid under the stated stationarity assumption that
    click rates measured in the SF batch are representative of the full run).

    mu, tB, eta : echoed from the originating ObservableSnapshot, useful for
                  bookkeeping / plotting in framework.py.
    N_target, N_sf : full target experiment size and the actual number of SF
                  pulses simulated (kept distinct so security_bounds.py can,
                  if ever needed, sanity check the scaling factor).
    """

    n_z: float
    n_z_ci: float
    Ez: float
    Ez_ci: float

    n_DM0_alphaalpha: float
    n_DM1_alphaalpha: float
    n_DM0_00: float
    n_DM1_00: float
    n_alphaalpha: float
    n_00: float

    n_DM0_0z: float
    n_DM1_0z: float
    n_DM0_1z: float
    n_DM1_1z: float

    # Total number of |0z> / |1z> SIGNAL pulses sent, scaled to N_target --
    # i.e. the correct denominators for turning n_DM*_0z / n_DM*_1z into
    # rates (Gao Eq. 7's G^{Mi}_{0z}=n^{Mi}_{0z}/n_{0z-trials}, NOT
    # n_DM0_0z/(n_DM0_0z+n_DM1_0z), which only ever sums to 1 by
    # construction and discards the actual signal-pulse denominator).
    # Previously computed inside apply_detector_physics as n_0z_target/
    # n_1z_target and then discarded -- now exposed here so Layer 3 doesn't
    # have to reconstruct (and get wrong) its own denominator.
    n_0z_total: float
    n_1z_total: float

    # rate (gain) versions of the monitor-line counts -- convenience fields
    # used directly by the phase-error-bound formulas (Eqs. 10-12 in Li 2024 /
    # Eq. 7 in Gao consume *gains*, i.e. rates, not raw counts)
    G_DM0_alphaalpha: float
    G_DM1_alphaalpha: float
    G_DM0_00: float
    G_DM1_00: float

    mu: float
    tB: float
    eta: float
    p_dark: float
    N_target: int
    N_sf: int
    effective_rep_rate_hz: float


# ----------------------------------------------------------------------------
# Dark count / dead time application
# ----------------------------------------------------------------------------

def _wilson_ci(successes: int, trials: int, confidence: float = 0.95):
    """
    Wilson score interval for a binomial proportion -- more reliable than the
    naive normal-approximation interval when success counts are small (as is
    typical for the vacuum-decoy dark-count-only channel, where the click
    probability can be ~1e-6 to 1e-8). Returns (p_hat, half_width) where
    half_width is in PROBABILITY units (not yet scaled to absolute counts).
    """
    if trials == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(0.5 + confidence / 2.0)
    p_hat = successes / trials
    denom = 1.0 + z**2 / trials
    centre = p_hat + z**2 / (2 * trials)
    half_width = z * np.sqrt(p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2))
    lower = (centre - half_width) / denom
    upper = (centre + half_width) / denom
    p_centre = (lower + upper) / 2.0
    half = (upper - lower) / 2.0
    return p_centre, half


def apply_dark_counts(
    clicks_signal: int,
    n_trials: int,
    dark_count_prob: float,
) -> tuple[float, float]:
    """
    Combine a genuine quantum click rate with an independent dark-count
    process. For a threshold detector, "click" = "at least one event" (real
    photon OR dark count OR both), and since the two processes are
    independent:

        P(click_total) = 1 - (1 - P(click_quantum)) * (1 - p_dark)

    This matches the (1 - p_d) multiplicative factors that appear throughout
    Gao's Appendix A gain formulas (e.g. G^{M0}_{alpha alpha} = (1-pd)^3
    [1 - (1-pd) e^{...}] e^{...}); that paper folds the dark-count factor
    directly into the analytic gain expression, which is exactly the
    operation performed here at the Monte-Carlo level.

    Returns (corrected_click_probability, corrected_click_count_estimate).
    """
    if n_trials == 0:
        return 0.0, 0.0
    p_quantum = clicks_signal / n_trials
    p_total = 1.0 - (1.0 - p_quantum) * (1.0 - dark_count_prob)
    return p_total, p_total * n_trials


def apply_dead_time_thinning(click_probability_per_pulse: float, dead_time_s: float, rep_rate_hz: float) -> float:
    """
    Dead-time handling: standard non-paralyzable-detector thinning, applied
    to a per-pulse click probability.

        r      = click_probability_per_pulse * rep_rate_hz   (implied click
                 rate, Hz, if this probability held at every pulse window)
        r_eff  = r / (1 + r * dead_time_s)                    (Knoll,
                 "Radiation Detection and Measurement," the standard
                 non-paralyzable dead-time model: after every click the
                 detector is blind for dead_time_s, so a fraction of
                 would-be clicks are lost in proportion to how much of the
                 mean inter-click interval that blind window consumes)
        p_eff  = r_eff / rep_rate_hz
               = click_probability_per_pulse
                 / (1 + click_probability_per_pulse * rep_rate_hz * dead_time_s)

    HISTORY -- why this was a no-op until now, and why that no longer
    applies. A prior attempt to apply exactly this formula was reverted
    because it made results "swing wildly, unphysically" -- but that
    instability traced to the `dead_time_s` value being used at the time
    (30 us), not to the formula itself. 30 us is a gated-InGaAs-APD
    afterpulsing hold-off figure; Cao et al. 2026's own primary source
    (arXiv:2601.06772, checked directly, main text + appendix) reports no
    numeric dead time at all, but does confirm D0/D1/D2 are all SNSPDs
    ("SPD: superconducting nanowire single-photon detector," Fig. 1
    caption) -- a detector family with reset times of order 10-100 ns, not
    tens of microseconds. Worse, 30 us was independently provably
    impossible for their real hardware: their own reported n_z at 25 km
    implies a ~63.2 kHz average data-line click rate, which EXCEEDS the
    ~33.3 kHz asymptotic ceiling (1/dead_time_s) a genuine 30 us
    non-paralyzable dead time permits -- a rate no real 30 us-dead-time
    detector could have produced, regardless of the true underlying rate.
    `COWParams.dead_time_s`'s default is now 100 ns, a
    literature-representative SNSPD reset time (see that field's own
    docstring for the citation trail); at 100 ns the correction above is
    small and well-behaved everywhere in the tested operating range
    (r*dead_time_s ~ 0.6-1.5% at Cao et al.'s real 25 km click rates, and
    smaller still at longer distances, where click rates only drop) --
    the magnitude that made the earlier attempt "swing wildly" was a
    symptom of the wrong dead-time number, not of the formula.

    Scope: applied only at call sites carrying the genuine-signal click
    probability for each state (see `apply_detector_physics`, the tau1/tau0
    slots where Alice's real photon physically sits), consistent with the
    existing architecture -- dead time is overwhelmingly set by real
    (comparatively high-rate) signal clicks, not the astronomically rarer
    dark-count-only slots, so thinning only the dominant contributor is a
    reasonable first-order treatment rather than building a joint
    all-four-slots detector-occupancy model.
    """
    if dead_time_s <= 0.0 or rep_rate_hz <= 0.0:
        return click_probability_per_pulse
    p = float(click_probability_per_pulse)
    denom = 1.0 + p * rep_rate_hz * dead_time_s
    return p / denom if denom > 0.0 else p


def effective_rep_rate_hz(rep_rate_hz: float, dead_time_s: float) -> float:
    """
    Diagnostic/reporting quantity: the asymptotic maximum sustainable click
    rate a non-paralyzable detector with this dead time permits
    (1/dead_time_s as rep_rate_hz -> infinity). Still NOT consumed by any
    rate computation anywhere in this project (confirmed by grep: zero call
    sites outside this module and `ScaledMetrics.effective_rep_rate_hz`,
    which is itself never read back) -- this is deliberately separate from
    `apply_dead_time_thinning` (see that function's docstring), which now
    genuinely applies the per-pulse non-paralyzable correction at the
    click-probability level. The two are not redundant if both were wired
    in: this quantity is a hardware CEILING figure (useful as a sanity
    check that reported click rates stay below it -- see
    COWParams.dead_time_s's docstring for the ~150x margin at Cao et al.'s
    real operating points), not itself a correction to apply to a
    rate/probability a second time.

    *** CORRECTION (retained from an earlier pass): an earlier version of
    this docstring claimed this quantity "is kept available for
    framework.py's bookkeeping/plotting (e.g. converting a per-pulse key
    rate into kbps using the actual achievable gate rate)." That usage does
    not exist anywhere in `framework.py` or any validation script -- every
    bps number this project reports uses the raw `rep_rate_hz` (or a fixed
    fraction of it, e.g. `ROUND_RATE_HZ = rep_rate_hz/2`), never this
    function's output. Retained purely so a caller CAN report the
    dead-time-limited maximum sustainable gate rate as a hardware-capability
    figure alongside the reported key rate, if that comparison is ever
    wanted -- not because anything in this codebase currently does so.
    """
    if dead_time_s <= 0:
        return rep_rate_hz
    return 1.0 / (1.0 / rep_rate_hz + dead_time_s)


# ----------------------------------------------------------------------------
# Scaling helper (propagates Wilson CIs from the SF sample to N_target)
# ----------------------------------------------------------------------------

def _scale_count(successes: int, trials: int, n_target_trials: float):
    """
    Scale a click count observed in `trials` SF-simulated rounds up to an
    equivalent count in `n_target_trials` rounds, assuming the underlying
    click probability is stationary (validated separately by running
    multiple independent SF batches -- see framework.py's stationarity
    check). Returns (scaled_count, scaled_count_95ci_halfwidth).

    Note on why the *relative* uncertainty does not shrink under scaling:
    if we observed a Wilson half-width of `h` on the probability estimate
    from `trials` SF samples, the absolute count uncertainty at the full
    scale is h * n_target_trials -- i.e. the absolute error bar GROWS
    proportionally with N_target even though the *relative* error stays
    fixed (it is set entirely by the finite SF sample size, not by
    N_target). This is the correct and expected behaviour: we are not
    pretending to have actually run N_target pulses, only extrapolating a
    rate estimated from a smaller, honestly-simulated sample.
    """
    p_centre, p_half = _wilson_ci(successes, trials)
    scaled = p_centre * n_target_trials
    scaled_ci = p_half * n_target_trials
    return scaled, scaled_ci


# ----------------------------------------------------------------------------
# Main entry point (the public interface of Layer 2)
# ----------------------------------------------------------------------------

def apply_detector_physics(
    obs: ObservableSnapshot,
    dark_count_rate_hz: float,
    dead_time_s: float,
    rep_rate_hz: float,
    N_target: int,
) -> ScaledMetrics:
    """
    Apply classical detector physics (dark counts, dead time) to a raw
    ObservableSnapshot from observable.py, then scale every resulting rate
    estimate up to `N_target` total pulses with propagated error bars.

    Parameters
    ----------
    obs : ObservableSnapshot
        Raw quantum click counts from a single SF run (Layer 1).
    dark_count_rate_hz : float
        Detector dark-count rate in counts/second (e.g. Cao et al.: 7/1/11 Hz
        for D0/D1/D2 at 500 MHz rep rate -> pd = rate / 500e6 per pulse window).
    dead_time_s : float
        Detector dead time in seconds -- e.g. ~100 ns, a literature-
        representative SNSPD reset time (see COWParams.dead_time_s's
        docstring for the citation trail; NOT a gated-InGaAs-SPAD hold-off
        figure, which would be 1-2 orders of magnitude longer and is the
        wrong detector family for Cao et al. 2026's own SNSPD-based setup).
        Applied via `apply_dead_time_thinning`'s non-paralyzable correction.
    rep_rate_hz : float
        Pulse repetition rate in Hz (used to convert dark_count_rate_hz into
        a per-pulse dark-count PROBABILITY, p_d = dark_count_rate_hz / rep_rate_hz).
    N_target : int
        Total number of pulses the experiment is meant to represent (e.g.
        1e8). All scaled quantities in the returned ScaledMetrics correspond
        to this size, even though only `obs.num_pulses_simulated` (e.g. 1e6)
        pulses were actually run through Strawberry Fields.

    Returns
    -------
    ScaledMetrics ready to be consumed by security_bounds.py.
    """
    p_dark = dark_count_rate_hz / rep_rate_hz

    n_sf = obs.num_pulses_simulated
    scale_factor = N_target / n_sf if n_sf > 0 else 0.0

    # ---- Data line: build the 4 gains G^{tau_j}_{wz}, dark-count corrected ----
    # |0z> at tau0 should be empty (vacuum) -> ideal click prob ~0, dark counts
    # are the ONLY source of clicks here.
    p_0z_tau0, _ = apply_dark_counts(obs.clicks_data_0z_tau0, obs.n_0z, p_dark)
    p_0z_tau1, _ = apply_dark_counts(obs.clicks_data_0z_tau1, obs.n_0z, p_dark)
    p_1z_tau0, _ = apply_dark_counts(obs.clicks_data_1z_tau0, obs.n_1z, p_dark)
    p_1z_tau1, _ = apply_dark_counts(obs.clicks_data_1z_tau1, obs.n_1z, p_dark)

    # Wilson half-widths (probability units) on the four RAW (quantum-only)
    # data-line counts, needed to build a genuine n_z/Ez CI below (see the
    # CORRECTION note near `n_z_ci` for why this was previously a disclosed
    # gap, hardcoded to 0.0). apply_dark_counts is the affine transform
    # p_total = p_dark + (1-p_dark)*p_quantum, so a half-width on p_quantum
    # propagates to a half-width on p_total by the same (1-p_dark) factor
    # -- no need to re-derive a separate CI formula for the dark-count-
    # corrected probability.
    _, _h_0z_tau0 = _wilson_ci(obs.clicks_data_0z_tau0, obs.n_0z)
    _, _h_0z_tau1 = _wilson_ci(obs.clicks_data_0z_tau1, obs.n_0z)
    _, _h_1z_tau0 = _wilson_ci(obs.clicks_data_1z_tau0, obs.n_1z)
    _, _h_1z_tau1 = _wilson_ci(obs.clicks_data_1z_tau1, obs.n_1z)
    h_0z_tau0 = (1.0 - p_dark) * _h_0z_tau0
    h_0z_tau1 = (1.0 - p_dark) * _h_0z_tau1
    h_1z_tau0 = (1.0 - p_dark) * _h_1z_tau0
    h_1z_tau1 = (1.0 - p_dark) * _h_1z_tau1

    # Apply dead-time thinning to the genuine-signal click probabilities
    # (p_0z_tau1, p_1z_tau0 -- the slots where Alice's real photon
    # physically sits for each state; these are now correctly identified
    # as the ERROR terms in Ez, per the fix above, but dead-time thinning
    # still applies to them on physical grounds regardless of the
    # error/correct bookkeeping, since dead time affects wherever the
    # genuine high-rate signal photon arrives) -- see
    # apply_dead_time_thinning's docstring for the non-paralyzable formula
    # now genuinely applied here (SNSPD-sourced dead_time_s).
    #
    # The thinning transform p_eff = p/(1+c*p) (c = rep_rate_hz*dead_time_s)
    # has derivative dp_eff/dp = 1/(1+c*p)^2 = (p_eff/p)^2 -- used below to
    # propagate each slot's Wilson half-width through the SAME nonlinear
    # transform just applied to its point estimate, rather than letting
    # h_0z_tau1/h_1z_tau0 silently go stale now that this is no longer an
    # identity operation.
    p_0z_tau1_pre_thinning = p_0z_tau1
    p_1z_tau0_pre_thinning = p_1z_tau0
    p_0z_tau1 = apply_dead_time_thinning(p_0z_tau1, dead_time_s, rep_rate_hz)
    p_1z_tau0 = apply_dead_time_thinning(p_1z_tau0, dead_time_s, rep_rate_hz)
    if p_0z_tau1_pre_thinning > 0.0:
        h_0z_tau1 *= (p_0z_tau1 / p_0z_tau1_pre_thinning) ** 2
    if p_1z_tau0_pre_thinning > 0.0:
        h_1z_tau0 *= (p_1z_tau0 / p_1z_tau0_pre_thinning) ** 2

    # Bit-error-rate components, Cao et al. 2026's OWN explicitly-stated
    # convention (arXiv:2601.06772v1, Sec. III "Measurement" paragraph):
    # "If the detection event occurs in the former time slot, Bob assigns
    # a logic bit value of 1. If it occurs in the latter time slot, Bob
    # assigns a logic bit value of 0."
    #
    # |0z> = |0>_{2k-1}|alpha>_{2k}: Alice's bit=0, photon physically at
    #   the LATTER slot (tau1). A click at tau1 (signal) -> Bob records
    #   bit 0 -> MATCHES Alice (correct term). A click at tau0 (dark-
    #   count only) -> Bob records bit 1 -> mismatch (error term).
    # |1z> = |alpha>_{2k-1}|0>_{2k}: Alice's bit=1, photon physically at
    #   the FORMER slot (tau0). A click at tau0 (signal) -> Bob records
    #   bit 1 -> MATCHES Alice (correct term). A click at tau1 (dark-
    #   count only) -> Bob records bit 0 -> mismatch (error term).
    #
    # So under CAO'S OWN stated convention, the genuine-signal click is
    # the CORRECT term for both states, and the dark-count-only click is
    # the ERROR term -- this is DIFFERENT from a literal reading of Gao
    # et al. 2022's prose ("click at T0 -> bit 0, T1 -> bit 1"), which
    # gives the opposite assignment. An earlier attempt at this fix used
    # Gao's convention (since security_bounds.py/observable.py reference
    # Gao's Eq. 4 extensively) and produced Ez~0.9997 -- but Cao, not Gao,
    # is the actual validation target and the paper whose real data this
    # framework is being checked against, and Cao states its own
    # convention explicitly and unambiguously in its own text, so it
    # takes precedence. Restored to match Cao's stated rule.
    G_tau0_0z = p_0z_tau0          # dark-count-only slot for |0z> -> error term
    G_tau1_0z = p_0z_tau1          # genuine-signal slot for |0z> -> correct term
    G_tau0_1z = p_1z_tau0          # genuine-signal slot for |1z> -> correct term
    G_tau1_1z = p_1z_tau1          # dark-count-only slot for |1z> -> error term

    error_terms = G_tau0_0z + G_tau1_1z
    total_terms = G_tau0_0z + G_tau1_0z + G_tau0_1z + G_tau1_1z
    Ez = error_terms / total_terms if total_terms > 0 else 0.0

    # *** CORRECTION (referee-report response, 2026-07-14) ***
    # An earlier version of this comment claimed clicks_data_0z_tau1 and
    # clicks_data_1z_tau0 (the genuine coherent-state signal click counts)
    # were "now EXACT analytic values, not binomial samples," and that
    # "every term feeding Ez and n_z is therefore now exact." This was
    # FALSE and directly contradicted by observable.py's `run()`, which
    # binomial-samples both counts (e.g. `self._rng.binomial(b_0z,
    # p_data_alpha)`). The underlying click PROBABILITY is an exact closed
    # form, 1-exp(-eta*tB*mu), but the resulting COUNT is a real sample of
    # it and carries genuine statistical noise -- this can be severe at
    # small mu*tB (e.g. mu=5e-4, tB=0.01, L=60km gave an expected count of
    # only ~0.13 out of ~500k SF trials, an 88% chance of a spurious zero).
    #
    # *** FIX (was: n_z_ci/Ez_half hardcoded to 0.0, a disclosed gap) ***
    # Ez = error_terms / total_terms = error_terms / (error_terms +
    # correct_terms), where error_terms and correct_terms are each the SUM
    # of two INDEPENDENT binomial draws (see h_0z_tau0..h_1z_tau1 above --
    # independent RNG calls in observable.py's run(), even though tau0/tau1
    # draws share the same trial pool). Propagated via the standard
    # multivariate delta method treating error_terms/correct_terms as
    # independent -- the same "independence approximation, not an exact
    # joint-CI derivation" already used elsewhere in this project for the
    # decoy-visibility CI (see verify_usd_sweep.py's Wilson-CI propagation
    # through V=(G0-G1)/(G0+G1)).
    correct_terms = G_tau1_0z + G_tau0_1z
    ci_error = float(np.sqrt(h_0z_tau0 ** 2 + h_1z_tau1 ** 2))
    ci_correct = float(np.sqrt(h_0z_tau1 ** 2 + h_1z_tau0 ** 2))
    if total_terms > 0:
        d_Ez_d_error = correct_terms / total_terms ** 2
        d_Ez_d_correct = -error_terms / total_terms ** 2
        Ez_half = float(np.sqrt(
            (d_Ez_d_error * ci_error) ** 2 + (d_Ez_d_correct * ci_correct) ** 2
        ))
    else:
        Ez_half = 0.0

    # n_z: total sifted (correct-slot) clicks on the data line, scaled to
    # N_target.
    #
    # CORRECTED: Gao's Eq.(2) defines Qz as the AVERAGE of all four gain
    # terms divided by 2 -- but that division applies at the GAIN level
    # (combining four dimensionless probabilities into one), not at the
    # COUNT level. n_z is a COUNT (Cao et al. 2026 state explicitly:
    # "n_z represents the number of detection events in the Z basis"),
    # and the correct count-level operation is to SUM each state's
    # contribution (each state's own pair of gain terms, scaled by that
    # state's own pulse count), not to average the two states' counts
    # together and discard half of them. Confirmed directly: averaging
    # (dividing the summed counts by 2) produced n_z exactly half of Cao
    # et al.'s real reported n_z=25,278,913 at L=25km (ratio 0.50,
    # verified numerically) -- removing the erroneous /2.0 brings this to
    # within real experimental agreement (~1.0x, consistent with the
    # other independently-validated bounds at this same operating point).
    n_0z_target = N_target * obs.n_0z / n_sf if n_sf > 0 else 0.0
    n_1z_target = N_target * obs.n_1z / n_sf if n_sf > 0 else 0.0
    n_z_from_0z = (G_tau0_0z + G_tau1_0z) * n_0z_target
    n_z_from_1z = (G_tau0_1z + G_tau1_1z) * n_1z_target
    n_z = n_z_from_0z + n_z_from_1z
    # Genuine propagated Wilson CI (was: hardcoded to 0.0, a disclosed
    # gap -- see the FIX note above `Ez`). n_z_from_0z/n_z_from_1z are each
    # built from two independent binomial draws (0z's tau0+tau1 clicks;
    # 1z's tau0+tau1 clicks respectively), and n_z_from_0z/n_z_from_1z are
    # themselves independent of each other (disjoint |0z>/|1z> pulse
    # sub-populations) -- half-widths combine in quadrature at each stage.
    n_z_from_0z_ci = float(np.sqrt(h_0z_tau0 ** 2 + h_0z_tau1 ** 2)) * n_0z_target
    n_z_from_1z_ci = float(np.sqrt(h_1z_tau0 ** 2 + h_1z_tau1 ** 2)) * n_1z_target
    n_z_ci = float(np.sqrt(n_z_from_0z_ci ** 2 + n_z_from_1z_ci ** 2))

    # ---- Monitor line: decoy gains, dark-count corrected, scaled ----
    # NOTE: monitor-line click counts in `obs` were generated using Gao et
    # al.'s closed-form gain expressions (Appendix A), which have
    # the dark-count probability pd ALREADY built in (see observable.py
    # module docstring) -- applying apply_dark_counts() again here would
    # double-count dark counts. We therefore just read off the observed
    # rates directly.
    p_DM0_aa = obs.clicks_mon_DM0_alphaalpha / obs.n_alphaalpha if obs.n_alphaalpha > 0 else 0.0
    p_DM1_aa = obs.clicks_mon_DM1_alphaalpha / obs.n_alphaalpha if obs.n_alphaalpha > 0 else 0.0
    p_DM0_00 = obs.clicks_mon_DM0_00 / obs.n_00 if obs.n_00 > 0 else 0.0
    p_DM1_00 = obs.clicks_mon_DM1_00 / obs.n_00 if obs.n_00 > 0 else 0.0

    n_alphaalpha_target = N_target * obs.n_alphaalpha / n_sf if n_sf > 0 else 0.0
    n_00_target = N_target * obs.n_00 / n_sf if n_sf > 0 else 0.0

    # ---- Monitor line: decoy gains, scaled with proper Wilson CI ----
    # All monitor-line counts are now genuine binomial samples from SF-derived
    # probabilities (run() samples N_sf rounds, not exact analytic values).
    # _scale_count correctly applies the Wilson CI to propagate statistical
    # uncertainty from the N_sf sample up to N_target.
    n_DM0_aa, n_DM0_aa_ci = _scale_count(obs.clicks_mon_DM0_alphaalpha, max(obs.n_alphaalpha, 1), n_alphaalpha_target)
    n_DM1_aa, n_DM1_aa_ci = _scale_count(obs.clicks_mon_DM1_alphaalpha, max(obs.n_alphaalpha, 1), n_alphaalpha_target)
    n_DM0_00, n_DM0_00_ci = _scale_count(obs.clicks_mon_DM0_00,         max(obs.n_00, 1),         n_00_target)
    n_DM1_00, n_DM1_00_ci = _scale_count(obs.clicks_mon_DM1_00,         max(obs.n_00, 1),         n_00_target)

    # ---- Monitor line: SIGNAL-state clicks, scaled ----
    # Same dark-count-already-included treatment as the decoy-state clicks
    # above (see observable.py's monitor_line_signal_gains_closed_form).
    n_0z_target = N_target * obs.n_0z / n_sf if n_sf > 0 else 0.0
    n_1z_target = N_target * obs.n_1z / n_sf if n_sf > 0 else 0.0

    # ---- Monitor line: SIGNAL-state clicks, scaled with proper Wilson CI ----
    # Genuine binomial samples -- apply Wilson CI.
    n_DM0_0z, _ = _scale_count(obs.clicks_mon_DM0_0z, max(obs.n_0z, 1), n_0z_target)
    n_DM1_0z, _ = _scale_count(obs.clicks_mon_DM1_0z, max(obs.n_0z, 1), n_0z_target)
    n_DM0_1z, _ = _scale_count(obs.clicks_mon_DM0_1z, max(obs.n_1z, 1), n_1z_target)
    n_DM1_1z, _ = _scale_count(obs.clicks_mon_DM1_1z, max(obs.n_1z, 1), n_1z_target)

    return ScaledMetrics(
        n_z=n_z,
        n_z_ci=n_z_ci,
        Ez=Ez,
        Ez_ci=Ez_half,
        n_DM0_alphaalpha=n_DM0_aa,
        n_DM1_alphaalpha=n_DM1_aa,
        n_DM0_00=n_DM0_00,
        n_DM1_00=n_DM1_00,
        n_alphaalpha=n_alphaalpha_target,
        n_00=n_00_target,
        n_DM0_0z=n_DM0_0z,
        n_DM1_0z=n_DM1_0z,
        n_DM0_1z=n_DM0_1z,
        n_DM1_1z=n_DM1_1z,
        n_0z_total=n_0z_target,
        n_1z_total=n_1z_target,
        G_DM0_alphaalpha=p_DM0_aa,
        G_DM1_alphaalpha=p_DM1_aa,
        G_DM0_00=p_DM0_00,
        G_DM1_00=p_DM1_00,
        mu=obs.mu,
        p_dark=p_dark,
        tB=obs.tB,
        eta=obs.eta,
        N_target=N_target,
        N_sf=n_sf,
        effective_rep_rate_hz=effective_rep_rate_hz(rep_rate_hz, dead_time_s),
    )
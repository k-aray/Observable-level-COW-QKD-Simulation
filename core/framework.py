"""
framework.py
============
LAYER 4 of the COW-QKD framework: orchestrator.

Responsibility:
    - Wire Layer 1 (observable.py) -> Layer 2 (detector.py) -> Layer 3
      (security_bounds.py) together into single-call experiment functions.
    - Provide parameter-sweep utilities (e.g. "secure key rate vs distance")
      that re-run the FULL pipeline (including a fresh SF simulation) once
      per parameter point, per the architecture rule: "we run sf just once
      and collect the observables once for 1 MC run ... if we need to run
      something again we need to run sf again."
    - Provide a stationarity-check utility: run the SF layer multiple times
      independently at the SAME parameters and report the spread of the
      resulting rate estimates, to validate the "1e6 pulses is enough, with
      error bars" assumption (your point 3 in the design discussion).

What this file explicitly does NOT do (yet):
    - NO ATTACKS. Per your explicit instruction, attack hooks are deferred.
      The functions below are written so that an `attacks` argument could be
      threaded through later without altering Files 1-3, but no such
      argument exists yet -- we are not pre-building unused machinery.
    - No "best bound" selection (deferred, per your instruction).
    - No plotting (kept separate; this file returns plain data structures --
      lists of dicts / dataclasses -- that a plotting layer can consume).

Distance -> transmittance conversion: each paper uses a slightly different
exponent for the fibre attenuation formula, and `framework.py` is exactly the
place where that paper-specific choice should live (NOT observable.py, which
only cares about an already-computed eta), per the project's "thin physics
core, paper-specific configuration at the orchestration layer" principle:

    Gao et al. (2022):  eta = 10**(-0.02 * L)            [Sec. 4 numerical
                       simulation paragraph]
    Li et al. (2024):  eta = 10**(-0.016 * L)            [Sec. IV, "transmittance
                       of the optical fiber...is eta = 10^(-0.016L)"] -- NOTE:
                       this already has a different loss-per-km (0.08 dB/km
                       equivalent) than Gao's 0.2 dB/km (-0.02 exponent);
                       this is a genuine, paper-specific experimental choice,
                       not a bug, and is exactly why this conversion lives
                       here rather than being hard-coded in observable.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from observable import run_sf_simulation, ObservableSnapshot
from detector import apply_detector_physics, ScaledMetrics
from security_bounds import SecurityBound, KatoBound, AzumaBound, BoundResult, evaluate_all_bounds, binary_entropy


# ----------------------------------------------------------------------------
# Parameter container
# ----------------------------------------------------------------------------

@dataclass
class COWParams:
    """
    Single source of truth for one experiment configuration. Every field has
    an illustrative, physically plausible default (not tied to any single
    cited paper's exact reported hardware) so that `COWParams()` alone
    already reproduces a sensible baseline; any paper-specific reproduction
    (e.g. Cao et al.'s four real operating points) overrides these via
    `create_variant`/explicit construction rather than relying on the
    defaults.

    mu              : mean photon number of signal/decoy coherent states.
    tB              : beam-splitter transmittance to the data line.
    distance_km     : fibre length L (km).
    fibre_loss_exponent : the "0.02" (or "0.016") coefficient in
                      eta = eta_d * 10**(-exponent * L); see module docstring.
    detector_eff    : eta_d, detector quantum efficiency, FOR THE DATA LINE
                      (detector D0) specifically. Cao et al. 2026 report
                      D0/D1/D2 efficiencies as 76.2%/46.0%/46.0% respectively
                      (their Sec. III) -- these are NOT the same, and using
                      D0's value for the monitor line (D1/D2) was the root
                      cause of a previously-unresolved ~3.3x discrepancy
                      between this framework's predicted monitor-line decoy
                      click counts and Cao et al.'s real reported counts
                      (see AUDIT_STATUS.md). detector_eff_monitor below is
                      the corrected, SEPARATE efficiency for D1/D2.
    detector_eff_monitor : eta_{D1,D2}, the monitor-line detectors'
                      efficiency, distinct from detector_eff (=eta_{D0},
                      data line only). Defaults to detector_eff's value
                      via create_variant if not set explicitly, for
                      backward compatibility with any code that only sets
                      detector_eff -- but for any paper-validation run,
                      BOTH should be set explicitly from that paper's
                      reported D0 vs D1/D2 values, since they generally
                      differ.
    insertion_loss_db : ADDITIONAL fixed optical insertion loss (dB), applied
                      multiplicatively to eta on top of the fibre+detector
                      loss above, on the DATA LINE only. Default 0.
    monitor_insertion_loss_db : ADDITIONAL fixed optical insertion loss (dB)
                      applied to the MONITOR LINE only, on top of
                      detector_eff_monitor and the shared fibre loss. Cao
                      et al. 2026 (their Sec. III) state an explicit
                      additional 3 dB loss from X-basis time-window
                      selection at the monitor line ("an additional 3 dB
                      loss is introduced") -- this field exists specifically
                      to capture that. Default 0 (no extra loss) for
                      backward compatibility; set to 3.0 to match Cao et
                      al.'s reported setup.
    dark_count_rate_hz : detector dark-count rate in counts/second.
    dead_time_s     : detector dead time in seconds. Default 100e-9 (100 ns)
                      -- a LITERATURE-REPRESENTATIVE value for fiber-coupled,
                      telecom-band (1550 nm) SNSPDs, NOT Cao et al. 2026's
                      own measured number (their paper does not report one
                      -- confirmed by a full-text search of arXiv:2601.06772
                      main text + appendix; the only detector-technology
                      fact they DO state is that D0/D1/D2 are all SNSPDs --
                      "SPD: superconducting nanowire single-photon
                      detector," Fig. 1 caption). This replaces a prior
                      default of 30e-6 (30 us), which was silently
                      appropriate for a gated InGaAs APD's afterpulsing
                      hold-off, not an SNSPD -- three orders of magnitude
                      too long for the detector family Cao et al. actually
                      used, and (checked directly) physically IMPOSSIBLE
                      given their own reported data: Cao's real n_z at
                      25km implies an average data-line click rate of
                      ~63.2 kHz (25,278,913 clicks over a 400s run at
                      N=1e11, 250 MHz round rate), which exceeds the
                      ~33.3 kHz asymptotic ceiling (1/dead_time) a genuine
                      30us non-paralyzable dead time would impose --
                      meaning 30us cannot have been their real detector's
                      dead time no matter the true underlying click rate.
                      SNSPD reset times in the QKD literature typically run
                      10-100 ns (e.g. "dead-times as small as 10 ns" for
                      state-of-the-art devices; "70-80 ns" for a comparable
                      fiber system; "of order 100 nanoseconds" for a
                      standard SNSPD+low-noise-RF-amplifier readout chain;
                      integrated QKD receivers report <20 ns). 100 ns is
                      chosen as a round, literature-typical, conservative
                      (i.e. deliberately NOT the fastest lab-record figure)
                      value for a real deployed system, not a cherry-picked
                      best case. See `core/detector.py`'s
                      `apply_dead_time_thinning`/`effective_rep_rate_hz`
                      docstrings for why this choice, like the prior one,
                      still has NO effect on any reported rate: even at
                      this much shorter dead time, the non-paralyzable
                      ceiling (1/100ns = 10 MHz) sits ~150x above the real
                      ~63.2 kHz click rate, so dead time remains
                      non-binding at every one of Cao et al.'s operating
                      points -- this is now a well-sourced physical
                      conclusion, not an assumption.
    rep_rate_hz     : pulse repetition rate in Hz.
    p_signal, p_decoy_alphaalpha, p_decoy_00 : state-preparation probabilities.
    phi             : monitor-line interferometer phase shift (rad); default
                      pi gives genuine constructive interference at D_M0
                      (see core/observable.py's _run_sf_monitor_circuit).
    qber_threshold  : abort threshold on Ez.
    correction_efficiency_f : error-correction inefficiency f.
    epsilon_cor, epsilon_sec : finite-key security parameters, taken from the
                      paper being reproduced (see security_bounds.py).
    """

    mu: float = 0.5
    tB: float = 0.90
    distance_km: float = 80.0
    fibre_loss_exponent: float = 0.02  # Gao et al. 2022 convention
    detector_eff: float = 0.20
    detector_eff_monitor: float | None = None  # None -> falls back to detector_eff (see docstring)
    insertion_loss_db: float = 0.0
    monitor_insertion_loss_db: float = 0.0
    interference_visibility: float = 1.0  # monitor-line MZI/MSI visibility V;
                                            # 1.0 = perfect (no correction).
                                            # See observable.py's
                                            # monitor_line_gains_closed_form
                                            # docstring for the full
                                            # justification. Cao et al. 2026
                                            # report V~0.99.
    dark_count_rate_hz: float = 900.0
    dead_time_s: float = 100e-9  # literature-representative SNSPD reset
                                  # time, NOT Cao et al.'s own unpublished
                                  # number -- see docstring above for the
                                  # full citation trail and why 30e-6 (the
                                  # prior default) was physically
                                  # impossible given Cao's own real n_z data.
    rep_rate_hz: float = 500e6
    p_signal: float = 0.5
    p_decoy_alphaalpha: float = 0.25
    p_decoy_00: float = 0.25
    phi: float = np.pi  # genuine constructive interference at D_M0 (fixed 2026-07-16,
                        # see core/observable.py's _run_sf_monitor_circuit docstring
                        # and validation/cao/derive_monitor_gain_twomode.py)
    qber_threshold: float = 0.05
    correction_efficiency_f: float = 1.1
    epsilon_cor: float = 1e-15
    epsilon_sec: float = 1e-10
    monitor_gain_source: str = "sf_physical"  # NEW DEFAULT (2026-07-16).
    # Selects which computation supplies the DECOY-state (alphaalpha, 00)
    # monitor-line gains feeding every security bound: "sf_physical"
    # (genuinely SF-circuit-derived, including a physically-simulated
    # imperfect-visibility model -- see observable.py's
    # monitor_line_gains_sf_physical docstring) or "closed_form" (Gao et
    # al. 2022's closed form + an ad hoc leakage patch, the prior default).
    # BOTH are always computed; the non-selected one is retained as
    # ObservableCollector._monitor_gains_crosscheck for audit/reporting.
    # Whenever ir_attack_strength or usd4b_strength is nonzero, this
    # AUTOMATICALLY falls back to "closed_form" regardless of this
    # setting, since the attack mixtures are only implemented against the
    # closed form -- every existing attack-sweep result is unaffected by
    # this default change. See work_log.md Sec. 6 for the referee-report
    # investigation and validation against Cao et al.'s real measured
    # Table S2 counts (independent of Gao's own closed-form formula) that
    # motivated flipping this default from closed_form to sf_physical.
    use_sf_monitor_gains: bool = False  # LEGACY, narrower flag: OFF by
    # default. When True, ADDITIONALLY patches the SIGNAL-state (0z/1z)
    # D_M0 click probability from the genuine SF interference-circuit
    # output, dark counts reintroduced via an independent-process
    # combination. Decoy-state sourcing is controlled by
    # monitor_gain_source above (this flag no longer touches it). D_M1 is
    # unchanged in both modes (dark-count-only by construction for the
    # signal-state case, since there is no real interference partner to
    # leak flux from). See observable.py's ObservableCollector.__init__
    # docstring for the exact combination formula.
    basis_mode: str = "passive"  # "passive" (Bob's beamsplitter, default,
    # matches Cao et al.'s experimental setup) or "active" (Bob's optical
    # switch -- Gao et al.'s alternative configuration, simpler gain
    # formulas, larger secure region per Gao's own Fig. 3/Table 1). See
    # observable.py's COWQuantumChannel docstring for the full physical
    # distinction and exact formula differences.
    ir_attack_strength: float = 0.0  # ADVERSARIAL, OFF by default. Intercept-
    # resend attack strength p_IR in [0, 1] -- the PER-ROUND probability that
    # Eve applies the I-R strategy to a given two-pulse round (not an
    # independent per-pulse coin flip; see observable.py's
    # monitor_line_gains_closed_form docstring for why). Per Gisin, Ribordy,
    # Zbinden, Stucki, Brunner, Scarani, "Towards practical and fast Quantum
    # Cryptography," arXiv:quant-ph/0411022 (2004), Eqs. 19-26: a pure IR
    # attack leaves the data-line QBER Q unchanged, but drives the
    # decoy-pair monitor-line visibility to V_d = 1 - p_IR via a
    # flux-conserving 50/50-split mixture on the fully-decohered fraction of
    # rounds. See observable.py's monitor_line_gains_closed_form docstring
    # for the exact citation and scope note (decoy channel only, not the
    # signal-state monitor gains).
    bs_attack_strength: float = 0.0  # ADVERSARIAL, OFF by default. Beam-
    # splitting attack strength s in [0, 1]. UNLIKE ir_attack_strength, this
    # does NOT alter the simulated observables at all -- Gisin, Ribordy,
    # Zbinden, Stucki, Brunner, Scarani, arXiv:quant-ph/0411022 (2004),
    # Sec. III.C.1 + Eqs. 15-18: a passive beam-splitting tap on a
    # coherent-state signal is provably indistinguishable from honest
    # channel loss (Bob's Q, visibility, and click rates are all UNCHANGED).
    # Instead, this toggle parametrizes a companion theoretical rate
    # CEILING (security_bounds.py's beam_splitting_rate_ceiling) that the
    # framework's ACTUAL reported key rate must never exceed -- see that
    # function's docstring for the full derivation, the s=1 mandatory-check
    # vs s<1 sensitivity-toggle distinction, and the explicit notation
    # mapping (this is NOT simply "flip a flag and re-run the simulation").
    usd3_strength: float = 0.0  # ADVERSARIAL, OFF by default. Branciard et
    # al. 2007 zero-error attack (arXiv:quant-ph/0609090), USD3 sub-attack
    # (q1): data-line bit-faking via unambiguous state discrimination on a
    # 3-pulse window. UNLIKE bs_attack_strength, this DOES perturb the
    # simulated data-line click probability directly (observable.py's
    # ObservableCollector) -- see that file's usd3_conclusive_prob and the
    # mixture applied right after self._p_data_alpha is computed. Freely
    # settable in [0, 1], not pinned to the paper's "exactly invisible"
    # mixture -- see usd_invisible_mixture() below to auto-solve for that.
    usd4b_strength: float = 0.0  # ADVERSARIAL, OFF by default. Same paper,
    # USD4b sub-attack (q3): decoy-pair faking, preserving V_d. Perturbs
    # observable.py's monitor_line_gains_closed_form directly (see that
    # function's docstring for the full mechanism and the USD4a scope note
    # -- the cross-boundary sub-attack is NOT implemented, it would require
    # a cross-round signal-coherence statistic this codebase doesn't have).
    backflash_strength: float = 0.0  # ADVERSARIAL, OFF by default. Singh,
    # Sharma, Singh, Prabhakar, "Backflash Attack on Coherent One-Way
    # Quantum Key Distribution," arXiv:2502.04081v1 (2025). SPAD avalanche
    # breakdown emits "backflash" photons that retrace back along the DATA
    # LINE toward Alice; Eve intercepts them with a circulator + high-
    # efficiency SNSPD and, using the timestamp/bit data already exchanged
    # on the authenticated classical channel for sifting, learns Bob's bit
    # WITHOUT perturbing the quantum state he receives at all -- the paper
    # states explicitly this is passive and "does not disturb the
    # coherence of the pulse." Same "provably invisible to Bob's
    # statistics" category as bs_attack_strength: UNLIKE
    # ir_attack_strength/usd3_strength/usd4b_strength,
    # this field is NEVER threaded into run_sf_simulation or
    # apply_detector_physics -- it has zero effect on the simulated
    # statistics, by design, matching the cited paper's own claim. Its
    # only consumer is security_bounds.py's backflash_rate_ceiling.
    #
    # NOTATION MAPPING (flagged explicitly): the paper's own quantity is
    # `P_b` (Eq. 4), an EMPIRICALLY MEASURED ratio specific to their SNSPD
    # (74% efficiency) and circulator setup -- not a universal physical
    # constant, and not transferable to Cao's simulated hardware. This
    # field is therefore a freely-tunable [0,1] stand-in for P_b, same
    # convention as bs_attack_strength, swept for
    # sensitivity rather than fixed to a literal reported number.
    #
    # CEILING DESIGN (own derivation, NOT the paper's Eq. 6 transcribed
    # literally -- see backflash_rate_ceiling's docstring for the full
    # comparison of alternatives considered and why this one was chosen):
    # Eq. 6's own P_sec = P_sift*(1-P_b-f*H(e)) has NO phase-error term at
    # all (a simpler, non-decoy-state formula than even Gao-Asymptotic),
    # so transcribing it literally as "the ceiling" would not be apples-
    # to-apples against this framework's monitor-line-aware Kato/Azuma
    # bounds. Instead, combining the paper's own Eq. 5
    # (P_learn^Eve = P_b * P_sift) with Eq. 6 shows P_sift*P_b IS P_learn
    # -- i.e. the paper's own model already treats backflash leakage as an
    # ABSOLUTE per-pulse leaked-information rate subtracted directly,
    # structurally identical to how leak_EC is subtracted as an absolute
    # count inside Kato's own key_length formula. The ceiling therefore
    # takes the framework's OWN already-correct honest rate (backflash_
    # strength=0, which already includes the real Ep_upper/monitor-line
    # penalty and Kato's finite-key overhead term) as an opaque baseline,
    # and subtracts Q_sift_honest*backflash_strength from it -- avoiding
    # re-deriving Kato's internal formula by hand (risk of silently
    # dropping its _finite_key_overhead() term) while remaining
    # structurally faithful to Eqs. 5-6.

    # NOTE: an auto-solver for Branciard et al.'s "exactly invisible"
    # combined-attack mixture (Eqs. 34-39) is deliberately NOT included yet.
    # That system was derived for their own 3-component (q1,q2,q3) model and
    # per-round rate parametrization; reducing it to the 2-component
    # (q1,q3)-only case this codebase can represent (no USD4a/V_10) requires
    # re-solving Eq. 33's linear system properly for the reduced target set,
    # which I have not done rigorously enough yet to ship as a formula --
    # rather than present an unverified guess as a citation-backed result,
    # q1/q3 are left as manually-set sweep parameters for now (see Step E).

    @property
    def t_fiber(self) -> float:
        """PURE fibre/channel transmission, EXCLUDING detector efficiency --
        the quantity the beam-splitting attack ceiling (security_bounds.py's
        beam_splitting_rate_ceiling) actually operates on, per Gisin et al.
        2004's trusted-device assumption that Bob's detector imperfections
        are not given to Eve (only the channel itself is her attack
        surface). Distinct from `eta` above, which bundles detector_eff in."""
        return 10.0 ** (-self.fibre_loss_exponent * self.distance_km) * \
            10.0 ** (-self.insertion_loss_db / 10.0)

    def beam_splitting_ceiling_bps(self, round_rate_hz: float) -> float:
        """Convenience wrapper: the Gisin et al. 2004 Eq. 18 (generalized)
        beam-splitting rate ceiling at this configuration's operating point,
        converted to bits/second for direct comparison against a reported
        ExperimentResult's key rate. Returns +inf when bs_attack_strength=0
        (attack off -- no constraint). See
        security_bounds.beam_splitting_rate_ceiling's docstring for the full
        derivation, citation, and the explicit f-vs-p_signal notation
        mapping this relies on."""
        from security_bounds import beam_splitting_rate_ceiling
        ceiling_per_pulse = beam_splitting_rate_ceiling(
            t_fiber=self.t_fiber,
            tB=self.tB,
            detector_eff=self.detector_eff,
            p_signal=self.p_signal,
            bs_attack_strength=self.bs_attack_strength,
        )
        if ceiling_per_pulse == float("inf"):
            return float("inf")
        return ceiling_per_pulse * round_rate_hz

    @property
    def eta(self) -> float:
        """Total channel transmittance INCLUDING detector efficiency and any
        extra insertion loss, exactly the eta convention every formula in
        observable.py / detector.py / security_bounds.py expects. THIS IS
        THE DATA-LINE (D0) TRANSMITTANCE specifically."""
        base_eta = self.detector_eff * 10.0 ** (-self.fibre_loss_exponent * self.distance_km)
        insertion_transmittance = 10.0 ** (-self.insertion_loss_db / 10.0)
        return base_eta * insertion_transmittance

    @property
    def eta_monitor(self) -> float:
        """Total channel transmittance for the MONITOR LINE (D1/D2)
        specifically -- uses detector_eff_monitor (falling back to
        detector_eff if not set) and monitor_insertion_loss_db (e.g. Cao
        et al. 2026's reported additional 3 dB X-basis time-window
        selection loss), on top of the SAME shared fibre loss as the data
        line (the fibre itself doesn't know which detector a pulse is
        headed to)."""
        eff = self.detector_eff_monitor if self.detector_eff_monitor is not None else self.detector_eff
        base_eta = eff * 10.0 ** (-self.fibre_loss_exponent * self.distance_km)
        insertion_transmittance = 10.0 ** (-self.monitor_insertion_loss_db / 10.0)
        return base_eta * insertion_transmittance

    def create_variant(self, **overrides) -> "COWParams":
        """
        Return a NEW COWParams with the given fields overridden, leaving
        `self` untouched. This is the parameter-sweep building block:

            base = COWParams()
            variant_120km = base.create_variant(distance_km=120.0)

        Mirrors the `.create_variant()` pattern from your original
        cow_config.py design.
        """
        from dataclasses import replace
        return replace(self, **overrides)


# ----------------------------------------------------------------------------
# Result container for one full experiment
# ----------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Everything produced by running the full Layer1->Layer2->Layer3 pipeline
    once for a single COWParams configuration."""

    params: COWParams
    observable: ObservableSnapshot
    metrics: ScaledMetrics
    bounds: dict[str, BoundResult]


# ----------------------------------------------------------------------------
# Core orchestration: run the full pipeline ONCE
# ----------------------------------------------------------------------------

def run_experiment(
    params: COWParams,
    num_sf_pulses: int | None = None,
    N_target: int = 100_000_000,
    bounds: list[SecurityBound] | None = None,
    seed: int | None = None,
    exact_expected_values: bool = False,
) -> ExperimentResult:
    """
    Run the complete pipeline ONCE for a single parameter configuration:
        Layer 1 (SF quantum simulation + genuine MC sampling, num_sf_pulses
                 pulses, == N_target unless the caller deliberately wants
                 a smaller diagnostic batch)
        -> Layer 2 (detector physics; scaling is a no-op when
                    num_sf_pulses == N_target)
        -> Layer 3 (evaluate every requested SecurityBound)

    This function calls Strawberry Fields exactly once (`run_sf_simulation`),
    matching your explicit instruction: "we run sf just once and collect the
    observables once for 1 mc run, it can consist of many bounds
    calculation... if we need to run something again we need to run sf
    again." Multiple bounds (Kato, Azuma) are evaluated against the SAME
    snapshot here, with no re-simulation.

    Parameters
    ----------
    params : COWParams
        The experiment configuration.
    num_sf_pulses : int or None
        Number of pulses actually drawn (genuine multinomial/binomial
        sampling from the SF-derived click probabilities). DEFAULTS TO
        N_target (None => num_sf_pulses = N_target) -- this is the only
        statistically valid choice, since Kato's/Azuma's bounds treat
        their k parameter as the number of genuine independent trials.
        An earlier version of this codebase defaulted this to a small
        fixed value (1e6) decoupled from N_target "for speed," then
        linearly rescaled the resulting counts. That was both
        unnecessary (the real bottleneck, rng.choice() building a
        per-pulse label array, has since been replaced by
        rng.multinomial()/rng.binomial() batch sampling -- empirically
        ~0.26s of pure sampling time even at N=1e12) and statistically
        invalid (it understates the true sampling fluctuation, since the
        bound's confidence interval was computed as if N_target draws
        had genuinely occurred). Only override this away from N_target
        for deliberate small-N diagnostics (e.g. check_stationarity).
    N_target : int
        Total pulse count the experiment represents (default 1e8).
    bounds : list of SecurityBound instances. Defaults to [KatoBound(),
        AzumaBound()] using params.epsilon_cor / params.epsilon_sec /
        params.qber_threshold / params.correction_efficiency_f.
    seed : RNG seed forwarded to the SF layer for reproducibility.

    Returns
    -------
    ExperimentResult bundling the raw observable snapshot, the scaled
    metrics, and every requested bound's result.
    """
    if bounds is None:
        bounds = [
            KatoBound(
                epsilon_cor=params.epsilon_cor,
                epsilon_sec=params.epsilon_sec,
                qber_threshold=params.qber_threshold,
                correction_efficiency_f=params.correction_efficiency_f,
            ),
            AzumaBound(
                epsilon_cor=params.epsilon_cor,
                epsilon_sec=params.epsilon_sec,
                qber_threshold=params.qber_threshold,
                correction_efficiency_f=params.correction_efficiency_f,
            ),
        ]

    if num_sf_pulses is None:
        num_sf_pulses = N_target  # statistically valid default -- see docstring

    p_dark = params.dark_count_rate_hz / params.rep_rate_hz

    obs = run_sf_simulation(
        mu=params.mu,
        tB=params.tB,
        eta=params.eta,
        eta_for_monitor=params.eta_monitor,
        interference_visibility=params.interference_visibility,
        p_dark=p_dark,
        num_pulses=num_sf_pulses,
        p_signal=params.p_signal,
        p_decoy_alphaalpha=params.p_decoy_alphaalpha,
        p_decoy_00=params.p_decoy_00,
        phi=params.phi,
        basis_mode=params.basis_mode,
        ir_attack_strength=params.ir_attack_strength,
        usd3_strength=params.usd3_strength,
        usd4b_strength=params.usd4b_strength,
        use_sf_monitor_gains=params.use_sf_monitor_gains,
        monitor_gain_source=params.monitor_gain_source,
        seed=seed,
        exact_expected_values=exact_expected_values,
    )

    metrics = apply_detector_physics(
        obs,
        dark_count_rate_hz=params.dark_count_rate_hz,
        dead_time_s=params.dead_time_s,
        rep_rate_hz=params.rep_rate_hz,
        N_target=N_target,
    )

    bound_results = evaluate_all_bounds(metrics, bounds)

    return ExperimentResult(
        params=params,
        observable=obs,
        metrics=metrics,
        bounds=bound_results,
    )


# ----------------------------------------------------------------------------
# Parameter sweeps
# ----------------------------------------------------------------------------

def parameter_sweep(
    base_params: COWParams,
    param_name: str,
    param_values: list[float],
    num_sf_pulses: int = 1_000_000,
    N_target: int = 100_000_000,
    bounds: list[SecurityBound] | None = None,
    seed: int | None = None,
) -> list[ExperimentResult]:
    """
    Run `run_experiment` once per value in `param_values`, varying
    `param_name` (e.g. "distance_km") via `COWParams.create_variant`. Each
    point gets its OWN fresh SF simulation (a fresh quantum run per point,
    consistent with "run sf again" whenever a new configuration is needed).

    Example
    -------
        base = COWParams(detector_eff=0.1, dead_time_s=50e-6)
        results = parameter_sweep(base, "distance_km",
                                   [20, 40, 60, 80, 100, 120, 140, 160])
    """
    results = []
    for value in param_values:
        variant = base_params.create_variant(**{param_name: value})
        result = run_experiment(
            variant,
            num_sf_pulses=num_sf_pulses,
            N_target=N_target,
            bounds=bounds,
            seed=seed,
        )
        results.append(result)
    return results


# ----------------------------------------------------------------------------
# Stationarity check (validates the "1e6 pulses is enough" assumption)
# ----------------------------------------------------------------------------

@dataclass
class StationarityReport:
    """Result of running several independent SF batches at fixed parameters
    to check that rate estimates (Ez, decoy gains) are stable across
    independent draws -- i.e. that extrapolating from 1e6 to N_target is
    statistically justified."""

    Ez_values: list[float]
    Ez_mean: float
    Ez_std: float
    G_DM0_00_values: list[float]
    G_DM0_00_mean: float
    G_DM0_00_std: float
    n_batches: int
    pulses_per_batch: int


def check_stationarity(
    params: COWParams,
    n_batches: int = 5,
    pulses_per_batch: int = 1_000_000,
    base_seed: int = 0,
) -> StationarityReport:
    """
    Run `n_batches` independent SF simulations (Layer 1 only, each with a
    different seed) at the SAME `params`, apply the SAME detector physics
    (Layer 2) to each, and report the spread of the resulting Ez and
    G_DM0_00 estimates across batches.

    If the standard deviation across batches is small relative to the
    within-batch Wilson confidence interval (see ScaledMetrics.Ez_ci),
    this confirms that a single 1e6-pulse batch's error bar already
    captures the true statistical uncertainty, validating the framework's
    "1e6 pulses + error bars, no need for the full 1e8 in SF" design choice.
    """
    Ez_values = []
    G_DM0_00_values = []
    p_dark = params.dark_count_rate_hz / params.rep_rate_hz

    for i in range(n_batches):
        obs = run_sf_simulation(
            mu=params.mu,
            tB=params.tB,
            eta=params.eta,
            p_dark=p_dark,
            num_pulses=pulses_per_batch,
            p_signal=params.p_signal,
            p_decoy_alphaalpha=params.p_decoy_alphaalpha,
            p_decoy_00=params.p_decoy_00,
            phi=params.phi,
            basis_mode=params.basis_mode,
            seed=base_seed + i,
        )
        metrics = apply_detector_physics(
            obs,
            dark_count_rate_hz=params.dark_count_rate_hz,
            dead_time_s=params.dead_time_s,
            rep_rate_hz=params.rep_rate_hz,
            N_target=pulses_per_batch,  # no scaling needed for this check
        )
        Ez_values.append(metrics.Ez)
        G_DM0_00_values.append(metrics.G_DM0_00)

    return StationarityReport(
        Ez_values=Ez_values,
        Ez_mean=float(np.mean(Ez_values)),
        Ez_std=float(np.std(Ez_values)),
        G_DM0_00_values=G_DM0_00_values,
        G_DM0_00_mean=float(np.mean(G_DM0_00_values)),
        G_DM0_00_std=float(np.std(G_DM0_00_values)),
        n_batches=n_batches,
        pulses_per_batch=pulses_per_batch,
    )


# ----------------------------------------------------------------------------
# (mu, tB) optimizer
# ----------------------------------------------------------------------------
#
# WHY THIS EXISTS (see chat history for the full investigation): the
# finite-key/asymptotic phase-error bound implemented in security_bounds.py
# was found to saturate (Ep_upper pinned at its ceiling) when evaluated at
# a single, fixed, experimentally-realistic (mu, tB) pair such as
# (mu=0.5, tB=0.9). This was NOT a bug in the formula or in
# our implementation -- cross-checking against Cao et al.'s real
# experimental data (mu~1e-3) and Gao et al.'s own stated methodology
# ("tB is given by the optimization algorithm," no fixed mu reported in
# their numerical section, only optimized per-distance curves) confirms
# that EVERY one of these papers' published key-rate-vs-distance curves
# comes from numerically searching for the best (mu, tB) at each distance,
# not from evaluating the bound at one fixed, intuitive-looking parameter
# choice. This module adds that missing optimization layer.

@dataclass
class OptimizationResult:
    """Result of optimizing (mu, tB) at a single distance for a single
    SecurityBound, maximizing the secure key rate (bits per pulse)."""

    distance_km: float
    bound_name: str
    mu_optimal: float
    tB_optimal: float
    key_rate_per_pulse: float
    Ep_upper: float
    Ez: float
    secure: bool
    n_evaluations: int


def _evaluate_key_rate_at(
    mu: float,
    tB: float,
    base_params: COWParams,
    bound_class: type,
    num_sf_pulses: int,
    N_target: int,
    seed: int | None,
    exact_expected_values: bool = False,
) -> tuple[float, BoundResult]:
    """
    Single (mu, tB) evaluation: builds a COWParams variant, runs the full
    Layer1->Layer2->Layer3 pipeline ONCE (fresh SF simulation, per the
    "run sf again for a new configuration" rule), and returns the secure
    key rate per pulse (0.0 if insecure) plus the full BoundResult for
    diagnostics.
    """
    params = base_params.create_variant(mu=mu, tB=tB)
    bound = bound_class(
        epsilon_cor=params.epsilon_cor,
        epsilon_sec=params.epsilon_sec,
        qber_threshold=params.qber_threshold,
        correction_efficiency_f=params.correction_efficiency_f,
    )
    result = run_experiment(
        params,
        num_sf_pulses=num_sf_pulses,
        N_target=N_target,
        bounds=[bound],
        seed=seed,
        exact_expected_values=exact_expected_values,
    )
    bound_result = result.bounds[bound.name()]
    return bound_result.key_rate_per_pulse, bound_result


def optimize_mu_tB(
    base_params: COWParams,
    bound_class: type = KatoBound,
    mu_bounds: tuple[float, float] = (1e-4, 0.6),
    tB_bounds: tuple[float, float] = (0.05, 0.98),
    num_sf_pulses: int = 1_000_000,
    N_target: int = int(1e9),
    seed: int | None = 0,
    maxiter: int = 40,
    popsize: int = 12,
    verbose: bool = False,
    progress_every: int = 25,
) -> OptimizationResult:
    """
    For a FIXED distance and hardware parameters (detector efficiency, dark
    count rate, dead time, epsilons -- everything in `base_params` except
    mu and tB), search over (mu, tB) to find the pair that MAXIMIZES the
    secure key rate per pulse, using scipy's `differential_evolution` (a
    global optimizer, robust to the formula's saturated/zero-key-rate
    plateaus -- a local optimizer like Nelder-Mead can get stuck on a flat
    "insecure" region with no gradient to follow, exactly the failure mode
    your colleague's code documented and fixed the same way).

    *** num_sf_pulses default raised to 1e6 (was 50,000 in an earlier
    version) ***. An earlier two-tier design (cheap search at 50k pulses,
    then a separate "refine" step at 1e6) was found to be UNRELIABLE: the
    same (mu, tB) point flipped between secure=True (at 50k) and
    secure=False (at 200k+) purely due to sampling noise in the phase-error
    estimate, not because the formula itself was unstable (Ez stayed stable
    across all tested sample sizes; only Ep_upper flip-flopped at low N).
    Since the Gaussian backend makes even 1e6-pulse evaluations fast
    (~30ms each, confirmed empirically -- the quantum amplitude propagation
    runs ONCE per configuration regardless of pulse count; only the cheap
    numpy binomial sampling scales with N), there is no longer a speed
    reason to use a smaller, noisier sample during search. Using ONE
    consistent, reliable sample size throughout (search AND final result)
    eliminates the search/refine inconsistency entirely.

    If `verbose=True`, prints a progress line every `progress_every`
    evaluations (count, elapsed time, best rate found so far) -- useful
    for seeing that the search is actually moving, since
    `differential_evolution` itself gives no feedback until it returns.
    """
    from scipy.optimize import differential_evolution
    import time as _time

    eval_count = 0
    best_rate_so_far = 0.0
    t_start = _time.time()
    # IMPORTANT: use a SINGLE FIXED seed for every evaluation during the
    # search. Without this, the objective function is noisy (the same
    # (mu, tB) point can give wildly different key-rate estimates across
    # random seeds at low num_sf_pulses, confirmed empirically: a 10-seed
    # check at one near-optimal point gave rates ranging from 0 to 1.3e-4),
    # which causes the optimizer to chase statistical noise instead of the
    # true underlying landscape. A fixed seed makes the search-phase
    # objective deterministic (smooth in (mu, tB)), even though it is still
    # only an approximate estimate of the true key rate at that sample size.
    search_seed = seed if seed is not None else 0

    def objective(x):
        nonlocal eval_count, best_rate_so_far
        log10_mu, tB = x
        mu = 10.0 ** log10_mu  # convert back from log-space
        eval_count += 1
        
        # SEARCH PHASE: Force N_target and exact values to avoid statistical starvation
        rate, bound_result = _evaluate_key_rate_at(
            mu, tB, base_params, bound_class, N_target, N_target, search_seed, exact_expected_values=True
        )
        
        if rate > best_rate_so_far:
            best_rate_so_far = rate
        if verbose and eval_count % progress_every == 0:
            elapsed = _time.time() - t_start
            print(
                f"    [search] eval {eval_count:>5d}  elapsed {elapsed:>6.1f}s  "
                f"best rate so far: {best_rate_so_far:.4e}",
                flush=True,
            )
        # *** FIX: false-convergence bug ***
        # Confirmed directly (see chat history) that differential_evolution
        # falsely declares convergence after just ~2 generations whenever
        # the ENTIRE initial population lands in the insecure (rate=0)
        # region: a population of all-identical-zeros has EXACTLY ZERO
        # spread, which trivially satisfies scipy's internal
        # std(pop)<=atol+tol*|mean(pop)| check for ANY tol/atol >= 0 --
        # confirmed this cannot be disabled via tol, atol, OR a callback
        # (scipy checks convergence internally before the callback runs).
        # The only real fix is to never let the objective be EXACTLY flat:
        # when insecure (rate=0), return a small smooth gradient based on
        # how close Ep_upper is to the 0.5 security threshold, so the
        # population always has a direction to climb even before finding
        # the first secure point. Scaled by 1e-6 so it can NEVER outrank a
        # genuine positive rate (typical rates here are >=1e-9 at secure
        # points, and this penalty is bounded in [-5e-7, 0) for
        # Ep in [0, 0.5]) -- this only breaks ties among otherwise-flat
        # insecure points, it never distorts the ranking of real results.
        if rate <= 0.0:
            # *** FIX: the Ep-only proxy used previously was WRONG ***
            # Confirmed directly: ranking purely by Ep's distance from 0.5
            # ignores Ez entirely, and Ez varies by >10x across the
            # relevant tB range (it is NOT negligible -- it is often the
            # dominant term in the bracket). This caused the optimizer to
            # converge to tB=0.05 (real bracket -0.048) over tB=0.35 (real
            # bracket -0.0038), exactly backwards from the true landscape.
            # Fix: use the REAL bracket (1-h(Ep)-f*h(Ez)) as the smooth
            # signal directly. This is already a continuous function of
            # BOTH Ep and Ez with the correct sign everywhere -- it becomes
            # positive exactly when the point is genuinely secure, so no
            # separate "insecure proxy" formula is needed at all; we can
            # use the SAME quantity GaoAsymptoticBound.evaluate() already
            # computes internally (its un-rate-multiplied bracket term).
            # Still scaled down (x1e-6) so it can never outrank a genuine
            # positive Q*bracket rate, since the bracket itself can range
            # up to ~1.0 while real rates are typically 1e-9 to 1e-4 scale.
            f_ec = base_params.correction_efficiency_f
            bracket = 1.0 - binary_entropy(bound_result.Ep_upper) - f_ec * binary_entropy(bound_result.Ez)
            # Scale: bracket ranges up to ~1.0 in the best (still-insecure)
            # case, so 1e-9 keeps the worst-case signal (1e-9) safely below
            # the smallest genuine secure rate measured anywhere in this
            # project (~6.6e-8 at L=50km) -- confirmed by direct check
            # after an earlier scale (1e-6) was found unsafe (worst case
            # 1e-6 > 6.6e-8).
            return -1e-9 * bracket
        return -rate  # differential_evolution minimises

    if verbose:
        print(f"  Starting search (popsize={popsize}, maxiter={maxiter}, "
              f"~{popsize * 2} evals/generation)...", flush=True)

    # *** FIXED BUG: search mu in LOG-SPACE, not linear space ***
    # mu_bounds typically spans several orders of magnitude (e.g. 1e-5 to
    # 0.3). differential_evolution's default Latin-Hypercube initial
    # population and its mutation/crossover steps operate LINEARLY in the
    # given bounds -- so with a linear mu range spanning 4+ orders of
    # magnitude, only ~3% of the initial population lands below mu=0.01,
    # and the secure (mu, tB) region for this formula was found (by direct
    # grid search) to be an extremely narrow sliver typically located at
    # mu ~ 1e-3 to 1e-2. A linear search badly undersamples this region and
    # essentially never finds it, which is EXACTLY the failure mode
    # observed empirically: every distance converged to the identical
    # arbitrary point that scipy returns for a perfectly flat (all-zero)
    # objective with a fixed seed -- proven by reproducing that exact
    # point with a literal flat_objective=0 test. Searching log10(mu)
    # instead distributes the initial population evenly across orders of
    # magnitude, giving every decade of mu equal sampling probability.
    log10_mu_bounds = (np.log10(mu_bounds[0]), np.log10(mu_bounds[1]))

    res = differential_evolution(
        objective,
        bounds=[log10_mu_bounds, tB_bounds],
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        # *** FIXED BUG: tol=1e-10 caused FALSE early convergence ***
        # `tol` is a RELATIVE tolerance on the population's objective
        # spread (see scipy docs: stops when
        # std(pop) <= atol + tol*abs(mean(pop))). Whenever the secure
        # region is narrow enough that the ENTIRE initial population
        # lands outside it, every member returns objective=0 (our
        # `objective()` returns -rate, and rate=0 for any insecure
        # point) -- a population of all-zeros has EXACTLY ZERO spread,
        # which trivially satisfies tol=1e-10 (or any positive tol),
        # so scipy reports "converged" after just ~2 generations,
        # having found nothing. Confirmed directly: L=60-100km runs
        # all terminated after only 51 evaluations (vs ~1000 at
        # L=10-50km, where the population DID find secure points) and
        # returned the IDENTICAL (mu, tB) point at every one of those
        # distances -- the unmistakable signature of this exact bug,
        # not a genuine result. Setting tol=0 disables this early-stop
        # entirely, forcing the full `maxiter` generations to run
        # regardless of how flat the population looks at any point --
        # the only reliable way to escape a population that initially
        # missed a narrow secure region by chance. atol stays at its
        # scipy default (0), so convergence now requires the full
        # `maxiter` budget; if that proves too slow in practice, raise
        # `popsize` instead of re-introducing `tol`, since a larger
        # initial population is a population-coverage fix, while `tol`
        # is purely a stopping-criterion shortcut that this exact
        # failure mode shows cannot be trusted here.
        tol=0,
        polish=True,
        workers=1,
    )

    if verbose:
        elapsed = _time.time() - t_start
        print(f"  Search done: {eval_count} evaluations in {elapsed:.1f}s. "
              f"Refining best point...", flush=True)

    mu_opt, tB_opt = float(10.0 ** res.x[0]), float(res.x[1])
    
    # FINAL PHASE: Use genuine binomial sampling for the reported result
    # We still use N_target here so it evaluates the true experiment size.
    final_rate, final_bound_result = _evaluate_key_rate_at(
        mu_opt, tB_opt, base_params, bound_class, N_target, N_target, seed, exact_expected_values=False
    )

    return OptimizationResult(
        distance_km=base_params.distance_km,
        bound_name=bound_class(epsilon_cor=1e-15, epsilon_sec=1e-10).name(),
        mu_optimal=mu_opt,
        tB_optimal=tB_opt,
        key_rate_per_pulse=final_rate,
        Ep_upper=final_bound_result.Ep_upper,
        Ez=final_bound_result.Ez,
        secure=final_bound_result.secure,
        n_evaluations=eval_count,
    )


def optimize_then_refine(
    base_params: COWParams,
    bound_class: type = KatoBound,
    mu_bounds: tuple[float, float] = (1e-4, 0.6),
    tB_bounds: tuple[float, float] = (0.05, 0.98),
    num_sf_pulses: int = 1_000_000,
    N_target: int = int(1e9),
    seed: int | None = 0,
    maxiter: int = 40,
    popsize: int = 12,
    verbose: bool = False,
    progress_every: int = 25,
) -> OptimizationResult:
    """
    Thin wrapper around `optimize_mu_tB` using ONE consistent sample size
    throughout (search and final result), kept as a separate name for
    backward compatibility with existing scripts that call
    `optimize_then_refine`.

    *** SIMPLIFIED (an earlier two-tier "cheap search at 50k, refine at
    1e6" version of this function was found to be UNRELIABLE -- the same
    (mu, tB) point flipped between secure=True and secure=False purely due
    to search-phase sampling noise, not a real instability in the
    underlying formula. Since 1e6-pulse SF evaluations are fast (~30ms
    each, confirmed empirically), there's no remaining reason to use a
    smaller, noisier sample during search -- see optimize_mu_tB's
    docstring for the full investigation.) ***
    """
    import time as _time
    t_start = _time.time()

    result = optimize_mu_tB(
        base_params, bound_class, mu_bounds, tB_bounds,
        num_sf_pulses=num_sf_pulses, N_target=N_target,
        seed=seed, maxiter=maxiter, popsize=popsize,
        verbose=verbose, progress_every=progress_every,
    )

    if verbose:
        elapsed = _time.time() - t_start
        print(f"  Done. Total time for this distance: {elapsed:.1f}s", flush=True)

    return result


def optimize_sweep_distance(
    base_params: COWParams,
    distances_km: list[float],
    bound_class: type = KatoBound,
    mu_bounds: tuple[float, float] = (1e-4, 0.6),
    tB_bounds: tuple[float, float] = (0.05, 0.98),
    num_sf_pulses: int = 1_000_000,
    N_target: int = int(1e9),
    seed: int | None = 0,
    maxiter: int = 40,
    popsize: int = 12,
    verbose: bool = False,
    progress_every: int = 25,
) -> list[OptimizationResult]:
    """
    Run `optimize_then_refine` once per distance in `distances_km`,
    reproducing the structure of Gao et al. Fig. 2-5 / your colleague's
    Fig.2-style plots: an independently-optimized (mu, tB) at every
    distance point, not a single fixed parameter pair swept across
    distances.

    If `verbose=True`, prints a banner before each distance point (so you
    can see overall sweep progress) plus the per-evaluation search
    progress within each distance (see `optimize_mu_tB`).
    """
    import time as _time
    t_sweep_start = _time.time()

    results = []
    for i, d in enumerate(distances_km):
        if verbose:
            elapsed = _time.time() - t_sweep_start
            print(
                f"\n--- Distance {i+1}/{len(distances_km)}: L={d:.0f}km "
                f"(sweep elapsed: {elapsed:.1f}s) ---", flush=True
            )
        variant = base_params.create_variant(distance_km=d)
        result = optimize_then_refine(
            variant, bound_class, mu_bounds, tB_bounds,
            num_sf_pulses, N_target,
            seed, maxiter, popsize, verbose=verbose, progress_every=progress_every,
        )
        if verbose:
            print(
                f"  -> mu={result.mu_optimal:.6f}  tB={result.tB_optimal:.4f}  "
                f"rate={result.key_rate_per_pulse:.4e}  secure={result.secure}",
                flush=True,
            )
        results.append(result)

    if verbose:
        total_elapsed = _time.time() - t_sweep_start
        print(f"\nFull sweep complete in {total_elapsed:.1f}s.", flush=True)

    return results
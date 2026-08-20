# ML-QEM Pipeline Peer Review — grover_v4

## 1. Feasibility Verdict

**Yes, conditionally feasible this week.** The architecture is mathematically sound — a residual correction `noisy + tanh(Δ)*0.5` can learn error deltas. The blocker is not the model; it's the **degenerate training signal from Tower B**.

---

## 2. Root-Cause Diagnosis

The zero-gradient / mode-collapse is caused by **two compounding issues**:

- **Static Choi → constant Tower B output.** `X_hardware_t` is `hw_tensor.repeat(500, 1)` ([Cell 7, line 461](file:///d:/Academic%20UOP/Internship/NMR/grover's%20v4/grover_v4%20(1).ipynb)). Every row is identical. Tower B's `Linear(576→8) → Tanh()` maps all 500 samples to the **same 8-D embedding**. Backprop sees zero variance → `∂L/∂W_B ≈ 0` → Tower B weights freeze, fusing a **constant bias** into the fusion layer. The model collapses to Tower-A-only regression.
- **Naive Gaussian noise is insufficient.** The current `x_hw + randn * 0.01` perturbation ([Cell 8, line 557](file:///d:/Academic%20UOP/Internship/NMR/grover's%20v4/grover_v4%20(1).ipynb)) adds i.i.d. noise that **violates CPTP** — perturbed Choi matrices are no longer positive-semidefinite or trace-preserving. The network learns to ignore unphysical noise → Tower B remains dead.

### Evidence from `inference_statistics.json`
- `std_raw_error: 0.0` and `std_mitigated_error: 0.0` confirm the IBM batch job returned **identical** expectation values across 20 runs (EstimatorV2 determinism), so the model always predicts the same correction. The reported "96.8% improvement" is an artifact of the model's constant correction coincidentally being close to the ideal for **one** circuit — it will not generalize.

---

## 3. Lowest-Code CPTP-Preserving Fix

### Strategy: **Depolarizing-channel perturbation of the physical Choi matrix**

Instead of adding raw Gaussian noise, **mix the measured Choi with the maximally-mixed channel** at varying strengths per training sample. This is guaranteed CPTP.

For a $d$-dimensional gate (d=2 for 1Q, d=4 for 2Q), the Choi matrix of the depolarizing channel is $\mathcal{J}_{\text{dep}} = \frac{1}{d} I_{d^2}$. A CPTP perturbation is:

$$\mathcal{J}_{\text{perturbed}} = (1 - p) \cdot \mathcal{J}_{\text{physical}} + p \cdot \frac{I_{d^2}}{d}$$

where $p \sim \text{Uniform}(0, p_{\max})$ with $p_{\max} \in [0.05, 0.20]$.

### Implementation — replace the `forward` perturbation with a **pre-training data augmentation** in Cell 7:

```python
# === INSERT AFTER line 460 (hw_tensor = ...) ===

def cptp_perturb_choi(hw_vec, p, gate_dims=[(4, 32), (4, 32), (16, 512)]):
    """Apply depolarizing mixing to each gate's Choi sub-vector.
    gate_dims: list of (d^2, num_real_imag_features) for H, X, CZ gates."""
    perturbed = hw_vec.copy()
    offset = 0
    for d_sq, n_feat in gate_dims:
        d = int(np.sqrt(d_sq))
        half = n_feat // 2  # split real / imag
        re = perturbed[offset : offset + half]
        im = perturbed[offset + half : offset + n_feat]
        # Choi of identity channel (maximally mixed): I/d, flattened
        choi_id_re = np.eye(d_sq).flatten() / d
        choi_id_im = np.zeros_like(choi_id_re)
        re[:] = (1 - p) * re + p * choi_id_re
        im[:] = (1 - p) * im + p * choi_id_im
        offset += n_feat
    return perturbed

# Build augmented hardware tensor with per-sample drift
P_MAX = 0.15
hw_list = []
for i in range(NUM_CIRCUITS):
    p = np.random.uniform(0, P_MAX)
    hw_list.append(cptp_perturb_choi(hardware_fingerprint, p))
X_hardware_t = torch.tensor(np.array(hw_list), dtype=torch.float32)
```

### Also remove the unphysical `forward` noise (Cell 8, line 556-557):

```diff
     def forward(self, x_circ, x_hw):
-        # INJECT SYNTHETIC DRIFT to prevent mode collapse on static Tomography data
-        if self.training:
-            x_hw = x_hw + torch.randn_like(x_hw) * 0.01
-
         circ_feats = self.circ_tower(x_circ)
```

---

## 4. Additional Quick Wins (≤5 lines each)

| Fix | Rationale |
|---|---|
| **Increase `P_MAX` to 0.20 for NMR** | SpinQ Gemini drift is higher; matches physical noise budget |
| **Add L2 auxiliary loss on Tower B embedding variance** `+ 0.01 * (1.0 / (hw_feats.var(dim=0).mean() + 1e-6))` | Forces Tower B to produce diverse embeddings across samples |
| **Use `shots=1024` in EstimatorV2** to get shot-noise variance across inference runs | Current `std = 0.0` makes Wilcoxon impossible; shot noise adds the stochastic variance needed for statistical testing |
| **Increase `NUM_EVAL_RUNS` to ≥30** | 20 is underpowered for Wilcoxon at the expected effect size |

---

## 5. Expected Outcome After Fix

With CPTP-augmented training data:
- Tower B gradients unfreeze → learns to map different noise levels to different embeddings
- Fusion layer gets a **noise-strength signal** it can correlate with circuit features
- Wilcoxon $p$-value should drop below 0.05 given the model can now discriminate noise regimes
- Cross-platform comparison with NMR next week becomes meaningful since the same Choi-based pipeline works on both platforms with only `P_MAX` tuning

> [!IMPORTANT]
> The `inference_statistics.json` showing `std_raw_error: 0.0` means you ran EstimatorV2 **without shot noise** (or identical seeding). You **must** add `shots=1024` or similar to get non-degenerate inference statistics for the paper's Wilcoxon test.

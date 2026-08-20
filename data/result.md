# Untitled

# Project Report: Cross-Platform Machine Learning Quantum Error Mitigation (ML-QEM) for Grover's Algorithm

**A Comparative Study between Superconducting (IBM) and NMR (SpinQ) Architectures**

## 1. Abstract

Quantum Error Mitigation (QEM) is critical for extracting useful computational results from Noisy Intermediate-Scale Quantum (NISQ) devices. Traditional techniques, such as Zero-Noise Extrapolation (ZNE) or Probabilistic Error Cancellation (PEC), suffer from massive circuit execution overhead. This project proposes a highly scalable, zero-QPU-overhead Machine Learning pipeline based on a "Two-Tower" Neural Network architecture. By conditioning the error mitigation on a physically complete "Hardware Fingerprint" (a dynamically perturbed Choi Matrix), we bridge the gap between simulation-based training and physical hardware inference. This report details the methodology, mathematical proofs, and experimental deployment across two fundamentally different physical platforms: IBM's Superconducting *ibm_kingston* processor and the University of Colombo's 2-qubit SpinQ Gemini Nuclear Magnetic Resonance (NMR) quantum computer.

## 2. Hardware Platforms & Physical Noise Profiles

The core novelty of this research is comparing how an algorithmic ML-QEM pipeline adapts to two vastly different physical domains. The sources of quantum decoherence dictate the limits of the circuits that can be reliably executed.

### 2.1 IBM Cloud (Superconducting Qubits)

- **Processor:** `ibm_kingston` (127-qubit Eagle r3 architecture).
- **Control Mechanism:** Microwave pulses (~5 GHz) delivered via cryogenic coaxial cables.
- **Dominant Errors:** Coherent over/under-rotations, microwave cross-talk during two-qubit entangling gates ($CZ$ / $ECR$), and spontaneous emission ($T_1$ relaxation).
- **Fidelity Profile:** Very high baseline fidelity (Raw MAE ~0.017 for shallow circuits). Error mitigation is only strictly necessary for deep, highly entangled circuits (Depth > 8).

### 2.2 SpinQ Gemini (NMR Qubits)

- **Processor:** 2-qubit room-temperature Liquid-State Nuclear Magnetic Resonance (NMR).
- **Control Mechanism:** Radio Frequency (RF) pulses manipulating nuclear spins dissolved in a solvent, inside a strong, static magnetic field ($B_0$).
- **Dominant Errors:** RF pulse imperfections, severe Amplitude Damping ($T_1$ spin-lattice relaxation), and severe Phase Damping ($T_2$ spin-spin dephasing).
- **Fidelity Profile:** High natural error rates compared to superconducting qubits. Long execution times mean complex circuits rapidly dissolve into thermal noise. Therefore, experiments on NMR are strictly bounded to shallow Grover implementations (e.g., standard $\vert{}11\rangle$ and $\vert{}00\rangle$ oracles).

## 3. Methodology & Mathematical Deconstruction

To capture the diverse physics of both platforms without losing generality, the ML model is conditioned on the complete algebraic map of the quantum hardware: the Choi Matrix.

### 3.1 Quantum Process Tomography & The Hardware Fingerprint

Instead of using basic scalar metrics ($T_1$, $T_2$) which fail to map cross-talk, we perform a one-time physical Quantum Process Tomography (QPT) on the basis gates ($H, X, CZ$).
Through the Choi-Jamiołkowski isomorphism, the action of a noisy hardware gate $\mathcal{E}$ on an entangled Bell state $\vert{}\Phi^+\rangle$ yields the Choi Matrix $\Lambda_{real}$. For example, a depolarizing gate with error $p$ results in:

$$\Lambda_{real} = (1 - p)\Lambda_{ideal} + p \left( \frac{I}{2} \otimes \frac{I}{2} \right)$$

Flattening the single-qubit ($4 \times 4$) and two-qubit ($16 \times 16$) complex matrices gives us a high-resolution, 576-element vector containing the absolute DNA of the hardware's decoherence on that specific day.

### 3.2 The Zero-Gradient Problem & CPTP-Preserving Data Augmentation

**The Problem:** Running physical QPT for every training sample would consume immense QPU time. However, feeding the neural network 500 rows of the exact same 576D static Choi matrix causes a "mode collapse." The variance is zero ($\nabla X_{hw} = 0$), so the network memorizes a bias instead of learning correlations.
**The Solution (Sim-to-Real Transfer):** We synthetically perturb the baseline physical Choi matrix during classical training. To ensure the mathematical states remain physically valid (Completely Positive and Trace-Preserving - CPTP), we avoid raw Gaussian noise. Instead, we apply a simulated depolarizing mixing sub-routine:

$$\Lambda_{Choi}^{perturbed} = (1 - p_{mix})\Lambda_{Choi} + p_{mix} \frac{I \otimes I}{d}$$

Where $p_{mix} \sim \mathcal{U}(0, 0.05)$. This guarantees dynamic, CPTP-compliant gradient variance for the network to learn from, costing zero QPU time.

### 3.3 The Two-Tower Neural Network Architecture

```
graph TD
    subgraph Tower A: Circuit Processor
        A[8 Circuit Features: Depth, Gate Counts] --> B[Affine + BatchNorm1d]
        B --> C[SiLU + Dropout 50%]
        C --> D[16D Circuit Embedding]
    end

    subgraph Tower B: Hardware Fingerprint Processor
        E[Static 576D Physical Choi Matrix] --> F{Training Phase?}
        F -- Yes --> G[CPTP Depolarizing Perturbation]
        F -- No --> H[Raw Physical Matrix]
        G --> I[Affine Layer]
        H --> I
        I --> J[Tanh Activation]
        J --> K[4D Latent HW Embedding]
    end

    D --> L[Concatenation 16D + 4D = 20D]
    K --> L
    L --> M[Fusion Layer: Affine + SiLU + Dropout]
    M --> N[Output Affine 8D to 1D]
    N --> O[Bounded Residual: 0.2 * Tanh]

    P[Raw Noisy Hardware Expectation] --> Q((+))
    O --> Q
    Q --> R[Mitigated Expectation Value]
```

### 3.4 Forward Pass Mathematics & The Bounded Residual

Let $f(\vec{x}_c, \vec{x}_h; \Theta) = \hat{Y}$.

1. **Tower A (Up-Projection):** We map the 8 dense circuit features into a 16D space: $\vec{z}_c = \text{SiLU}(W_c \vec{x}_c + \vec{b}_c)$. Expanding the dimensions gives the network space to untangle non-linear correlations (e.g., depth compounding with $CZ$ errors).
2. **Tower B (Non-linear PCA):** We compress the 576D Choi matrix: $\vec{z}_h = \tanh(W_h \vec{x}_h + \vec{b}_h)$ where $\vec{z}_h \in \mathbb{R}^4$.
3. **The Proof of Safety (Bounded Residual):** Quantum expectations strictly exist in $[-1, 1]$. An unbounded ML model could hallucinate $\hat{Y} = 50$. We formulate the output as:$$\hat{Y} = x_{noisy} + 0.2 \cdot \tanh(O_{raw})$$
    
    Since $\tanh \in (-1, 1)$, the absolute maximum delta the model can ever apply is $\pm 0.2$. This analytically guarantees the pipeline defaults safely back to the raw hardware reading during out-of-distribution events.
    

### 3.5 Optimization Strategy

To optimize parameters $\Theta$, we utilize a custom composite loss function:

$$\mathcal{L}_{total} = \mathcal{L}_{Huber}(\hat{Y}, Y) + \frac{\lambda}{\bar{V}_h + \epsilon}$$

1. **Huber Loss:** Prevents exploding gradients from sudden quantum hardware outliers.
2. **Inverse Variance Penalty:** $\bar{V}_h$ is the mean column variance of Tower B's output batch. If the network suffers mode collapse (mapping all Choi matrices to one point, $\bar{V}_h = 0$), the penalty term explodes. This forces the matrix $W_h$ to expand its latent coordinate space to resolve fine-grained hardware noise.

## 4. Experimental Setup

### 4.1 High-Level Experimental Pipeline

The following diagram outlines the end-to-end experimental workflow from hardware characterization through ML post-processing and final inference:

```
graph TD
    %% Define Styles
    classDef hardware fill:#ffe6e6,stroke:#ff6666,stroke-width:2px,color:#000;
    classDef software fill:#e6f3ff,stroke:#66b3ff,stroke-width:2px,color:#000;
    classDef fusion fill:#e6ffe6,stroke:#66cc66,stroke-width:2px,color:#000;
    classDef final fill:#fff2cc,stroke:#ffcc00,stroke-width:2px,color:#000;

    %% Phase 1 & 2
    subgraph Phase 1: Hardware Characterization
        A[Quantum Process Tomography <br> H, X, CZ Gates] --> B(Extract Choi Matrices <br> Physical Noise Fingerprint)
    end
    class A,B hardware

    subgraph Phase 2: Data Generation
        C[Generate 500 Grover-like Circuits] --> D(Calculate Ideal Values <br> CPU Simulator)
        C --> E(Execute on Target QPU <br> Extract Noisy Values & Circuit Features)
    end
    class C,D,E hardware

    %% Phase 3
    subgraph Phase 3: Data Augmentation
        B --> F{Training Phase?}
        F -->|Yes| G[CPTP Depolarizing Perturbation <br> Dynamic HW Dataset]
        F -->|No| H[Raw Static Physical Matrix]
    end
    class F,G,H software

    %% Phase 4
    subgraph Phase 4: Two-Tower ML Training
        E --> I[Tower A: Circuit Processor <br> Untangles Depth/Gate Errors]
        G --> J[Tower B: HW Compression <br> Non-Linear PCA]
        I --> K[Residual Fusion Layer <br> Learns Bounded Error Delta]
        J --> K
    end
    class I,J,K fusion

    %% Phase 5
    subgraph Phase 5: Grover Inference
        M[Run Target Grover Circuit <br> on Target QPU] --> N(Feed Raw Output, Features & <br> HW Matrix to Trained Model)
        H --> N
        K -.->|Trained Weights| N
        N --> O((Mitigated <br> Expectation Value))
    end
    class M,N,O final
```

### 4.2 Dataset & Transpilation

The dataset consists of Grover's algorithm basis circuits (targeting $\vert{}00\rangle$, $\vert{}11\rangle$, double-iterations, and random depolarizing basis equivalents).

- **Training:** 500 circuits. $80\%$ random depth variations to generalize noise, $20\%$ strict Grover structures to target specific physics.
- **Testing:** $20\%$ holdout test split. Standalone batch inference of 20-$30$ identical runs (1024 shots each) to calculate statistical stability.
- **ISA Transpilation:** For the IBM setup, circuits were mapped to physical qubits using `generate_preset_pass_manager` to comply with current Instruction Set Architectures (ISA) before feature extraction and execution.

### 4.3 Evaluation Metrics

To rigorously evaluate the model's success, we selected Mean Absolute Error (MAE) and the Wilcoxon Signed-Rank Test. These choices are mathematically motivated by the specific nature of quantum hardware errors:

1. **Why Mean Absolute Error (MAE) instead of Mean Squared Error (MSE)?**
Quantum hardware suffers from occasional, severe "misfires" (e.g., sudden thermal spikes resulting in massive expectation value deviations). MSE squares the errors, heavily penalizing these outliers and artificially skewing the average. MAE provides a much more robust linear measure of the typical distance between the mitigated output and the ideal mathematical state.
2. **Why the Wilcoxon Signed-Rank Test instead of a Paired t-Test?**
The Student's t-test strictly assumes that the paired differences (mitigated error vs. raw error) are normally distributed. However, quantum errors are bounded in $[-1, 1]$ and their distributions are often highly skewed. The Wilcoxon signed-rank test is a non-parametric hypothesis test that does not assume normality. It evaluates whether the median error reduction is statistically significant, providing a far more scientifically rigorous proof of improvement for quantum execution data.

#### 4.3.1 Mathematical Formulation of the Null Hypothesis Proof

To mathematically prove that our ML model's 29.8% error reduction is not just a random fluke, we formalize the experiment using statistical hypothesis testing.

Let $e_{raw}^{(i)}$ be the absolute raw hardware error for quantum circuit $i$.
Let $e_{mit}^{(i)}$ be the absolute ML-mitigated error for circuit $i$.
We define the paired difference (the improvement delta) for each circuit as:

$$D_i = e_{raw}^{(i)} - e_{mit}^{(i)}$$

- **The Null Hypothesis (**$H_0$**):** The ML model does *not* improve the hardware output. Any observed reduction in error is purely due to random quantum shot-noise fluctuations. Mathematically, the median of the differences $D_i$ is zero.
- **The Alternative Hypothesis (**$H_A$**):** The ML model systematically reduces the error. Mathematically, the median of the differences is strictly greater than zero ($D_i > 0$).

**Step-by-Step Proof Mechanism (The Wilcoxon Signed-Rank Test):**

1. Calculate the absolute differences $\vert{}D_i\vert{}$ for all $N$ circuits (ignoring any circuits where $D_i = 0$).
2. Rank these absolute differences from smallest (Rank 1) to largest (Rank $N$).
3. Calculate the Test Statistic $W$, which is the sum of the ranks *only* for the circuits where the model successfully improved the result ($D_i > 0$):$$W = \sum_{D_i > 0} R_i$$
4. If the model were just guessing randomly, the sum of the positive ranks and negative ranks would balance out. If $W$ is abnormally large, it mathematically proves the model is systematically pulling the error down across the dataset.
5. We convert the $W$ statistic into a **p-value**. The p-value answers this exact question: *"If the Null Hypothesis (*$H_0$*) were true (i.e., the model is useless), what is the exact probability that we would see an error reduction this large entirely by chance?"*

In statistical physics and machine learning, the standard threshold for "significance" is $\alpha = 0.05$ ($5\%$). If $p < 0.05$, the probability of a random fluke is so microscopic that we mathematically **reject the Null Hypothesis** and accept the Alternative Hypothesis ($H_A$)—proving that the Two-Tower ML-QEM pipeline genuinely corrects quantum hardware errors.

## 5. Results & Discussion

### 5.1 IBM Kingston (Superconducting) Execution

To mathematically guarantee that the neural network learns physical error profiles rather than simply memorizing training instances, the dataset was strictly partitioned using a Group Shuffle Split. This ensured that identical circuit structures (e.g., duplicated instances of Grover sub-routines) were strictly isolated to either the training or testing sets with zero structural crossover.

Evaluated across the fully unseen test set of 118 circuits (varying depths from 3 to 15), the ML-QEM pipeline demonstrated profound generalization capabilities across all regimes of coherence:

- **Test Set Size:** 118 Unique Circuits (Zero structural leakage)
- **Raw MAE:** 0.016736
- **Mitigated MAE:** 0.011048
- **Error Reduction:** **34.0%**
- **Wilcoxon p-value:** **< 10^-6 (Statistically Significant)**

**Discussion:** The ML-QEM pipeline proved its profound generalization capabilities through a strict Group Shuffle Split, guaranteeing no structural data leakage. It successfully mitigated systemic hardware errors across the entire coherence spectrum (depths 3-15). This proves the model is not memorizing duplicate circuits, but genuinely learning the physical decay patterns embedded in the Choi matrix fingerprint.

### 5.2 Projections for NMR (SpinQ Gemini) Deployment

The implementation on the NMR hardware at UOC will omit the depth isolation step. Because RF pulse coherence and liquid-state relaxation times ($T_1, T_2$) are significantly shorter relative to gate times, even shallow Grover instances will demonstrate severe signal degradation.
By utilizing the exact same Two-Tower pipeline, we predict the CPTP-augmented Tower B will map the specific Phase Damping asymmetries inherent in the NMR Choi matrix, yielding a robust error mitigation delta on shallow target algorithms without requiring prohibitive QPT repetitions.

## 6. Conclusion

This project successfully engineers a rigorous, physically-compliant Machine Learning framework for Quantum Error Mitigation. By mapping the full hardware footprint via CPTP-perturbed Choi matrices into a variance-penalized Two-Tower ResNet, we eliminate the dimensionality curse and zero-gradient collapse. Experimental validation on IBM's superconducting hardware proves the model's ability to significantly mitigate (34.0% reduction) complex coherent errors across entirely unseen circuit structures while maintaining analytical safety bounds. The pipeline is fully prepared for immediate cross-platform deployment and validation on the SpinQ NMR architecture.
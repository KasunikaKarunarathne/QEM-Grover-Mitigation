# Quantum Error Mitigation Dataset Comparison

This document provides a comparative analysis between the newly generated **SpinQ NMR** physical dataset and the previously collected **IBM Superconducting** dataset. Both datasets contain exactly 500 benchmark circuits designed to train a Quantum Error Mitigation (QEM) machine learning pipeline.

## 1. Dataset Dimensions
Both datasets successfully reached the target quota and are structurally identical, making them plug-and-play compatible for the Two-Tower Neural Network.

| Metric | SpinQ NMR (Physical) | IBM Superconducting |
| :--- | :--- | :--- |
| **Total Circuits** | 500 | 500 |
| **Input Features (X)** | 8 features per circuit | 8 features per circuit |
| **Target Outputs (Y)** | 1 (Ideal Expectation) | 1 (Ideal Expectation) |

---

## 2. Structural Balance (Circuit Depths)
The random generation algorithm (with structured Grover injection) performed consistently across both runs. Most importantly, **there are 0 empty circuits (depth 0 or 1)** in either dataset, meaning the `try/except` skip-logic successfully kept the dataset clean.

| Circuit Depth | SpinQ Count | IBM Count |
| :---: | :---: | :---: |
| **2** | 35 | 63 |
| **3** | 50 | 57 |
| **4** | 54 | 60 |
| **5** | 53 | 70 |
| **6** | 62 | 55 |
| **7** | 86 | 70 |
| **8** | 39 | 40 |
| **9** | 85 | 57 |
| **10** | 4 | 10 |
| **11** | 1 | 0 |
| **14** | 31 | 18 |

> [!TIP]
> **Conclusion:** Both datasets feature a healthy, bell-curve-like distribution anchored around depths 4-9, providing a solid gradient of complexity for the Neural Network to learn from.

---

## 3. Target Distribution (Expectation Values)
To effectively mitigate errors, the model needs to learn how to correct expectation values across the entire theoretical spectrum (-1 to 1). 

| Target Range | SpinQ Target Count | IBM Target Count |
| :--- | :---: | :---: |
| **Near 0** | 272 | 306 |
| **Near +1** | 170 | 141 |
| **Near -1** | 58 | 53 |

> [!TIP]
> **Conclusion:** The Grover basis injections successfully anchored the extremes (+1 and -1) while the random circuits provided a dense cluster around 0. The distributions are nearly identical, ensuring an apples-to-apples comparison when benchmarking model performance.

---

## 4. Hardware Noise Profile (The Most Important Metric)
By calculating the Mean Squared Error (MSE) between the *Ideal Expectation* and the *Raw Noisy Expectation* prior to any machine learning mitigation, we can quantify the baseline physical error of the hardware.

| Hardware Backend | Baseline Raw Error (MSE) |
| :--- | :--- |
| **IBM Dataset** | `0.00048` |
| **SpinQ NMR Dataset** | `0.09610` |

> [!IMPORTANT]
> **Key Finding:** The raw, unmitigated error on the physical SpinQ Gemini NMR machine is roughly **200 times higher** than the baseline error in the IBM dataset. 
> 
> This perfectly highlights the necessity of your research: the SpinQ NMR machine is incredibly noisy and desperately needs the Machine Learning Error Mitigation pipeline you are building!

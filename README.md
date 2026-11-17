# Science

## A Comprehensive Collection of Quantum Physics, Mathematics, and Advanced Topics in R

![Author](https://img.shields.io/badge/Author-John%20Akwei-blue)
![R](https://img.shields.io/badge/Language-R-276DC3?logo=r)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)
![Erdős Problem 30](https://img.shields.io/badge/Mathematics-Erd%C5%91s%20Problem%2030-8A2BE2)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

This repository collects the scientific work of John Akwei, Senior Data Scientist at [ContextBase](https://contextbase.github.io). It has three parts:

1. **Mathematics.** Research papers and a long-form report on [Erdős Problem 30](https://www.erdosproblems.com/30), the maximum size of a Sidon set in {1,…,N}.
2. **Physics.** R Markdown documents that pair derivations in quantum field theory and condensed-matter physics with computations and visualizations in R.
3. **Tools.** A Python multi-agent pipeline that retrieves and summarizes new quantum-physics papers from arXiv.

---

## Contents

| File | What it is |
|---|---|
| [`limit_vector_smoothing.pdf`](limit_vector_smoothing.pdf) | **The limit of vector-valued smoothing for Sidon sets is 2√2/3.** Current version, 16 Nov 2026, 38 pp. |
| [`A_Data_Science_Analysis_of_Erdos_Problem_30_V6.pdf`](A_Data_Science_Analysis_of_Erdos_Problem_30_V6.pdf) | **A Data Science Analysis of Erdős Problem 30**, Version 6, 21 Oct 2026, 72 pp. |
| [`The_limit_of_vector-valued_smoothing_for_Sidon_sets.pdf`](The_limit_of_vector-valued_smoothing_for_Sidon_sets.pdf) | Earlier version of the limit paper (author block added 5 Oct 2026), 30 pp. |
| [`Sidon_smoothing_limit.pdf`](Sidon_smoothing_limit.pdf) | First version of the limit paper (dated 26 Sep 2026, posted 1 Oct 2026), 23 pp. |
| [`Sidon_smoothing_barrier.pdf`](Sidon_smoothing_barrier.pdf) | *A barrier for vector-valued smoothing certificates for Sidon sets* (25 Sep 2026), 16 pp. The precursor that proved γ* ≥ 0.8991. |
| `Quantum_Field_Theory_in_R.Rmd` | Quantum Field Theory in R |
| `Quantum_Chromodynamics_in_R.Rmd` | Quantum Chromodynamics: Mathematical Foundations and Computational Analysis |
| `QuasiParticles_in_R.Rmd` | Quasiparticles in R |
| `Fracton_Codes_in_R.Rmd` | Fracton Codes: Foundations and Analysis in R |
| `Chiral_Graviton_Modes_in_R.Rmd` / `.html` | Chiral Graviton Modes in Fractional Quantum Hall Liquids |
| `QET_Subsystem_Symmetry_in_R.Rmd` / `.html` | Quantum Energy Teleportation under Subsystem Symmetry |
| `arxiv_quantum_agent.py` | arXiv Quantum Physics Paper Triage & Summarization Agent |

---

## 🔢 Mathematics: Erdős Problem 30

A **Sidon set** is a set of integers whose pairwise sums a + b (a ≤ b) are all distinct, or equivalently whose nonzero differences are all distinct. Let h(N) be the largest size of a Sidon set in {1,…,N}. Erdős Problem 30 asks whether

> h(N) = N^(1/2) + O_ε(N^ε) for every ε > 0.

The problem has been open since Erdős and Turán (1941) and carries a $1000 prize. The lower half of the conjecture reduces to gaps between primes and is widely believed. The difficulty is the upper bound h(N) ≤ N^(1/2) + γ·N^(1/4) + O(1). In 85 years the exponent 1/4 has not moved, but the constant γ has:

| Upper bound | γ | Year |
|---|---|---|
| Erdős–Turán; Lindström | 1 | 1941, 1969 |
| Balogh–Füredi–Roy | ≈ 0.998 | 2021 |
| O'Bryant | 0.99703 | 2022 |
| Carter–Hunter–O'Bryant | 0.98183 | 2023 |
| Carter, Georgiev, Gómez-Serrano, Hunter, O'Bryant, Tao, Wagner (announced) | 0.97633 | 2026 |
| Hou–Zhao, vector-valued smoothing | 0.9435 | 2026 |
| Carter–O'Bryant–Tao, for sets of at least 483 elements | 0.94304 | 2026 |
| **This work**, verified symmetric certificate | **0.9428095** | 2026 |
| Chen, continuous kernel, additive term 1, N ≥ 4.6×10⁶ | 2√2/3 | 2026 |
| **This work**, limit of finite smoothing certificates | **2√2/3 = 0.9428090…** | — |

*Source: Table 1 of `limit_vector_smoothing.pdf`. The 2026 bounds, including those in this repository, are preprints and have not been refereed.*

### The limit of vector-valued smoothing for Sidon sets is 2√2/3

**File:** [`limit_vector_smoothing.pdf`](limit_vector_smoothing.pdf) (current version, 16 November 2026, 38 pages)

The strongest recent upper bounds come from the vector-valued smoothing framework of Hou and Zhao (arXiv:2607.01169). In that framework a bound is certified by kernels, boundary weights and mixing weights that satisfy finitely many linear inequalities, and the constant is found by running an optimizer. This paper determines exactly how far that approach can go.

**Main results**

- **Theorem 1 (the limit).** Every certificate, symmetric or two-sided, has γ ≥ 2√2/3 = 0.94280904…. The bound is sharp in both settings. In the two-sided framework the linear kernel ρ(x) = 2x attains it. In Hou and Zhao's original symmetric framework, mixtures of centred uniform kernels approach it at rate O(1/m²). Hence γ\*_sym = γ\* = 2√2/3.
- **The proof.** A single dual multiplier proves the lower bound: twice the renewal measure of the uniform distribution. The weight it places on the kernels' autocorrelation is constant on its first window, so the bound depends on the kernels only through their L² norm. That constancy reduces to the identity ∫₀^∞ (r(x) − 2)² dx = 1/3 for the renewal density r, which the paper proves. The argument extends to signed vector-valued kernels (Corollary 4.6, due to Ariel Williams).
- **Theorem 2 (a verified constant).** A symmetric certificate at resolution m = 1024 with 512 kernels and boundary depth L = 6 satisfies all 6145 covering inequalities in exact rational arithmetic and has γ < 0.942809493. This gives

  > h(N) ≤ N^(1/2) + 0.9428095·N^(1/4) + O(1),

  using only Hou and Zhao's Theorem 2.1. The result is about 4.5×10⁻⁷ above the proved limit. The earlier two-sided certificate (linear kernel, m = 512, L = 8, 8194 inequalities) gives γ < 0.942809522.
- **Theorem 3 (keeping the variance term).** The level-one moment (pair-statistics) relaxation keeps the variance term that smoothing discards. Its value is still N^(1/2) + (2√2/3)·N^(1/4) + O(1), and adding distinct-sum and Sherali–Adams constraints does not change that.
- **Related methods hit the same wall.** The window–table method of Carter, O'Bryant and Tao also stops at 2√2/3 (Remark 4.7), and so does the fixed-kernel energy method of H. Chen.

**Consequence.** Arguments of this type cannot push the constant below 2√2/3 and cannot improve the exponent 1/4. To go further, an argument must keep information that one- and two-point statistics of the set do not capture.

**Earlier versions** are kept for the record. [`Sidon_smoothing_barrier.pdf`](Sidon_smoothing_barrier.pdf) proved γ\* ≥ √((π²+16)/32) = 0.899124… and conjectured 2√2/3. [`Sidon_smoothing_limit.pdf`](Sidon_smoothing_limit.pdf) (the first version) and [`The_limit_of_vector-valued_smoothing_for_Sidon_sets.pdf`](The_limit_of_vector-valued_smoothing_for_Sidon_sets.pdf) proved the limit for two-sided certificates but left the symmetric case and the pair-level statement open. The current version settles both, and its theorem numbering follows the first version.

### A Data Science Analysis of Erdős Problem 30, Version 6

**File:** [`A_Data_Science_Analysis_of_Erdos_Problem_30_V6.pdf`](A_Data_Science_Analysis_of_Erdos_Problem_30_V6.pdf) (dated 21 October 2026, posted 26 October 2026, 72 pages)

This report collects a staged investigation of the problem from both sides, in twelve parts and four appendices. It supersedes Versions 1–5.

- **The upper-bound wall (Parts III–VI, Appendices B–D).** Three barriers, each with a written proof, stop at exactly 2√2/3: the smoothing framework (symmetric and two-sided), the pair-level moment relaxation ĥ₂(N), and the three-point relaxation ĥ₃(N) = √N + (2√2/3)·N^(1/4) + O(1) (Theorem 6.8). Keeping the variance term, keeping the difference slack, or adding three-point constraints each gains only O(1).
- **Exact values (Part I).** Optimal Golomb rulers through 28 marks (OGR-28, length 585) determine h(N) exactly for every N ≤ 586.
- **The construction survey (Part II, Stage 0).** For the Singer, Bose and Ruzsa families in every characteristic, the excess h(N) − √N obtained by cutting a cyclic Sidon set at its largest gap grows logarithmically, not like a power of N.
- **Extreme values of Singer constructions (Part VIII).** Every multiplier class of the characteristic-2 Singer sets was enumerated for q = 2¹¹ to 2¹⁴, and 2×10⁶ classes were sampled for q = 2¹⁵ to 2¹⁸, reaching N ≈ 6.9×10¹⁰. The tail is exponential and the best excess grows like c·ln N with c ≈ 1.6–2.0. This family has the strongest structural effect, so it is the likeliest algebraic source of a power law, and it shows none.
- **Gaps over multipliers (Part VII, Stage 1).** Every Singer set, under every multiplier, has a gap of at least 2q, so h(q² − q + 2) ≥ q + 1. Every Bose–Chowla set has a gap of at least 2q − 1. It remains open whether the best multiplier's gap is unbounded relative to q, which would refute h(N) = √N + O(1) (O'Bryant's Open Question 5).
- **The lower side for typical N (Part IX).** √N − h(N) ≤ N^(1/43+ε) for all N outside a set of density zero. In the almost-all sense, Erdős #30 is a question about the upper bound alone.
- **Arithmetic tests (Part VI).** In a cyclic model, integer counts on a single quotient certify 98 of 105 impossible configurations. Three interval analogues certify nothing, which suggests that useful arithmetic must be tied to positions, not to the difference set alone.
- **What data can decide (Part X).** The statement is Π₃, and no finite dataset decides it. An explicit Bayes ledger moves P(true) from an illustrative 0.60 to 0.74, with a ceiling of 0.746.
- **Formalization (Part XI).** Lean 4 statements of the barrier side are written on the Hulak–Ramos–de Queiroz Singer library. They have not yet been compiled.

**Bottom line.** The evidence favours a true answer, and the known construction families will not refute the conjecture. On the upper-bound side, every method built on one-, two- and three-point statistics has been followed to the same wall at (2√2/3)·N^(1/4). That wall is a theorem about those methods, not about h(N).

### Certificates, data and joint work

- **Certificates for Theorem 2(a):** h(N) ≤ √N + 0.9428095·N^(1/4) + O(1). Plain-text certificate with two independent exact checkers and negative controls. [doi:10.5281/zenodo.24138215](https://doi.org/10.5281/zenodo.24138215)
- **Two-kernel smoothing certificates for Sidon sets.** [zenodo.org/records/24163953](https://zenodo.org/records/24163953)
- **An effective four-unit bracket for noncyclic Sidon pair relaxations**, with Ariel Williams. [doi:10.5281/zenodo.24015162](https://doi.org/10.5281/zenodo.24015162)
- *The sharp WT-triple bound: stability and witness width*, with Ariel Williams. In preparation.

### Related independent work

- H. Chen, *An explicit second-order bound for Sidon sets and the limit of the fixed-kernel energy method* (v3, Zenodo, Oct 2026), [doi:10.5281/zenodo.23273526](https://doi.org/10.5281/zenodo.23273526). Chen proves h(N) ≤ √N + (2√2/3)·N^(1/4) + 1 for N ≥ 4.6×10⁶ with the same extremal kernel. The two results were reached independently.
- D. Carter, K. O'Bryant and T. Tao, *On the diameter and distribution of Sidon sets*, arXiv:2610.36207.
- A. M. Williams, *Effective sharpness of a pair-moment relaxation for Sidon sets* (Zenodo), [doi:10.5281/zenodo.23607543](https://doi.org/10.5281/zenodo.23607543).

*None of the papers in this repository has been refereed. As the papers state, much of the computation, proof checking and drafting was done with AI assistance (Claude, Anthropic). The author checked the arguments and is responsible for any errors.*

---

## ⚛️ Physics in R

Each document is a self-contained R Markdown file that combines mathematical derivations with executable R code and visualizations. Pre-rendered HTML is included for the Chiral Graviton Modes and QET documents. Knit the others in RStudio to view them.

### 📘 Quantum Field Theory in R
`Quantum_Field_Theory_in_R.Rmd`

Shows how quantum fields arise from combining quantum mechanics with special relativity. Covers the Klein–Gordon equation and scalar fields, canonical quantization and mode expansion, the Dirac equation, virtual particles and vacuum fluctuations, quantization of the electromagnetic field, and a computational check of the spin-statistics theorem.

### 🔬 Quantum Chromodynamics: Mathematical Foundations and Computational Analysis
`Quantum_Chromodynamics_in_R.Rmd`

The SU(3) gauge theory of the strong force. Covers Gell-Mann matrices, the β-function and a proof of asymptotic freedom, the running coupling, colour confinement and the QCD potential, parton distribution functions, chiral symmetry breaking, lattice QCD concepts, and the evidence from deep inelastic scattering.

### ⚛️ Quasiparticles in R
`QuasiParticles_in_R.Rmd`

Emergent excitations in condensed matter from 2005 to 2025: spinons, magnons, anyons (including fractional quantum Hall anyons and Fibonacci anyon braiding), skyrmions and excitons. Includes dispersion relations, phase diagrams, material systems and timeline visualizations.

### 🌀 Fracton Codes: Foundations and Analysis in R
`Fracton_Codes_in_R.Rmd`

Fracton topological order and quantum error correction with immobile excitations. Includes an implementation of the X-cube model, lattice visualizations, and applications to topological quantum computing.

### 🧲 Chiral Graviton Modes in Fractional Quantum Hall Liquids
`Chiral_Graviton_Modes_in_R.Rmd`, `Chiral_Graviton_Modes_in_R.html`

*Quantum Geometry, Spectral Sum Rules, and Polarization Selection Rules in R.* Chiral graviton modes are the long-wavelength spin-2 limit of the Girvin–MacDonald–Platzman magnetoroton: quanta of fluctuations of Haldane's emergent guiding-centre metric. The document shows that graviton chirality follows as a theorem with a topological input. Non-negativity of two spectral densities forces S₄ ≥ |𝒮 − 1|/8, with equality exactly when one circular-polarization channel is empty. Topics include the Wen–Zee shift and Haldane's bound, the Golkar–Nguyen–Son chiral sum rules, Laughlin saturation and Jain-sequence shifts, the ν = 1 control, and recent experiments.

### 🔋 Quantum Energy Teleportation under Subsystem Symmetry
`QET_Subsystem_Symmetry_in_R.Rmd`, `QET_Subsystem_Symmetry_in_R.html`

*A Selection Rule for Fracton-Like Phases.* Uses exact diagonalization of the 2D plaquette Ising (Xu–Moore) model, with the transverse-field Ising model as a control, to test whether restricted mobility changes Quantum Energy Teleportation (QET).

- Commuting-projector stabilizer ground states teleport exactly zero energy despite substantial entanglement.
- Subsystem symmetry blocks the standard two-party protocol completely, in every direction and at every field strength. This falsifies the anisotropy conjecture.
- The blockade lifts only when Alice's observable has odd X-parity in exactly Bob's row and column. That turns QET into a four-party protocol on the corners of a rectangle, which remains genuine LOCC.
- The selection rule is verified against all 31 candidate observables on a 3×3 lattice and all 793 candidates of size ≤ 4 on a 4×4 lattice.

---

## 📚 arXiv Quantum Physics Triage Agent

`arxiv_quantum_agent.py`

This agent addresses information overload in quantum-physics research. It queries the arXiv API in the `quant-ph` category, then runs the results through a sequential pipeline of five Gemini-backed agents:

1. **Paper Retriever:** fetches matching papers from arXiv.
2. **Abstract Analyzer:** extracts the key claims.
3. **Mathematical Notation Identifier:** picks out the important equations.
4. **Relevance Scorer:** ranks the papers against the query and the stored research preferences.
5. **Summary Generator:** writes the final summary.

The script also keeps user research preferences for the session, tracks per-agent metrics, and prints an evaluation report. It falls back through several Gemini model names until one is available.

```bash
python -m pip install google-generativeai
export GEMINI_API_KEY="your-key"        # PowerShell: $env:GEMINI_API_KEY = "your-key"
python arxiv_quantum_agent.py
```

When it runs, the script asks for a query (or uses a default about quantum error correction) and for the number of papers to retrieve (1–20).

---

## Getting started

```bash
git clone https://github.com/johnakwei/Science.git
cd Science
```

**Papers:** open the PDFs in any viewer.

**R Markdown documents:** you need R ≥ 4.0 and, ideally, RStudio.

```r
install.packages(c(
  "ggplot2", "plotly", "viridis", "scales", "reshape2",
  "gridExtra", "dplyr", "tidyr", "latex2exp", "rmarkdown"
))

# Render one document to HTML
rmarkdown::render("Quantum_Field_Theory_in_R.Rmd")
```

In RStudio you can instead open any `.Rmd` and click **Knit**. The optional 4×4 lattice check in the QET document (Hilbert dimension 65,536) also needs `Matrix` and `RSpectra` and is not evaluated by default.

---

## Citation

To cite the limit paper:

> J. Akwei, *The limit of vector-valued smoothing for Sidon sets is 2√2/3*, preprint, 16 November 2026. https://github.com/johnakwei/Science

To cite the report:

> J. Akwei, *A Data Science Analysis of Erdős Problem 30*, Version 6, 21 October 2026. https://github.com/johnakwei/Science

To cite the repository as a whole:

> J. Akwei (2026). *Science: Quantum Physics, Mathematics, and Advanced Topics in R.* GitHub repository. https://github.com/johnakwei/Science

## Author

**John Akwei**, Senior Data Scientist, [ContextBase](https://contextbase.github.io)
Scientific computing, quantum physics, additive combinatorics, and agentic AI systems.

## License

MIT.

---

## ⭐ Star this repository if you find it useful!  
Last updated: November 2026

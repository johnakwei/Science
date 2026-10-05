# Science

## A Comprehensive Collection of Quantum Physics, Mathematics, and Advanced Topics in R

![Static Badge](https://img.shields.io/badge/Author-John%20Akwei-blue)  

![Static Badge](https://img.shields.io/badge/Language-R-276DC3?logo=r)  

![Static Badge](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)  

![Static Badge](https://img.shields.io/badge/Mathematics-Erd%C5%91s%20Problem%2030-8A2BE2)  

![Static Badge](https://img.shields.io/badge/License-MIT-green.svg)  

## Overview

This repository contains a comprehensive collection of scientific documents exploring cutting-edge topics in theoretical physics and quantum mechanics, authored by John Akwei, Senior Data Scientist. Each document combines rigorous mathematical foundations with computational implementations in R, providing both theoretical derivations and interactive visualizations. The repository also includes research papers in additive combinatorics (PDF) and a Python multi-agent tool for triaging new arXiv papers.

### Repository Contents

#### 📘 Quantum Field Theory in R
File: Quantum_Field_Theory_in_R.Rmd  

A complete mathematical proof and computational implementation of key concepts in Quantum Field Theory (QFT). This document demonstrates how quantum fields emerge from the marriage of quantum mechanics and special relativity.  

Topics Covered:  
Klein-Gordon equation and scalar field theory  
Canonical quantization and mode expansion  
Dirac equation for spin-1/2 fermions  
Virtual particles and vacuum fluctuations  
Electromagnetic field quantization  
Spin-statistics theorem verification  
The unity of QFT framework  

Key Features:  
Interactive visualizations of field evolution  
Computational verification of theoretical principles  
Implementation of quantum field modes  
Analysis of vacuum fluctuations and zero-point energy  

### Prerequisites:  
R (version ≥ 4.0.0)  
RStudio (recommended for R Markdown rendering)  
A PDF viewer for the mathematics papers  

Required R Packages:
```r
install.packages(c(
  "ggplot2",      # Data visualization
  "plotly",       # Interactive plots
  "viridis",      # Color palettes
  "reshape2",     # Data reshaping
  "gridExtra",    # Multiple plot arrangements
  "dplyr",        # Data manipulation
  "tidyr",        # Data tidying
  "latex2exp"     # LaTeX expressions in plots
))
```

Optional, for the 4×4 verification in the QET document: `Matrix` and `RSpectra`.  

#### 🔬 Quantum Chromodynamics in R
File: Quantum_Chromodynamics_in_R.Rmd  

An in-depth analysis of Quantum Chromodynamics (QCD), the quantum field theory describing the strong nuclear force between quarks and gluons.  

Topics Covered:  
Mathematical framework of QCD (SU(3) gauge theory)  
Asymptotic freedom and beta function analysis  
Color confinement mechanism  
Gell-Mann matrices and SU(3) structure  
Running coupling constant evolution  
Parton distribution functions  
Chiral symmetry breaking  
Lattice QCD concepts  

Key Features:  
Proof of asymptotic freedom  
Visualization of QCD potential and confinement  
Analysis of quark mass hierarchy  
Experimental verification through deep inelastic scattering  
Computational demonstration of quantum corrections  

#### ⚛️ Quasiparticles in R
File: QuasiParticles_in_R.Rmd  

A comprehensive analysis of quasiparticle physics covering developments from 2005-2025, exploring emergent phenomena in condensed matter physics.  

Quasiparticles Analyzed:  
Spinons - Fractional spin excitations  
Magnons - Quantized spin waves  
Anyons - Exotic particles with fractional statistics  
Fractional Quantum Hall Anyons-Trions  
Skyrmions - Topologically protected spin textures  
Excitons - Bound electron-hole pairs  
Additional emergent quasiparticles  

Key Features:  
Historical progression and experimental advances (2005-2025)  
Material systems and properties  
Dispersion relations and phase diagrams  
Timeline visualizations  
Interactive plots and comparative analyses  

#### 🌀 Fracton Codes in R
File: Fracton_Codes_in_R.Rmd  

An exploration of fracton topological order and quantum error correction codes, representing one of the most exciting recent developments in quantum information theory.  

Topics Covered:  
Fracton phases of matter  
X-cube model implementation  
Quantum error correction with immobile excitations  
Topological quantum computing applications  
Lattice implementations and visualizations  

#### 🧲 Chiral Graviton Modes in R
Files: Chiral_Graviton_Modes_in_R.Rmd, Chiral_Graviton_Modes_in_R.html  

Chiral Graviton Modes in Fractional Quantum Hall Liquids: Quantum Geometry, Spectral Sum Rules, and Polarization Selection Rules. Chiral graviton modes are the long-wavelength, spin-2 limit of the Girvin–MacDonald–Platzman magnetoroton, the quanta of fluctuations of Haldane's emergent guiding-centre metric. The document shows that graviton chirality follows as a theorem with a topological input: non-negativity of two spectral densities forces S₄ ≥ |𝒮 − 1|/8, with equality exactly when one circular polarization channel is empty.  

Topics Covered:  
Quantum geometry and the guiding-centre metric  
Wen–Zee shift, guiding-centre spin, and Haldane's bound  
Golkar–Nguyen–Son chiral spectral sum rules  
Laughlin saturation and Jain-sequence shifts  
Long-wavelength structure factor and the ν = 1 control  
Polarization selection rules and recent experiments  
Technical connections to the QCD, Fracton Codes, and Quasiparticles documents  

#### 🔋 Quantum Energy Teleportation under Subsystem Symmetry in R
Files: QET_Subsystem_Symmetry_in_R.Rmd, QET_Subsystem_Symmetry_in_R.html  

Quantum Energy Teleportation under Subsystem Symmetry: A Selection Rule for Fracton-Like Phases. Using exact diagonalization of the 2D plaquette Ising (Xu–Moore) model, with the transverse-field Ising model as a control, this study tests whether restricted mobility changes Quantum Energy Teleportation (QET). Subsystem symmetry blocks the standard two-party protocol entirely, in every direction and at every field strength. The blockade is lifted only when Alice's measured observable has odd X-parity in exactly Bob's row and column, which turns QET into a four-party protocol on the corners of a rectangle.  

Key Results:  
Commuting-projector stabilizer ground states teleport exactly zero energy despite substantial entanglement  
The anisotropy conjecture is falsified: subsystem symmetry blocks bipartite QET completely  
A derived selection rule, verified against all 31 candidate observables on a 3×3 lattice and all 793 candidates of size ≤ 4 on a 4×4 lattice  
The four-party protocol remains genuine LOCC  

#### 🔢 Erdős Problem 30: Sidon Sets and Smoothing Certificates

Two research papers on Erdős Problem 30, which asks whether the maximum size h(N) of a Sidon set in {1,…,N} satisfies h(N) = N^(1/2) + O(N^ε) for every ε > 0. The best current upper bounds have the form h(N) ≤ N^(1/2) + γ·N^(1/4) + O(1), with γ obtained from numerical certificates in the vector-valued smoothing framework of Hou and Zhao. The two papers locate the limit of that method.

**A barrier for vector-valued smoothing certificates for Sidon sets**  
File: [Sidon_smoothing_barrier.pdf](Sidon_smoothing_barrier.pdf) (16 pages)  

Proves that every certificate in the Hou–Zhao framework, and in a two-sided extension admitting asymmetric kernels, has γ ≥ √((π²+16)/32) = 0.899124…, using an explicit dual multiplier built from the renewal measure of the uniform distribution. It conjectures that the true limit is 2√2/3.  

**The_limit_of_vector-valued_smoothing_for_Sidon_sets**  
File: [The_limit_of_vector-valued_smoothing_for_Sidon_sets.pdf](The_limit_of_vector-valued_smoothing_for_Sidon_sets.pdf) (23 pages)  

Determines the limit of the method exactly. Every certificate, symmetric or two-sided, has γ ≥ 2√2/3 = 0.94280904…, and for two-sided certificates this is sharp: the linear kernel ρ(x) = 2x attains 2√2/3 in the continuum limit.  

Key Results:  
A single dual multiplier proves the lower bound, via the identity ∫₀^∞ (r(x) − 2)² dx = 1/3 for the renewal density of the uniform distribution  
A certificate verified in exact rational arithmetic gives h(N) ≤ N^(1/2) + 0.9428096·N^(1/4) + O(1), within 5×10⁻⁷ of the limit  
The natural semidefinite relaxation that keeps the variance term has the same second-order constant, with explicit feasible points tested up to N = 10⁷  
Consequence: arguments of this type cannot push the constant below 2√2/3, nor improve the exponent 1/4  

#### 📚 ArXiv Quantum Physics Triage Agent

File: arxiv_quantum_agent.py

by John Akwei, Senior Data Scientist, ContextBase, https://contextbase.github.io  

This AI agent solves the problem of information overload in quantum physics research by automatically retrieving, analyzing, and summarizing recent papers from ArXiv.  

Architecture: Multi-agent sequential pipeline with 5 specialized agents  

Paper Retriever Agent: Fetches papers from ArXiv API  
Abstract Analyzer Agent: Extracts key claims from abstracts  
Mathematical Notation Identifier: Identifies important equations  
Relevance Scorer Agent: Ranks papers by relevance  
Summary Generator Agent: Creates comprehensive summaries  

Requirements Demonstrated:
✅ Multi-agent system (Sequential agents)
✅ Custom tools (ArXiv API, LaTeX parser)
✅ Built-in tools (Google Search, Code Execution)
✅ Sessions & Memory (InMemorySessionService, user preferences)
✅ Observability (LoggingPlugin, custom metrics)
✅ Bonus: Gemini integration, deployment-ready architecture

### Getting Started

Clone the repository:
```bash
git clone https://github.com/johnakwei/Science.git
cd Science
```

Open in RStudio:  
Open any .Rmd file in RStudio  
Click "Knit" to generate HTML output with all visualizations  

View generated HTML files:  
After knitting, HTML files will be created in the same directory  
Open in any web browser for interactive viewing. Pre-rendered HTML is included for the Chiral Graviton Modes and QET documents.  

Running Code Chunks  
Each document is organized with executable R code chunks that can be run independently:
```r
# Example: Run all chunks in sequence
knitr::knit("Quantum_Field_Theory_in_R.Rmd")

# Or render to HTML
rmarkdown::render("Quantum_Field_Theory_in_R.Rmd")
```

#### Document Structure  

All R Markdown documents follow a consistent professional format:  
- **Abstract/Introduction** - Overview and motivation  
- **Theoretical Foundation** - Mathematical derivations  
- **R Implementation** - Computational analysis  
- **Visualizations** - Interactive plots and figures  
- **Experimental Verification** - Comparison with data  
- **Conclusions** - Key insights and implications  

#### Key Features  
✨ **Mathematical Rigor** - Complete derivations from first principles  
🎨 **Rich Visualizations** - Interactive plots using ggplot2 and plotly  
💻 **Reproducible Research** - All code included with detailed comments  
📊 **Computational Analysis** - Numerical implementations of theoretical concepts  
🔬 **Experimental Context** - Connection to real-world observations  

#### Applications  
These documents are valuable for:  
- **Graduate Students** - Learning advanced quantum physics with computational tools  
- **Researchers** - Reference implementations of complex theories  
- **Educators** - Teaching materials with interactive visualizations  
- **Data Scientists** - Applications of scientific computing in physics  
- **Physicists** - Quick reference for QFT and QCD calculations  
- **Mathematicians** - Research on Sidon sets and Erdős Problem 30  

#### Topics Covered  
#### Fundamental Physics  
- Quantum Field Theory  
- Quantum Chromodynamics  
- Gauge Theory (SU(3))  
- Special Relativity  
- Quantum Mechanics  

#### Advanced Concepts  
- Asymptotic Freedom  
- Color Confinement  
- Chiral Symmetry Breaking  
- Spin-Statistics Theorem  
- Virtual Particles  
- Quasiparticle Physics  
- Topological Order  
- Fracton Physics  
- Fractional Quantum Hall Quantum Geometry  
- Chiral Graviton Modes  
- Quantum Energy Teleportation  
- Subsystem Symmetry  

#### Mathematics  
- Sidon Sets (B₂ Sets)  
- Erdős Problem 30  
- Smoothing Certificates and Quadratic Programming Duality  
- Renewal Theory  

#### Computational Methods  
- Numerical field evolution  
- Mode expansion algorithms  
- Lattice simulations  
- Statistical analysis  
- Data visualization techniques  
- Exact diagonalization  
- Exact rational arithmetic verification  

#### Future Additions  
Planned additions to this repository:  
- String Theory implementations  
- Quantum Computing applications  
- Topological Quantum Field Theory  
- Non-equilibrium dynamics  
- Many-body quantum systems  
- Quantum information theory  

#### Contributing  
Contributions, suggestions, and discussions are welcome! Please feel free to:  
- Open an issue for questions or suggestions  
- Submit pull requests for improvements  
- Share how you've used these documents  

#### Citation  
If you use these materials in your research or teaching, please cite:  

Akwei, J. (2026). Science: Quantum Physics, Mathematics, and Advanced Topics in R.  
GitHub repository: https://github.com/johnakwei/Science  
Author  
John Akwei  
Senior Data Scientist  
Specializing in scientific computing, quantum physics, and data visualization  
License  
This project is licensed under the MIT License - see the LICENSE file for details.  
Acknowledgments  

Theoretical foundations based on established physics literature  
R visualization techniques inspired by the R community  
Computational methods following best practices in scientific computing  

Contact  
For questions, collaborations, or discussions:  
GitHub: @johnakwei  
Repository: Science  

## ⭐ Star this repository if you find it useful!  
Last updated: October 2026

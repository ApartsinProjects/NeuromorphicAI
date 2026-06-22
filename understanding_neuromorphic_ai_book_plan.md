# Understanding Neuromorphic AI
## From Spiking Neurons to Intelligent Edge Systems

**Proposed book type:** Comprehensive graduate-level textbook with plain-language explanations, intuition, diagrams, mathematical depth, programming demonstrations, assignments, and research-oriented projects.  
**Primary audience:** Advanced undergraduate students, graduate students, AI researchers, engineers, robotics researchers, edge-AI developers, and instructors building courses in neuromorphic computing or brain-inspired AI.  
**Pedagogical style:** Formal and rigorous, but every concept is introduced in plain language before mathematical and programming treatment. Every claim about efficiency or accuracy is evaluated critically, not taken at face value.

---

## 1. Book Vision

Neuromorphic AI studies intelligent systems that compute through sparse, temporal, event-driven signals, often using specialized hardware inspired by the nervous system. Unlike conventional deep learning, where computation is usually dense, synchronous, and frame-based, neuromorphic systems are designed to compute only when informative events occur.

The book presents neuromorphic AI as a practical and theoretical field connecting:

- Spiking neural networks
- Event-driven computation
- Low-power edge AI
- Neuromorphic hardware
- Event-based vision and sensing
- Local learning and plasticity
- Hybrid deep-learning/neuromorphic systems
- Robotics and autonomous systems
- Brain-inspired computation
- Hardware-software co-design

The main message of the book:

> Neuromorphic AI is not simply an attempt to copy the brain. It is an engineering and scientific approach to building AI systems that process time, events, sparsity, and energy constraints more naturally than conventional dense neural networks.

---

## 2. Book Positioning

### 2.1 What This Book Is

This book is a complete introduction and advanced guide to neuromorphic AI. It is usable as:

- A basic neuromorphic computing textbook
- A graduate course textbook on spiking neural networks
- A practical guide to event-based AI
- A research preparation book for neuromorphic AI students
- A bridge between deep learning, neuroscience-inspired models, and neuromorphic hardware
- A project-based course book for AI, robotics, or edge computing programs
- A practitioner's software handbook for building, training, profiling, reproducing, and deploying neuromorphic systems (Part XIII)

### 2.2 What This Book Is Not

The book avoids becoming only:

- A neuroscience textbook
- A hardware-only manual
- A pure mathematical treatment of dynamical systems
- A narrow book about spiking neural networks only
- A speculative book about artificial brains

Instead, it combines theory, algorithms, hardware, sensors, software, applications, and research methodology.

---

## 3. Learning Outcomes

After completing the book, readers should be able to:

1. Explain the difference between conventional AI and neuromorphic AI.
2. Understand spikes, membrane potential, thresholds, reset, leakage, and temporal dynamics.
3. Apply basic dynamical systems analysis (fixed points, stability, F-I curves) to spiking neuron models.
4. Implement basic spiking neurons from scratch.
5. Build, train, and evaluate spiking neural networks.
6. Understand major spike-coding strategies including rate, temporal, phase, and population coding.
7. Explain why training SNNs is difficult and how surrogate gradients, eligibility traces, and direct spike-timing methods address the problem.
8. Compare STDP, surrogate-gradient learning, eligibility-trace learning, ANN-to-SNN conversion, and hybrid approaches.
9. Process event-camera and event-stream data including stereo and 3D representations.
10. Understand the design principles of neuromorphic hardware, including digital chips, analog systems, memristive devices, and FPGAs.
11. Analyze accuracy-latency-energy tradeoffs using appropriate metrics.
12. Design neuromorphic edge-AI applications.
13. Evaluate neuromorphic systems using appropriate benchmarks and metrics.
14. Critically assess limitations, hype, and open problems, including the accuracy gap between SNNs and ANNs.
15. Read and understand modern neuromorphic AI research papers.
16. Build a final project involving SNNs, event-based sensing, edge inference, or hardware-aware neuromorphic design.
17. Select an appropriate, currently-maintained software stack and move a model across frameworks and hardware backends using NIR.
18. Engineer SNN training for performance (time-axis vectorization, `torch.compile`/JAX, gradient checkpointing) and verify correctness (gradient checking, property tests).
19. Produce reproducible neuromorphic experiments (environment lockfiles, config, tracking) and use AI coding assistants responsibly, with a reproducibility trail.
20. Take a trained SNN through the full deployment pipeline to edge hardware and report the simulation-to-hardware gap.

---

## 4. Pedagogical Principles

Each chapter follows a consistent structure:

1. **Plain-language motivation**  
   Explain why the topic matters.

2. **Core intuition**  
   Use analogies before formulas.

3. **Formal model**  
   Introduce equations and precise definitions.

4. **Algorithmic view**  
   Explain how the concept becomes an algorithm.

5. **Programming demonstration**  
   Use Python, PyTorch, snnTorch, Norse, Brian2, Nengo, Lava, or SpikingJelly.

6. **Visualization**  
   Include diagrams, spike rasters, membrane traces, event streams, architecture diagrams, and timing diagrams.

7. **Common mistakes**  
   Explain likely misunderstandings.

8. **Hype calibration box**  
   Each chapter's domain has at least one "what the evidence actually shows" callout box addressing common overclaims in the literature. Students who cannot distinguish evidence from advocacy are not graduate-ready.

9. **Mini-assignment**  
   Provide practice exercises.

10. **Research connection**  
    Link the topic to modern literature.

11. **Forward links**  
    Preview where the concept will appear again.

12. **Runnable recipe**  
    Where a chapter introduces a deployable technique, it ends with a self-contained, copy-pasteable **Recipe** (environment, data, model, train/eval, expected numbers, common failure points). Recipes target real, currently-maintained libraries and pinned versions, and are designed to run as-is on a single consumer GPU or a free cloud GPU.

### 4.1 Software-first philosophy

This book treats software as a first-class subject, not an afterthought to the math. The thesis: **in neuromorphic AI the gap between a published equation and a working, measured system is where most of the real difficulty lives** (training instability, sparsity that does not materialize, energy claims that do not survive measurement). A reader who finishes the book should be able to clone a repo, reproduce a result, profile it, change it, and ship it.

Concretely this adds the following recurring elements throughout the book, and a dedicated software part (Part XIII):

- **Library Spotlight** boxes: for each major technique, which maintained library implements it well in 2026, what version, and what it is good and bad at. The book commits to a small, current **core teaching stack** (snnTorch for learning, SpikingJelly for performance, Tonic for event data, NeuroBench for evaluation, NIR for interoperability) and treats everything else as situational.
- **Recipe / Cookbook** boxes (principle 12 above) and a consolidated cookbook chapter.
- **Vibe-Coding callouts**: short notes on using AI coding assistants (Claude Code, Copilot, Cursor) to implement, debug, and refactor neuromorphic code responsibly, with the reproducibility discipline that makes AI-assisted research code trustworthy.
- **Reproducibility checkpoints**: seeds, determinism, environment lockfiles, experiment tracking, and the gradient-correctness test that surrogate-gradient code specifically requires.
- **"What broke and why"** debugging notes drawn from real framework behavior (stale-PyPI vs active-GitHub gaps, deprecated SDKs, dead-neuron collapse, congestion on hardware).

### 4.2 Two reading paths

The book is explicitly designed for two audiences, signposted in every part:

- **Researcher path** (graduate students, researchers): emphasizes derivations, theory chapters (Parts II, XI, XII), open problems, and paper-review assignments. Each frontier chapter ends with a stated set of open problems.
- **Practitioner path** (engineers, edge-AI developers): emphasizes the Recipe boxes, Part XIII (software engineering), the deployment pipeline (Ch. 67), evaluation (Part XI), and the cookbook. A practitioner can read the recipe spine and the software part largely standalone.

A "How to read this book" front-matter page maps both paths chapter by chapter.

---

## 5. Required Diagram and Illustration Types

The book is illustration-heavy. Diagrams include:

- ANN neuron vs spiking neuron comparison
- Dense computation vs event-driven computation
- Action potential shape with labeled phases (brief, biological motivation only)
- Ion channel cartoon (qualitative, not Nernst-equation level)
- Phase plane diagram for 1D and 2D neuron dynamics
- Fixed point stability illustration
- F-I curve (firing rate vs input current)
- Bifurcation diagram (saddle-node)
- Timeline of spike trains
- Membrane potential charging and firing
- LIF neuron diagram
- Refractory period diagram
- Short-term plasticity facilitation and depression curves
- Rate coding vs temporal coding vs phase coding
- Raster plot explanation
- Stochastic vs deterministic spike train comparison
- Sparse coding dictionary visualization
- Predictive coding architecture diagram
- Feedforward SNN architecture
- Recurrent SNN architecture
- WTA circuit diagram
- CNN-to-SNN conversion pipeline
- BPTT unrolling diagram for SNNs
- Dead neuron illustration
- Surrogate function comparison (sigmoid, fast sigmoid, piecewise linear, triangle, arctan)
- STDP learning window
- Voltage-based STDP illustration
- Eligibility trace mechanism diagram
- Three-factor learning rule diagram
- Tempotron learning illustration
- Normalization pipeline (TDBN)
- Event-camera pixel event generation
- Event stream as sparse spatiotemporal data
- Address-event representation routing
- DAVIS combined frame+event camera
- Event-based stereo camera setup
- Event-based depth estimation pipeline
- Neuromorphic SLAM overview
- Neuromorphic chip block diagram
- Loihi 2 architecture block diagram
- FPGA SNN streaming dataflow diagram
- Crossbar array diagram
- Memristive synapse I-V curve
- RRAM vs PCM vs FeFET comparison table
- Edge AI system architecture
- Robotics perception-action loop
- Hardware-software co-design workflow
- Accuracy-latency-energy tradeoff triangle
- SOP (synaptic operations) counting illustration
- ANN/SNN/hybrid system comparison
- Accuracy gap chart: SNN vs ANN on standard benchmarks
- Course project pipeline
- Framework-selection decision tree (PyTorch vs JAX vs vendor SDK by goal/hardware)
- NIR interoperability graph: one model, many backends
- BPTT-over-time memory profile with and without gradient checkpointing
- Reproducible-experiment stack: environment / config / tracking / data layers
- Surrogate-gradient `gradcheck` workflow diagram
- AI-assisted development loop: spec → generate → test → trust, with the reproducibility trail
- SNN deployment pipeline: trained model → quantize → map → route → verify → profile (sim-to-hardware gap)

---

## 6. Full Book Structure

### 6.0 Master chapter map

Chapters are numbered in their original sequence; several later-numbered chapters (64-69) are placed inside earlier parts by topic, and Part XIII (70-76) is the software spine. The reading order is the part order below.

| Part | Chapters | Theme |
|---|---|---|
| I — Foundations of Neuromorphic AI | 1, 2, 3, 4, 5 | What/why, energy, ANN→SNN, biology, time |
| II — Mathematical Foundations of Neural Dynamics | 6 | Dynamical systems, F-I curve, populations |
| III — Spiking Neuron Models | 7, 8, 9, 10, 11 | LIF, richer models, synapses, noise, spike trains |
| IV — Neural Coding | 12, 13, 14, 15 | Coding schemes, encoding data, sparse/predictive |
| V — Spiking Neural Network Architectures | 16, 17, 18, 19, 20, 21, **64** | FF/RNN/Conv, reservoirs, generative, graphs, HDC |
| VI — Learning in Spiking Neural Networks | 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, **65, 66, 67** | Difficulty, STDP, supervised, surrogate, norm, conversion, deep, e-prop, RL, continual; no-backprop, compression, deployment |
| VII — Neuromorphic Hardware | 32, 33, 34, 35, 36, 37, 38 | Why, digital, analog, memristors, AER, FPGA/ASIC, co-design |
| VIII — Event-Based Sensing | 39, 40, 41, 42, 43 | Event cameras, algorithms, stereo/3D, audio, tactile/multimodal |
| IX — Neuromorphic AI Applications | 44, 45, 46, 47, 48, **68** | Edge, robotics, biomedical, industrial, autonomous; optimization |
| X — Hybrid Neuromorphic and Deep Learning Systems | 49, 50, 51, 52 | Hybrids/ODE/LTC, spiking Transformers/SSMs, distillation, foundation models |
| XI — Evaluation and Benchmarking | 53, 54, 55, 56, **69** | Metrics, datasets/NeuroBench, fair energy, limitations; explainability |
| XII — Research Frontiers | 57, 58, 59, 60, 61, 62, 63 | On-chip, scale, neuroscience, security, theory, photonics, future |
| **XIII — The Neuromorphic AI Software Stack (new)** | **70, 71, 72, 73, 74, 75, 76** | **Ecosystem/selection, NIR interop, performance, reproducibility, AI-assisted dev, cookbook, capstone** |
| Appendices | A-E | Math, deep learning, PyTorch, notation, glossary |

---

# Part I — Foundations of Neuromorphic AI

## Chapter 1 — What Is Neuromorphic AI?

### Purpose
Introduce neuromorphic AI as event-driven, sparse, temporal, energy-aware AI.

### Topics
- Definition of neuromorphic AI
- Neuromorphic computing vs conventional computing
- Brain-inspired computation vs brain simulation
- Dense AI vs sparse AI
- Clock-driven vs event-driven systems
- Frame-based vs event-based perception
- Why energy matters in AI
- Why memory movement is expensive
- Neuromorphic AI in edge devices
- Main promises and current limitations
- Brief history: Carver Mead's original vision through modern chips

### Key intuition
A conventional AI model behaves like a factory where every machine runs at every clock tick. A neuromorphic system behaves more like a security system: it reacts only when something changes.

### Hype calibration box
Neuromorphic AI is often described as achieving "1000x energy reduction." This chapter explains what that claim requires to be true, and why it often is not measured on real hardware with fair comparisons.

### Programming demo
Demonstrate the difference between dense frame processing and sparse event processing with a toy signal.

### Assignment
Explain three situations where event-driven computation may be better than dense computation, and three situations where it may not help.

---

## Chapter 2 — The Computational Problem: Energy, Time, and Sparsity

### Topics
- Why modern AI is computationally expensive
- FLOPs vs memory access costs (the memory wall)
- Latency and power consumption
- Edge AI constraints
- Size, weight, and power limitations
- Always-on sensing
- Sparse input streams
- Temporal redundancy in video and sensor data
- Event-driven opportunities
- Quantitative estimates: energy per MAC, energy per memory access

### Core concept
Neuromorphic AI is most useful when information is sparse in time and space.

### Programming demo
Compare processing all frames of a synthetic video versus processing only changed pixels. Measure operation counts.

### Assignment
Given a simple sensor stream, estimate how many operations are saved when processing only changes.

---

## Chapter 3 — From Artificial Neurons to Spiking Neurons

### Topics
- Artificial neuron: weighted sum and activation function
- Spiking neuron: spike as an event
- Membrane potential, threshold, reset, leakage
- Refractory period
- Time as part of computation
- Key comparison table: ANN vs SNN properties

### Key comparison
| Conventional ANN | Spiking Neural Network |
|---|---|
| Continuous activation | Discrete spike event |
| Usually synchronous | Often event-driven |
| Often dense | Often sparse |
| Static layer output | Time-dependent dynamics |
| Easy backpropagation | Harder training |
| Clock-driven | Event-driven or clocked |

### Programming demo
Implement a simple threshold neuron and then extend it into a leaky integrate-and-fire neuron.

### Assignment
Plot membrane potential over time for different input currents and thresholds.

---

## Chapter 4 — Biological Neurons: What We Borrow and Why

### Purpose
Provide the biological grounding needed to understand why neuromorphic design choices exist. This chapter is not a neuroscience course — it provides exactly the biological depth required to understand the engineering decisions made throughout the rest of the book, no more and no less.

### Topics
- The action potential: depolarization, threshold, repolarization, hyperpolarization (qualitative)
- Ion channels as voltage-controlled switches (conceptual, not Nernst/Hodgkin-Huxley)
- Why action potentials are all-or-nothing: the spike as a reliable digital event on an analog substrate
- Absolute and relative refractory periods and their functional purposes
- Synaptic transmission: neurotransmitter release, receptor binding, postsynaptic potential
- Excitatory (glutamate) vs inhibitory (GABA) synapses
- Dale's law: one neuron type, one neurotransmitter
- Excitatory/inhibitory balance and why it matters for stability
- Why biological neurons fire sparsely (~10 Hz average): metabolic cost of spikes
- Dendritic integration: spatial summation and temporal summation
- What neuromorphic AI borrows: spiking, locality, asynchrony, sparsity, event-driven communication
- What it does not borrow: exact ion-channel kinetics, detailed dendritic morphology, neuromodulator biochemistry

### Key intuition
An architect studying traditional buildings does not need to be a materials scientist, but must understand why adobe walls are thick and why straw is mixed into mud bricks. Similarly, understanding *why* biological neurons have refractory periods, or why cortex uses inhibitory interneurons, tells us what constraints shaped the neuromorphic design choices we study in this book.

### Hype calibration box
Neuromorphic AI is not building brains. The biological analogies are engineering inspiration, not engineering constraints. Every analogy in this book should be understood as a design principle extracted from biology, not as a commitment to biological accuracy.

### Assignment
For each of five neuromorphic design features (spiking communication, refractory period, local learning, sparse firing, event-driven processing), identify the biological origin and explain the engineering rationale for including or excluding it in a practical system.

---

## Chapter 5 — Time as a Computational Resource

### Topics
- Static input vs temporal input
- Time steps and event timing
- Spike latency and temporal integration
- State variables as short-term memory
- Dynamical systems view (preview of Part II)
- Short-term memory in neuron dynamics

### Key intuition
In a standard neural network, a neuron produces a number. In a spiking network, a neuron produces a sequence of events over time. Two spike trains with identical spike counts but different timing can produce completely different downstream responses.

### Programming demo
Simulate two spike trains with the same number of spikes but different timing. Show that timing changes the downstream response.

### Assignment
Create three spike trains with equal spike counts and compare how a leaky neuron responds to each.

---

# Part II — Mathematical Foundations of Neural Dynamics

## Chapter 6 — Dynamical Systems for Neural Computation

### Purpose
Provide the mathematical engine behind all spiking neuron models. Without this foundation, students can simulate LIF neurons but cannot reason about their stability, bifurcations, or population behavior. This chapter establishes tools that will be used in every subsequent chapter on neuron models, network dynamics, and learning.

### Topics
- Ordinary differential equations: first-order linear ODEs, existence and uniqueness, stability
- Phase plane analysis: 1D and 2D systems, nullclines, vector fields
- Fixed points and their classification: stable node, unstable node, saddle, spiral
- Linear stability analysis: Jacobian eigenvalues near fixed points
- Bifurcations relevant to neurons:
  - Saddle-node bifurcation (type I excitability, found in LIF-like systems)
  - Andronov-Hopf bifurcation (type II excitability, oscillatory onset)
- The F-I curve: derivation from the LIF equation (closed-form expression for firing rate as a function of input current, time constant, and threshold)
- Mean-field approximation: approximating a population of spiking neurons as a rate equation
- Lyapunov stability: qualitative introduction, applied to recurrent networks
- Euler method and its error: why discrete-time simulation approximates continuous dynamics
- Population dynamics and Wilson-Cowan equations:
  - The Wilson-Cowan model: coupled ODEs for excitatory (E) and inhibitory (I) population firing rates
  - Nullclines, fixed points, and limit cycles in the E-I plane
  - How the Wilson-Cowan model predicts oscillations, bistability, and winner-take-all dynamics
  - The mean-field limit: when does a population of LIF neurons reduce to a rate equation?
- Fokker-Planck equation for stochastic LIF populations:
  - Probability density of membrane potentials across a population
  - Steady-state population firing rate as a function of input statistics
  - Why this matters: large-scale neuromorphic simulation and network design
- Neural mass models and their connection to EEG: Jansen-Rit, Wendling models (brief)

### Mathematical core
Derive the LIF firing rate formula from first principles. Given the LIF ODE:

```
tau * dV/dt = -(V - V_rest) + R * I
```

When V reaches threshold V_th, a spike fires and V resets to V_r. The inter-spike interval T can be computed exactly, giving firing rate f = 1/T as an explicit function of I, tau, V_th, V_r, and V_rest.

### Programming demo
Plot the phase plane of a 2D neuron model (e.g., FitzHugh-Nagumo as a simple illustration). Show nullclines, vector field, and fixed point. Then plot the F-I curve of the LIF analytically and compare it with simulation.

### Assignment
For the standard LIF neuron parameters, compute the minimum input current required to make the neuron fire (the rheobase current). Verify analytically and experimentally.

---

# Part III — Spiking Neuron Models

## Chapter 7 — The Leaky Integrate-and-Fire Neuron

### Topics
- Integrate-and-fire neuron history (Lapicque 1907)
- Leaky integrate-and-fire neuron: derivation, parameters
- Membrane time constant and its physical interpretation
- Threshold crossing and the reset mechanism
- Soft reset vs hard reset (important for hardware)
- Refractory period implementation
- Discrete-time simulation: Euler method applied to LIF
- F-I curve from Chapter 6: plotting and interpretation
- Effect of input noise on firing threshold (preview of stochastic LIF)

### Mathematical core
Introduce the LIF equation step by step:

```
Change in membrane potential = input contribution - leak contribution
tau_m * dV/dt = -(V - V_rest) + R_m * I(t)
if V >= V_th:  fire spike; V <- V_r; enter refractory for t_ref
```

Derive the discrete-time update. Compute the F-I curve analytically. Derive the rheobase current.

### Programming demo
Implement LIF simulation from scratch in NumPy. Plot membrane potential, spike times, and the F-I curve.

### Assignment
Study how firing rate changes when input current, threshold, leak time constant, and refractory period are varied. Verify against the analytical F-I curve.

---

## Chapter 8 — Beyond LIF: Richer Spiking Neuron Models

### Topics
- Spike Response Model (SRM): Gerstner's generalized formulation, kernel-based description
  - Why SRM matters: it provides a unified framework for many neuron types and is the theoretical basis for several learning rules
- Exponential integrate-and-fire (EIF): sharp threshold, better biological approximation
- Adaptive exponential integrate-and-fire (AdEx): spike-frequency adaptation, bursting
- Izhikevich model: two-variable system, wide range of firing patterns with low compute cost
- Hodgkin-Huxley model at high level: historical importance, four-variable conductance-based system
- Current-based vs conductance-based synapses: formal difference and computational implications
- Type I vs Type II excitability: classification table (LIF, EIF, AdEx, Izhikevich, HH)
- Biological realism vs computational efficiency: the tradeoff for edge deployment

### Programming demo
Compare LIF, AdEx, and Izhikevich firing patterns for the same input current. Show bursting and adaptation in AdEx that LIF cannot reproduce.

### Assignment
Explain which neuron model is suitable for edge AI (justify with computational cost), which for neuroscience simulation (justify with biological realism), and which for expressive temporal processing (justify with dynamics).

---

## Chapter 9 — Synapses, Delays, and Network Dynamics

### Topics
- Synaptic weights: excitatory and inhibitory
- Current-based vs conductance-based synapses: formal derivation of both
- Synaptic delay: propagation delay, its role in temporal computation
- Postsynaptic current models: exponential, alpha function, double-exponential
- Short-term plasticity (STP): synaptic facilitation and depression (Tsodyks-Markram model)
  - Formal ODE for STP variables
  - Why STP matters for temporal pattern recognition and burst detection
- Gap junctions and electrical synapses: brief treatment, role in oscillations
- Recurrent connections: feedback, oscillation, stability
- Excitatory/inhibitory balance: why imbalance leads to runaway activity

### Programming demo
Build a two-neuron excitatory/inhibitory circuit with synaptic delay. Demonstrate oscillation and quenching.

### Assignment
Add short-term depression to a synapse in your simulation. Show how depression limits sustained high-frequency input and acts as a temporal high-pass filter.

---

## Chapter 10 — Stochastic Spiking Networks and Noise

### Purpose
Noise is not a nuisance to be minimized: it is a fundamental property of biological and silicon spiking systems. This chapter provides the formal tools to reason about variability, stochastic models, and probabilistic inference with spikes.

### Topics
- Sources of noise: thermal noise, channel noise, input variability, network noise
- Ornstein-Uhlenbeck process as a model for noisy synaptic input
- Stochastic LIF: membrane potential diffusion between threshold crossings
- Escape noise model: spike probability as a soft exponential function of membrane potential
  - This is the probabilistic/Bayesian interpretation of spiking
  - Connection to the logistic function and sigmoid surrogate gradients (preview of Chapter 25)
- Population codes as noise averaging: why many neurons coding the same variable reduce noise
- Spike train statistics: Poisson assumption, Fano factor, coefficient of variation
- Noise robustness vs training difficulty tradeoff
- Bayesian brain hypothesis: spikes as samples from a posterior distribution (brief, conceptual)
- Stochastic SNNs in practice: Monte Carlo dropout via stochastic neurons
- Variability in hardware: device mismatch in analog and digital chips (preview of Part VII)

### Programming demo
Simulate a stochastic LIF neuron with Ornstein-Uhlenbeck input. Plot the distribution of inter-spike intervals and compare to a Poisson process.

### Assignment
Implement the escape noise model (soft threshold) and compare its surrogate gradient properties to the hard-threshold model. Plot how the probability of firing changes with membrane potential for different noise levels.

---

## Chapter 11 — Spike Trains and Raster Plots

### Topics
- Spike train representation: binary time series, event list
- Raster plots: construction and interpretation
- Inter-spike interval distribution
- Firing rate: instantaneous, mean, population
- Spike count statistics
- Synchrony and correlation across neurons
- Population activity and population vectors
- Noise in spike trains: measurement and characterization

### Programming demo
Generate raster plots for a small SNN under different input conditions.

### Assignment
Analyze spike patterns under different input noise levels. Compute the coefficient of variation of inter-spike intervals and compare to a Poisson process.

---

# Part IV — Neural Coding

## Chapter 12 — How Spikes Represent Information

### Topics
- Why coding matters: the same information can be represented many ways
- Rate coding: firing rate as the signal
- Temporal coding: precise spike timing carries information
- Latency coding: time-to-first-spike
- Rank-order coding: which neuron fires first
- Time-to-first-spike coding
- Population coding: a vector of firing rates represents a stimulus
- Sparse coding: few active neurons represent each stimulus
- Phase coding: spike phase relative to a background oscillation
  - Relevance to theta-gamma oscillations in hippocampus
  - Engineering relevance: phase encoding in clocked neuromorphic chips
- Burst coding: number of spikes in a burst encodes stimulus intensity
- The binding problem: how do neurons represent composed objects?
- Information-theoretic analysis: channel capacity of spike trains, entropy of spike timing

### Programming demo
Encode a scalar value using rate coding, latency coding, and population coding. Compare information content.

### Assignment
Compare latency, spike count, robustness to noise, and decoding accuracy for three coding schemes on the same input.

---

## Chapter 13 — Encoding Static Data into Spikes

### Topics
- Image-to-spike conversion
- Poisson encoding: rate proportional to pixel intensity
- Deterministic rate coding
- Latency encoding
- Population encoding with Gaussian tuning curves
- Normalization and scaling for stable training
- Encoding tabular data
- Encoding text features

### Programming demo
Convert MNIST images into spike trains using three encoding methods.

### Assignment
Train a small SNN using Poisson encoding and latency encoding. Compare accuracy and spike count.

---

## Chapter 14 — Encoding Temporal and Event Data

### Topics
- Sensor streams and event streams
- Spike representation for audio
- Spike representation for motion
- Event-camera data format
- Temporal batching and windowing
- Event accumulation strategies
- Sparse tensors for event data

### Programming demo
Convert a synthetic moving object trajectory into event-like spike data and visualize.

### Assignment
Design an encoding pipeline for ECG, audio, or accelerometer data.

---

## Chapter 15 — Sparse Coding and Predictive Coding

### Purpose
Sparse and predictive coding bridge neural coding theory with unsupervised learning, explain why sparsity is computationally efficient, and provide the conceptual foundation for local learning rules. This chapter pays off in Part VI (learning) and Part XII (neuroscience and cognitive models).

### Topics
- Sparse representations: motivating example from natural image statistics
- Olshausen and Field (1996): sparse coding and dictionary learning
  - Sparse coding as an optimization problem: minimize reconstruction error subject to sparse activations
  - Connection to LASSO and compressive sensing
- Why sparse codes are computationally efficient: fewer synaptic operations per inference step
- Lateral inhibition and competition as sparsity enforcement in networks
- Winner-Take-All (WTA) circuits: k-WTA and softmax-WTA, implementation in SNNs
- Inhibitory interneurons as sparsity regulators
- Predictive coding: each layer predicts the layer below, passes only residuals upward
  - Rao and Ballard (1999): hierarchical predictive coding in visual cortex
  - Predictive coding as an energy-based model
  - Connection to variational autoencoders and self-supervised learning
- Free energy principle: conceptual introduction only (Friston)
  - Active inference brief treatment: perception as minimizing prediction error
- Connection to modern deep learning: contrastive self-supervised methods as approximate predictive coding

### Programming demo
Implement sparse dictionary learning on image patches. Visualize learned receptive fields and compare to Gabor filters and early visual cortex responses.

### Assignment
Build a WTA circuit in a spiking network. Show that it learns selectivity to repeated input patterns without explicit supervision.

---

# Part V — Spiking Neural Network Architectures

## Chapter 16 — Feedforward Spiking Neural Networks

### Topics
- Layered SNNs: input, hidden, output
- Output decoding strategies:
  - Spike-count decoding
  - Time-to-first-spike decoding
  - Membrane potential decoding (last step)
  - Rate decoding with temporal pooling
- Classification pipeline with SNNs
- Depth vs temporal length tradeoffs
- Skip connections in deep SNNs

### Programming demo
Build a feedforward SNN for digit classification on N-MNIST.

### Assignment
Compare output decoding methods on the same trained network.

---

## Chapter 17 — Recurrent Spiking Neural Networks

### Topics
- Recurrent dynamics and the role of feedback
- Memory through state: attractor networks
- Lyapunov stability applied to spiking recurrent networks (using tools from Chapter 6)
- Balanced E/I networks: when excitation and inhibition cancel to produce stability
- Chaotic regimes and edge-of-chaos: computational power vs. stability tradeoff
- Echo state property: preview of reservoir computing (Chapter 19)
- Temporal sequence processing with recurrent SNNs
- Spiking RNNs vs classical RNNs: formal comparison

### Programming demo
Build a recurrent SNN for a simple sequence task. Analyze how recurrent weight scaling affects stability.

### Assignment
Compute the dominant eigenvalue of the recurrent weight matrix and relate it to observed dynamics (stable, oscillating, chaotic).

---

## Chapter 18 — Convolutional Spiking Neural Networks

### Topics
- Convolution in SNNs: shared weights applied to spike maps
- Spiking feature maps across time
- Spatial and temporal pooling for SNNs
- Event-based convolution: processing only at spike locations
- Frame-based vs event-based convolution: operation count comparison
- CNN-to-SNN conversion pipeline for convolutional layers
- Spiking object recognition benchmarks

### Programming demo
Train a convolutional SNN on CIFAR10-DVS using SpikingJelly or snnTorch.

### Assignment
Compare a standard CNN and a convolutional SNN on accuracy, total spike count, and estimated synaptic operations.

---

## Chapter 19 — Reservoir Computing and Liquid State Machines

### Topics
- Reservoir computing: the separation property and approximation property
- Echo state networks: formal echo state property and sufficient conditions
- Eigenvalue analysis of the reservoir weight matrix: spectral radius and its role
- Liquid state machines: spiking reservoirs
- Random recurrent spiking networks
- Readout training: linear regression on reservoir states
- When readout-only training suffices vs. when it fails
- Temporal feature extraction: what the reservoir does and does not learn
- Edge-friendly sequence processing

### Programming demo
Build a liquid state machine for temporal classification. Compare with training the full network.

### Assignment
Vary the spectral radius of the reservoir weight matrix and measure classification accuracy. Identify the critical transition near spectral radius 1.

---

## Chapter 20 — Generative and Unsupervised Spiking Models

### Purpose
Classification is not the only task. This chapter provides the generative modeling foundations needed for on-device learning, anomaly detection, and representation learning.

### Topics
- Winner-Take-All circuits as unsupervised feature detectors
  - k-WTA, lateral inhibition implementation
- Competitive learning: Hebbian + WTA = self-organization
- Self-organizing maps with spiking neurons
- Spiking restricted Boltzmann machines (sRBM): brief treatment
- Spiking variational autoencoders: reconstruction with spike-based latent variables
- Generative replay for continual learning (preview of Chapter 31)
- Why generative models matter for edge adaptation: on-device learning without labeled data
- Anomaly detection as density estimation with spiking models

### Programming demo
Build a WTA-based unsupervised feature learner. Train on MNIST and visualize learned prototypes.

### Assignment
Use the WTA model as a feature extractor followed by a linear readout. Compare accuracy with the STDP baseline from Chapter 23.

---

## Chapter 21 — Spiking Graphs and Structured Data

### Topics
- Graph neural networks and SNNs: motivation
- Event propagation on graphs
- Spiking message passing
- Temporal graphs
- Neuromorphic graph processing
- Sensor networks as temporal graphs

### Programming demo
Create a small spiking graph network for signal propagation over a sensor network graph.

### Assignment
Model a multimodal sensor network as a spiking graph. Analyze how connectivity affects propagation latency.

---

## Chapter 64 — Hyperdimensional Computing and Vector-Symbolic Architectures

### Purpose
Hyperdimensional computing (HDC) is a biologically inspired, hardware-friendly computing paradigm distinct from spiking neural networks but closely related in its hardware targets and application domains. It is gaining rapid traction in edge AI, neuromorphic IoT, and low-power classification. A 2026 graduate student will encounter HDC alongside SNN-based approaches and needs a principled comparison.

### Topics
- The core idea: represent information as high-dimensional (D = 10,000) binary or bipolar vectors called hypervectors
  - Analogy: the brain likely encodes information in high-dimensional distributed patterns, not low-dimensional compact representations
- Vector-symbolic algebra: three operations that make HDC work
  - Binding (element-wise XOR or multiplication): creates a new vector that represents a composed concept
  - Superposition (element-wise addition + thresholding): bundles multiple vectors into one
  - Permutation (cyclic shift): encodes sequence position or temporal order
- Similarity measurement: Hamming distance for binary, cosine for bipolar
- Encoding pipelines: how sensory inputs become hypervectors
  - Random projection encoding for continuous values
  - Level hypervectors for scalar encoding
  - Temporal binding for time-series
- Classification with hypervectors: associative memory (class hypervectors) and cosine-similarity lookup
- Learning in HDC: online single-pass learning, iterative refinement
- Why HDC is hardware-friendly:
  - All operations are bitwise (XOR, OR, popcount) — no multiplication
  - Memory-centric: inference is a table lookup, not a matrix multiply
  - Natural mapping to SRAM, TCAM, and neuromorphic memory
- HDC vs SNN: comparison table on accuracy, latency, energy, interpretability, task fit
- HDC implementations on neuromorphic hardware: FPGA and custom ASIC demonstrations
- Applications: ECG classification, gesture recognition, language detection, drug discovery encoding
- Limitations: accuracy ceiling on complex tasks, dimensionality vs power tradeoff
- Emerging HDC + SNN hybrids: using SNN as encoder, HDC as classifier

### Programming demo
Implement an HDC classifier for gesture recognition or EMG signal classification in Python. Compare accuracy and inference speed with a small SNN classifier.

### Assignment
Design an HDC encoding pipeline for ECG anomaly detection. Justify the choice of D and the similarity threshold. Estimate memory footprint and compare to an equivalent SNN.

---

# Part VI — Learning in Spiking Neural Networks

## Chapter 22 — Why Training SNNs Is Difficult

### Topics
- Non-differentiable spike function: the hard threshold problem
- The dead neuron problem: when a neuron never fires, its gradient is always zero
- Gradient cliffs near spike events
- Discrete events and the credit assignment problem over time
- Backpropagation through time (BPTT) for SNNs:
  - Formal unrolling of the SNN computation graph
  - Vanishing and exploding gradients across time steps
  - Diagram of gradient flow paths and dead zones
- Training-time vs inference-time sparsity mismatch
- The accuracy gap: SNNs trained from scratch currently trail ANNs by 5-15% on most benchmarks
  - This fact should be stated plainly and returned to throughout the book
  - Understanding why this gap exists is a research frontier

### Key intuition
Backpropagation expects smooth, differentiable changes. A spike is a discontinuous jump from zero to one and back. Standard gradient descent cannot navigate through that discontinuity without additional machinery.

### Hype calibration box
Many neuromorphic AI papers report accuracy on converted ANNs, not trained-from-scratch SNNs. Conversion-based accuracy is an upper bound set by the original ANN, not a demonstration of what SNNs can do natively. Students should check which training method was used before accepting accuracy comparisons.

### Programming demo
Show explicitly why the derivative of a hard threshold function is zero everywhere except at the threshold itself, and why this kills gradients in a training loop.

### Assignment
Implement a minimal SNN training loop without any surrogate gradient. Measure gradient magnitudes and identify dead neurons.

---

## Chapter 23 — Hebbian Learning and STDP

### Topics
- Hebbian learning: formal rule, "neurons that fire together wire together"
- Spike-timing-dependent plasticity (STDP): the classical exponential window
- Causal (pre before post) vs anti-causal (post before pre) timing
- Voltage-based STDP (Clopath et al. 2010): more biologically realistic, weight-dependent updates
- BCM learning rule (Bienenstock-Cooper-Munro): sliding threshold for stability
- Triplet STDP: three-spike interactions for matching natural image statistics
- Unsupervised feature learning with STDP
- Local learning rule: no global error signal needed
- Stability problems: weight explosion, normalization strategies (soft weight bounds, multiplicative normalization)
- Connection to sparse coding (Chapter 15): STDP + WTA drives sparse representations

### Programming demo
Implement classical STDP for a simple pattern-learning task. Visualize weight evolution.

### Assignment
Use STDP to make a neuron become selective to a repeated input pattern. Add multiplicative weight normalization and show it prevents runaway potentiation.

---

## Chapter 24 — Supervised Spike-Timing Learning: SpikeProp, Tempotron, and ReSuMe

### Purpose
A decade of important supervised learning algorithms precedes surrogate gradients. Understanding them clarifies the problem structure and provides temporal credit assignment intuition.

### Topics
- SpikeProp (Bohte et al. 2000): first supervised SNN training via gradient descent on spike times
  - The key insight: spike time is differentiable with respect to input timing and weights
  - Limitation: only works when neurons are guaranteed to fire
- The tempotron (Gütig & Sompolinsky 2006): fire-or-not decision learning
  - Learning rule: potentiate if the neuron should have fired but did not; depress otherwise
  - Connection to perceptron learning
  - Elegant formulation, limited to binary output
- ReSuMe (remote supervised method, Ponulak 2005): online learning of target spike trains
  - Teacher forcing with a Hebbian/anti-Hebbian combination
  - Can learn arbitrary spike timing patterns
- Why these methods matter even though surrogate gradients now dominate:
  - They isolate the temporal credit assignment problem clearly
  - They motivate the surrogate approach (what problem is being solved)
  - They are still used in constrained hardware scenarios
- Limitations: sensitivity to initialization, difficulty scaling to deep networks

### Programming demo
Implement the tempotron on a toy classification problem. Compare convergence speed with an STDP baseline.

### Assignment
Implement SpikeProp for a two-layer network with guaranteed output firing. Identify when the algorithm fails and why.

---

## Chapter 25 — Surrogate Gradient Learning

### Topics
- The surrogate gradient idea: forward pass uses hard spikes, backward pass uses smooth approximation
- Straight-through estimator (STE) connection: the original idea from binary neural networks
- Surrogate function families and their properties:
  - Sigmoid: smooth, nonzero almost everywhere, but saturates far from threshold
  - Fast sigmoid (BoxcarPseudoDerivative): limited support, sharp
  - Piecewise linear: simple, hardware-friendly
  - Triangle: symmetric, common in hardware implementations
  - Arctan: avoids saturation in tails
- Visual comparison of surrogate functions and their derivatives
- Approximation error analysis: how far does the surrogate deviate from the true subgradient?
- Backpropagation through time for spiking networks: formal derivation
- Loss functions for SNNs: cross-entropy on spike counts, temporal losses on spike timing, membrane potential loss
- Dead neuron handling: what each surrogate does when a neuron never fires
- Practical training tricks: gradient clipping, learning rate schedules, weight initialization
- State-of-the-art accuracy with surrogate gradients: what is achievable and on what benchmarks

### Programming demo
Train an SNN with surrogate gradients using snnTorch. Compare at least three surrogate functions on N-MNIST or CIFAR10-DVS.

### Assignment
Implement two different surrogate functions from scratch in PyTorch (as custom autograd functions). Compare training curves and final accuracy.

---

## Chapter 26 — Normalization in Spiking Networks

### Purpose
Normalization is not a minor detail. Without it, SNN training is unstable and ANN-to-SNN conversion loses accuracy. This chapter unifies normalization concepts scattered across training chapters.

### Topics
- Why standard batch normalization (BN) is problematic for SNNs:
  - BN normalizes across the batch at a single time step; spike rates vary across time
- Threshold-Dependent Batch Normalization (TDBN, Zheng et al.):
  - Formal derivation: normalize membrane potential, adjust threshold accordingly
  - Connection between BN and threshold scaling
- Temporal batch normalization: normalizing across both batch and time dimensions
- Online normalization: computing running statistics during inference for hardware deployment
- Weight normalization for ANN-to-SNN conversion:
  - Weight normalization to prevent saturation at high spike rates
  - Threshold balancing: setting thresholds to match ANN activation ranges
- Neuron-level vs layer-level normalization: tradeoffs for hardware
- Practical training recipe: initialization, learning rate, normalization order

### Programming demo
Train the same deep convolutional SNN with and without TDBN. Compare training stability and final accuracy.

### Assignment
Implement threshold balancing for ANN-to-SNN conversion. Measure accuracy as a function of the number of time steps for the converted SNN.

---

## Chapter 27 — ANN-to-SNN Conversion

### Topics
- Motivation: trained ANNs reach higher accuracy than from-scratch SNNs in most settings
- The conversion procedure: training an ANN, then mapping activations to firing rates
- Activation function compatibility: ReLU maps to firing rate naturally; others do not
- Weight normalization for conversion
- Threshold balancing (using tools from Chapter 26)
- Latency-accuracy tradeoff: more time steps yield higher accuracy but higher latency
- Limitations of conversion:
  - Conversion requires many time steps to converge to ANN accuracy
  - Temporal dynamics (recurrence, short-term plasticity) are not captured
  - Quantization noise grows with network depth
- Conversion for CNNs: handling pooling, batch normalization
- Recent hybrid methods: conversion initialization followed by fine-tuning with surrogate gradients

### Programming demo
Train a small ANN and convert it to an SNN. Plot accuracy vs number of time steps.

### Assignment
Measure how many time steps are needed for the converted SNN to reach ANN accuracy. Explain the tradeoff.

---

## Chapter 28 — Direct Training of Deep SNNs

### Topics
- Deep SNN training with surrogate gradients
- Temporal loss functions: counting spikes at the output vs matching spike timing
- Spike regularization: L1 penalty on spike count to enforce sparsity
- Gradient stabilization: gradient clipping, adaptive learning rates
- Residual SNNs: adding skip connections to enable deeper training
- Normalization for deep SNNs (using Chapter 26)
- Training convolutional SNNs: activation maps as spike rate maps
- Training recurrent SNNs: truncated BPTT, gradient truncation
- Temporal Efficient Training (TET, arXiv 2202.11946): gradient reweighting toward flatter minima; the now-standard baseline (~83% on DVS-CIFAR10) every 2024-2026 paper benchmarks against — teach it before the reversible methods.
- Parallel and scalable SNN training:
  - The sequential bottleneck of BPTT: each time step depends on the previous
  - Temporal reversible SNNs: storing only activations at checkpoints, recomputing others
  - T-RevSNN (ICML 2024, arXiv 2405.16466): disables temporal dynamics in most neurons with multi-level reversible interactions at selected neurons — O(L) training memory, O(1) inference, ~8.6x memory reduction and ~2x training speedup with competitive ImageNet accuracy. The headline reversible-training result.
  - ParaRevSNN (2025, arXiv 2508.01223): parallel + reversible combined (newer, less vetted)
  - Constant-time parallel training via reversible architectures
  - Parallelism across the time dimension vs. across the batch dimension
  - Practical speedups and memory savings on multi-GPU training; cross-reference the performance-engineering treatment in Ch. 72

### Programming demo
Train a multi-layer convolutional SNN on CIFAR10-DVS. Add spike-count regularization and measure its effect on spike rate and accuracy.

### Assignment
Study the effect of network depth on training stability without and with skip connections. Estimate the memory savings from gradient checkpointing across the temporal dimension.

---

## Chapter 29 — Online Learning and Eligibility Traces

### Purpose
Surrogate gradient learning requires unrolling the full computation graph across time, which is computationally expensive and biologically implausible. Eligibility traces provide a mechanism for credit assignment in continuous time and are the leading approach for on-chip and online learning. This is arguably the most important missing piece in standard SNN education.

### Topics
- The temporal credit assignment problem in continuous time: why BPTT is expensive
- Eligibility traces: a neuron remembers a decaying record of recent activity
- Three-factor learning rule: presynaptic trace × postsynaptic trace × neuromodulatory signal
  - The neuromodulatory signal as a global reward or error broadcast
  - Connection to dopaminergic reward signals in biology
- e-prop (Bellec et al. 2020, Nature Communications): formal derivation
  - Forward computation of eligibility traces
  - Approximation to the exact gradient: what is gained and what is lost
  - Empirical performance: competitive with BPTT on temporal tasks
- Real-time recurrent learning (RTRL): exact online gradient, O(n^4) complexity
- Forward gradient methods: alternatives to BPTT that compute gradients in forward time
- Connection to STDP: e-prop with a reward signal reduces to a form of reward-modulated STDP
- When eligibility traces work: tasks with temporal structure, online settings
- When they fail: very long-range dependencies, tasks requiring precise multi-layer credit
- Hardware implementation of eligibility traces: local storage, trace decay circuits

### Programming demo
Implement e-prop for a simple temporal classification task. Compare convergence with BPTT using surrogate gradients.

### Assignment
Implement a three-factor learning rule for a single spiking neuron receiving a delayed reward. Show that the weight update correctly reinforces the spike that preceded the reward.

---

## Chapter 30 — Reinforcement Learning with SNNs

### Topics
- Sequential decision making and the reward signal
- Spiking policies: state encoding, action selection via spike counts
- Reward-modulated STDP (R-STDP): formal derivation using eligibility traces from Chapter 29
- Policy gradients with SNNs: REINFORCE applied to spiking policies
- Actor-critic SNNs: separate spiking value and policy networks
- Neuromorphic control: low-latency sensorimotor loops
- Robotics applications: locomotion, obstacle avoidance, reaching

### Programming demo
Train a simple spiking policy on a toy control problem (e.g., CartPole).

### Assignment
Compare convergence speed and policy quality between a conventional neural policy and a spiking neural policy.

---

## Chapter 31 — Continual, Online, and Local Learning

### Topics
- Online learning: updating weights from a stream of data without storing all data
- Continual learning: adapting to new tasks while preserving old ones
- Catastrophic forgetting: the core challenge of continual learning
- Generative replay with spiking models (uses Chapter 20): rehearsing old data via generation
- Plasticity and stability tradeoff
- Eligibility traces for online continual learning (uses Chapter 29)
- Neuromodulation as a gating mechanism: dopamine-like signals control when learning occurs
- Local learning vs global optimization: what can be done with only local information
- On-chip learning: hardware requirements and constraints (preview of Chapter 57)
- Federated learning with SNNs:
  - Why federated learning matters for neuromorphic edge devices: data privacy, no cloud upload
  - Federated averaging with spiking models: aggregating local SNN updates
  - Communication efficiency: spikes as compressed gradients
  - Heterogeneous SNN architectures across devices: challenges for aggregation
  - Privacy-preserving on-chip learning without raw data sharing
- Neuromorphic continual learning as a recognized subfield (survey: arXiv 2410.09218):
  - CG-SNN: context-gated spiking network for task-incremental learning
  - ALADE-SNN: adaptive latent-space replay with differential evolution
  - Replay4NCL: generative replay with neuromorphic constraints (arXiv 2503.17061)
  - Neuromodulated SNNs: dopamine-inspired gating of plasticity
  - Comparison table: rehearsal, regularization, architectural, and parameter-isolation strategies applied to SNNs
  - Why SNNs are well-suited for continual learning: event-driven updates avoid interference between tasks

### Programming demo
Train an SNN sequentially on two tasks using only local learning rules. Measure forgetting. Compare with a replay-based method.

### Assignment
Design a local learning rule for a changing sensor environment using eligibility traces and a reward signal. Add a simple generative replay mechanism and measure forgetting reduction.

---

## Chapter 65 — Biologically Plausible Learning Without Backpropagation

### Purpose
Backpropagation requires a global error signal propagated backward through every layer — a mechanism with no known biological analogue. This chapter surveys the growing family of learning algorithms designed to work without backpropagation, motivated both by biological plausibility and by on-chip learning constraints.

### Topics
- The biologically implausible aspects of backpropagation:
  - Weight transport problem: the backward pass requires exact transposed weights
  - Update locking: layers cannot update until the forward and backward passes are complete
  - Non-local learning signals: a synapse must "know" about error at the output
- Local learning rules that sidestep these problems:
  - STDP + WTA (already in Chapter 23): local but unsupervised
  - Eligibility traces + neuromodulation (Chapter 29): local credit with delayed reward
- Forward-Forward algorithm (Hinton 2022):
  - Trains each layer independently to distinguish "positive" data (real) from "negative" data (corrupted)
  - No backward pass needed: each layer optimizes its own local objective
  - Formal specification: goodness function, contrastive update rule
  - Accuracy on MNIST and CIFAR relative to backpropagation
  - Hardware implications: layers can update in parallel, no gradient buffers needed
- Contrastive Hebbian learning (CHL) and equilibrium propagation (Scellier & Bengio 2017):
  - Networks settle to a free phase and a nudged phase; synapses update from the difference
  - Connection to energy-based models
  - Hardware realization: continuous-time settling replaces discrete forward-backward passes
- Target propagation and difference target propagation:
  - Each layer receives a local target rather than a gradient
  - Decouples layers from the output error
- Dendritic error learning (Guerguiev et al. 2017, Sacramento et al. 2018):
  - Apical dendrites receive top-down error signals, basal dendrites receive bottom-up input
  - Two-compartment neuron model as the hardware primitive
  - Why this is relevant to neuromorphic chip design
- Comparison table: accuracy, hardware cost, biological plausibility, scalability
- When local learning is enough and when it is not

### Programming demo
Implement the Forward-Forward algorithm for a two-layer network on MNIST. Compare convergence and final accuracy with surrogate-gradient backpropagation.

### Assignment
Identify two hardware constraints that make standard backpropagation difficult to implement on a neuromorphic chip. For each, explain which of the local learning rules above addresses it.

---

## Chapter 66 — SNN Model Compression: Pruning and Quantization

### Purpose
Deploying SNNs on memory- and compute-constrained hardware requires model compression. Pruning and quantization for SNNs have important differences from their ANN counterparts due to spike sparsity, temporal dynamics, and binary activations.

### Topics
- Why compression matters for neuromorphic deployment: core count, routing budget, memory limits
- Structured vs unstructured pruning:
  - Weight magnitude pruning: removing small weights
  - Neuron pruning: removing entire neurons and their connections
  - Layer-wise pruning: removing entire layers
  - Spike-activity-based pruning: removing neurons that rarely fire
- Pruning SNNs: interactions with sparsity and temporal dynamics
  - Pruned weight = no routing event: double benefit on neuromorphic hardware
  - Sensitivity analysis: which layers tolerate pruning and which do not
  - Lottery ticket hypothesis applied to SNNs
- Quantization for SNNs:
  - Weights are already binary or sparse (spikes): the weight quantization problem
  - Fixed-point weight quantization: bit-width selection for neuromorphic cores
  - Quantization-aware training (QAT) for SNNs: training with simulated fixed-point noise
  - Threshold quantization: rounding thresholds to hardware-representable values
  - Post-training quantization vs QAT: accuracy tradeoffs
- Combining pruning and quantization: joint compression pipelines
- Neural architecture search (NAS) for SNNs: automatically finding efficient architectures
  - Spike-NAS and SNN-NAS approaches
  - Hardware-aware NAS: optimizing for core utilization and routing load
- Practical compression recipes: step-by-step for Loihi 2 and FPGA targets

### Programming demo
Apply magnitude-based weight pruning to a trained convolutional SNN. Measure accuracy vs. sparsity curve. Apply fixed-point quantization and measure the combined accuracy-memory tradeoff.

### Assignment
Starting from a trained SNN, apply progressive pruning and quantization to achieve a 10× reduction in model size while losing less than 2% accuracy. Document each compression step and its individual accuracy cost.

---

## Chapter 67 — SNN Deployment Pipeline: From Trained Model to Hardware

### Purpose
Training an SNN is only half the problem. Getting a trained model onto actual neuromorphic hardware involves a pipeline of conversion, compilation, mapping, and verification steps that no other chapter covers explicitly. This chapter is primarily for engineers and students targeting real deployment.

### Topics
- The deployment gap: why a model that performs well in simulation may behave differently on chip
  - Numerical precision differences between simulation and hardware
  - Timing discretization: continuous-time simulation vs. fixed time steps on hardware
  - Routing timing: spike delivery delays not captured in simulation
- The deployment pipeline, step by step:
  1. Model specification: ensuring the trained model uses hardware-supported neuron models
  2. Weight quantization: mapping floating-point weights to hardware precision (Chapter 66)
  3. Architecture mapping: assigning neurons to cores, synapses to on-chip memory
  4. Routing table generation: encoding the connectivity in the chip's routing fabric
  5. Timing analysis: verifying that spike delivery meets timing constraints
  6. Functional verification: comparing chip output to simulation output on test inputs
  7. Performance profiling: measuring latency, energy, and core utilization on-chip
- Compiler toolchains by platform:
  - Intel Lava: Python-based, maps to Loihi 2 cores
  - Nengo + NengoLoihi: model-level compilation with automatic partitioning
  - SpiNNaker PyNN interface: standard neuroscience simulation language to hardware
  - Sinabs/Rockpool for SynSense Xylo
- Common failure modes and how to diagnose them:
  - Accuracy loss at deployment time: causes and mitigations
  - Spike congestion: routing saturation and its symptoms
  - Silent cores: neurons that stop firing after mapping
- Testing strategies: simulation-hardware co-testing, input replay
- Documentation and reproducibility for deployed neuromorphic systems

### Programming demo
Use the Lava framework (Intel Loihi 2) or a Loihi-style simulator to take a previously trained SNN through the full deployment pipeline: quantize, map, route, verify, and profile.

### Assignment
Given a 4-layer convolutional SNN trained for DVS Gesture recognition, write the deployment checklist: list every step from trained weights to verified chip execution. Identify two deployment-specific failure modes and their diagnosis procedure.

---

# Part VII — Neuromorphic Hardware

## Chapter 32 — Why Neuromorphic Hardware Exists

### Topics
- Von Neumann architecture and the memory wall
- Data movement cost: energy per byte moved far exceeds energy per operation
- Energy breakdown of conventional inference: compute, memory read, memory write, communication
- Why CMOS energy per MAC differs from energy per memory access by 2-3 orders of magnitude at modern technology nodes
- Parallelism and its limits in GPU-style systems
- Asynchronous computation: removing the global clock
- Event routing as an alternative to broadcasting
- Compute-near-memory and in-memory computing
- Low-power inference: what the target numbers look like in practice

### Assignment
Given a published ANN inference energy measurement, break down the energy into compute and memory contributions. Explain where neuromorphic hardware targets each component.

---

## Chapter 33 — Digital Neuromorphic Chips

### Topics
- Digital spiking processors: architecture overview
- Cores, compartments, and neuron models
- Spike routing: packet-switched networks on chip
- Programmable neuron models: configurable LIF parameters
- Event packets and address-event representation
- Intel Loihi and Loihi 2:
  - Architecture: 128 cores (Loihi), graded spikes, on-chip learning (Loihi 2)
  - Programming model: Lava framework — **note the 2026 deprecation**: the `lava-nc/lava` repository was archived read-only (May 2026); Intel has stated it is moving its next-generation SDK to open-standard AI frameworks (covered in Ch. 67 and Ch. 71). Lava is taught historically and with a deprecation warning, not as a forward-looking tool.
- Intel Loihi 3 (reportedly in development): **hype-calibration case study.** As of mid-2026 Intel has published no official datasheet, neuron count, process node, or launch date. Widely circulated "8M neurons / 64B synapses / 4nm / Jan 2026" figures trace only to SEO blogs and syndicated press releases, not to Intel. This chapter uses Loihi 3 as a live worked example of how to NOT cite hardware specs: present it as "reportedly in development, unconfirmed," and teach students to track the primary source.
- Intel Hala Point (April 2024, deployed at Sandia National Laboratories):
  - 1152 Loihi 2 chips, largest neuromorphic system to date, ~2.6 kW
  - 1.15 billion neurons, 128 billion synapses, 140,544 neuromorphic cores (owl-cortex scale)
  - Used for large-scale optimization and scientific computing
- IBM TrueNorth: 4096 cores, 256 neurons per core, fixed LIF model
- IBM NorthPole (Science 2023, Modha et al.): inference-focused, near-memory architecture
  - Not strictly spiking, but shares neuromorphic principles (no off-chip memory during inference)
  - 25x more energy-efficient and 22x faster than a V100 at 12 nm; cite the Science paper, treat "2026 production" claims as weakly sourced
- SpiNNaker and SpiNNaker2: ARM-based, flexible neuron models, scalable routing
  - **SpiNNcloud commercialization (2024-2025): the first commercial neuromorphic supercomputer.** Boards of 48 chips x 152 Arm cores; systems deployed at Sandia (June 2025) and Leipzig University (July 2025). The most significant commercial hardware story of the period; SpiNNaker2 also demonstrated event-based backpropagation and e-prop on-chip (cross-reference Ch. 29, 57).
- BrainScaleS (digital coordination layer): note connection to Chapter 34
- Akida (BrainChip): inference-focused, ANN-to-SNN deployment via MetaTF
  - Akida 2 / TENNs: temporal state-space (event-native) models, 8-bit quantization, vision-transformer support
  - Akida Pico: sub-milliwatt always-on coprocessor (wake-word / keyword class); AKD1500 co-processor (announced Nov 2025); Akida Cloud eval access (Aug 2025)
- Innatera Pulsar (2025): "first mass-market neuromorphic microcontroller," 28 nm, radar presence detection at ~600 µW; Best-in-Show COMPUTEX 2025. A strong edge-commercialization example to contrast with research chips.
- SynSense (cross-reference Ch. 34, 39): Xylo (audio edge SoC) and Speck (event-vision SoC, sensor + SNN compute co-located)
- Samsung NM-500 / Orca (Synaptics), GrAI NeuronFlow (note: GrAI/Snap line largely dormant since 2023): historical / emerging platforms
- Comparison table: cores, neurons/core, synapses/core, precision, on-chip learning support, SDK maturity and license, technology node, commercial availability
- Hardware constraints: communication bandwidth, routing table size, neuron model flexibility
- **Key reference for the whole part:** Kudithipudi et al., "Neuromorphic computing at scale," *Nature* 637 (Jan 2025) — the field's definitive roadmap; sparsity and scale as the central open challenges.

### Programming demo
Use Lava (Intel Loihi 2 SDK) or a Loihi-style abstraction to map a small SNN to hardware-constrained deployment.

### Assignment
Research one digital neuromorphic platform in depth. Summarize: architecture, programming model, demonstrated applications, and limitations.

---

## Chapter 34 — Analog and Mixed-Signal Neuromorphic Systems

### Topics
- Analog computation: exploiting physical device behavior for computation
- Mixed-signal circuits: digital spike communication, analog neuron dynamics
- Subthreshold CMOS circuits: transistors operating in weak inversion for ultra-low power
- Noise and device mismatch: sources, effects, and mitigation
- Calibration: why analog systems require per-chip tuning
- Energy efficiency of analog computation: sub-picojoule operations
- BrainScaleS (Heidelberg): mixed-signal wafer-scale integration, 1000x faster than biology
- DYNAP-SE (Zurich): event-driven analog/digital mixed-signal chip
- Design principles of analog neuron circuits: subthreshold integrator, comparator for threshold

### Hype calibration box
Analog neuromorphic chips often claim extreme energy efficiency. These numbers are real but come with critical caveats: device mismatch degrades performance, calibration costs energy, and the numbers are usually for single-chip inference, not full system operation.

### Assignment
Explain why analog systems can be efficient but difficult to control. List three engineering mitigations for device mismatch.

---

## Chapter 35 — Memristors, Crossbars, and In-Memory Computing

### Topics
- Memristive devices: formal I-V relationship (Strukov et al. 2008), state variable model
- Crossbar arrays: parallel synaptic weight storage and matrix-vector multiplication
- How a crossbar computes: Ohm's law and Kirchhoff's current law as analog dot products
- Resistive memory (RRAM): read/write endurance, retention, variability
- Phase-change memory (PCM): multi-level storage, slower write, better retention
- Ferroelectric FET (FeFET): fast write, moderate retention, emerging
- SRAM and DRAM comparison for synaptic storage: digital reference
- Device variability: stuck-at-fault tolerance, training with noise injection
- Write noise and its effect on training stability
- STDP-like device behavior: some memristive devices naturally implement local plasticity
- Hybrid CMOS-memristor systems: current research landscape

### Programming demo
Simulate a memristive crossbar with device variability. Train a simple weight matrix and measure accuracy degradation as a function of device noise level.

### Assignment
Draw a crossbar array and explain how it computes weighted sums. Identify three sources of error and propose one mitigation for each.

---

## Chapter 36 — Address-Event Representation and Spike Routing

### Topics
- Address-event representation (AER): origin, protocol, packet structure
- Event packets: neuron address, timestamp, polarity
- Routing tables: how a spike reaches its target neurons
- Network-on-chip for neuromorphic systems
- Communication bottlenecks: fan-in, fan-out, congestion
- Spike congestion and its effects on timing
- Time-division multiplexing vs event-driven routing
- Comparison: AER vs conventional bus communication

### Programming demo
Simulate spike routing between layers with configurable connectivity. Measure routing load as a function of network density and spike rate.

### Assignment
Analyze how network connectivity (random, local, hub-and-spoke) affects routing load and timing jitter.

---

## Chapter 37 — FPGA and Custom ASIC Implementations of SNNs

### Purpose
FPGAs are the primary research vehicle for custom SNN hardware. Engineers who will design or evaluate neuromorphic systems need this chapter.

### Topics
- Why FPGAs for SNN research: flexibility, reconfigurability, realistic timing
- The key operation: for SNNs, multiply-accumulate (MAC) reduces to accumulate (AC) when inputs are binary spikes
  - AC-only implementation saves area and power vs MAC
- Pipelined LIF implementation: dataflow from spike detection to membrane update to threshold compare
- Memory layout for weight storage: BRAM vs distributed RAM tradeoffs
- Fixed-point precision for spiking networks: membrane potential, weights, thresholds
- Spike encoding in hardware: event queues, time-step counters
- Routing fabric: configurable connectivity vs hard-wired
- Power estimation from activity: dynamic vs static power in FPGA implementations
- Custom ASIC flow for SNN chips: conceptual overview (synthesis, place-and-route, tapeout)
- Representative designs: ODIN (Louvain), MorphIC (EPFL), and related academic chips
- Toolchain: PyGeNN for CUDA-accelerated spiking networks, HDL generation flows

### Programming demo
Implement a single LIF neuron layer in a hardware-description-level Python abstraction (e.g., Amaranth HDL or Migen, or a behavioral VHDL sketch) and simulate it.

### Assignment
Estimate the FPGA resource utilization (LUTs, BRAMs, DSPs) required for a 100-neuron LIF layer with 10,000 synapses. Justify each estimate.

---

## Chapter 38 — Hardware-Software Co-Design

### Topics
- Why models must fit hardware: mapping constraints are non-trivial
- Formal mapping problem: integer programming formulation (neuron placement, core assignment)
- Quantization: weight precision, activation precision, their effects on spike dynamics
- Weight sharing: reducing memory cost
- Spike-rate constraints: hardware cannot handle unbounded firing rates
- Placement and scheduling: temporal multiplexing and its latency implications
- Compiler toolchains: Lava compiler, Nengo compiler, BindsNET-to-Loihi flow
- Hardware-aware loss functions: penalizing spike rates, memory usage, and routing load during training
- Energy-aware training: estimating synaptic operations (SOPs) as a training penalty

### Assignment
Modify an SNN architecture to reduce spike traffic by 50% while preserving >95% of baseline accuracy. Document each constraint applied.

---

# Part VIII — Event-Based Sensing

## Chapter 39 — Event Cameras and Dynamic Vision Sensors

### Topics
- Frame camera vs event camera: the fundamental difference
- How a Dynamic Vision Sensor (DVS) pixel works: independent per-pixel threshold comparison
- Event format: timestamp, pixel address (x, y), polarity (ON/OFF)
- High dynamic range: why event cameras handle bright + dark scenes better than frame cameras
- Low latency: microsecond-level event timestamps
- Sparse visual representation: most pixels are silent most of the time
- DAVIS camera: combining events and standard frames on one sensor — why hybrid sensors exist
- Prophesee Metavision sensor line: industrial event camera overview
  - GenX320 (320x320, ~3x4 mm, microwatt-class): the edge/mobile event sensor — Raspberry Pi 5 starter kit (2025), Qualcomm mobile partnership
  - IMX636 (HD 1280x720, Sony-Prophesee, EVK4): the high-resolution reference sensor
  - Tooling/licensing note: since Metavision SDK 5.3 the full SDK requires a development license; the open modules live in OpenEB — teach OpenEB for reproducibility
- SynSense Speck: event-vision SoC with sensor + SNN compute co-located (ultra-low power, no host needed for inference) — the must-cover in-sensor-compute example
- Comparison table: event camera vs frame camera latency, dynamic range, power, data rate
- Limitations: high noise at low contrast, blurring at very high speeds, no texture in static scenes
- In-sensor and near-sensor computing:
  - The next step beyond chip + sensor: computation inside the sensor pixel array
  - 3D-stacked in-sensor computing (3DS-ISC): processing logic stacked beneath the pixel array
  - Near-sensor corner detection and feature extraction: reducing data transfer to the host
  - Why this matters: latency reduction, bandwidth reduction, further power savings
  - Current research vs. commercial readiness

### Programming demo
Load or simulate event-camera data using the tonic library. Visualize raw event streams, polarity maps, and accumulated event frames.

### Assignment
Explain why event cameras are useful for fast motion and low-latency robotics. Identify two tasks where frame cameras would still be preferred and explain why. Estimate the bandwidth savings from in-sensor corner detection for a 640×480 DVS at 1 Mev/s.

---

## Chapter 40 — Event-Based Vision Algorithms

### Topics
- Event accumulation strategies: fixed-count, fixed-time, surface of active events
- Time surfaces (Lagorce et al. 2017): formal definition, exponential decay kernel
- Event histograms and voxel grids
- Event-based feature detection: e-FAST (Mueggler et al.), e-Harris corner detection
- Asynchronous convolutional processing: applying filters at event locations only
- Optical flow from events: variational formulation, event-based Lucas-Kanade
- Object tracking with events: Kalman filter driven by events
- Object recognition: event-to-frame conversion followed by CNN vs end-to-end event processing
- Modern event-based detection backbones (deep-learning side, taught alongside SNN approaches):
  - RVT (Recurrent Vision Transformer): standard event-detection baseline (~47% mAP on Gen1, <12 ms on a T4)
  - SAST (Scene-Adaptive Sparse Transformer): window-token co-sparsification at lower compute
  - SMamba (Sparse Mamba 2D selective scan) and PMRVT (2025): newest detectors reportedly surpassing SAST on Gen1/1Mpx
- Gesture recognition: DVS Gesture dataset, spike-based classifiers
- Event-based segmentation: brief treatment
- Hybrid frame-event models: using both sources for complementary information
- Toward event "foundation models" (honest framing): there is no GPT-scale event-vision foundation model yet; the current frontier is self-supervised pretraining — Masked Event Modeling, OpenESS (open-vocabulary event semantics), EASE (free-energy embodied event perception). State this as an open problem, not a solved one.

### Programming demo
Build an event-based gesture classifier on the DVS Gesture dataset using snnTorch or SpikingJelly with tonic.

### Assignment
Implement a time surface representation and compare its temporal resolution advantage over a fixed-time event accumulation.

---

## Chapter 41 — Event-Based Stereo Vision and 3D Perception

### Purpose
Monocular event vision covers most published work, but depth perception is critical for robotics. This chapter is essential for readers targeting autonomous systems.

### Topics
- Stereo event cameras: hardware setup, synchronization between two DVS sensors
- DAVIS 346 stereo: combined frame+event stereo sensor
- Event-based stereo matching: disparity estimation from asynchronous event streams
  - The challenge: events from left and right sensors are not synchronized
  - Matching by timestamp proximity, surface-of-active-events approach
- Event-based depth estimation: from disparity to depth
  - DERD-Net (NeurIPS 2025): depth from event-based ray densities; >=42% MAE reduction over prior SOTA on stereo MVSEC/DSEC, with monocular results matching earlier stereo methods. A current strong-result anchor.
- Structure from motion with event cameras: keyframe-based vs continuous formulations
- Event-based visual odometry: ego-motion estimation from events alone
- Neuromorphic SLAM: simultaneous localization and mapping with event cameras
  - Front-end: feature tracking with events
  - Back-end: pose graph optimization
- Depth completion: combining sparse event-based depth with dense frame-based depth
- Current limitations: stereo calibration difficulty, matching ambiguity at low texture

### Programming demo
Simulate a stereo event camera pair observing a moving object. Implement a simple disparity estimator using timestamp-matched events.

### Assignment
Explain the calibration procedure required for a stereo event camera system. What parameters must be estimated and how do errors in each affect depth accuracy?

---

## Chapter 42 — Neuromorphic Audio and Speech

### Topics
- Biological cochlea: tonotopic organization, gammatone filters as a model
- Cochleagram: gammatone filterbank output as a neuromorphic audio representation
- Dynamic Audio Sensor (DAS): event-driven audio sensor hardware
- Spiking audio encoders: converting cochleagram to spike trains
- Binaural sound localization: interaural time difference (ITD) and interaural level difference (ILD)
  - Jeffress model and its spiking network implementation
- Spoken digit recognition: NTIDIGITS dataset
- Spiking keyword spotting: SHD and SSC datasets
- Always-on low-power audio: wakeword detection with minimal energy
- Low-power wake-word detection pipeline

### Programming demo
Encode audio into spikes using a gammatone filterbank. Train a simple SNN classifier on SHD (Spiking Heidelberg Digits).

### Assignment
Compare spectrogram-based audio classification (CNN) and spike-based audio classification (SNN) on SHD. Report accuracy and estimated inference energy.

---

## Chapter 43 — Tactile, Olfactory, and Multimodal Neuromorphic Sensors

### Topics
- Neuromorphic tactile sensing: event-driven pressure and vibration sensors
- Robotic touch: fingertip sensors, slip detection with events
- Olfactory-inspired sensing: electronic nose with spiking processing
- Multimodal event streams: combining vision, audio, and touch
- Sensor fusion with SNNs: temporal alignment and joint encoding
- Asynchronous sensor fusion: handling events from sensors with different time constants

### Assignment
Design a multimodal neuromorphic perception system for a robotic gripper combining tactile and vision events.

---

# Part IX — Neuromorphic AI Applications

## Chapter 44 — Neuromorphic Edge AI

### Topics
- Always-on edge devices: the target deployment class
- TinyML and neuromorphic AI: positioning and overlap
- Battery-powered sensing: power budget analysis
- Latency-critical inference: where frame-based AI is too slow
- Smart cameras, wearables, IoT devices, industrial sensors
- System-level power accounting: sensor, encoder, SNN inference, output
- Synaptic operations (SOP) counting: formal definition and use as a proxy metric
  - AC vs MAC: why SNN inference is accumulate-only when inputs are binary
  - Limitations of SOP counting: ignores memory access, communication, control

### Programming demo
Build a small SNN and measure model size, spike count, estimated SOP count, and latency.

### Assignment
Design a neuromorphic edge system for one real-world application. Provide a complete system-level power budget with justified numbers.

---

## Chapter 45 — Neuromorphic Robotics

### Topics
- Perception-action loops: sensing, encoding, processing, actuation
- Event-based obstacle avoidance: DVS-driven reactive control
- Drone navigation with event cameras: low latency, high dynamic range
- Spiking motor control: CPG-inspired oscillators for locomotion
- Reflexive behavior: hardwired spiking circuits for fast reaction
- Low-latency sensorimotor integration: event-to-action in microseconds
- Embodied AI and sensorimotor contingencies
- Neuromorphic quadruped and humanoid control: research landscape

### Programming demo
Create a toy robot controller using spiking neurons in a simulation (Gymnasium + PyBullet). Implement event-driven collision avoidance.

### Assignment
Design a neuromorphic drone obstacle-avoidance pipeline. Specify the sensor, encoder, SNN architecture, output decoding, and actuation. Estimate end-to-end latency.

---

## Chapter 46 — Biomedical and Brain-Computer Applications

### Topics
- EEG processing: spike-based artifact detection and classification
- ECG monitoring: event-driven arrhythmia detection
- Brain-computer interfaces (BCIs): neural signal decoding with SNNs
- Prosthetics: spiking control of motorized prosthetic limbs
- Neural signal decoding: spike sorting and population vector decoding
- Low-power medical devices: implantable constraints
- Event-driven health monitoring: always-on vital sign detection

### Assignment
Design an SNN-based always-on cardiac arrhythmia monitor. Specify the encoding, architecture, decision threshold, and power budget.

---

## Chapter 47 — Industrial Monitoring and Anomaly Detection

### Topics
- Sparse sensor events in industrial settings
- Predictive maintenance: detecting anomalies before failure
- Cyber-physical systems: neuromorphic edge-cloud architectures
- Fault detection with spiking reservoir models (uses Chapter 19)
- Streaming anomaly detection: one-class and density-based approaches
- Edge deployment with limited retraining
- Neuromorphic temporal pattern recognition for vibration, pressure, and temperature signals

### Programming demo
Use a liquid state machine for anomaly detection in a synthetic industrial sensor stream.

### Assignment
Compare conventional time-series anomaly detection (LSTM or autoencoder) with spike-based detection on the same synthetic dataset.

---

## Chapter 48 — Neuromorphic AI for Autonomous Systems

### Topics
- Autonomous vehicles: perception under extreme latency constraints
- UAVs: size, weight, and power constraints
- Space systems: radiation tolerance and ultra-low power
- Mobile robots: continuous adaptation to environment
- Low-power perception stack design
- Real-time decision making under uncertainty
- Robustness under environmental changes: lighting, weather, sensor degradation

### Assignment
Build a system design document for an autonomous neuromorphic perception stack for a small UAV. Identify every component, justify the sensor choice, and estimate the power budget.

---

## Chapter 68 — Neuromorphic Computing for Combinatorial Optimization

### Purpose
One of the most compelling demonstrated use cases for neuromorphic hardware is combinatorial optimization — a class of problems (scheduling, routing, constraint satisfaction) that maps naturally onto energy-minimizing spiking dynamics. This application domain is distinct from pattern recognition and deserves dedicated treatment.

### Topics
- Combinatorial optimization problems: traveling salesman, graph coloring, Boolean satisfiability, QUBO (Quadratic Unconstrained Binary Optimization)
- Why conventional hardware struggles: exponential search spaces, no gradient to follow
- Spiking networks as energy-minimizing systems:
  - Hopfield networks: energy function, attractor dynamics, capacity
  - Spiking Hopfield networks: discrete spike-based energy minimization
  - Connection to Boltzmann machines and simulated annealing
- Mapping optimization to spiking networks:
  - Each binary variable → one or two neurons (x and not-x)
  - Constraints → synaptic inhibitory connections
  - Objective function → synaptic weight magnitudes
- Loihi demonstrations of optimization:
  - Constraint satisfaction problems
  - Sparse coding as optimization
  - Graph coloring and scheduling benchmarks
- Stochastic spiking networks for optimization: noise as a randomization mechanism (simulated annealing analogy)
- Comparison with classical and quantum optimization: where spiking networks stand
- Limitations: problem size, solution quality, convergence guarantees

### Programming demo
Formulate a small graph-coloring problem as a spiking Hopfield network. Simulate the network and observe energy minimization to a solution.

### Assignment
Map a 10-city TSP instance to a Hopfield-style spiking network. Analyze how network size scales with problem size and identify the main limitation.

---

# Part X — Hybrid Neuromorphic and Deep Learning Systems

## Chapter 49 — ANN-SNN Hybrid Models

### Topics
- Why hybrid systems are the practical reality
- SNN front-end with ANN back-end: event processing in SNN, classification in ANN
- ANN feature extractor with SNN temporal head: pretrained features, spiking classifier
- Event camera with CNN/Transformer backbone: published architectures
- Spiking encoders as a preprocessing stage
- Hybrid edge-cloud systems: SNN at the sensor, ANN in the cloud
- Practical deployment strategies: when to use each component
- Neural ODEs and continuous-depth networks (Chen et al. 2018 NeurIPS):
  - ODE-based hidden state dynamics as a generalization of residual networks
  - Connection to LIF dynamics: the neuron membrane IS an ODE integrator
  - Adjoint method for gradient computation: backpropagation through an ODE solver
- Liquid Time-Constant (LTC) networks (Hasani et al. 2021):
  - Input-dependent time constants: the network's temporal scale adapts to the input
  - Inspired by C. elegans wiring: sparse, interpretable connectivity
  - Closed-form Continuous-time (CfC) networks: tractable approximation of LTCs
  - Comparison with SNNs: both are event-friendly, but LTCs use continuous activations
  - Edge AI deployment: LTCs on microcontrollers for time-series classification

### Programming demo
Build a hybrid model using a spiking DVS-processing front-end and a conventional linear classifier back-end.

### Assignment
Identify which components of a video processing pipeline benefit from SNN processing and which require ANN-level accuracy. Justify each decision quantitatively.

---

## Chapter 50 — Spiking Transformers, Attention, and State Space Models

### Topics
- Why attention is expensive: quadratic complexity in sequence length
- Sparse attention mechanisms: reducing the quadratic cost
- Event-driven attention: computing attention only at event locations
- Spiking self-attention: binary query/key/value mechanism
  - SpikFormer (Zhou et al. 2022): formal formulation of spiking attention
  - How spike-based Q, K, V enable accumulate-only attention
- The spiking-Transformer SOTA lineage (teach as a progression, with the number that matters):
  - Spikformer -> Spikformer V2 (~82.4% ImageNet) -> Spike-driven Transformer V1/V2 -> **V3 (IEEE T-PAMI 2025): 86.2% top-1 on ImageNet-1k**, via Efficient Spike Firing Approximation training. This is the current "how close are SNNs to ANNs on ImageNet" reference number.
  - QKFormer (NeurIPS 2024): spike-based Q/K attention with spiking patch embedding and deformable shortcuts
  - QSD-Transformer (Quantized Spike-driven Transformer, ICLR 2025): pushes the efficiency frontier
- Positional encoding with spikes: challenges and current solutions
- Energy analysis: does spiking attention actually save synaptic operations over standard attention?
- Current accuracy gaps vs standard Transformers: honest benchmark comparison
- State Space Models (SSMs) as an alternative to Transformers for sequences:
  - Mamba selective SSM: input-dependent state transitions, linear complexity in sequence length
  - Why SSMs are naturally compatible with spiking dynamics: both are recurrent, event-friendly
  - SpikingSSMs (AAAI 2025): sparse + parallel spiking SSM for long sequences; LIF with deterministic reset plus a surrogate-dynamic network dropped at inference — the anchor paper for spiking SSMs
  - P-SpikeSSM (ICLR 2025): probabilistic spiking SSM for long-range dependencies; SPikE-SSM (sparse/precise/efficient variant)
  - Diagonal structured SSM deployed on Loihi 2 (2024): a clean algorithm + hardware crossover example
  - Compute-in-memory SSM for event sequences (Nature Communications 2025): hardware realization
  - Comparison: Transformer attention vs SSM recurrence for event-based temporal tasks
- Limitations and open questions for both spiking Transformers and spiking SSMs

### Hype calibration box
Early spiking Transformer papers report accuracy competitive with ANNs on ImageNet. These results use conversion or hybrid training, not purely spike-driven training. Readers should check the training method before comparing numbers. The same applies to spiking SSM results.

### Assignment
Derive the operation count for standard dot-product attention, spiking attention (binary Q/K/V), and an SSM recurrent step. Under what sequence-length and sparsity conditions does each architecture save operations relative to the others?

---

## Chapter 51 — Knowledge Distillation for SNNs

### Purpose
Knowledge distillation is one of the most effective techniques for closing the accuracy gap between SNNs and ANNs. It is under-covered in most SNN courses.

### Topics
- Knowledge distillation basics: soft targets, temperature scaling, Hinton et al. 2015
- ANN teacher to SNN student: distillation as a training objective
- Activation matching: aligning ANN hidden-layer activations with SNN membrane potentials
- Logit matching: soft cross-entropy from ANN output probabilities
- Online distillation: simultaneously training ANN and SNN, updating the teacher
- Hybrid conversion + fine-tuning: using distillation to recover accuracy lost during conversion
- Why distillation improves SNN accuracy: the smooth teacher provides gradient signal that hard spikes cannot
- Comparison of accuracy: from-scratch SNN, conversion, conversion + distillation fine-tuning
- Practical pipeline: step-by-step from pretrained ANN to distilled SNN

### Programming demo
Distill a pretrained ResNet-20 into a spiking counterpart. Compare accuracy with conversion-only and from-scratch training.

### Assignment
Implement activation-matching distillation for a two-layer SNN. Measure the accuracy improvement over standard surrogate-gradient training.

---

## Chapter 52 — Neuromorphic AI and Foundation Models

### Purpose
The convergence of spiking networks and large-scale foundation models is one of the fastest-moving areas as of 2025-2026. This chapter provides a rigorous survey of what has been demonstrated, what is plausible, and what remains speculative.

### Topics
- Spiking large language models: the emerging subfield
  - SpikeGPT (2023): autoregressive language model using spiking neurons in a GPT-style architecture
  - SpikeLM (arXiv 2406.03287): general spike-driven language modeling via elastic bi-spiking
  - SpikeLLM (ICLR 2025): first SNN scaled to 7-70B params; Generalized Integrate-and-Fire neuron + Optimal Brain Spiking; largest gains at extreme low bit-width (e.g., +8.10% over OmniQuant at W2A16 on LLaMA-2-7B), framed as a spiking-vs-quantization comparison
  - **SpikingBrain-7B / -76B (arXiv 2509.05276, 2025): the defining brain-inspired large-model result.** Hybrid linear + sliding-window + full attention, adaptive spiking neurons, sparse MoE, converted from a pretrained Transformer with <2% of typical training tokens, trained on a non-NVIDIA (MetaX) cluster. Reports >100x speedup in time-to-first-token at 4M-token context and ~69% activation sparsity. Hype-calibration: the speedup is a long-context inference claim and accuracy is "comparable," not superior, to open Transformer baselines.
  - **SpikeMLLM (arXiv 2604.18610, 2026): first spike-based multimodal LLM**; modality-specific temporal scales + temporal compression; ~25.8x power efficiency and ~9x throughput vs FP16 on a custom RTL accelerator with <1.2% accuracy gap. Frame energy numbers as projected-on-accelerator (RTL simulation), not commodity-GPU wall-power.
  - NSLLM: MatMul-free FPGA accelerator for a ~1.5B-scale spiking LLM (verify primary source before quoting throughput)
  - Darkit / DarwinKit toolkit (arXiv 2412.15634): pip-installable open-source tools wrapping GPT/BERT/Llama for spiking LLMs — the "tooling sidebar" of this chapter
  - Accuracy vs energy tradeoff in spiking LLMs: current numbers and trends
- The quantization cousin (taught for contrast, not conflation): **BitNet b1.58** ternary {-1,0,1} LLMs
  - BitNet attacks the same energy target as spiking LLMs (replace FP matmul with integer/additive ops) but from the quantization side; SNNs attack it from the temporal-sparsity side. They are complementary, and SpikeLLM explicitly benchmarks against quantization. b1.58 2B4T (2025) is the first natively-trained open 1-bit LLM at scale; bitnet.cpp gives large CPU energy/speed wins.
- Sparse inference for foundation models: mixture-of-experts and event-driven forward passes
- Neuromorphic encoders for large models: efficient tokenization from event-based sensors
- Edge foundation-model interfaces: neuromorphic preprocessing to reduce model input dimensionality
- Neuromorphic preprocessing for audio, vision, and multimodal models
- Low-power adaptation: fine-tuning a large model at the edge with local plasticity
- Can Transformers be fully spiked? What architectural constraints prevent it
- Realistic near-term engineering: what is achievable in 2-5 years
- Long-term research vision: what would a truly spiking large model require at scale?

### Hype calibration box
Most "spiking LLM" papers achieve accuracy well below their ANN counterparts, require many time steps, and have not been demonstrated on real neuromorphic hardware. The field is moving fast, but the gap between published SpikeGPT accuracy and GPT-2 accuracy remains large. Students should track benchmark numbers carefully.

### Discussion
Every claim in this chapter should be labeled as engineering-near-term, research-medium-term, or speculative-long-term. Students should leave this chapter able to evaluate future papers that claim to combine large models with neuromorphic processing, rather than accepting claims at face value.

---

# Part XI — Evaluation and Benchmarking

## Chapter 53 — Metrics for Neuromorphic AI

### Topics
- Accuracy: top-1, top-5, F1
- Latency: first-spike latency, inference latency, batch vs streaming
- Energy: theoretical energy from SOP count vs measured chip energy
- Synaptic operations (SOPs): formal definition, AC vs MAC distinction
- Energy-delay product: combined metric
- Spike count: total and per-layer
- Time-to-first-decision
- Robustness: accuracy under input noise, sensor degradation, adversarial perturbation
- Memory footprint: model parameters, synaptic memory
- Throughput: events per second, inferences per second
- Event sparsity: fraction of silent neurons
- Hardware utilization: core occupancy, routing saturation

### Hype calibration box
Many neuromorphic AI papers report only accuracy. A paper that claims efficiency should also report measured energy (not SOP count), real hardware latency (not simulation), and a fair ANN comparison at the same accuracy level.

### Assignment
Evaluate a published SNN model using at least five metrics, not accuracy alone. Identify which metrics the original paper omitted and explain why they matter.

---

## Chapter 54 — Datasets and Benchmarks

### Datasets
- Static datasets converted to spikes: MNIST, Fashion-MNIST, CIFAR-10 (via Poisson encoding)
- Native event datasets: N-MNIST, N-Caltech101 (DVS recordings of static images)
- DVS Gesture: 11-class hand gesture recognition
- CIFAR10-DVS: event-camera recording of CIFAR-10 images
- SHD: Spiking Heidelberg Digits (audio)
- SSC: Spiking Speech Commands
- Event-camera driving datasets: MVSEC, DSEC (DSEC remains the canonical optical-flow/depth benchmark)
- Newer driving/SLAM datasets (2024-2026): ECMD (long-range driving SLAM), NSAVP (stereo event + thermal + RGB with ground-truth poses), WECMD (wearable multisensor)
- Event-based optical-flow datasets
- NTIDIGITS: spike-encoded audio digits

### Topics
- Why benchmarks are difficult: no standard evaluation protocol
- Frame-to-event conversion artifacts: synthetic events lack physical realism
- Real event data vs synthetic event data: distribution mismatch
- Benchmark leakage: training-set contamination in published results
- Dataset bias: most neuromorphic vision datasets are small by deep learning standards
- Reproducibility: fixed seeds, reported hyperparameters, code availability
- NeuroBench (Yik et al., framework paper in Nature Communications 2025): the standardized neuromorphic benchmark platform — actively maintained (2.x series through 2026)
  - Motivation: lack of standard evaluation protocol has made neuromorphic results incomparable
  - Two tracks: algorithm track (accuracy + footprint metrics) and system track (deployed hardware)
  - Metrics: accuracy, synaptic operations, activation sparsity, membrane potential updates
  - Benchmark tasks: keyword spotting, ECG classification, gesture recognition, edge inference
  - Community adoption: how NeuroBench relates to MLPerf for conventional AI
  - How to use NeuroBench to evaluate a new SNN: step-by-step (this is the eval spine the recipes and Ch. 73 build on)

### Assignment
Evaluate a trained SNN using the NeuroBench protocol on a supported task. Report all required metrics and compare with a published baseline from the NeuroBench leaderboard.

---

## Chapter 55 — Energy Measurement and Fair Comparison

### Topics
- The problem with theoretical energy claims: SOPs ignore memory, communication, control
- Simulation vs hardware measurement: what each can and cannot tell you
- Dynamic energy vs static (leakage) energy vs total system energy
- Energy budget breakdown: sensor, encoding, SNN inference, output communication
- Batch size effects: why per-sample energy numbers change with batch size
- Fair comparison with GPUs, CPUs, and edge accelerators:
  - Same task, same accuracy, same input format
  - Why comparing SNN energy on chip X with ANN energy on GPU is invalid
  - The correct comparison: same chip family, or same technology node
- Energy per correct inference: the unified metric
- Published fair comparisons: what they find

### Assignment
Write a complete evaluation protocol for fairly comparing an SNN and an ANN system for always-on keyword spotting. Specify: hardware platform, input format, batch size, accuracy operating point, and energy measurement method.

---

## Chapter 56 — Limitations and Failure Modes

### Topics
- The accuracy gap: quantitative assessment of where SNNs lag ANNs and by how much
- Training instability: root causes and known mitigations
- Hardware availability: most neuromorphic chips require special access
- Toolchain immaturity: debugging difficulty, framework fragmentation
- Sparse computation overhead on standard hardware: sparsity helps only on event-driven hardware
- Mismatch between simulated and deployed behavior: numerical precision, routing timing
- Scaling problems: accuracy gaps grow with dataset complexity
- When neuromorphic AI is not appropriate: dense static images, large language models, tasks without temporal structure

### Hype calibration box
This chapter is intentionally critical. Neuromorphic AI is a promising research direction with real applications in edge sensing and temporal processing. It is not a replacement for conventional deep learning for most current tasks. A researcher who cannot articulate the limitations of their field is not ready to do research in it.

### Assignment
Write a critical review of a neuromorphic AI paper. Identify: the claims, the evidence, the missing comparisons, and the unstated assumptions.

---

## Chapter 69 — Explainability and Interpretability of Spiking Neural Networks

### Purpose
SNNs are increasingly targeted for safety-critical domains: medical monitoring, autonomous vehicles, and industrial fault detection. Deploying AI in these domains requires understanding why the model made a decision. SNN explainability is a young field, but practitioners cannot wait for it to mature — this chapter gives the current best tools.

### Topics
- Why explainability matters for neuromorphic AI: safety-critical deployment, regulatory requirements
- Definitions: interpretability (the model is inherently transparent) vs explainability (post-hoc explanation of an opaque model)
- Inherent interpretability of small SNNs: spike raster as a decision trace
  - Spike timing as a natural explanation: which neurons fired, when, and in what order
  - For small networks, the firing sequence IS the explanation
- Saliency and attribution methods for SNNs:
  - Temporal saliency maps: which time steps contributed most to the output
  - Spike-rate gradient saliency: adapting GradCAM to spiking feature maps
  - Integrated gradients through time: attributing output to input events
- Concept-based explanations: which learned feature maps are semantically meaningful
- Attention maps from spiking Transformers (Chapter 50): interpretability side benefit
- Anomaly explanation: for industrial monitoring applications, explaining which sensor event triggered the anomaly flag
- Faithfulness and completeness of SNN explanations: evaluation methodology
- Regulatory context: EU AI Act and explainability requirements for high-risk AI systems
- Open challenges: temporal explanations are harder than static-input explanations

### Programming demo
Apply temporal gradient saliency to a trained SNN classifier. Visualize which input events and time steps contributed most to the classification decision.

### Assignment
For an SNN trained on ECG arrhythmia detection, produce a temporal saliency map for one true positive and one false positive. Use the explanation to identify potential failure modes.

---

# Part XII — Research Frontiers

## Chapter 57 — On-Chip Learning

### Topics
- Why on-chip learning matters: privacy, personalization, adaptation to sensor drift
- Local plasticity: what learning rules are feasible in hardware
- Hardware constraints on learning: memory for eligibility traces, precision for weight updates
- Event-based backpropagation on chip: approximate methods
- Eligibility-trace-based on-chip learning (uses Chapter 29): e-prop on Loihi 2
- Continual adaptation: updating a deployed model without full retraining
- Privacy-preserving learning: raw sensor data never leaves the chip

### Assignment
Design an on-chip learning rule for a sensor device that adapts to sensor drift without storing raw data.

---

## Chapter 58 — Neuromorphic Systems at Scale

### Topics
- Scaling SNNs: does more neurons and more layers help the way it helps ANNs?
- Multi-chip systems: routing between chips, latency, bandwidth
- Distributed event processing: challenges at the network level
- SpiNNaker2: 152 ARM cores per chip, designed for scalable neural simulation
- Large-scale biological neural simulation: Blue Brain Project, Human Brain Project
- Large recurrent systems and their stability
- Cloud access to neuromorphic hardware: Intel DevCloud (Loihi), SpiNNaker access
- Main barriers to scaling neuromorphic AI

### Assignment
Explain the three main barriers to scaling neuromorphic AI to the size of current large language models.

---

## Chapter 59 — Neuromorphic AI, Neuroscience, and Cognitive Models

### Topics
- What AI can borrow from neuroscience: principles vs mechanisms
- What should not be copied blindly: biological constraints are not engineering constraints
- Attention and working memory: spiking attractor network models
- Predictive coding networks as deep generative models (extends Chapter 15)
- Active inference and the free energy principle: the computational (not philosophical) version
- Dendritic computation: apical vs basal dendrites as segregated learning signal pathways
  - Körding & König (2001), Guerguiev et al. (2017): dendritic error signals
- Cognitive architectures: ACT-R, SPAUN, and their neuromorphic interpretations
- Memory systems: hippocampus-inspired episodic memory in SNNs
- Place cells and grid cells: spatial representation for neuromorphic robotics
- Embodied cognition: intelligence as a property of agent-environment interaction

### Assignment
Compare engineering-driven and neuroscience-driven neuromorphic research on a specific topic. Identify one insight from neuroscience that has transferred productively to engineering and one that has not.

---

## Chapter 60 — Security, Reliability, and Robustness

### Topics
- Adversarial attacks on SNNs: transferability from ANN attacks, spike-specific attacks
- Noise robustness: SNNs vs ANNs under Gaussian and salt-and-pepper noise
- Fault tolerance: hardware faults (stuck-at, bit flips) and their effect on SNN inference
- Hardware variability: device mismatch effects on network accuracy
- Sensor attacks: malicious event injection into event camera streams
- Event-stream spoofing: fabricating events to fool an event-based classifier
- Robust edge deployment: defense strategies

### Assignment
Design an attack scenario against an event-based vision classifier. Propose a defense and evaluate its cost in extra computation.

---

## Chapter 61 — Theoretical Limits and Expressivity of SNNs

### Purpose
A graduate student should be able to answer: "What can an SNN compute that an ANN cannot, and vice versa?" This chapter provides the formal tools.

### Topics
- Universal approximation theorems for spiking networks:
  - Maass (1997): liquid state machines are universal approximators in the sense of approximating any time-invariant filter
  - Maass et al. (2002): real-time computing without stable states
  - Comparison with universal approximation for standard MLPs
- Computational power of recurrent spiking circuits: Turing completeness of sufficiently complex spiking networks (Siegelmann & Sontag style results for spiking case)
- Expressivity vs sample complexity: more expressive models may require more data
- PAC learning theory applied to SNNs: what VC dimension bounds are known
- Computational complexity: what functions can T-step SNNs compute?
- SNNs vs ANNs on temporal tasks: formal separation results (sparse temporal patterns that SNNs represent efficiently but ANNs do not)
- When SNNs are provably equivalent to ANNs: the rate-coding limit
- Open theoretical questions: the main unsolved problems

### Assignment
Explain why an SNN with temporal coding can represent certain temporal patterns with fewer parameters than an equivalent ANN, and under what conditions this advantage vanishes.

---

## Chapter 62 — Neuromorphic Photonics and Optical Computing

### Purpose
Silicon electronics is approaching fundamental limits. Neuromorphic photonics — spiking computation using light rather than electrons — has emerged as a distinct research frontier with a dedicated roadmap and early hardware demonstrations. A 2026 graduate student will encounter this literature and needs a grounding.

### Topics
- Why photonics for neuromorphic AI:
  - Speed: photonic spikes can propagate at the speed of light
  - Bandwidth: optical channels carry far more information than electronic channels
  - Energy: some optical operations cost less energy than their electronic equivalents
  - Fan-out: one light source can broadcast to many detectors without the energy penalty of electronic fan-out
- Building blocks of photonic spiking neurons:
  - Vertical-Cavity Surface-Emitting Lasers (VCSELs): semiconductor lasers that exhibit excitability
  - Resonant Tunneling Diodes (RTDs): fast-switching devices combined with lasers
  - Micro-ring resonators: wavelength-selective optical filters
  - Mach-Zehnder modulators: all-optical signal routing
- Photonic integrate-and-fire neuron: analogy with electronic LIF, key differences
- Photonic synapses: weight encoding via optical attenuators and phase-change materials
- GHz-speed spiking photonic chips with in-situ training (arXiv 2506.14272): first demonstration of backpropagation-like training in a photonic spiking network
- The roadmap on neuromorphic photonics (arXiv 2501.07917): community consensus on open problems
- Integrated photonic neuromorphic computing: monolithic vs heterogeneous integration
- Limitations: fabrication complexity, thermal sensitivity, integration with electronic readout
- Current state vs. electronic neuromorphic chips: where photonic systems lead and where they lag
- Applications: high-bandwidth signal processing, lidar processing, radar, optical communications

### Hype calibration box
Photonic neuromorphic computing operates at GHz speeds but is not yet competitive with electronic systems on tasks requiring large weight counts or on-chip learning. Most published demonstrations involve small networks (tens of neurons). The roadmap honestly identifies the remaining engineering challenges.

### Assignment
Compare the energy per spike operation for an electronic LIF neuron on a 28 nm CMOS chip and a VCSEL-based photonic spiking neuron. Identify the conditions under which the photonic system wins on energy.

---

## Chapter 63 — The Future of Neuromorphic AI

### Topics
- Neuromorphic edge intelligence: the realistic near-term trajectory
- In-sensor AI: computing at the pixel, the cochlear hair cell, the pressure transducer
- Memristive learning systems: on-chip plasticity with non-volatile synapses
- Event-driven robotics: where the field is heading
- Neuromorphic multimodal AI: integrating vision, audio, and touch events
- Spiking large models: what would have to be true for this to work
- Sustainable AI: neuromorphic as a response to the energy cost of large models
- Open benchmarks: what the field needs for scientific progress
- Open toolchains: fragmentation and standardization efforts
- Research challenges: the five hardest open problems

### Final discussion
The field should be presented as promising but unfinished. Neuromorphic AI has demonstrated genuine advantages in edge AI, event-driven sensing, temporal processing, and ultra-low-power inference on specialized hardware. It still faces major challenges in training (the accuracy gap), software maturity (fragmented toolchains), hardware availability (limited access), and fair benchmarking (inconsistent evaluation protocols). A student who completes this book should be able to contribute to solving one of these problems.

---

# Part XIII — The Neuromorphic AI Software Stack

This part is the book's software spine. It is written so a practitioner can read it largely standalone, and so a researcher gains the engineering discipline that separates a reproducible result from an unreproducible one. It is deliberately placed late so it can reference real techniques from every earlier part, but the "How to read this book" page routes practitioners here early.

## Chapter 70 — The Neuromorphic Software Ecosystem and Framework Selection

### Purpose
The single biggest practical obstacle in neuromorphic AI is not the math, it is choosing tools that are alive, fast enough, and able to reach the hardware you target. This chapter is an honest, dated map of the 2026 ecosystem and a decision procedure for picking a stack.

### Topics
- The landscape, grouped by job, with maintenance status and "teach/use in 2026?" verdicts:
  - Gradient-trained SNNs (PyTorch family): snnTorch (the pedagogical default), SpikingJelly (fastest, custom CUDA; note the confusing even=stale-PyPI / odd=active-GitHub versioning), Norse (functional, `torch.compile`, low memory), BindsNET (bio/STDP/reservoir, low release cadence)
  - JAX family (real momentum, still pre-1.0): Spyx (fast, but no release since early 2024 — flag), Slax (online-training approximations: OSTL, RTRL, OTTT, OTPE, FPTT), SNNAX, jaxsnn (event-driven, EventProp, BrainScaleS-2)
  - Neuroscience simulators: Brian2 (biophysical realism), NEST (HPC point-neuron), GeNN/PyGeNN (CUDA codegen), Nengo (NEF)
  - Vendor/hardware SDKs: SynSense Rockpool + Sinabs (most practitioner-deployable; Xylo/Speck), BrainChip MetaTF (convert-a-CNN-to-edge path), SpiNNaker sPyNNaker/PyNN, and Intel Lava — **taught with its 2026 deprecation** (repo archived read-only May 2026; successor SDK unannounced)
- A framework-selection decision tree: by goal (learn vs publish-fast vs deploy-to-chip), by hardware target, by team JAX/PyTorch preference, by need for online/local learning
- Why "fastest" is a time-axis story: SNN training speed comes from vectorizing across the time dimension (fused CUDA kernels, XLA/`torch.compile`, JAX `scan`), not raw FLOPs — this reframes every benchmark the student will read
- Reading framework benchmarks critically (the Open Neuromorphic 11-library comparison as a worked example)
- The "is this library dead?" checklist: last release date, CI status, open-vs-closed issue ratio, PyPI-vs-GitHub divergence, NIR support as a longevity signal

### Library Spotlight
The committed core teaching stack for the rest of the book: snnTorch + SpikingJelly + Tonic + NeuroBench + NIR, with Rockpool/Sinabs and MetaTF for deployment.

### Assignment
Given three project briefs (a course exercise, an ImageNet-scale research result, an always-on edge product), select and justify a stack for each. For one library, audit its current maintenance health from primary sources and write a one-paragraph verdict.

---

## Chapter 71 — Cross-Framework Interoperability with NIR

### Purpose
Framework and hardware fragmentation is the field's defining software problem. The Neuromorphic Intermediate Representation (NIR) is the closest thing to an "ONNX for neuromorphic," and with Intel's Lava archived in favor of open-standard frameworks, a vendor-neutral IR is now strategically central. This chapter deserves dedicated treatment.

### Topics
- The fragmentation problem made concrete: the same LIF network expressed five incompatible ways
- What NIR is (Nature Communications 2024, arXiv 2311.14641): a backend-agnostic set of composable primitives defined as hybrid continuous-time + discrete-event systems; a model defined once, translated to many backends
- The NIR graph model: nodes, edges, supported primitives (LIF, conv, affine, etc.)
- Current adoption (dated, from NIR docs): read+write in Nengo, Norse, Rockpool, Sinabs, snnTorch, Spyx, hxtorch; read-only in Lava-DL, jaxsnn, SpiNNaker2; hardware targets Loihi 2, SpiNNaker2, Speck, Xylo, BrainScaleS-2
- NIRData: standardized exchange of spiking data
- The portability workflow: train in snnTorch -> export NIR -> import to Rockpool -> deploy to Speck, with the round-trip-fidelity checks that catch silent semantic drift
- Limits: which primitives do and do not round-trip; precision and timing semantics that NIR does not fully fix (cross-reference Ch. 67)
- Comparison with ONNX and why neuromorphic needed its own IR

### Programming demo
Define one SNN, export it to NIR, load it into a second framework, and verify numerically identical (or characterized-divergent) outputs on a fixed input.

### Assignment
Take a model trained in one framework and deploy it to a different backend via NIR. Document every primitive that did not round-trip and explain why.

---

## Chapter 72 — Performance Engineering for SNN Training

### Purpose
BPTT over time is memory- and compute-hungry, and naive SNN code is dominated by Python per-step dispatch. This chapter teaches the techniques that turn a correct-but-slow training loop into a fast one, and how to measure that you actually improved it.

### Topics
- Profiling first: PyTorch profiler, identifying the temporal-loop dispatch bottleneck, GPU utilization vs memory-bound diagnosis
- Vectorizing the time axis: step-mode/fused kernels (SpikingJelly CuPy backend), `torch.compile` kernel fusion (biggest wins on functional/stateless neuron designs, Norse-style), JAX `scan` for the unrolled time loop (Spyx/Slax)
- Memory techniques for long sequences: gradient checkpointing (`torch.utils.checkpoint`) to trade compute for memory; combining with mixed precision (AMP)
- `vmap`/functorch for batched per-sample gradients and Jacobians without Python loops
- Reversible/parallel training (cross-reference Ch. 28): T-RevSNN-style O(L) memory and the practical multi-GPU recipe
- Multi-GPU and large-batch SNN training: what scales and what does not
- Apple Silicon and non-CUDA paths (e.g., mlx-snn) as a practical sidebar
- The measurement discipline: report wall-clock and peak memory, fix seeds, and never claim a speedup without a before/after profile

### Programming demo
Take a baseline snnTorch training loop and apply, in sequence, `torch.compile`, AMP, and gradient checkpointing — measuring wall-clock and peak memory after each, and confirming unchanged accuracy.

### Assignment
Reproduce a published framework benchmark on your own GPU. Explain any discrepancy between your numbers and the paper's in terms of dispatch overhead, kernel fusion, and the time-axis vectorization story.

---

## Chapter 73 — Reproducible Research Engineering for Neuromorphic AI

### Purpose
Surrogate-gradient code is unusually easy to get subtly wrong, and neuromorphic energy claims are unusually easy to make unreproducibly. This chapter teaches the engineering hygiene that makes a result trustworthy, with the SNN-specific tests that generic ML-engineering advice omits.

### Topics
- The environment layer: reproducible Python environments with uv (pure-Python, `uv.lock`) and pixi (GPU/CUDA + native libs via conda-forge); why this layer is distinct from everything below it
- The config layer: Hydra / OmegaConf for composable configs and multirun sweeps over neuron, surrogate-gradient, and time-step hyperparameters
- The tracking layer: Weights & Biases / MLflow / Aim; logging spike rasters, firing rates, sparsity, and surrogate-gradient diagnostics, not just loss
- The data layer: DVC for versioning event datasets and pipeline DAGs
- Determinism and seeding: what is and is not reproducible on GPU; controlling nondeterministic kernels
- Testing scientific/SNN code (the core of the chapter):
  - Gradient correctness as the central SNN test: `torch.autograd.gradcheck` / `gradgradcheck` on custom surrogate-gradient autograd functions (hand-defined backward passes are the #1 silent bug)
  - Property-based testing (Hypothesis) over neuron parameters and time-step counts: invariants like "zero input -> no spikes," "firing rate monotonic in input current"
  - Unit tests for neuron-update math; golden/regression tests pinning a fixed-seed loss curve
- Open-science release standards: NeurIPS reproducibility checklist, NERVE-ML (neural-engineering ML reproducibility checklist), Papers with Code linkage, and licensing (code vs weights vs event-dataset terms, which often differ)

### Programming demo
Wrap an SNN experiment in a uv/pixi + Hydra + W&B scaffold, add a `gradcheck` test for a custom surrogate, add one Hypothesis property test, and produce a one-command reproducible run.

### Assignment
Take a provided "messy" SNN training script and make it reproducible: pin the environment, externalize config, add seeding, add a gradient-correctness test, and produce a filled-in reproducibility checklist.

---

## Chapter 74 — AI-Assisted Development ("Vibe Coding") for Neuromorphic Research

### Purpose
By 2026, researchers routinely use AI coding assistants to write, debug, and refactor scientific software. Done well this is a large productivity multiplier; done badly it silently corrupts reproducibility. This chapter teaches AI-assisted development as a first-class, disciplined research skill rather than a guilty shortcut.

### Topics
- What AI assistants are good and bad at in neuromorphic code: boilerplate, dataloaders, plotting, refactors, and test scaffolding (good) vs novel surrogate-gradient math and subtle temporal semantics (verify everything)
- A workflow that stays reproducible:
  - Pin and record the assistant model + version in the repo; commit prompts/transcripts alongside generated code
  - Generate -> test -> trust: never accept AI-written numerical code without `gradcheck` and property tests (ties to Ch. 73)
  - Use the assistant to write the tests and the docstring-level spec, then the implementation against them
- Published guidance and its application here: LLM-for-software-engineering guidelines (declare usage, report model/version/config, capture traces) adapted to research code
- The verification bottleneck and reproducibility failure modes: nondeterminism of commercial models, missing-code/ambiguous-doc traps; what ResearchCodeBench-style evaluations say about realistic expectations of AI-implemented ML research
- Practical neuromorphic recipes: "implement this surrogate from the paper and prove it with gradcheck," "port this snnTorch model to NIR and verify the round-trip," "profile and speed up this temporal loop and show the before/after"
- Honesty and authorship norms: disclosing AI assistance in papers and code

### Vibe-Coding Capstone
Use an AI assistant end-to-end to implement, test, and profile a small SNN component, while producing the full reproducibility trail (pinned model, committed prompts, passing gradient tests). Submit the artifact, not just the result.

### Assignment
Implement a specified surrogate-gradient function with AI assistance. Deliver: the code, the `gradcheck`/property tests that prove it correct, the prompt transcript, and a one-paragraph reflection on where the assistant was wrong and how the tests caught it.

---

## Chapter 75 — The Neuromorphic AI Cookbook

### Purpose
A consolidated, copy-pasteable recipe collection: the fastest path from "I want to do X" to a running, measured baseline. Each recipe is self-contained, version-pinned, and ends with expected numbers and the failure points that trip practitioners.

### Recipes (each: environment, data, model, train, evaluate, expected result, common failures)
1. From-scratch surrogate-gradient SNN on N-MNIST (snnTorch) with a NeuroBench-style metric report
2. Convolutional SNN on CIFAR10-DVS with TET + TDBN (SpikingJelly), including the sparsity/accuracy tradeoff
3. ANN-to-SNN conversion + distillation fine-tuning to recover accuracy
4. Event-camera gesture recognition on DVS Gesture with Tonic + a time-surface representation
5. Always-on keyword spotting on SHD with an energy-and-SOP report
6. e-prop online learning on a temporal task, compared against BPTT
7. Train-once / deploy-many via NIR: snnTorch -> NIR -> Rockpool -> Speck-class target
8. Fair SNN-vs-ANN energy comparison protocol on one task at a matched accuracy operating point
9. A spiking-Transformer fine-tune (Spike-driven-Transformer-family) baseline
10. A performance-engineering recipe: profile -> `torch.compile` -> AMP -> checkpointing on the same model

### Assignment
Extend one cookbook recipe to a new dataset or hardware target. Document every change required and whether the expected-result numbers held.

---

## Chapter 76 — Capstone: Building a Neuromorphic AI System End-to-End

### Purpose
Integrate everything: a single project carried from problem framing to a measured, reproducible, optionally hardware-deployed system. This is the practitioner counterpart to the research final project (Section 11).

### Topics
- Project framing: is this task actually a fit for neuromorphic AI? (temporal, sparse, low-power, low-latency) — and the honest "no" cases
- The full pipeline as one artifact: sensor/encoding -> model -> training (reproducible) -> evaluation (multi-metric, fair) -> compression (Ch. 66) -> deployment (Ch. 67) -> profiling on target
- System-level power and latency accounting end to end, not just inference SOPs
- Documentation: model card, reproducibility checklist, README that lets a stranger rerun it
- Optional hardware track: take it onto Speck/Xylo (Rockpool/Sinabs) or Akida (MetaTF), or a Loihi-style simulator, and report the simulation-to-hardware gap

### Deliverable
A complete, version-pinned, documented repository implementing one application from Section 11, with a NeuroBench-style evaluation, an energy/latency budget, an AI-assistance disclosure, and a short report on what worked, what did not, and what the limitations are.

---

## Appendices

### Appendix A — Mathematical Prerequisites

This appendix is a compact reference for the mathematical background assumed by the main text. It is not a replacement for a full course in each area, but provides enough to follow every derivation in the book.

#### A.1 Ordinary Differential Equations
- First-order linear ODEs: form, solution, stability
- Euler method: derivation, local truncation error, step size selection
- Stability of Euler method applied to the LIF equation

#### A.2 Linear Algebra
- Vectors and matrices: multiplication, transpose, inverse
- Eigenvalues and eigenvectors: definition, diagonalization
- Spectral radius: definition and relevance to recurrent network stability
- Jacobian matrix: definition, use in linearization of nonlinear systems

#### A.3 Probability and Stochastic Processes
- Random variables: discrete and continuous, expectation, variance
- Poisson process: definition, inter-arrival time distribution, connection to rate coding
- Gaussian distribution: PDF, cumulative distribution
- Ornstein-Uhlenbeck process: SDE, stationary distribution, relevance to neural noise

#### A.4 Information Theory
- Entropy: Shannon entropy, binary entropy function
- Mutual information: definition, connection to coding efficiency
- Channel capacity: relevance to spike train coding

#### A.5 Signal Processing Basics
- Convolution: discrete and continuous
- Matched filters: connection to template matching in event streams
- Fourier transform: frequency representation of spike trains

---

### Appendix B — Deep Learning Prerequisites

This appendix provides the deep learning background assumed when comparing SNNs with ANNs.

#### B.1 Feedforward Neural Networks
- Layers, activations, loss functions
- Forward pass computation
- Backpropagation: chain rule, gradient computation

#### B.2 Convolutional Neural Networks
- Convolutional layers, pooling, padding
- Parameter sharing and translation equivariance

#### B.3 Recurrent Neural Networks
- Vanilla RNN, LSTM, GRU
- Backpropagation through time (BPTT)
- Vanishing gradient problem

#### B.4 Transformers and Attention
- Dot-product attention: scaled, multi-head
- Positional encoding
- Why attention is expensive: O(n^2) complexity

#### B.5 Optimization
- SGD, momentum, Adam
- Learning rate schedules
- Batch normalization

---

### Appendix C — Python and PyTorch Quickstart

For readers new to PyTorch, this appendix covers:
- Tensors and autograd
- Custom nn.Module and custom autograd Functions (the basis of surrogate gradients)
- Training loop structure
- GPU usage
- Reproducible environment setup with uv / pixi; installation of snnTorch, Norse, SpikingJelly, tonic
- Pointers forward to Part XIII for performance, testing, interoperability, and AI-assisted workflows

---

### Appendix D — Notation and Symbol Reference

A complete table of all symbols used in the book:
- $V$, $V_m$: membrane potential
- $V_{th}$: threshold potential
- $V_r$: reset potential
- $V_{rest}$: resting potential
- $\tau_m$: membrane time constant
- $R_m$: membrane resistance
- $I(t)$: input current
- $S(t)$: spike indicator (0 or 1)
- $t_{ref}$: refractory period
- $W$: weight matrix
- $w_{ij}$: synaptic weight from neuron j to neuron i
- $T$: number of time steps
- $\Delta t$: time step size
- $f$: firing rate
- $\lambda$: eligibility trace decay rate
- $e_{ij}$: eligibility trace for synapse (i,j)
- SOP: synaptic operations count
- AC: accumulate operation
- MAC: multiply-accumulate operation

---

### Appendix E — Glossary

150-term glossary covering all field-specific vocabulary from neuroscience, engineering, machine learning, and hardware. Every technical term used before its definition chapter should appear here.

---

## 7. Programming Track

The book contains a progressive programming track.

| Module | Programming Goal | Suggested Tools |
|---|---|---|
| 1 | Simulate basic spike trains and raster plots | NumPy, Matplotlib |
| 2 | Implement LIF neuron and F-I curve | NumPy |
| 3 | Phase plane analysis of spiking neuron | NumPy, SciPy |
| 4 | Stochastic LIF with OU input | NumPy |
| 5 | Build small SNN layers | PyTorch |
| 6 | Train SNN with surrogate gradients | snnTorch, SpikingJelly, Norse |
| 7 | Implement STDP and WTA | NumPy/PyTorch |
| 8 | Implement e-prop eligibility traces | PyTorch |
| 9 | Encode images into spikes | PyTorch, torchvision |
| 10 | Train on N-MNIST or DVS Gesture | tonic, snnTorch, SpikingJelly |
| 11 | Process event-camera streams | tonic, DSEC/MVSEC loaders |
| 12 | Build SNN reservoir | NumPy/PyTorch |
| 13 | ANN-to-SNN conversion with distillation | PyTorch, snnTorch |
| 14 | Estimate energy and SOP count | custom Python |
| 15 | Build hybrid ANN-SNN system | PyTorch |
| 16 | Hardware-aware simulation | Lava, Nengo, Loihi-style abstractions |
| 17 | Implement HDC classifier | NumPy/PyTorch |
| 18 | Forward-Forward algorithm | PyTorch |
| 19 | SNN pruning and quantization pipeline | PyTorch |
| 20 | NeuroBench evaluation on a trained SNN | NeuroBench library |
| 21 | Cross-framework round-trip via NIR | snnTorch, NIR, Rockpool |
| 22 | Performance engineering: profile → compile → AMP → checkpoint | PyTorch profiler, torch.compile |
| 23 | Reproducible experiment scaffold + gradient/property tests | uv/pixi, Hydra, W&B, gradcheck, Hypothesis |
| 24 | AI-assisted surrogate implementation with full reproducibility trail | AI assistant + pytest |
| 25 | End-to-end capstone: train → compress → deploy → profile | snnTorch, Sinabs/Rockpool or MetaTF |

---

## 8. Suggested Python Libraries and Frameworks

### Core numerical tools
- NumPy
- SciPy
- Matplotlib
- PyTorch
- torchvision

### Spiking neural network tools (with 2026 status; see Ch. 70 for the full dated map)
- snnTorch — **primary teaching library** (PyTorch, surrogate gradients, NIR support)
- SpikingJelly — performance track (custom CUDA; note even-PyPI-stale / odd-GitHub-active versioning)
- Norse — functional PyTorch building blocks, `torch.compile`, low memory
- BindsNET — bio/STDP/reservoir focus, lower release cadence
- Brian2 — computational-neuroscience modeling (biophysical realism)
- Nengo — NEF/cognitive modeling, NengoLoihi backend
- NEST, GeNN/PyGeNN, pyNN — large-scale simulation / abstraction layer
- **JAX family (emerging, pre-1.0):** Spyx (fast JIT; flag: no release since early 2024), Slax (online-training approximations), SNNAX, jaxsnn (event-driven, EventProp)

### Hardware SDKs (deployment; see Ch. 70-71)
- SynSense Rockpool + Sinabs — most practitioner-deployable (Xylo audio, Speck event-vision); NIR support
- BrainChip MetaTF — convert TensorFlow/Keras CNNs to Akida edge deployment
- SpiNNaker sPyNNaker / PyNN — SpiNNaker2 / SpiNNcloud
- Intel Lava + Lava-DL — **deprecated/archived (May 2026); taught historically only**

### Interoperability (dedicated chapter, Ch. 71)
- NIR (Neuromorphic Intermediate Representation) — the "ONNX for neuromorphic"; train once, deploy to many backends

### Event-based data tools
- tonic (datasets and transforms for event-based data) — primary
- Prophesee Metavision SDK / OpenEB (open modules; full SDK needs a license since 5.3)
- AEStream (real-time event streaming), expelliarmus (fast .raw/.dat decoding)
- DSEC and MVSEC dataset loaders; custom sparse tensor pipelines

### Evaluation, reproducibility, and engineering tooling (Ch. 72-74)
- NeuroBench (standardized neuromorphic benchmark, 2.x)
- uv / pixi (reproducible environments), Hydra / OmegaConf (config), Weights & Biases / MLflow / Aim (tracking), DVC (data versioning)
- torch.compile, AMP, gradient checkpointing, vmap/functorch, JAX scan (performance)
- torch.autograd.gradcheck, Hypothesis (testing surrogate-gradient and SNN code)
- AI coding assistants (Claude Code, Copilot, Cursor) used with reproducibility discipline

### Robotics and simulation extensions
- Gymnasium
- PyBullet
- ROS/ROS2 concepts
- simple custom robot simulators

---

## 9. Suggested Labs

### Lab 1 — Dense vs Event-Driven Processing
Simulate a signal where most values do not change. Compare dense processing with event-only processing. Count operations saved.

### Lab 2 — A Single Spiking Neuron
Implement an integrate-and-fire neuron and visualize threshold firing.

### Lab 3 — LIF Dynamics and the F-I Curve
Add leakage, reset, refractory period, and time constants. Derive and verify the F-I curve analytically.

### Lab 4 — Phase Plane Analysis
Plot the nullclines and vector field of a two-variable spiking neuron. Identify fixed points and their stability.

### Lab 5 — Stochastic LIF
Add Ornstein-Uhlenbeck noise. Plot ISI distribution and compare to Poisson.

### Lab 6 — Spike Coding
Encode scalar values and images using rate coding, latency coding, and population coding.

### Lab 7 — Sparse Coding
Implement dictionary learning on image patches. Visualize learned basis functions.

### Lab 8 — A Feedforward SNN
Build a simple SNN for classification on MNIST or N-MNIST.

### Lab 9 — Surrogate Gradient Training
Train an SNN end-to-end with surrogate gradients using snnTorch.

### Lab 10 — STDP Pattern Learning
Implement STDP and WTA. Visualize learned selectivity.

### Lab 11 — Eligibility Traces and e-prop
Implement e-prop for a temporal classification task. Compare with BPTT.

### Lab 12 — ANN-to-SNN Conversion
Train a small ANN and convert it to an SNN. Add distillation fine-tuning.

### Lab 13 — Event Camera Data
Load DVS Gesture data with tonic. Build a time-surface representation and classify.

### Lab 14 — Neuromorphic Audio
Convert audio into spikes using a gammatone filterbank. Train a keyword classifier on SHD.

### Lab 15 — Reservoir Computing
Build a liquid state machine for temporal classification. Measure the effect of spectral radius.

### Lab 16 — Neuromorphic Edge Evaluation
Evaluate a model using accuracy, latency, spike count, SOP count, memory, and energy proxy.

### Lab 17 — Hybrid ANN-SNN Model
Build a system where the event-processing front-end is spiking and the classifier back-end is conventional.

### Lab 18 — Final Project Prototype
Build a complete neuromorphic AI system for an application domain.

### Lab 19 — Cross-Framework Round-Trip with NIR (Ch. 71)
Define one SNN, export to NIR, import into a second framework, and verify identical (or characterized-divergent) outputs on a fixed input.

### Lab 20 — Performance Engineering (Ch. 72)
Take a baseline snnTorch training loop and apply, in sequence, `torch.compile`, AMP, and gradient checkpointing. Measure wall-clock and peak memory after each step and confirm unchanged accuracy.

### Lab 21 — Reproducible Experiment Scaffold (Ch. 73)
Wrap an SNN experiment in a uv/pixi + Hydra + W&B scaffold. Add a `gradcheck` test for a custom surrogate and one Hypothesis property test. Produce a one-command reproducible run.

### Lab 22 — AI-Assisted Surrogate Implementation (Ch. 74)
Implement a surrogate-gradient function with an AI assistant. Deliver the code, the gradient/property tests proving it correct, the prompt transcript, and a reflection on what the assistant got wrong.

### Lab 23 — Deploy to Edge Hardware (Ch. 67, 70, 76)
Take a trained SNN through quantization and mapping to a SynSense Speck/Xylo (Sinabs/Rockpool) or Akida (MetaTF) target, or a Loihi-style simulator, and report the simulation-to-hardware gap.

---

## 10. Assignment Types

### Conceptual assignments
- Explain dense vs sparse computation.
- Compare ANN and SNN neurons.
- Explain spike coding methods including phase and burst coding.
- Discuss when neuromorphic AI is useful and when it is not.
- Explain the accuracy gap and its known causes.

### Mathematical assignments
- Derive discrete-time LIF update from the continuous ODE.
- Derive the F-I curve analytically.
- Analyze fixed-point stability via Jacobian eigenvalues.
- Plot the STDP learning window.
- Derive the surrogate gradient backward pass for Euler LIF.
- Analyze spectral radius and reservoir stability.
- Derive the SOP count formula for a two-layer SNN.

### Programming assignments
- Implement spiking neurons.
- Build SNN classifiers.
- Train with surrogate gradients.
- Implement STDP.
- Implement e-prop.
- Process event-camera streams.
- Compare ANN and SNN models.
- Implement distillation from ANN to SNN.

### Experimental assignments
- Compare coding schemes.
- Compare SNN depth vs temporal length.
- Study spike regularization.
- Measure robustness to input noise.
- Estimate energy using SOP count.
- Measure accuracy vs time steps in ANN-to-SNN conversion.

### Paper-review assignments
- Review a paper on Loihi 2.
- Review a paper on SpiNNaker2.
- Review a paper on event cameras.
- Review a paper on surrogate-gradient SNN training.
- Review a paper on memristive neuromorphic systems.
- Review a paper on e-prop or eligibility trace learning.
- Review a paper claiming neuromorphic energy efficiency: evaluate the measurement methodology.

### Design assignments
- Design neuromorphic drone perception with power budget.
- Design always-on audio sensing with event-based hardware.
- Design industrial anomaly detection pipeline.
- Design biomedical monitoring system.
- Design event-based smart camera system.
- Design an on-chip learning rule for sensor adaptation.

### Software-engineering assignments (Part XIII)
- Select and justify a framework stack for three different project briefs; audit one library's maintenance health from primary sources.
- Port a model across backends via NIR; document every primitive that did not round-trip.
- Profile and speed up a temporal training loop; report before/after wall-clock and peak memory with unchanged accuracy.
- Make a "messy" SNN script reproducible: pin the environment, externalize config, add seeding, add a gradient-correctness test, fill in a reproducibility checklist.
- Implement a surrogate with an AI assistant and prove it correct with `gradcheck` plus property tests; submit the full reproducibility trail.
- Trace a hyped claim (a hardware spec or an energy number) to its primary source and write a one-paragraph hype-calibration verdict.

---

## 11. Final Project Ideas

1. Event-based gesture recognition on DVS Gesture
2. SNN classifier for N-MNIST with full metric evaluation
3. Low-power wake-word detection on SHD
4. Spiking anomaly detector for synthetic sensor data
5. Event-camera object tracker using time surfaces
6. ANN-to-SNN conversion study with distillation
7. SNN vs CNN energy-latency comparison using SOP metric
8. STDP-based unsupervised feature learning with WTA
9. Neuromorphic obstacle avoidance for a toy robot in simulation
10. Hybrid event-camera + lightweight CNN model
11. Spiking reservoir for time-series classification
12. Neuromorphic biomedical signal monitor
13. Hardware-aware SNN optimization for a chip with core constraints
14. Spike-count regularization for efficient inference
15. Robustness of SNNs to noisy and corrupted event streams
16. Eligibility trace learning for a temporal task
17. Stereo event camera depth estimation (simulation)
18. Fair energy comparison: SNN vs ANN on keyword spotting

### Practitioner / software-track capstones (Part XIII)
19. Train-once / deploy-many: one model shipped to two backends via NIR, with a round-trip-fidelity report
20. Performance-engineering study: reproduce a published framework benchmark and explain the gap
21. Fully reproducible SNN result: environment lockfile, config, tracking, gradient tests, and a filled reproducibility checklist
22. AI-assisted reimplementation of a recent SNN paper with a complete reproducibility trail
23. End-to-end edge deployment (Ch. 76): train → compress → deploy to Speck/Xylo/Akida → profile the sim-to-hardware gap

---

## 12. Course Mapping Options

### 12.1 Short 7-Week Module

| Week | Topic |
|---|---|
| 1 | Neuromorphic AI overview and spiking neurons (Ch. 1-3) |
| 2 | Neural dynamics, LIF models, spike coding (Ch. 6-7, 12-13) |
| 3 | SNN architectures (Ch. 16-18) |
| 4 | Learning: STDP, surrogate gradients, eligibility traces (Ch. 23, 25, 29) |
| 5 | Event cameras and event-based sensing (Ch. 39-40) |
| 6 | Neuromorphic hardware and edge AI (Ch. 32-33, 44) |
| 7 | Applications, limitations, and projects (Ch. 45-48, 56) |

### 12.2 Standard 13-Week Graduate Course

| Week | Lecture Topic | Lab Topic |
|---|---|---|
| 1 | What is neuromorphic AI? Biological motivation (Ch. 1, 4) | Dense vs event-driven processing |
| 2 | Dynamical systems, LIF neuron (Ch. 6, 7) | LIF simulation and F-I curve |
| 3 | Richer neuron models, stochastic SNNs (Ch. 8, 10) | Stochastic LIF |
| 4 | Spike coding (Ch. 12, 13) | Encoding images into spikes |
| 5 | Feedforward and recurrent SNNs (Ch. 16, 17) | SNN classifier |
| 6 | STDP and sparse coding (Ch. 15, 23) | STDP pattern learning |
| 7 | Surrogate gradients and normalization (Ch. 25, 26) | Train SNN with PyTorch |
| 8 | ANN-to-SNN conversion, distillation (Ch. 27, 51) | Conversion experiment |
| 9 | Eligibility traces, e-prop (Ch. 29) | e-prop temporal task |
| 10 | Event cameras and algorithms (Ch. 39, 40) | Event-data pipeline |
| 11 | Neuromorphic hardware (Ch. 32-33, 37) | Hardware-aware constraints |
| 12 | Edge AI, robotics, evaluation (Ch. 44, 45, 53) | Low-power evaluation |
| 13 | Hybrid systems, limitations, future directions (Ch. 49-50, 56, 62) | Final project presentations |

Software thread for the 13-week course: weave Ch. 70 (framework selection) into week 5, Ch. 72-73 (performance + reproducibility) into weeks 7-9 as lab discipline, and Ch. 71/76 (NIR + deployment) into weeks 11-13.

### 12.3 Advanced 26-Week Course

- Semester 1: Parts I-VI (Foundations, dynamics, neuron models, coding, architectures, learning)
- Semester 2: Parts VII-XIII (Hardware, event sensing, applications, hybrid systems, benchmarking, research frontiers, and the software stack)

### 12.4 Practitioner / Industry Track (software-first, ~10 weeks)

For engineers and edge-AI developers who want a working system fast. Reads the recipe spine and Part XIII largely standalone, pulling theory only as needed.

| Week | Topic | Hands-on |
|---|---|---|
| 1 | Why neuromorphic, when it fits (Ch. 1-3, 56) | Dense vs event-driven |
| 2 | LIF and surrogate-gradient training in practice (Ch. 7, 25) | Train an SNN with snnTorch |
| 3 | The 2026 ecosystem and stack selection (Ch. 70) | Pick and justify a stack |
| 4 | Event data and sensors (Ch. 39-40) | Tonic + DVS Gesture pipeline |
| 5 | Evaluation that survives scrutiny (Ch. 53-55) | NeuroBench-style report |
| 6 | Performance engineering (Ch. 72) | Profile → compile → AMP → checkpoint |
| 7 | Reproducible engineering + testing (Ch. 73) | Lockfile, config, gradcheck |
| 8 | AI-assisted development (Ch. 74) | AI-assisted component with test trail |
| 9 | Interop and deployment (Ch. 71, 66, 67) | NIR round-trip; compress; map to chip |
| 10 | Capstone (Ch. 75-76) | Ship one cookbook recipe end to end |

---

## 13. Recommended Book Features

### 13.1 Chapter boxes
Each chapter includes boxed elements:

- **Plain-English idea**
- **Mathematical form**
- **Engineering meaning**
- **Common misconception**
- **Hype calibration** (what the evidence actually shows)
- **When this matters in real systems**
- **Programming checkpoint**
- **Research connection**
- **Library Spotlight** (which maintained 2026 library implements this, version, strengths/weaknesses)
- **Recipe** (self-contained, version-pinned, with expected numbers and common failure points)
- **Vibe-Coding callout** (using AI assistants responsibly, with the reproducibility discipline)
- **Open problems** (frontier chapters: the stated unsolved questions)

### 13.2 Recurring examples
Use several recurring examples throughout the book:

1. A single spiking neuron (introduced in Part I, used through Part VI)
2. A simple MNIST/N-MNIST image classifier (Parts IV-VI)
3. A moving-dot event camera example (Parts IV, VIII, IX)
4. A low-power keyword detector (Parts VIII, IX)
5. A drone obstacle-avoidance system (Parts VIII, IX, XII)
6. An industrial sensor anomaly detector (Part IX)
7. A hybrid ANN-SNN edge system (Part X)

### 13.3 Visual learning aids
Include:

- Phase plane diagrams
- F-I curves
- Step-by-step diagrams
- Timelines
- Raster plots
- Spike histograms
- Surrogate function comparison plots
- BPTT unrolling diagrams
- Hardware block diagrams
- Model architecture diagrams
- Training workflow diagrams
- Energy-latency-accuracy tradeoff plots
- SOP counting diagrams
- Accuracy gap benchmark tables

---

## 14. Important Conceptual Threads Across the Book

The book repeatedly returns to these central themes:

1. **Time is information**  
   In neuromorphic AI, timing is not just an implementation detail. It can carry meaning.

2. **Sparsity is power**  
   Sparse events can reduce computation, communication, and energy — but only on the right hardware.

3. **Hardware matters**  
   SNNs are not automatically efficient on ordinary hardware. Efficiency requires suitable hardware, and comparing SNN energy on a neuromorphic chip with ANN energy on a GPU is not a valid comparison.

4. **Training is the bottleneck**  
   Spikes are powerful but hard to train. The accuracy gap between SNNs and ANNs is real and should not be dismissed or hidden.

5. **Neuromorphic AI is application-dependent**  
   It is strongest for temporal, sparse, sensor-driven, low-power, and low-latency tasks. It is not competitive with ANNs on static dense tasks like language modeling.

6. **Hybrid systems are practical**  
   Real systems may combine SNNs, CNNs, Transformers, event cameras, and edge accelerators. Purity is not a goal.

7. **Accuracy alone is not enough**  
   Neuromorphic systems must be evaluated using accuracy, latency, energy, spike count, and robustness together.

8. **Efficiency is not automatic**  
   SNNs save energy only when (a) the right hardware exists, (b) the network is actually sparse, and (c) the comparison is fair. Theoretical SOP counts are not the same as measured system energy.

9. **The accuracy gap is real**  
   SNNs trained from scratch currently trail ANNs on most standard benchmarks. Understanding why is a research frontier. Students should be able to state the gap quantitatively, not dismiss it.

10. **Deployment is not trivial**  
    A trained SNN and a deployed SNN behave differently. Numerical precision, timing discretization, routing delays, and hardware mapping all introduce discrepancies. The deployment pipeline is a first-class engineering concern, not a footnote.

11. **Neuromorphic AI is not only spiking**  
    Hyperdimensional computing, liquid time-constant networks, neural ODEs, and neuromorphic photonics are all brain-inspired approaches that share the same hardware and application domains but operate differently from SNNs. A student who knows only spiking networks has an incomplete picture.

12. **Software is where the difficulty lives**  
    The distance between a published equation and a working, measured, reproducible system is large and is where most practitioners get stuck. Tool choice, performance engineering on the time axis, gradient-correctness testing, interoperability (NIR), and reproducible workflows are not peripheral skills; they are the difference between a result and an anecdote. Surrogate-gradient code in particular is easy to get silently wrong, so gradient checking is a non-negotiable habit.

13. **Track the primary source**  
    This field moves fast and is heavily hyped. Hardware specs (Loihi 3), energy numbers (often CPU/RTL projections, not wall-power), and accuracy claims (often conversion or hybrid, not from-scratch) must be traced to primary sources with dates. The book models this discipline rather than repeating press releases.

---

## 15. Suggested Reference Themes

### Core neuromorphic computing
- Neuromorphic computing reviews and surveys
- Brain-inspired computing surveys
- Digital and analog neuromorphic architectures

### Spiking neuron models and dynamics
- LIF, AdEx, Izhikevich model papers
- Gerstner's spike response model
- Dynamical systems and bifurcation theory for neurons

### SNN training
- Surrogate gradient methods
- STDP and voltage-based STDP
- ANN-to-SNN conversion
- e-prop and eligibility-trace learning
- Knowledge distillation for SNNs
- SpikeProp, tempotron, ReSuMe

### Neuromorphic hardware
- Loihi and Loihi 2 (Intel)
- IBM TrueNorth
- SpiNNaker and SpiNNaker2
- BrainScaleS
- DYNAP-SE
- Akida
- Memristive systems (RRAM, PCM)
- FPGA neuromorphic implementations

### Event-based sensing
- Dynamic vision sensors
- Event-based vision algorithms
- Stereo event cameras and 3D perception
- Event-based robotics
- Neuromorphic audio and tactile sensing

### Theory
- Universal approximation theorems for SNNs
- Liquid state machine computation theory
- PAC learning theory for spiking networks

### Hyperdimensional computing
- HDC/VSA surveys and hardware implementations
- HDC for edge AI and biosignal classification
- HDC + SNN hybrid systems

### Biologically plausible learning
- Forward-Forward algorithm (Hinton 2022)
- Equilibrium propagation (Scellier & Bengio 2017)
- Contrastive Hebbian learning
- Dendritic error learning (Guerguiev et al., Sacramento et al.)

### Model compression and deployment
- SNN pruning and quantization-aware training
- NAS for SNNs
- Deployment pipeline: Lava, NengoLoihi, Sinabs, Rockpool

### Neuromorphic optimization
- Hopfield networks and spiking energy minimization
- Loihi constraint satisfaction demonstrations
- QUBO mapping to spiking networks

### Explainability
- Temporal saliency maps for SNNs
- Spike-rate GradCAM
- Faithfulness evaluation for temporal explanations

### Spiking foundation models and LLMs
- SpikeGPT, SpikeLLM, NSLLM, SpikeLM
- Darkit toolkit
- Spiking SSMs and Mamba hybrids

### Neuromorphic photonics
- VCSEL and RTD-based photonic spiking neurons
- Roadmap on neuromorphic photonics
- GHz photonic spiking chip demonstrations

### Neuromorphic continual learning
- CG-SNN, ALADE-SNN, Replay4NCL
- Survey: arXiv 2410.09218

### In-sensor computing
- 3D-stacked in-sensor computing (3DS-ISC)
- Near-sensor feature extraction for event cameras

### Applications
- Edge AI
- Robotics
- Biomedical monitoring
- Industrial anomaly detection
- Autonomous systems

### Software, interoperability, and reproducibility
- NIR (Neuromorphic Intermediate Representation), Nature Communications 2024
- SNN framework benchmarking (Open Neuromorphic), SpikingJelly (Science Advances), Spyx (JAX)
- Reproducibility checklists (NeurIPS, NERVE-ML) and LLM-assisted-coding guidelines for research software
- Roadmaps and surveys: Kudithipudi et al., "Neuromorphic computing at scale," Nature 637 (2025); "The road to commercial success for neuromorphic technologies," Nature Communications 2025

### Spiking foundation and efficient large models
- SpikingBrain-7B/76B, SpikeLLM, SpikeLM, SpikeMLLM, Darkit
- BitNet b1.58 (quantization cousin, taught for contrast)

---

## 16. Working Bibliography Starters

These are not complete references, but useful source categories to build the bibliography:

1. Reviews of neuromorphic chip architectures and event-based non-von-Neumann computation.
2. Reviews comparing deep and spiking neural networks for edge AI.
3. Papers on Loihi 2 and programmable neuromorphic cores.
4. Papers on SpiNNaker2 and large-scale event-based asynchronous machine learning.
5. Papers on surrogate-gradient training of SNNs.
6. Papers on STDP and local plasticity (classical and voltage-based).
7. Papers on e-prop and eligibility-trace-based online learning.
8. Papers on ANN-to-SNN conversion and knowledge distillation.
9. Papers on event-based cameras and neuromorphic vision.
10. Papers on stereo event cameras and 3D reconstruction.
11. Papers on memristive crossbars and in-memory computing.
12. Papers on neuromorphic robotics and UAV perception.
13. Papers on FPGA implementations of spiking neural networks.
14. Maass (1997) and related papers on computational power of spiking networks.
15. Papers on fair energy measurement and benchmark methodology for neuromorphic AI.

---

## 17. Suggested Writing Style

The book uses the following style:

- Formal but plain-language explanations
- Step-by-step derivations with no missing steps
- Many analogies, but not vague metaphors
- Every equation introduced with intuition first
- Every algorithm connected to implementation
- Every hardware concept connected to AI behavior
- Every advanced topic connected to a simple example
- Research-level depth without assuming neuroscience background
- Hype calibration as a standard feature, not an exception
- Negative results and limitations discussed openly

Example style:

> A spike is not a number like 0.73 in a standard neural network. A spike is an event: it happened or it did not happen. The meaning comes not only from which neuron fired, but also from when it fired, how often it fired, and which other neurons fired nearby in time.

---

## 18. Proposed Subtitle Variants

1. **Understanding Neuromorphic AI: From Spiking Neurons to Intelligent Edge Systems**
2. **Understanding Neuromorphic AI: Event-Driven Learning for Low-Power Intelligent Systems**
3. **Understanding Neuromorphic AI: Spikes, Sensors, Hardware, and Edge Intelligence**
4. **Understanding Neuromorphic AI: A Practical Guide to Spiking Neural Networks and Event-Based Computing**
5. **Understanding Neuromorphic AI: From Sparse Spikes to Autonomous Edge Systems**

Recommended subtitle:

> **From Spiking Neurons to Intelligent Edge Systems**

---

## 19. Complete One-Paragraph Book Description

**Understanding Neuromorphic AI: From Spiking Neurons to Intelligent Edge Systems** is a comprehensive graduate-level guide to the theory, algorithms, hardware, and applications of neuromorphic artificial intelligence. Starting from the basic idea of spikes as time-based computational events, the book develops mathematical foundations in dynamical systems, formal spiking neuron models including stochastic variants, spike coding theory, spiking neural network architectures including generative models, and a complete treatment of learning algorithms spanning STDP, surrogate gradients, eligibility traces, and knowledge distillation. Hardware coverage extends from digital neuromorphic chips and analog mixed-signal systems to memristive devices and FPGA implementations. Event-based sensing chapters cover monocular and stereo event cameras, neuromorphic audio, and tactile sensing. Applications span edge AI, robotics, biomedical monitoring, and autonomous systems. Evaluation chapters address fair energy measurement, benchmark methodology, and limitations including the accuracy gap between SNNs and ANNs. Recent developments receive full treatment, including spiking Transformers and state space models, spiking and brain-inspired large models (SpikeLLM, SpikingBrain, SpikeMLLM), reversible and temporally-efficient training, the NeuroBench evaluation standard, and 2025-2026 hardware (SpiNNcloud, Innatera, Akida, SynSense). A dedicated software part treats the field as an engineering discipline: framework selection, cross-backend interoperability through the Neuromorphic Intermediate Representation (NIR), performance engineering, reproducible workflows, AI-assisted ("vibe coding") development, an end-to-end recipe cookbook, and a deployment capstone. Each concept is explained first in plain language and then developed formally through equations, diagrams, runnable version-pinned recipes, and assignments, with Hype Calibration boxes throughout that teach students to trace claims to primary sources. The book is designed to serve both as a complete introduction to neuromorphic computing and as an advanced graduate-level and practitioner resource for students, researchers, and engineers building energy-efficient, event-driven, and brain-inspired AI systems.

---

## 20. Short Back-Cover Description

Modern AI is powerful, but often computationally expensive, energy-hungry, and poorly suited for always-on edge intelligence. Neuromorphic AI offers a different path: sparse, temporal, event-driven computation inspired by the nervous system and implemented through spiking neural networks, event-based sensors, and specialized hardware. This book explains neuromorphic AI from the ground up, beginning with dynamical systems foundations and spiking neuron models, progressing through neural coding, spiking network architectures, and the full range of learning algorithms from STDP and surrogate gradients to eligibility traces. Hardware coverage spans digital neuromorphic chips, analog systems, memristive devices, and FPGAs. Event-based sensing chapters cover stereo and 3D vision, audio, and multimodal fusion, and a full software part takes you from framework choice and cross-backend interoperability through performance engineering, reproducible workflows, AI-assisted development, and edge deployment. With plain-language intuition, rigorous derivations, diagrams, runnable recipes, and assignments, it provides a self-contained foundation for understanding, building, and shipping the next generation of low-power intelligent systems — while teaching students to evaluate claims critically rather than accepting efficiency numbers at face value.

---

## 21. Final Recommended Scope

The book supports:

- A basic neuromorphic computing course
- An advanced spiking neural networks course
- A neuromorphic hardware/software co-design course
- A neuromorphic robotics or event-based vision course
- A research seminar on energy-efficient AI
- Project-based graduate instruction
- A software-first practitioner / industry track (Part XIII, see Section 12.4)

The scope:

> **Neuromorphic AI = spiking computation + temporal learning + event-based sensing + specialized hardware + a reproducible software stack + energy-aware intelligent systems.**

**Chapter count:** 76 chapters in 13 parts plus 5 appendices. (Part XIII — The Neuromorphic AI Software Stack, Ch. 70-76 — is the practitioner/software spine added in this revision.)

**Approximate page count at standard textbook density:** 1150-1450 pages.

**Two-semester graduate course:** Parts I through VI in Semester 1; Parts VII through XIII in Semester 2 (practitioner-track courses can pull Part XIII forward and read it alongside Part VI).

**Chapter index of additions relative to original 50-chapter outline:**
- Ch. 4: Biological Neurons — What We Borrow and Why (new)
- Ch. 6: Dynamical Systems — expanded with Wilson-Cowan, Fokker-Planck, neural mass models (expanded)
- Ch. 10: Stochastic Spiking Networks and Noise (new)
- Ch. 15: Sparse Coding and Predictive Coding (new)
- Ch. 20: Generative and Unsupervised Spiking Models (new)
- Ch. 24: SpikeProp, Tempotron, ReSuMe (new)
- Ch. 26: Normalization in Spiking Networks (new)
- Ch. 28: Direct Training — expanded with parallel/reversible SNN training (expanded)
- Ch. 29: Online Learning and Eligibility Traces / e-prop (new)
- Ch. 31: Continual Learning — expanded with neuromorphic continual learning subfield and federated learning (expanded)
- Ch. 33: Digital Chips — expanded with Loihi 3, Hala Point, NorthPole, Akida 2.0 (expanded)
- Ch. 37: FPGA and Custom ASIC Implementations (new)
- Ch. 39: Event Cameras — expanded with in-sensor computing (expanded)
- Ch. 41: Event-Based Stereo Vision and 3D Perception (new)
- Ch. 49: ANN-SNN Hybrid Models — expanded with Neural ODEs and LTC networks (expanded)
- Ch. 50: Spiking Transformers, Attention, and State Space Models — expanded with Mamba/SSMs (expanded)
- Ch. 51: Knowledge Distillation for SNNs (new)
- Ch. 52: Neuromorphic AI and Foundation Models — expanded with SpikeGPT, SpikeLLM, NSLLM (expanded)
- Ch. 54: Datasets and Benchmarks — expanded with NeuroBench (expanded)
- Ch. 62: Neuromorphic Photonics (new)
- Ch. 64: Hyperdimensional Computing and Vector-Symbolic Architectures (new)
- Ch. 65: Biologically Plausible Learning Without Backpropagation (new)
- Ch. 66: SNN Model Compression: Pruning and Quantization (new)
- Ch. 67: SNN Deployment Pipeline: From Trained Model to Hardware (new)
- Ch. 68: Neuromorphic Computing for Combinatorial Optimization (new)
- Ch. 69: Explainability and Interpretability of SNNs (new)
- **Part XIII — The Neuromorphic AI Software Stack (new, Ch. 70-76):**
  - Ch. 70: The Neuromorphic Software Ecosystem and Framework Selection (new)
  - Ch. 71: Cross-Framework Interoperability with NIR (new)
  - Ch. 72: Performance Engineering for SNN Training (new)
  - Ch. 73: Reproducible Research Engineering for Neuromorphic AI (new)
  - Ch. 74: AI-Assisted Development ("Vibe Coding") for Neuromorphic Research (new)
  - Ch. 75: The Neuromorphic AI Cookbook (new)
  - Ch. 76: Capstone — Building a Neuromorphic AI System End-to-End (new)

**2026 currency updates folded into existing chapters in this revision:**
- Ch. 28: Temporal Efficient Training (TET) and T-RevSNN / ParaRevSNN reversible training
- Ch. 33: Loihi 3 reframed as unverified hype-calibration case; corrected Hala Point figures; SpiNNcloud commercial launch, Innatera Pulsar, Akida 2/Pico; Lava deprecation; Kudithipudi et al. Nature 2025 roadmap
- Ch. 39: Prophesee GenX320 / IMX636, OpenEB licensing, SynSense Speck in-sensor compute
- Ch. 40-41: RVT → SAST → SMamba/PMRVT detection lineage; DERD-Net depth; event self-supervised pretraining as an open problem
- Ch. 50: Spike-driven Transformer V3 (86.2% ImageNet), QKFormer, QSD-Transformer; SpikingSSMs (AAAI 2025), P-SpikeSSM
- Ch. 52: SpikingBrain-7B/76B, SpikeMLLM, BitNet b1.58 contrast
- Ch. 54: NeuroBench 2.x; ECMD/NSAVP/DSEC datasets

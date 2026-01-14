# Dissertation: Networks with Complex Weights

## Context
- 14 weeks total
- Goal: Learn valuable skills, keep career options open
- Willing to learn quantum computing
- Solid undergrad math, can stretch to graduate level

## Learning Preferences
- **Task-based sessions** (not time-based): "Achieve X, Y, Z" with logical breaks
- Can focus for extended periods
- Big picture first, then details
- Active work (computation) over passive reading
- Clear deliverables per session

---

## Starting Point: The Core Mathematical Question

**What does it mean to have complex weights in a network?**

A network with complex weights has adjacency matrix A ∈ C^{n×n} where entries a_ij ∈ C.

**Why would we want this?**
1. **Phase information** - magnitude encodes strength, phase encodes direction/delay/offset
2. **Directed graphs** - asymmetric A has complex eigenvalues anyway
3. **Quantum systems** - amplitudes are naturally complex
4. **Signal processing** - frequency domain representations

**Key mathematical consequences:**
- A may not be diagonalizable over R
- Eigenvalues can be complex (spectral theory changes)
- Inner products need conjugation: ⟨u,v⟩ = u*v
- Hermitian (A = A*) is the "nice" case → real eigenvalues

---

## Possible Directions (to evaluate during weeks 1-2)

| Direction | Core Question | Math Flavor |
|-----------|--------------|-------------|
| Spectral theory | How do complex weights change eigenvalue structure? | Pure math |
| Quantum walks | How do quantum dynamics behave on complex networks? | Math physics |
| Machine learning | When do complex representations help learning? | Applied math |
| Synchronization | How do coupled oscillators behave with complex coupling? | Dynamical systems |

**Decision point:** End of week 2, after reading foundational material.

---

## 14-Week Schedule

### Phase 1: Explore (Weeks 1-2)
| Week | Goal |
|------|------|
| 1 | Read Porter et al. 2024 + skim surveys from spectral/quantum/ML |
| 2 | Identify tractable direction; discuss with supervisor; commit |

**Deliverable:** 1-page summary + chosen direction

### Phase 2: Deep Dive (Weeks 3-5)
| Week | Goal |
|------|------|
| 3 | Read 3-4 key papers in chosen direction |
| 4 | Identify specific research question / gap |
| 5 | Formulate approach, start background writing |

**Deliverable:** Clear problem statement + draft background section

### Phase 3: Research (Weeks 6-10)
| Week | Goal |
|------|------|
| 6-8 | Main technical work (proofs, analysis, or implementation) |
| 9-10 | Experiments / validation / examples |

**Deliverable:** Core results (5 weeks instead of 4)

### Phase 4: Write Up (Weeks 11-14)
| Week | Goal |
|------|------|
| 11 | Draft main results section |
| 12 | Draft intro, conclusion |
| 13 | Full draft complete |
| 14 | Polish, proofread, submit |

---

## Fields & Reading List

### Field 1: Spectral Graph Theory
**Core question:** How do eigenvalues/eigenvectors of graph matrices encode structure?

**Key techniques:**
- Laplacian eigenvalue analysis (algebraic connectivity, clustering)
- Spectral decomposition: A = VΛV*
- Cheeger inequalities (eigenvalue ↔ graph cuts)
- Perturbation theory (how eigenvalues change with edge modifications)

**Papers to read:**
1. **Chung, "Spectral Graph Theory" (1997)** - Ch 1-2 for foundations
   - [Free online](https://mathweb.ucsd.edu/~fan/research/revised.html)
2. **Porter et al. 2024** - "Complex Networks with Complex Weights"
   - Sections on spectral properties of complex adjacency matrices

**What to extract:** How does spectral theory change when weights are complex? What breaks, what still works?

---

### Field 2: Quantum Walks
**Core question:** How do quantum dynamics propagate on graphs?

**Key techniques:**
- Continuous-time QW: |ψ(t)⟩ = e^{-iHt}|ψ(0)⟩ where H is Hermitian
- Discrete-time QW: coin + shift operators
- Amplitude interference (not probability diffusion)
- Perfect state transfer, mixing times

**Papers to read:**
1. **Kempe 2003** - "Quantum random walks: An introductory overview"
   - [arXiv:quant-ph/0303081](https://arxiv.org/abs/quant-ph/0303081)
   - Focus: Sections 1-3 (skip quantum computing applications)
2. **Godsil 2012** - "When can perfect state transfer occur?"
   - [arXiv:1011.0231](https://arxiv.org/abs/1011.0231)
   - Shows how spectral properties determine quantum dynamics

**What to extract:** What graph properties enable/prevent quantum phenomena? How does Hermitian structure matter?

---

### Field 3: Machine Learning on Graphs
**Core question:** How do we learn representations that capture graph structure?

**Key techniques:**
- Message passing: h_v^{(k+1)} = AGG({h_u : u ∈ N(v)})
- Spectral convolution: filter in eigenspace
- Positional encodings (Laplacian eigenvectors)
- Expressivity analysis (Weisfeiler-Leman hierarchy)

**Papers to read:**
1. **Bronstein et al. 2021** - "Geometric Deep Learning" (survey)
   - [arXiv:2104.13478](https://arxiv.org/abs/2104.13478)
   - Sections 5.1-5.3 on GNNs (skim rest)
2. **Zhang et al. 2021** - "MagNet" (complex weights for directed graphs)
   - [arXiv:2102.11391](https://arxiv.org/abs/2102.11391)
   - The main ML paper on complex-weighted graphs

**What to extract:** Why did ML need complex weights? What problems does it solve/not solve?

---

### Field 4: Dynamical Systems on Networks
**Core question:** How do coupled systems behave on network topology?

**Key techniques:**
- Kuramoto model: θ̇_i = ω_i + Σ K_ij sin(θ_j - θ_i)
- Stability analysis (Lyapunov, linearization)
- Synchronization conditions
- Master stability function

**Papers to read:**
1. **Arenas et al. 2008** - "Synchronization in complex networks"
   - [Physics Reports 469(3)](https://arxiv.org/abs/0805.2976)
   - Sections 1-3 for foundations
2. **Porter et al. 2024** - "Complex Networks with Complex Weights"
   - Sections on Schrödinger-Lohe model (quantum Kuramoto)

**What to extract:** How does complex coupling change synchronization? What's the physics intuition?

---

## Reading Strategy

**Week 1 (Days 1-7):**
| Day | Read | Time |
|-----|------|------|
| 1-2 | Porter et al. 2024 (full) | 4-6 hrs |
| 3 | Chung Ch 1-2 (spectral foundations) | 2-3 hrs |
| 4 | Kempe 2003 Sections 1-3 (quantum walks) | 2-3 hrs |
| 5 | Bronstein 5.1-5.3 (GNN basics) | 2 hrs |
| 6 | Arenas 2008 Sections 1-3 (sync) | 2-3 hrs |
| 7 | Write 1-page summary of landscape | 2 hrs |

**Week 2 (Days 8-14):**
- Deep read in 1-2 most promising directions
- Read second paper from those fields
- Identify candidate research questions
- Talk to supervisor

---

## What to Look For While Reading

In each paper, note:
1. **Definitions:** How do they define complex-weighted networks?
2. **Tools:** What mathematical machinery do they use?
3. **Open questions:** What do they say is unsolved?
4. **Connections:** Do they cite papers from other fields?

**Novelty signal:** If Field A uses technique T but Field B doesn't mention it → potential bridge

---

## Questions to Resolve by End of Week 2

1. Which field's questions interest me most?
2. Which field's techniques feel most natural?
3. Where is the gap I can fill?

*(Specific problem identified in weeks 3-5)*

---

# COMPREHENSIVE FIELD ANALYSIS

## Dissertation Requirements (from guidelines)
- **7,500 words** (25-30 pages)
- **50% Mathematical Content**: Difficulty (ambitious scope) + Correctness + Comprehensiveness
- **25% Content**: Coherence + Individuality (novelty)
- **25% Presentation**: Narrative + Clarity
- **For 80+ marks**: "Likely to contain original material - new propositions, examples, or calculations"

---

## FIELD 1: SPECTRAL GRAPH THEORY

### Techniques Available
| Technique | Maturity | Applicable to Complex? |
|-----------|----------|----------------------|
| Laplacian eigenvalue analysis | Mature | Yes (Hermitian case) |
| Cheeger inequalities | Mature | Partial (signed/directed, NOT general complex) |
| Spectral clustering | Mature | Yes via magnetic Laplacian |
| Perturbation theory | Mature | Yes for Hermitian |
| Gershgorin bounds | Mature | Yes |

### What's KNOWN
- Hermitian adjacency matrices for digraphs (Guo-Mohar 2015)
- Magnetic Laplacian preserves spectral theorem
- T-gain graphs (unit circle weights) spectral properties
- Cheeger extended to signed graphs and directed graphs (STOC 2023)
- Frustration index bounds via eigenvalues

### What's UNKNOWN (Gaps)
| Gap | Difficulty | Novelty |
|-----|------------|---------|
| Cheeger inequality for general T-gain graphs | High | Very High |
| Spectral characterization: which T-gain graphs are determined by spectrum? | Medium-High | High |
| Centrality measures for complex-weighted networks | Medium | High |
| Random T-gain graph universality | High | Very High |
| PT-symmetric graph Laplacians | Medium | High |

### Feasibility for Dissertation
**RATING: ⭐⭐⭐⭐ (4/5)**

✅ **Pros:**
- Clean mathematical framework (linear algebra you know)
- Specific open problems with clear statements
- "Spectral characterization of T-gain graphs" explicitly noted as tractable
- Can restrict to specific graph families (circulants, Cayley) for achievable scope

⚠️ **Cons:**
- Some problems require algebraic number theory
- May need representation theory for full generality

**Best tractable problems:**
1. Extend Cheeger to specific T-gain graph families
2. Spectral clustering analysis for magnetic Laplacian
3. Perturbation bounds for complex-weighted graphs

---

## FIELD 2: QUANTUM WALKS

### Techniques Available
| Technique | Maturity | Complexity |
|-----------|----------|------------|
| CTQW via matrix exponential | Mature | Moderate |
| Spectral characterization of PST | Mature | Moderate |
| Strong cospectrality | Developed | Moderate |
| Mixing time analysis | Partial | High |
| Discrete-time QW | Mature | Higher |

### What's KNOWN
- Perfect state transfer (PST) spectral conditions
- PST on: hypercubes, P2, P3, Cartesian products
- Pretty good state transfer (PGST) relaxation
- Fractional revival characterization (2024)
- Laplacian vs Adjacency QW equivalence on regular graphs only

### What's UNKNOWN (Gaps)
| Gap | Difficulty | Novelty |
|-----|------------|---------|
| Combinatorial characterization of strong cospectrality | Medium | High |
| Does C9 admit uniform mixing? | Unknown | Medium |
| Time-independent uniform mixing characterization | Hard | Very High |
| PST on digraphs (systematic theory) | Medium-High | High |
| QW on simplicial complexes | Medium | High |
| Economic PST (polynomial size graphs) | Hard | Very High |

### Feasibility for Dissertation
**RATING: ⭐⭐⭐⭐⭐ (5/5)**

✅ **Pros:**
- Explicit open problem list exists (Coutinho & Guo 2024)
- Problems have clear difficulty ratings
- Can attack restricted graph classes (Cayley, distance-regular)
- Spectral methods you'd learn transfer everywhere
- Active field with recent papers to build on

⚠️ **Cons:**
- Some problems need number theory (rationality conditions)
- Full generality is hard; must restrict scope

**Best tractable problems:**
1. PST/PGST on specific Cayley graphs (use representation theory)
2. Strong cospectrality for vertex-transitive graphs
3. Uniform mixing on small families (attack C9 or similar)
4. Laplacian vs Adjacency QW on irregular graph families

---

## FIELD 3: GRAPH NEURAL NETWORKS (Complex)

### Techniques Available
| Technique | Maturity | Theory vs Empirical |
|-----------|----------|---------------------|
| Magnetic Laplacian GNN | Developed | Mostly empirical |
| Complex spectral convolution | Developed | Some theory |
| Sheaf neural networks | Emerging | Theoretical |
| Positional encodings | Active | Mixed |
| Expressivity (WL hierarchy) | Mature | Theoretical |

### What's KNOWN
- MagNet: Hermitian complex Laplacian for directed graphs
- CWCN (2024): Universal node classification with complex weights
- Complex weights empirically help on heterophilic directed graphs
- Sheaf diffusion separates more classes than trivial sheaves

### What's UNKNOWN (Gaps)
| Gap | Difficulty | Novelty |
|-----|------------|---------|
| Uniform expressivity results (fixed-parameter) | High | Very High |
| Generalization bounds for complex GNNs | High | Very High |
| k-WL comparison: complex vs real weights | Medium-High | High |
| Optimization landscape with complex weights | Medium | High |
| Learnable phase/charge assignment | Medium | Medium |

### Feasibility for Dissertation
**RATING: ⭐⭐⭐ (3/5)**

✅ **Pros:**
- Hot field, high-impact if successful
- CWCN 2024 provides theoretical foundation to build on
- Can combine theory + implementation naturally

⚠️ **Cons:**
- Expressivity proofs are hard (open for years)
- "Uniform setting" analysis noted as major open challenge
- Field moves fast - risk of being scooped
- Less mathematically clean than spectral/QW

**Best tractable problems:**
1. Expressivity comparison on specific graph families
2. Connect magnetic Laplacian to quantum walk dynamics (bridge paper)
3. Spectral filter design with theoretical guarantees

---

## FIELD 4: DYNAMICAL SYSTEMS / SYNCHRONIZATION

### Techniques Available
| Technique | Maturity | Applicable to Complex Coupling? |
|-----------|----------|-------------------------------|
| Kuramoto model | Mature | Extended (2024) |
| Master Stability Function | Mature | NOT for complex weights |
| Lyapunov analysis | Mature | Partially |
| Schrödinger-Lohe model | Developing | Native |
| Bifurcation analysis | Mature | Yes |

### What's KNOWN
- Complexified Kuramoto: purely imaginary K does NOT synchronize
- Complex coupling can cause finite-time blow-up
- Schrödinger-Lohe: phase sync without space sync possible
- Higher-order interactions increase linear stability but shrink basins
- Non-Hermitian sync: PT-symmetry effects emerging

### What's UNKNOWN (Gaps)
| Gap | Difficulty | Novelty |
|-----|------------|---------|
| MSF for complex-weighted networks | High | Very High |
| Complexified Kuramoto for general N | Medium-High | High |
| Higher-order + complex coupling combined | Medium | High |
| Time delays + complex coupling | Medium | High |
| Quantum-classical crossover structure | High | Very High |

### Feasibility for Dissertation
**RATING: ⭐⭐⭐ (3/5)**

✅ **Pros:**
- Complexified Kuramoto (2024) is brand new with low-hanging fruit
- Physics intuition helps
- Numerical experiments are natural

⚠️ **Cons:**
- Requires dynamical systems background
- Bifurcation analysis can be technically demanding
- Less algebraic, more analytic flavor
- Harder to get "clean" theorems

**Best tractable problems:**
1. Complexified Kuramoto phase diagram for small N
2. Higher-order + complex coupling interaction
3. Non-Hermitian sync on specific network topologies

---

## CROSS-FIELD COMPARISON

| Criterion | Spectral | Quantum Walks | GNNs | Dynamics |
|-----------|----------|---------------|------|----------|
| **Math clarity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Open problem clarity** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **14-week achievability** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Skill transferability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Novelty potential** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Risk level** | Medium | Low-Medium | High | Medium-High |

---

## TOP RECOMMENDED PROBLEMS (Ranked by Feasibility)

### Tier 1: High Confidence (Can definitely produce dissertation)

**1. PST/PGST on Cayley Graphs of Specific Groups** (Quantum Walks)
- Extend recent extraspecial group results to dihedral/symmetric groups
- Tools: Representation theory, spectral decomposition
- Clear methodology, bounded scope
- **Feasibility: 95%**

**2. Spectral Properties of T-gain Circulant Graphs** (Spectral)
- Explicit eigenvalue formulas exist for circulants
- Extend to T-gain setting
- Clean algebra, achievable proofs
- **Feasibility: 90%**

**3. Strong Cospectrality for Vertex-Transitive Graphs** (Quantum Walks)
- Exploit symmetry to attack combinatorial characterization
- Connects to Problem 2.3 in open problem list
- **Feasibility: 85%**

### Tier 2: Good Confidence (Should work with focused effort)

**4. Magnetic Laplacian ↔ CTQW Formal Correspondence** (Bridge: GNN + QW)
- Both use Hermitian complex matrices
- Formalize the connection rigorously
- High novelty, clear contribution
- **Feasibility: 80%**

**5. Cheeger-type Inequality for Specific T-gain Families** (Spectral)
- Restrict to signed+directed or specific phase values
- Build on STOC 2023 directed Cheeger
- **Feasibility: 75%**

**6. Complexified Kuramoto Phase Diagram for N=3,4** (Dynamics)
- N=2 done in 2024 paper
- Extend bifurcation analysis to small N
- Numerical + analytical
- **Feasibility: 75%**

### Tier 3: Ambitious (High reward, higher risk)

**7. Uniform Mixing Characterization for Distance-Regular Graphs** (QW)
- Attack Problem 3.3 on restricted family
- Could be hard or could crack open
- **Feasibility: 60%**

**8. Expressivity: Complex vs Real GNN on WL-equivalent Graphs** (GNN)
- Find graphs that complex weights distinguish but real can't
- High impact if successful
- **Feasibility: 50%**

---

## RECOMMENDATION

**Primary: Quantum Walks (Field 2)**

Reasons:
1. Best-defined open problem list with difficulty ratings
2. Clean mathematical framework (spectral + algebra)
3. Natural restriction strategies (specific graph families)
4. Skills transfer to quantum computing, spectral methods, ML
5. Low risk of "no results" - can always characterize specific families

**Backup: Spectral Graph Theory (Field 1)**

If quantum walks feel too unfamiliar, spectral theory is equally clean and uses more familiar linear algebra.

**Avoid for first dissertation: GNNs and Dynamics**

Both have less clear problem statements and higher risk of getting stuck in technical weeds.

---

# CHOSEN DIRECTION: QUANTUM WALKS ON GRAPHS

## Why Quantum Walks

1. **Explicit open problem list**: Coutinho & Guo 2024 paper lists problems with difficulty
2. **Spectral methods**: Core technique is eigenvalue analysis (transferable skill)
3. **Restriction strategy**: Can always limit to specific graph families for guaranteed results
4. **Growing field**: Active research with 2024-2025 papers to build on
5. **Quantum relevance**: Skills applicable to quantum computing

---

## Core Concepts You'll Need

| Concept | What It Is | Where to Learn |
|---------|------------|----------------|
| **CTQW** | U(t) = e^{-iAt} evolution on graph | Kempe survey §1-3 |
| **Perfect State Transfer** | |⟨v|U(t)|u⟩| = 1 for some t | Godsil notes |
| **Strong Cospectrality** | Vertices share eigenvalue support + eigenvector sign pattern | Coutinho-Godsil |
| **Pretty Good State Transfer** | Fidelity → 1 as t → ∞ | Recent papers (2024) |
| **Uniform Mixing** | Equal probability at all vertices at some time | Hard problem |

---

## Specific Problem Options

### Option A: PST on Cayley Graphs (SAFEST)

**Problem:** Characterize perfect state transfer on Cayley graphs of specific groups

**What's known:**
- PST on Cayley graphs of abelian groups: well-understood
- PST on extraspecial 2-groups: recent results (2024)
- General non-abelian: wide open

**Your contribution:**
- Pick a family: dihedral groups D_n, symmetric groups S_n, or quaternion groups
- Compute eigenvalues via representation theory
- Determine PST conditions

**Feasibility:** 95% - bounded scope, clear methodology

**Math needed:** Representation theory basics, eigenvalue computation

---

### Option B: Strong Cospectrality Characterization (MODERATE)

**Problem:** Give combinatorial conditions for when vertices are strongly cospectral

**What's known:**
- Spectral definition exists
- Automorphism-equivalent vertices are always strongly cospectral
- Converse is false

**Your contribution:**
- For vertex-transitive graphs: exploit symmetry
- Find necessary/sufficient combinatorial conditions
- Provide new examples or counterexamples

**Feasibility:** 85% - could crack or hit wall, but partial results still valuable

**Math needed:** Spectral graph theory, some group theory

---

### Option C: CTQW on Directed Graphs via Hermitian Adjacency (NOVEL)

**Problem:** Develop systematic PST/PGST theory for directed graphs

**What's known:**
- Hermitian adjacency matrix for digraphs exists (Guo-Mohar)
- Some sporadic results (2023)
- No systematic theory

**Your contribution:**
- Define PST for digraphs using Hermitian adjacency
- Find digraph families with PST
- Connect to magnetic Laplacian literature

**Feasibility:** 80% - novel direction, less precedent but clear approach

**Math needed:** Complex spectral theory, digraph combinatorics

---

### Option D: Uniform Mixing on Specific Families (AMBITIOUS)

**Problem:** Characterize uniform mixing time-independently

**What's known:**
- Uniform mixing only on few families (K_4, hypercubes, Hamming)
- Even cycles: NO uniform mixing
- C_9: unknown (explicitly open)

**Your contribution:**
- Attack C_9 or similar small cases
- Or: find new families with/without uniform mixing

**Feasibility:** 60% - could be hard; but negative results also valuable

**Math needed:** Eigenvalue rationality, algebraic number theory

---

## Recommended Choice: Option A or C

| | Option A (Cayley PST) | Option C (Directed QW) |
|---|----------------------|----------------------|
| **Safety** | Very high | High |
| **Novelty** | Medium | High |
| **Math style** | Representation theory | Complex linear algebra |
| **Connection to other fields** | Group theory | GNNs, magnetic Laplacian |

**If you want safe + solid:** Option A
**If you want novel + broader connections:** Option C

---

## Refined 14-Week Schedule (Quantum Walks Focus)

### Phase 1: Foundation (Weeks 1-2)
| Week | Goal | Reading |
|------|------|---------|
| 1 | Understand CTQW basics | Kempe survey, Godsil notes |
| 2 | Study PST/PGST conditions | Coutinho-Guo open problems, recent PST papers |

**Deliverable:** Summary of what's known about PST, choice of specific problem

### Phase 2: Background + Problem Setup (Weeks 3-4)
| Week | Goal |
|------|------|
| 3 | Deep read 3-4 papers on chosen subproblem |
| 4 | Identify specific graph family to study, formulate precise question |

**Deliverable:** Problem statement + relevant background written

### Phase 3: Core Research (Weeks 5-9)
| Week | Goal |
|------|------|
| 5-6 | Compute eigenvalues/eigenvectors for chosen family |
| 7-8 | Prove main results (PST conditions, characterizations) |
| 9 | Work examples, find counterexamples if needed |

**Deliverable:** Main theorem(s) + supporting examples

### Phase 4: Validation + Writing (Weeks 10-14)
| Week | Goal |
|------|------|
| 10 | Numerical verification (if applicable) |
| 11 | Draft main results section |
| 12 | Draft intro + background |
| 13 | Complete draft |
| 14 | Polish, proofread, submit |

---

## Key References (Quantum Walks)

**Essential:**
1. Kempe 2003 - "Quantum random walks: An introductory overview" [arXiv:quant-ph/0303081]
2. Coutinho & Guo 2024 - "Selected Open Problems in CTQW" [arXiv:2404.02236]
3. Godsil - "Graph Spectra and Continuous Quantum Walks" [Waterloo notes]

**For specific problems:**
- Option A: Papers on Cayley graph PST, representation theory
- Option C: Guo-Mohar Hermitian adjacency, recent digraph QW papers
- General: Godsil & Zhan textbook (2023)

---

## Next Steps

1. **Week 1:** Read Kempe survey (sections 1-3) + Coutinho-Guo open problems
2. **End of Week 1:** Decide between Option A (Cayley PST) or Option C (Directed QW)
3. **Week 2:** Deep dive into chosen option's literature
4. **Meet supervisor:** Discuss specific problem scope

---

# TODAY'S SESSION PLAN (Task-Based)

**Progress:** Completed Kempe Section 1 (Introduction)

## Session 1: Classical Foundations
**Goal:** Understand how classical random walks work spectrally

**Tasks:**
1. Read Kempe Section 2 (Classical Random Walks)
2. Extract: transition matrix P, stationary distribution, spectral gap
3. Compute: transition matrix P for path graph P_4

**Done when:** You can explain why eigenvalues of P control mixing time

---

## Session 2: Quantum Setup
**Goal:** Have the mathematical tools for CTQW

**Tasks:**
1. Read Kempe Section 3 (QM Primer)
2. Master the formula: `e^{-iHt} = Σⱼ e^{-iλⱼt} |vⱼ⟩⟨vⱼ|`
3. Verify: if H is Hermitian, then e^{-iHt} is unitary

**Done when:** You can compute matrix exponential via spectral decomposition

---

## Session 3: CTQW Core (Critical)
**Goal:** Understand CTQW and PST definition

**Tasks:**
1. Read Kempe Section 5 (CTQW)
2. Write down: transition amplitude formula
3. Write down: PST spectral condition

**Done when:** You can state what eigenvalue conditions are needed for PST

---

## Session 4: Hands-On Verification
**Goal:** Prove PST exists on P_2

**Tasks:**
1. Compute eigenvalues/eigenvectors of A for P_2
2. Compute U(t) = e^{-iAt}
3. Find time τ where |⟨1|U(τ)|0⟩| = 1
4. (Stretch) Check C_4 for PST

**Done when:** You have a worked example proving PST on paper

---

## Session 5: Synthesis
**Goal:** Consolidate understanding

**Tasks:**
1. Write half-page: What spectral properties enable PST?
2. Skim Coutinho-Guo open problems list
3. Note which problems look interesting

**Done when:** You can articulate one potential dissertation direction

---

## End-of-Day Checklist
- [ ] Can compute CTQW evolution U(t) for small graphs
- [ ] Can state PST condition in terms of eigenvalues
- [ ] Have worked P_2 example on paper
- [ ] Have initial sense of open problems

---

# PROGRESS LOG (as of last session)

## What Has Been Completed
- ✅ Kempe Section 1 (Introduction)
- ✅ Kempe Sections on DTQW and CTQW (understood both)
- ✅ Beginner introduction to quantum computing
- ✅ Understanding of connection between CTQW and complex weights

## Key Concepts Mastered

### Core CTQW Formulas
```
CTQW Evolution:       U(t) = e^{-iHt} = Σⱼ e^{-iλⱼt} |vⱼ⟩⟨vⱼ|
Transition Amplitude: ⟨v|U(t)|u⟩ = Σⱼ e^{-iλⱼt} ⟨v|vⱼ⟩⟨vⱼ|u⟩
PST Condition:        |⟨v|U(τ)|u⟩| = 1 for some τ > 0
```

### Classical Random Walk (5 Key Points)
1. **State = probability distribution**: p(t) = Pᵗ p(0)
2. **Transition matrix P**: P_ij = 1/deg(i) if edge exists
3. **Stationary distribution π**: Fixed point where Pπ = π
4. **Mixing time**: How long to reach equilibrium
5. **Spectral gap controls everything**: λ₁ - λ₂ determines convergence rate

### Worked Example: Transition Matrix for P₄
Path graph P₄: `0 — 1 — 2 — 3`

Adjacency matrix A:
```
    0  1  2  3
  ┌            ┐
0 │ 0  1  0  0 │
1 │ 1  0  1  0 │
2 │ 0  1  0  1 │
3 │ 0  0  1  0 │
  └            ┘
```

Transition matrix P = D⁻¹A:
```
    0    1    2    3
  ┌                  ┐
0 │ 0    1    0    0 │
1 │ 1/2  0   1/2   0 │
2 │ 0   1/2   0   1/2│
3 │ 0    0    1    0 │
  └                  ┘
```

Stationary distribution: π = (1/6, 1/3, 1/3, 1/6)

---

# CONNECTION: CTQW ↔ COMPLEX WEIGHTS

## The Core Link
CTQW requires H to be **Hermitian** (H = H†) for unitary evolution.
- Real symmetric matrices (undirected graphs) satisfy this automatically
- Directed graphs have asymmetric A, so A ≠ A† — can't use directly as Hamiltonian

## The Solution: Conjugate-Symmetric Complex Weights
For a directed edge i → j, set:
```
A_ij = e^{iθ}
A_ji = e^{-iθ}
```
Now A = A† (Hermitian) despite encoding direction. Phase θ carries directional information.

## Three Frameworks Using This

| Framework | Weight Structure | Used In |
|-----------|-----------------|---------|
| **Hermitian adjacency** (Guo-Mohar) | A_ij = e^{iθ}, A_ji = e^{-iθ} | Digraph spectral theory |
| **Magnetic Laplacian** | L = D - e^{iΘ} ⊙ A | GNNs (MagNet), physics |
| **T-gain graphs** | Edges weighted by unit circle T | Algebraic graph theory |

## Why This Matters for Dissertation
Standard CTQW/PST theory assumes real symmetric A. The moment you want quantum dynamics on directed graphs or graphs with phase information, you're forced into complex weights.

**Option C** (CTQW on directed graphs via Hermitian adjacency) is exactly this: extend PST theory to the complex-weighted setting where almost nothing is known.

**Mathematical payoff:** Same spectral machinery (eigenvalues, eigenvectors, matrix exponential) but richer structure. Phases can enable or destroy PST in ways that don't exist for real weights.

---

# CTQW APPLICATIONS & MOTIVATION

## What CTQWs Are Used For

1. **Quantum search algorithms** — Grover-like speedups on graphs. Finding marked vertices in O(√n) instead of O(n).

2. **Quantum transport** — Modeling excitation transfer in physical systems. Photosynthesis uses quantum coherent transport (experimentally verified).

3. **Perfect State Transfer** — Quantum communication. Move a qubit's state from node A to node B with 100% fidelity, no measurement needed. Critical for quantum networks.

4. **Graph discrimination** — CTQWs can distinguish some graphs that are classically indistinguishable (same spectrum, different structure).

## Why Directed Graphs Matter

Most real networks are directed:
- Web links (A links to B ≠ B links to A)
- Neural connections (synapses are directional)
- Citation networks
- Metabolic pathways (reactions have direction)
- Social media (follows aren't symmetric)

## The Practical Gap

| Setting | Theory Status | Real-World Relevance |
|---------|---------------|---------------------|
| Undirected CTQW | Well-developed | Limited (idealized) |
| Directed CTQW | Almost nothing | High (actual networks) |

## Concrete Example
A future quantum internet will route quantum states through networks. Physical channels may have directional constraints (easier to send A→B than B→A). PST theory for directed graphs answers: *which network topologies allow perfect quantum communication despite asymmetry?*

This is an unsolved problem with real engineering implications—and requires complex weights to even formulate mathematically.

---

# NEXT READING (Immediate Priority)

## 1. Godsil's PST Notes
**Paper:** "When can perfect state transfer occur?" [arXiv:1011.0231](https://arxiv.org/abs/1011.0231)

**Focus:** Theorems 1-3 (skip detailed proofs on first pass)

**Goal:** Learn the spectral conditions for PST:
- Eigenvalue support conditions
- Ratio conditions
- Strong cospectrality

## 2. Coutinho-Guo 2024 Open Problems
**Paper:** "Selected Open Problems in CTQW" [arXiv:2404.02236](https://arxiv.org/abs/2404.02236)

**Focus:** Problems rated "tractable" or "medium"

**Goal:** Find your dissertation menu — know exactly what's unsolved and how hard each problem is

## 3. Based on Direction Choice

| If you choose... | Read next |
|------------------|-----------|
| Option A (Cayley PST) | Papers on PST for abelian Cayley graphs, then extraspecial groups |
| Option C (Directed QW) | Guo-Mohar 2017 on Hermitian adjacency matrices |

## Deliverable After This Reading
- List of 2-3 specific open problems you could attack
- For each: what's known, what's unknown, what graph families might be tractable

---

# ALTERNATIVE DIRECTIONS (If Quantum Walks Don't Work Out)

The other fields remain viable. All use complex weights but with different focus:

| Field | Best Entry Problem | Key Paper |
|-------|-------------------|-----------|
| **Spectral** | T-gain circulant eigenvalues | Chung Ch 1-2 |
| **GNNs** | Magnetic Laplacian ↔ CTQW bridge | MagNet (Zhang 2021) |
| **Dynamics** | Complexified Kuramoto for N=3,4 | Porter et al. 2024 |

The spectral direction (Field 1) is the safest backup — uses familiar linear algebra and has clear tractable problems.

---

# KEY REFERENCES (Complete List)

## Essential (Read First)
1. Kempe 2003 - "Quantum random walks: An introductory overview" [arXiv:quant-ph/0303081]
2. Godsil 2012 - "When can perfect state transfer occur?" [arXiv:1011.0231]
3. Coutinho & Guo 2024 - "Selected Open Problems in CTQW" [arXiv:2404.02236]

## For Specific Directions
- **Option A (Cayley PST):** Cayley graph PST papers, representation theory
- **Option C (Directed QW):** Guo-Mohar 2017 Hermitian adjacency
- **General:** Godsil & Zhan textbook (2023)

## Background/Alternative Fields
- Chung "Spectral Graph Theory" (1997) - [Free online](https://mathweb.ucsd.edu/~fan/research/revised.html)
- Porter et al. 2024 - "Complex Networks with Complex Weights"
- Zhang et al. 2021 - "MagNet" [arXiv:2102.11391]
- Bronstein et al. 2021 - "Geometric Deep Learning" [arXiv:2104.13478]

---

# DISSERTATION REQUIREMENTS REMINDER

- **7,500 words** (25-30 pages)
- **50% Mathematical Content**: Difficulty + Correctness + Comprehensiveness
- **25% Content**: Coherence + Individuality (novelty)
- **25% Presentation**: Narrative + Clarity
- **For 80+ marks**: Original material — new propositions, examples, or calculations

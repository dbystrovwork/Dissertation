# Today's Tasks: Directed PST Research

## Morning: Proof Verification (2-3 hours)

### 1. Verify C_3 PST Calculation [CRITICAL]
**Location:** Plan file, Theorem 1, Step 5

**Check these:**
- [ ] Eigenvalues λ_k = -2sin(2πk/3) → {0, -√3, √3}
- [ ] Transition amplitude formula: ⟨1|U(t)|0⟩ = (1/3)[1 + e^{i(√3t + 2π/3)} + e^{i(-√3t + 4π/3)}]
- [ ] At t = 4π/(3√3): verify √3t = 4π/3
- [ ] Confirm all phases align to 0 (mod 2π)

**Hand calculation:** Write out exp(-iHt) explicitly for 3×3 H.

### 2. Verify Undirected C_3 Has NO PST [CRITICAL]
**Location:** Plan file, Theorem 1, Step 6

**Check:**
- [ ] Undirected eigenvalues: 2cos(2πk/3) = {2, -1, -1}
- [ ] Show |e^{-2it} - e^{it}| ≤ 2 < 3
- [ ] This proves max amplitude = 2/3 < 1

**This is THE key novelty:** Direction enables PST that doesn't exist undirected.

### 3. Verify Bipartite Equivalence H² = A²
**Location:** Plan file, Theorem 2

**Check:**
- [ ] Block matrix multiplication is correct
- [ ] H² = diag(BB^T, B^TB) = A²
- [ ] Conclude: same eigenvalue magnitudes

**Test on K_{2,2}:** Compute H, A, H², A² explicitly.

---

## Midday: Learn Key Concepts (1-2 hours)

### 4. Strong Cospectrality
**What it is:** Vertices u,v are strongly cospectral if for all eigenspaces, u and v have equal projection norms.

**Why it matters:** Necessary condition for PST.

**Resource:** Godsil notes Section 3 - https://www.math.uwaterloo.ca/~cgodsil/

**Quick check:** Verify C_3 vertices are strongly cospectral (they are, by symmetry).

### 5. Eigenvalue Ratio Condition
**What it is:** For PST, all eigenvalue differences must have rational ratios.

**Why C_5 fails:** λ_1/λ_2 = golden ratio φ ∉ ℚ

**Compute:** Write out eigenvalues of C_5, verify the ratio is φ.

### 6. Godsil's "Periodic" vs "PST"
**Periodic:** |⟨u|U(τ)|u⟩| = 1 (returns to itself)
**PST:** |⟨v|U(τ)|u⟩| = 1 for u ≠ v (transfers to another)

**Relationship:** PST implies both u and v are periodic with same period.

---

## Afternoon: Implementation/Computation (2 hours)

### 7. Compute K_4 Eigenvalues
**Setup:** K_4 = complete graph on 4 vertices, contains 4 triangles.

**Task:**
- [ ] Choose consistent orientation (e.g., tournament)
- [ ] Write out 4×4 Hermitian adjacency H
- [ ] Compute eigenvalues
- [ ] Check which pairs have PST

**Expected:** More complex than C_3, but should have some PST pairs.

### 8. Numerical Verification Script (Optional)
```python
import numpy as np
from scipy.linalg import expm

def directed_cycle_H(n):
    """Hermitian adjacency for directed n-cycle."""
    H = np.zeros((n, n), dtype=complex)
    for j in range(n):
        H[j, (j+1) % n] = 1j
        H[(j+1) % n, j] = -1j
    return H

def check_pst(H, u, v, t_max=10, steps=1000):
    """Check for PST between vertices u and v."""
    for t in np.linspace(0.01, t_max, steps):
        U = expm(-1j * H * t)
        amp = abs(U[v, u])
        if amp > 0.999:
            return t, amp
    return None, 0

# Test C_3
H3 = directed_cycle_H(3)
t, amp = check_pst(H3, 0, 1)
print(f"C_3 PST: t = {t:.4f}, amplitude = {amp:.4f}")
print(f"Theory: t = {4*np.pi/(3*np.sqrt(3)):.4f}")
```

---

## Evening: Reading & Planning (1 hour)

### 9. Priority Reading
1. **Coutinho "Quantum State Transfer in Graphs"** - PhD thesis, Chapters 2-3
   - Focus on: spectral characterization of PST

2. **Godsil "State Transfer on Graphs"** - survey paper
   - Focus on: open problems section

### 10. Outline Chapter 3
Draft bullet points for "PST on Directed Cycles" chapter:
- [ ] Introduction: why cycles matter
- [ ] Setup: Hermitian adjacency definition
- [ ] Main theorem statement
- [ ] Proof for n = 3
- [ ] Proof for n = 4
- [ ] Non-existence for n ≥ 5
- [ ] Comparison with undirected

---

## Quick Reference

**Key formulas:**
```
Directed C_n eigenvalues: λ_k = -2sin(2πk/n)
C_3 PST time: τ = 4π/(3√3) ≈ 2.418
Phase condition for C_3: θ ≡ π/6 (mod π/3)
```

**The main result:**
> The triangle (C_3) is the unique minimal structure where directing edges enables PST that doesn't exist in the undirected case.

---

## Checklist Summary

### Must Do
- [ ] Verify C_3 PST by hand
- [ ] Verify undirected C_3 has NO PST
- [ ] Understand strong cospectrality
- [ ] Run numerical verification

### Should Do
- [ ] Verify bipartite equivalence
- [ ] Compute K_4 case
- [ ] Read Godsil survey

### Nice to Have
- [ ] Outline Chapter 3
- [ ] Look at Coutinho thesis

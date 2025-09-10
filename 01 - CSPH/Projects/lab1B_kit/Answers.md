
# Problem 1: Absolute value of an array

## 1.1 Mapping Scalar → Vector Code

| Scalar C (per element) | Fake-intrinsics SIMD (per block of `VECTOR_LENGTH`) | What it does                         |
| ---------------------- | --------------------------------------------------- | ------------------------------------ |
| `x = in[i];`           | `__vfloat x = _vload(&in[i]);`                      | Load a block of inputs               |
| `if (x < 0)`           | `__vbool m = _vlt(x, _vbcast(0.f));`                | Build a lane mask where x < 0        |
| `y = -x;`              | `__vfloat y = _vsub(_vbcast(0.f), x, m);`           | On masked lanes, compute `y = -x`    |
| `else y = x;`          | `y = _vcopy(y, x, _vnot(m));`                       | On non-negative lanes, copy `x` to y |
| `out[i] = y;`          | `_vstore(&out[i], y);`                              | Store the block of results           |

> [!note]  One scalar `if/else` maps to **two vector operations**: a masked `SUB` + a masked `COPY`.

---

## 1.2 Vector Code with Comments

```cpp
for (int i = 0; i < N; i += VECTOR_LENGTH) {
    __vfloat x     = _vload(&in[i]);            // load VECTOR_LENGTH inputs
    __vfloat zeros = _vbcast(0.f);              // broadcast 0.0 to all lanes
    __vbool  mneg  = _vlt(x, zeros);            // mask: lanes where x < 0
    __vfloat y     = _vsub(zeros, x, mneg);     // masked: y = -x on negative lanes
    y              = _vcopy(y, x, _vnot(mneg)); // masked: y = x on non-negative lanes
    _vstore(&out[i], y);                        // store VECTOR_LENGTH outputs
}
```

---

## 1.3 Results for Different

| VECTOR_LENGTH | Total Vector Instructions | Cycles | Max Active Lanes | Vector Utilization |
|---------------|---------------------------|--------|------------------|--------------------|
| **8**         | ~70                       | ~80    | 560              | ~0.857 (≈ 6/7)     |
| **4**         | ~140                      | ~160   | 560              | ~0.857             |
| **2**         | ~280                      | ~320   | 560              | ~0.857             |

**Explanation**

- Per block, the kernel issues 7 vector instructions: 5 are unmasked (all lanes active); 2 are masked (SUB and COPY), but together they cover all lanes.
- This means per block we have ~6×VL active lanes out of 7×VL possible → **utilization = 6/7**, independent of VL. Larger VL values reduce the number of loop iterations and vector instructions, so **execution time improves almost linearly** with vector width.

# Problem 2:  Vectorizing Code using our “fake” SIMD Intrinsics

## 2.1 - Vectorize the function

```cpp
for (int i = 0; i < N; i += VECTOR_LENGTH){

// Load one SIMD block
__vfloat vals = _vload(&values[i]); // bases (float)
__vint exps = _vload(&exponents[i]); // exponents (int)

// Constants
const __vfloat one_f = _vbcast(1.0f);
const __vfloat clamp_f = _vbcast(9.999999f);
const __vint zero_i = _vbcast(0);
__vint one_i = _vbcast(1);

// Start: result = 1.0 for all lanes (covers exp == 0 case)
__vfloat result = one_f;

// For exp > 0, set result = x and count = exp - 1
__vbool mNZ = _vgt(exps, zero_i); // lanes with exponent > 0
result = _vcopy(result, vals, mNZ); // result = x where exp>0
exps = _vsub(exps, one_i, mNZ); // count = exp - 1 (note: vector - vector)

// Loop while any lane still needs multiplies
__vbool mActive = _vgt(exps, zero_i);

while (_vpopcnt(mActive) > 0){

result = _vmul(result, vals, mActive); // result *= x on active lanes
exps = _vsub(exps, one_i, mActive); // exps-- on active lanes
mActive = _vgt(exps, zero_i); // recompute activity

}

// Clamp to 9.999999
__vbool mClamp = _vgt(result, clamp_f);
result = _vcopy(result, clamp_f, mClamp);

// Store
_vstore(&out[i], result);

}
```

With VECTOR_LENGTH = 8, we implemented clampedExpVector using masked SIMD and `_vpopcnt` for loop termination, plus a final clamp at 9.999999. On the provided 16-element input, the simulator reports **Total Vector Instructions = 96** and **Vector Utilization ≈ 0.717**. Utilization is lower than in Problem 1 due to intra-block divergence (lanes finish at different times), while execution time benefits from processing 8 elements per iteration.

---

## 2.2 Effect of Changing Vector Width

| VECTOR_LENGTH | Total Vector Instructions | Cycles | Max Active Lanes | Total Active Lanes | Vector Utilization |
|---------------|---------------------------|--------|------------------|--------------------|--------------------|
| **4**         | 152                       | 172    | 608              | 471                | ~0.775             |
| **8**         | 96                        | 106    | 768              | 551                | ~0.717             |
| **16**        | 50                        | 55     | 800              | 567                | ~0.709             |

**Observations**

- **Vector Utilization:** stays roughly constant (≈ 0.7–0.8). This happens because divergence (different exponents needing different numbers of multiplications) causes some SIMD lanes to go idle, and that fraction doesn’t improve with a wider vector.  
- **Speedup (execution time):** increases with wider vectors. Larger VL reduces the number of loop iterations and vector instructions, so the total cycles drop significantly (e.g., 172 → 55).  
- **Takeaway:** SIMD utilization is limited by divergence, but throughput (speed) still scales with vector width since more elements are processed in parallel per instruction.

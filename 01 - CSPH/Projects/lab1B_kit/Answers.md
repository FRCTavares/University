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

**Why do the max active lanes remain the same?**

> The max active lanes remain the same because it is calculated as (vector width × number of vector instructions). As vector width increases, the number of vector instructions decreases proportionally, so the product stays constant.

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
 __vbool mNZ = _veq(exps, zero_i); // mask for exps != 0
 mNZ = _vnot(mNZ);
 
 result = _vcopy(result, vals, mNZ); // result = vals if exp > 0
 exps = _vsub(exps, one_i, mNZ); // count = exp - 1 (note: vector - vector)
 
 // Loop while any lane still needs multiplies
 __vbool mActive = _veq(exps, zero_i);
 mActive = _vnot(mActive); // mask for exps > 0
   
 // What does popcount do?
 // It counts the number of active lanes (lanes where the exponent > 0)
 while (_vpopcnt(mActive) > 0){
  result = _vmul(result, vals, mActive); // multiply on active lanes
  exps = _vsub(exps, one_i, mActive); // decrement exps where active
  
  mActive = _veq(exps, zero_i); // recompute activity
  mActive = _vnot(mActive);
 }
 
 // Clamp to 9.999999
 __vbool mClamp = _vlt(clamp_f, result);
 result = _vcopy(result, clamp_f, mClamp);
 
 // Store
 _vstore(&out[i], result);
}
```

With `VECTOR_LENGTH = 8`, we implemented `clampedExpVector` using masked SIMD and `_vpopcnt` for loop termination, plus a final clamp at `9.999999`. On the provided 16-element input, the simulator reports **Total Vector Instructions = 117**, **Cycles = 127**, and **Vector Utilization ≈ 0.768**.  

---

## 2.2 Effect of Changing Vector Width

| VECTOR_LENGTH | Total Vector Instructions | Cycles | Max Active Lanes | Total Active Lanes | Vector Utilization |
|---------------|---------------------------|--------|------------------|--------------------|--------------------|
| **4**         | 184                       | 204    | 736              | 599                | ~0.814             |
| **8**         | 117                       | 127    | 936              | 719                | ~0.768             |
| **16**        | 61                        | 66     | 976              | 743                | ~0.761             |

**Observations**

- **Vector Utilization:** decreases slightly as VL increases (≈0.81 → 0.76) due to divergence—wider vectors pack more heterogeneous exponents, so more lanes idle while the slowest lane finishes.  
- **Speedup (execution time):** improves with larger VL (cycles drop 204 → 127 → 66) because more elements are processed per instruction and setup costs are paid fewer times.  
- **Takeaway:** Utilization is bounded by divergence, but throughput still scales with vector width.

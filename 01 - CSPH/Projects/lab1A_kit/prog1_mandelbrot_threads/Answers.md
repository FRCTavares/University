
## Task 1 — 2-thread Spatial Decomposition

| Config | Time (ms) | Speedup | Efficiency |
| ------ | --------: | ------: | ---------: |
| Serial |   232.113 |   1.00× |       1.00 |
| 2T     |   118.856 |   1.95× |      0.976 |

> [!note] Implementation  
> Rows split into two contiguous halves:  

> - Thread 0 → `[0 … H/2)`  
> - Thread 1 → `[H/2 … H)`  

**Discussion:** Almost ideal 2× speedup. Small loss from thread creation overhead, caches, and minor imbalance.

---

## Task 2 — Extend to N Threads (contiguous blocks)

```cpp
int rowsPerThread = args->height / args->numThreads;
int startRow = args->threadId * rowsPerThread;

// Treat the last thread specially to cover all rows
int numRows = rowsPerThread;

if (args->threadId == args->numThreads - 1){
 numRows = args->height - startRow;
}

// Call the serial function to compute this thread's portion
mandelbrotSerial(args->x0, args->y0, args->x1, args->y1,
args->width, args->height,
startRow, numRows,
args->maxIterations, args->output);
```

### Results — VIEW 1

| Threads | Serial (ms) | Parallel (ms) |   Speedup | Eff. |
| ------: | ----------: | ------------: | --------: | ---: |
|       2 |     232.069 |       118.579 | **1.96×** | 0.98 |
|       3 |     231.984 |       147.967 | **1.57×** | 0.52 |
|       4 |     231.884 |        98.630 | **2.35×** | 0.59 |
|       5 |     232.206 |       100.584 | **2.31×** | 0.46 |
|       6 |     231.859 |        74.030 | **3.13×** | 0.52 |
|       7 |     232.058 |        72.318 | **3.21×** | 0.46 |
|       8 |     232.054 |        61.563 | **3.77×** | 0.47 |
|      16 |     231.953 |        36.758 | **6.31×** | 0.39 |

### Results — VIEW 2

| Threads | Serial (ms) | Parallel (ms) |   Speedup | Eff. |
| ------: | ----------: | ------------: | --------: | ---: |
|       2 |     123.985 |        75.858 | **1.63×** | 0.81 |
|       3 |     123.926 |        63.415 | **1.95×** | 0.65 |
|       4 |     123.629 |        52.083 | **2.37×** | 0.59 |
|       5 |     123.756 |        46.663 | **2.65×** | 0.53 |
|       6 |     123.943 |        41.125 | **3.01×** | 0.50 |
|       7 |     123.645 |        36.833 | **3.36×** | 0.48 |
|       8 |     124.021 |        32.949 | **3.76×** | 0.47 |
|      16 |     124.080 |        19.523 | **6.36×** | 0.40 |

> [!warning] Odd-thread anomaly  
> In **VIEW 1**, 3 threads perform badly (1.57×) because the central rows are costlier. Horizontal symmetry means middle blocks are heavier.  
> In **VIEW 2**, rows are more uniform, so scaling is smoother.

---

## Task 3 — Per-Thread Timing

```cpp
// Record start time for this thread
double startTime = CycleTimer::currentSeconds();

.
. // The thread code goes here
.
.
  
// Record end time and calculate elapsed time for this thread
double endTime = CycleTimer::currentSeconds();
double elapsedTime = (endTime - startTime) * 1000; // Convert to milliseconds

```

**3T (VIEW 1):**

- T0: [0–399] → 48.6 ms  
- T1: [400–799] → 147.2 ms ← bottleneck  
- T2: [800–1199] → 49.2 ms  

**4T (VIEW 1):**

- T0: [0–299] → 24.2 ms  
- T1: [300–599] → 99.6 ms  
- T2: [600–899] → 100.6 ms  
- T3: [900–1199] → 24.5 ms  

**3T (VIEW 2):**

- T0: [0–399] → 64.2 ms  
- T1: [400–799] → 38.4 ms  
- T2: [800–1199] → 36.1 ms  

> [!important] **Conclusion**  
> Load imbalance confirmed: in VIEW 1, middle rows are ~3× costlier → bottleneck effect.  
> VIEW 2 distributes cost more evenly → better efficiency.

---

## Task 4 — Block-Cyclic (interleaved)

```cpp
int rowsProcessed = 0;

for (int row = args->threadId; row < args->height; row += args->numThreads){
 // Call the serial function to compute one row at a time
 mandelbrotSerial(args->x0, args->y0, args->x1, args->y1,
 args->width, args->height,
 row, 1, // startRow = row, numRows = 1
 args->maxIterations, args->output);
 rowsProcessed++;
}
```

| Threads | Parallel (ms) | Speedup | Eff. |
| ------: | -------------: | ------: | ---: |
| 1  | 354.407 | 1.00×  | 1.00 |
| 2  | 177.512 | 2.00×  | 1.00 |
| 3  | 123.371 | 2.87×  | 0.96 |
| 4  | 92.547  | 3.83×  | 0.96 |
| 5  | 75.746  | 4.68×  | 0.94 |
| 6  | 63.210  | 5.60×  | 0.93 |
| 7  | 55.535  | 6.38×  | 0.91 |
| 8  | 49.562  | 7.15×  | 0.89 |
| 16 | 26.919  | 13.16× | 0.82 |

> [!note] Fix achieved  
> Interleaving rows balances heavy/light regions. The 3-thread anomaly vanishes (2.87×). Targets: **7.15× @8T** and **13.2× @16T**.

---

## Task 5 — 32 Threads (Oversubscription)

| Config | Time (ms) | Speedup | Eff. |
| ------ | --------: | -------: | ---: |
| Serial | 354.277   | 1.00×    | 1.00 |
| 16T    | 26.919    | 13.16×   | 0.82 |
| 32T    | 29.445    | 12.03×   | 0.38 |

> [!danger] Oversubscription  
> With 32 software threads on an 8C/16T CPU, performance drops:

> - Scheduler time-slicing → context-switch overhead  
> - SMT pairs share ports → contention  
> - Cache/memory bandwidth saturated  
> - Some threads spike to ~18–22 ms  

**Best point:** 16 threads (≈26.9 ms, 13.16×, 82% eff.).  
**32 threads:** slower (29.4 ms, 12.03×, 38% eff.).

---

# Takeaways

- **Decomposition matters:** Contiguous blocks cause imbalance (esp. VIEW 1).  
- **Block-cyclic interleaving fixes it:** smooth scaling up to hardware limit.  
- **Scaling limit:** Beyond 16 threads, oversubscription and memory contention dominate.  
- **Methodology:** Per-thread timing + scaling plots = clear diagnosis.

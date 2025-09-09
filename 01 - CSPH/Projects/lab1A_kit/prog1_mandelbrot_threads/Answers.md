# Mandelbrot Parallelization Lab

## Task 1 — 2-thread Spatial Decomposition

### Implementation

We split the image rows into two halves:

- Thread 0 → rows [0 … H/2)
- Thread 1 → rows [H/2 … H)

Each thread writes to disjoint rows, so no synchronization is required beyond `join()`.

### Results (MacBook M4 Pro, VIEW 1)

- **Serial**: 232.113 ms
- **2 threads**: 118.856 ms
- **Speedup**: 1.95×
- **Efficiency**: 97.6%

### Discussion

The 1.95× speedup is nearly ideal for 2 threads. The small deviation from perfect 2× scaling is due to thread overhead and minor load imbalance.

---

## Task 2 — Extend Mandelbrot Parallelization to Multiple Threads

### Setup

- Static row-block decomposition into **N** contiguous blocks.
- Runs for **t ∈ {2,3,4,5,6,7,8,16}**.
- Speedup = T_serial / T_parallel.

### Results

#### VIEW 1

| Threads | Serial (ms) | Parallel (ms) | Speedup | Efficiency |
|-------:|------------:|--------------:|--------:|-----------:|
| 2 | 232.069 | 118.579 | **1.96×** | 0.98 |
| 3 | 231.984 | 147.967 | **1.57×** | 0.52 |
| 4 | 231.884 | 98.630 | **2.35×** | 0.59 |
| 5 | 232.206 | 100.584 | **2.31×** | 0.46 |
| 6 | 231.859 | 74.030 | **3.13×** | 0.52 |
| 7 | 232.058 | 72.318 | **3.21×** | 0.46 |
| 8 | 232.054 | 61.563 | **3.77×** | 0.47 |
| 16 | 231.953 | 36.758 | **6.31×** | 0.39 |

#### VIEW 2

| Threads | Serial (ms) | Parallel (ms) | Speedup | Efficiency |
|-------:|------------:|--------------:|--------:|-----------:|
| 2 | 123.985 | 75.858 | **1.63×** | 0.81 |
| 3 | 123.926 | 63.415 | **1.95×** | 0.65 |
| 4 | 123.629 | 52.083 | **2.37×** | 0.59 |
| 5 | 123.756 | 46.663 | **2.65×** | 0.53 |
| 6 | 123.943 | 41.125 | **3.01×** | 0.50 |
| 7 | 123.645 | 36.833 | **3.36×** | 0.48 |
| 8 | 124.021 | 32.949 | **3.76×** | 0.47 |
| 16 | 124.080 | 19.523 | **6.36×** | 0.40 |

### Analysis and Discussion

- **Scaling limitations**: Speedup increases with threads but efficiency decreases due to overheads and hardware limits.

- **Load imbalance in VIEW 1**: The 3-thread case shows poor performance (1.57× speedup) because static row partitioning creates uneven workloads. The Mandelbrot's computational complexity varies by region.

- **VIEW 2 performs better**: More uniform workload distribution leads to smoother scaling (3-thread case achieves 1.95× speedup).

- **Beyond 8 threads**: Continued improvement up to 16 threads, but with diminishing returns due to thread overhead and memory bandwidth limits.

---

## Task 3 — Measure Per-Thread Execution Time

### Code Implementation

Added timing instrumentation to `workerThreadStart()`:

- Record start time using `CycleTimer::currentSeconds()`
- Record end time after computation completes
- Print per-thread timing with row range assignment

### Timing Results

#### 3-Thread Load Imbalance (VIEW 1)

```text
Thread 0: computed rows [0-399] in 48.6 ms
Thread 1: computed rows [400-799] in 147.2 ms    ← Slowest thread
Thread 2: computed rows [800-1199] in 49.2 ms
Speedup: 1.60×
```

#### 4-Thread Load Distribution (VIEW 1)

```text
Thread 0: computed rows [0-299] in 24.2 ms
Thread 1: computed rows [300-599] in 99.6 ms    ← Heavy workload
Thread 2: computed rows [600-899] in 100.6 ms   ← Heavy workload  
Thread 3: computed rows [900-1199] in 24.5 ms
Speedup: 2.39×
```

#### 3-Thread Better Balance (VIEW 2)

```text
Thread 0: computed rows [0-399] in 64.2 ms
Thread 1: computed rows [400-799] in 38.4 ms
Thread 2: computed rows [800-1199] in 36.1 ms
Speedup: 2.03×
```

### Analysis

- **VIEW 1 load imbalance confirmed**: Middle rows (400-799) take ~3× longer than top/bottom rows due to higher computational complexity in the Mandelbrot's central region.

- **Bottleneck effect**: Overall execution time is limited by the slowest thread. In the 3-thread case, Thread 1 becomes the bottleneck.

- **VIEW 2 more balanced**: Workload is more evenly distributed, leading to better parallel efficiency.

## 🎯 Project Objective

The goal of this project is to investigate whether **simplified, lower-dimensional regression models** can accurately predict **power losses in a power grid**, using only partial information (i.e., reduced versions of the input matrix `X`).

---

## 💡 Motivation

Full-feature models (such as those including all quadratic and cross terms) are **accurate but complex** and **not scalable** for larger networks.

To reduce computational cost and improve scalability, simpler models are often considered, such as:

- **Squares-only models**: only terms like `P_i^2`
- **Grouped-injections models**: terms like `(P_1 + P_2 + P_3)^2`
- **Edge-reduced models**: only include `2*P_i*P_j` where lines exist

These models are **lower-dimensional** but typically suffer from **reduced accuracy**.

---

## 🔍 Approach

To improve the performance of these reduced models without increasing their complexity, we hypothesize that:

> If power losses evolve **smoothly and predictably over time** (i.e., are **temporally correlated**), then even simple models can achieve high prediction accuracy.

To test this, we:
- Simulate datasets where power injections vary **smoothly over time**
- This induces **correlation in the resulting losses**
- We then **train reduced models** on this structured data and evaluate their performance

---

## ✅ What We Aim to Demonstrate

This project aims to demonstrate that:
- **Temporal structure in the data** can compensate for the lack of model complexity
- Under stable or predictable conditions, **low-dimensional models may still yield accurate loss predictions**
- This can make reduced models **more practical for real-world deployment**, especially in large or distributed grid environments

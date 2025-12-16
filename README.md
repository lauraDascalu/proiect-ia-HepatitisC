# Evolutionary SVM Optimization: Hepatitis C Case Study

### Project Overview

This project implements a hybrid methodology for binary classification using **Support Vector Machines (SVM)**, where the standard quadratic optimization problem is solved using a **Real-Coded Evolutionary Algorithm (EA)**. The objective is to enhance the robustness and accuracy of the SVM model by directly optimizing the Lagrange multipliers.

The core problem addressed is the binary classification of patient samples into two critical categories: **Non-Infected** (Blood Donors) and **Infected** (Hepatitis, Fibrosis, Cirrhosis).

### Dataset: [Hepatisis C](https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset)

#### 1. Evolutionary Optimization of SVM Dual Problem (GATE Algorithm)
* **Chromosome Representation:** Each chromosome is a vector of real-valued Lagrange multipliers, $\boldsymbol{\alpha} = (\alpha_1, \ldots, \alpha_L)$, constrained to the interval $[0, C]$.
* **Constraint Handling:** An adjustment algorithm is implemented to iteratively ensure compliance with the critical equality constraint: $\sum_{i=1}^{L} \alpha_i y_i = 0$. 
* **EA Operators:** The algorithm uses Elitism for selection, **Arithmetic Crossover** for generating offspring, and **Swap Mutation** (adapted for real-coded genes) for exploration.

#### 2. Data Preprocessing
* **Target Encoding:** The multi-class target variable was encoded into a binary format: Non-Infected (-1) and Infected (1).
* **Missing Values:** Handled via median imputation.
* **Scaling:** All features were scaled using StandardScaler.
* **Splitting:** Data was split into 80% Training and 20% Testing sets.

#### 3.Kernel Evaluation
The system was evaluated using two primary kernel functions: **Linear Kernel** and **Gaussian Kernel** .

### Conclusion
The results confirm the efficacy of the non-linear approach for this dataset. The superior performance of the **Gaussian Kernel** suggests that the Hepatitis C data is **non-linearly separable** in its original feature space, highlighting the power of kernel methods combined with global optimization via Evolutionary Algorithms.

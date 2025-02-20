# Foundation Model for Phase Field Dynamics  

This repository presents a **Fourier Neural Operator (FNO)** model designed to approximate solutions of the **Allen-Cahn equation**, which describes phase separation in binary systems. The model is trained across varying **$\epsilon$ values** and initial conditions to generalize across different phase field dynamics.

## Problem Statement  
The **Allen-Cahn equation** is given by:

```math
\frac{\partial u}{\partial t} = \Delta_x u - \frac{1}{\epsilon^2}(u^3 - u),
```

where \( u(x, t) \) represents the phase variable, \( \epsilon \) controls the interface width, and \( \Delta_x \) is the Laplacian operator. The equation is solved for \( t \in [0,1] \) and \( x \in [-1,1] \) under periodic boundary conditions.

## Methodology  
### **Data Generation**
- Training data is generated using **finite-difference solvers**, covering multiple \( \epsilon \) values:  
  - **In-distribution:** \( \epsilon = \{0.1, 0.07, 0.05, 0.02\} \).  
  - **Out-of-distribution:** \( \epsilon = \{0.15, 0.03, 0.01\} \).  
- **Initial Conditions:**  
  - Fourier series, Gaussian Mixture Models (GMM), and piecewise linear functions.

### **Model Architecture**  
- **Fourier Neural Operator (FNO)** extended with **conditional batch normalization**:  
  ```math
  \text{output} = \gamma(t) \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta(t), \quad \gamma(t), \beta(t) = \text{MLP}(t)
  ```
- **All-to-All Training:** Combines all \( t \) and \( \epsilon \) values to learn a general solution.  
- **Loss Function:** Weighted sum of **MSE loss**, **periodicity constraints**, and **smoothness regularization**:

  ```math
  \mathcal{L} = \| u_{\text{pred}} - u_{\text{true}} \|_2^2 + \lambda_1 \| u(x_1, t) - u(x_M, t) \|_2^2 + \lambda_2 \| \Delta_x u_{\text{pred}} \|_2^2.
  ```
  with \( \lambda_1, \lambda_2 = 0.05 \).

## Results  
- **Generalization Performance:**  
  - In-distribution errors remain **below 5%** across different \( \epsilon \) and time steps.  
  - Out-of-distribution errors increase significantly (**10-20%**), particularly for sharper transitions (low \( \epsilon \)).  
- **Challenges:**  
  - **Extrapolation to unseen \( \epsilon \)** values leads to high-frequency artifacts.  
  - **Sharp interfaces** are difficult to predict with high accuracy.  

## Report  
For detailed methodology and results, refer to the [project report](foundational-neural-operator-report.pdf).


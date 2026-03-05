---
title: "Calibrated Physics-Informed Uncertainty Quantification for Neural-PDE surrogates"
date: 10/03/2025 #MM/DD/YYYY
format: html
bibliography: bibliography.bib
csl: diabetologia.csl

---

# Making Neural PDE Solvers Trustworthy: Physics-Informed Uncertainty Quantification

Neural networks have revolutionized how we solve partial differential equations (PDEs), offering speed-ups of up to six orders of magnitude compared to traditional numerical solvers. But there's a critical problem: these neural PDE solvers often produce confident predictions that violate fundamental physical laws. Without reliable uncertainty estimates, deploying them in critical applications like fusion reactor design or weather forecasting remains risky.

**[Placeholder: Figure showing neural PDE framework with uncertainty quantification loop]**

## The Core Innovation

Researchers have developed CP-PRE (Conformal Prediction with Physics Residual Errors), a novel framework that provides guaranteed uncertainty bounds for neural PDE solvers without requiring any labeled data. The key insight is elegant: instead of measuring how well predictions match training data, measure how well they satisfy the underlying physics.

For any PDE of the form $D(u) = 0$, where $D$ is a differential operator and $u$ is the solution, the physics residual error is simply $|D(\hat{u})|$—how much the neural network's prediction $\hat{u}$ violates the governing equation. By computing this residual across multiple predictions and applying conformal prediction theory, CP-PRE provides statistically guaranteed coverage bounds.

## How It Works

The method leverages a clever computational trick: implementing finite difference stencils as convolutional kernels. For example, the 2D Laplacian operator becomes:

$$\nabla^2 \approx \frac{1}{h^2}\begin{bmatrix}0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0\end{bmatrix}$$

This allows fast, GPU-accelerated gradient estimation—achieving 1000x speedups over traditional implementations—making the framework computationally practical.

**[Placeholder: Visualization showing marginal vs joint coverage bounds]**

The framework offers two flavors of uncertainty quantification:
- **Marginal coverage**: Cell-wise error bars identifying problematic regions within a single prediction
- **Joint coverage**: Domain-wide bounds enabling accept/reject decisions for entire predictions

## Real-World Impact

The researchers validated CP-PRE across diverse applications, from classical PDEs (wave equation, Navier-Stokes, magnetohydrodynamics) to fusion reactor applications. In plasma modeling experiments using the JOREK code, CP-PRE identified physically inconsistent predictions while maintaining guaranteed coverage—all while being 10-100x faster than data-dependent uncertainty methods.

**[Placeholder: Results showing coverage guarantees across different PDEs]**

Perhaps most compelling: the framework works with *any* neural PDE solver (FNOs, PINNs, DeepONets) without architectural modifications. It's truly post-hoc, model-agnostic uncertainty quantification.

## The Bottom Line

CP-PRE transforms neural PDE solvers from fast but unreliable tools into trustworthy surrogates. By grounding uncertainty in physical consistency rather than data fit, it provides interpretable bounds indicating when predictions violate conservation laws. This enables a practical workflow: use the fast neural solver when predictions pass coverage thresholds, fall back to expensive numerical solvers only when necessary.

The framework's data-free nature is particularly valuable in scientific domains where generating labeled simulation data is expensive or when exploring out-of-distribution scenarios. As one researcher put it: "All models are wrong, but some are useful"—CP-PRE provides a principled measure of that usefulness.
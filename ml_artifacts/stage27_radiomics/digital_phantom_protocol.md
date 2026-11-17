# Digital Phantom Protocol and Analytical Derivations

**Project**: OsteoGuard AI Multi-Class Differential Diagnosis  
**Document Status**: Synthetic Digital Phantom Protocol  
**Date**: November 16, 2026  

---

## 1. Phantom Suite Specifications

To verify radiomic calculations without accessing patient data, six deterministic synthetic phantoms were designed:

### Phantom 1: Constant Uniform Field
- **Dimension**: $64 \times 64$ pixels with 8-pixel border margin.
- **ROI**: Central $48 \times 48$ square ($2,304$ pixels) with intensity $I(x, y) = 100.0$.
- **Analytical Truth**: $\mu = 100.0$, $\sigma^2 = 0.0$, $\text{Range} = 0.0$, $\text{GLCM Contrast} = 0.0$, $\text{GLCM Homogeneity} = 1.0$.

### Phantom 2: Symmetric Checkerboard
- **Dimension**: $64 \times 64$ pixels with $8 \times 8$ tiles.
- **Intensities**: Alternating tiles of $10.0$ and $90.0$.
- **Analytical Truth**: $\mu = 50.0$, $\sigma^2 = 1600.0$, $\text{Range} = 80.0$, $\text{Median} = 50.0$.

### Phantom 3: Monotonic Linear Ramp
- **Dimension**: $64 \times 64$ pixels.
- **Intensities**: $I(x, y) = 10.0 + 40.0 \times \frac{y}{63}$.
- **Analytical Truth**: $\mu = 30.0$, $\min = 10.0$, $\max = 50.0$, $\text{Range} = 40.0$.

### Phantom 4: Discrete 4-Level Step Wedge
- **Dimension**: $64 \times 64$ pixels.
- **Intensities**: 4 equal vertical columns with levels $[10.0, 30.0, 50.0, 70.0]$.
- **Analytical Truth**: $\mu = 40.0$, $\text{Range} = 60.0$, exactly 25% area per level.

### Phantom 5: Geometric Circular Annulus
- **Dimension**: $64 \times 64$ pixels.
- **ROI**: Centered at $(32, 32)$ with $r_{\text{inner}} = 10$, $r_{\text{outer}} = 24$, intensity $75.0$.
- **Analytical Truth**: Verifies non-rectangular mask masking and curved border interpolation.

### Phantom 6: Anisotropic Voxel Spacing Phantom
- **Dimension**: $64 \times 64$ pixels with spacing $(\Delta y = 0.5\text{ mm}, \Delta x = 1.5\text{ mm})$.
- **Analytical Truth**: Pixel area $= 0.75\text{ mm}^2$. Verifies physical dimension scaling in total energy.

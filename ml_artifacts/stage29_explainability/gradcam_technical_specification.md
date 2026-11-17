# Grad-CAM Technical Specification & Implementation Protocol

**Document Version**: 1.0.0  
**Project**: OsteoGuard AI: Explainability Engine  
**Module**: `osteoguard.explainability.gradcam`  
**Date**: November 16, 2026  
**Status**: **MATHEMATICALLY VERIFIED ON SYNTHETIC FIXTURES**

---

## 1. Mathematical Formulation

Gradient-weighted Class Activation Mapping (Grad-CAM) computes visual explanations for decisions made by convolutional neural networks by using the gradients of target class logits with respect to convolutional feature activation maps.

### 1.1 Forward Feature Capture
Let $A^k \in \mathbb{R}^{H \times W}$ represent the activation map of the $k$-th channel of target convolutional layer $L$ for a given input image $I$.

### 1.2 Neuron Importance Weights ($\alpha_k^c$)
The importance weight $\alpha_k^c$ of channel $k$ for target class $c$ is computed by global average pooling of the gradients:
$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial y^c}{\partial A_{i, j}^k}$$
where $Z = H \times W$ is the spatial area of the feature map, and $y^c$ is the unnormalized logit score for class $c$.

### 1.3 Rectified Weighted Combination ($L_{\text{Grad-CAM}}^c$)
The coarse Grad-CAM localization map is computed via a linear combination of activation maps weighted by $\alpha_k^c$, followed by a Rectified Linear Unit (ReLU) operation:
$$L_{\text{Grad-CAM}}^c = \operatorname{ReLU}\left( \sum_{k=1}^C \alpha_k^c A^k \right)$$
*Purpose of ReLU*: Filters out features that contribute negatively to the class of interest, isolating positive evidence.

### 1.4 Normalization & Interpolation Protocol
1. **Zero-Gradient Handling**: If $\max(L_{\text{Grad-CAM}}^c) \le 10^{-8}$ or gradients are zero, the engine outputs an all-zero map without division-by-zero or NaN creation.
2. **Bilinear Upsampling**: The coarse map is resized to target spatial dimensions (e.g. $512 \times 512$ or input size) using bilinear interpolation with `align_corners=False`.
3. **Safe Min-Max Scaling**:
   $$\operatorname{Norm}(L) = \frac{L - \min(L)}{\max(L) - \min(L)} \quad \text{if } \max(L) > \min(L) \text{ else } 0.0$$
   The output is strictly clamped to $[0.0, 1.0]$.

---

## 2. Production Engineering Standards

### 2.1 Hook Lifecycle Management
- Hooks are registered using `register_forward_hook` and `register_full_backward_hook`.
- Handles are stored in `_fwd_handle` and `_bwd_handle`.
- Explicit `remove_hooks()` method, `__del__` destructor, and `__enter__` / `__exit__` context manager prevent persistent memory leaks and duplicate hook accumulation.

### 2.2 Model State & Weight Invariance
- Preserves model evaluation mode: stores `was_training = model.training`, sets `model.eval()`, and restores original state upon completion.
- Guarantees zero weight modifications: gradients are zeroed cleanly via `model.zero_grad(set_to_none=True)`. Model weights remain identical before and after explainability calls.

### 2.3 Feasibility Assessment for N-Dimensional (3D CT) Generalization
For future 3D volumetric CT backbones ($D \times H \times W$):
$$\alpha_k^c = \frac{1}{D \cdot H \cdot W} \sum_{z=1}^D \sum_{y=1}^H \sum_{x=1}^W \frac{\partial y^c}{\partial A_{z, y, x}^k}$$
$$L_{\text{Grad-CAM3D}}^c = \operatorname{ReLU}\left( \sum_{k=1}^C \alpha_k^c A^k_{3D} \right)$$
*Feasibility Verdict*: Structurally feasible and mathematically consistent. In this stage, OsteoGuard AI is verified on 2D architectures; 3D extensions are reserved for approved volumetric CT pipelines.

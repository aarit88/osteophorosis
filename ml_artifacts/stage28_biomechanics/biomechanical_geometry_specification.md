# Cross-Sectional Biomechanical Geometry Specification

**Document Version**: 1.0.0  
**Project**: OsteoGuard AI  
**Module**: `osteoguard.biomechanics.geometry_engine`  
**Date**: November 16, 2026  
**Status**: **MATHEMATICALLY VERIFIED**

---

## 1. Theoretical Formulations

The 2D Cross-Sectional Geometry Engine computes the geometric resistance of bone cross-sections from segmented binary cortical masks M(y, x) in {0, 1} with physical pixel spacing dy, dx in millimeters (mm).

### 1.1 Cross-Sectional Cortical Area (A)
$$A = \sum_{y=0}^{H-1} \sum_{x=0}^{W-1} M(y, x) \cdot \Delta y \cdot \Delta x \quad [\text{mm}^2]$$

### 1.2 Centroid Coordinates (x_bar, y_bar)
$$\bar{x} = \frac{\sum_{y, x} (x \cdot \Delta x) M(y, x)}{\sum_{y, x} M(y, x)} \quad [\text{mm}]$$
$$\bar{y} = \frac{\sum_{y, x} (y \cdot \Delta y) M(y, x)}{\sum_{y, x} M(y, x)} \quad [\text{mm}]$$

### 1.3 Central Second Moments of Area (I_x, I_y, I_xy)
Computed about the cross-sectional centroid:
$$I_x = \sum_{y, x} (y \cdot \Delta y - \bar{y})^2 \cdot M(y, x) \cdot \Delta y \cdot \Delta x \quad [\text{mm}^4]$$
$$I_y = \sum_{y, x} (x \cdot \Delta x - \bar{x})^2 \cdot M(y, x) \cdot \Delta y \cdot \Delta x \quad [\text{mm}^4]$$
$$I_{xy} = \sum_{y, x} (x \cdot \Delta x - \bar{x})(y \cdot \Delta y - \bar{y}) \cdot M(y, x) \cdot \Delta y \cdot \Delta x \quad [\text{mm}^4]$$

*Sign Convention*: In classical beam theory, I_x represents resistance to bending about the horizontal centroidal axis, and I_y represents resistance to bending about the vertical centroidal axis.

### 1.4 Polar Second Moment of Area (J_p)
By the perpendicular axis theorem:
$$J_p = I_x + I_y = \sum_{y, x} \left[(x \cdot \Delta x - \bar{x})^2 + (y \cdot \Delta y - \bar{y})^2\right] M(y, x) \cdot \Delta y \cdot \Delta x \quad [\text{mm}^4]$$

### 1.5 Principal Moments of Area (I_max, I_min) and Orientation (theta_p)
$$I_{\max}, I_{\min} = \frac{I_x + I_y}{2} \pm \sqrt{\left(\frac{I_x - I_y}{2}\right)^2 + I_{xy}^2} \quad [\text{mm}^4]$$
$$\theta_p = \frac{1}{2} \operatorname{atan2}(-2 I_{xy}, I_x - I_y) \quad [\text{degrees}]$$

### 1.6 Geometric Retention Ratios
When evaluated alongside an intact reference cross-section:
$$\text{Retention}_A = \frac{A_{\text{defect}}}{A_{\text{intact}}}, \quad \text{Retention}_{J_p} = \frac{J_{p, \text{defect}}}{J_{p, \text{intact}}}, \quad \text{Retention}_{I_{\min}} = \frac{I_{\min, \text{defect}}}{I_{\min, \text{intact}}}$$

---

## 2. Critical Physical and Clinical Distinctions

> [!WARNING]
> **1. Polar Second Moment of Area vs. Mass Moment of Inertia**:
> - J_p has units of mm^4. It is a purely geometric measure of cross-sectional resistance to torsional shear.
> - Mass moment of inertia has units of kg*m^2 and depends on tissue density distribution.
>
> **2. 2D Geometry vs. 3D CT Rigidity Analysis (CTRA)**:
> - CTRA integrates calibrated volumetric Hounsfield Units into tissue modulus E(HU) across multiple consecutive 3D axial CT slices to calculate axial rigidity (EA), bending rigidity (EI), and torsional rigidity (GJ).
> - A 2D geometric moment engine does NOT reproduce CTRA and does NOT account for bone mineral density attenuation or spatial trabecular architecture.
>
> **3. Geometric Retention vs. Clinical Fracture Probability**:
> - Geometric degradation (J_defect / J_intact) quantifies loss of cross-sectional shape stiffness.
> - It does NOT directly equal patient fracture risk, which depends on patient body mass index, ambulatory loading, muscular forces, and tumor biological aggressiveness.

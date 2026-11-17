# OsteoGuard AI — Medical Image Ingestion and Preprocessing Pipeline

## 1. Executive Summary & Architectural Separation

OsteoGuard AI implements an isolated, modality-aware medical image ingestion and preprocessing engine. In strict compliance with medical physics and clinical imaging guidelines:

> [!IMPORTANT]
> **Modality Strategy Rule**: Computed Tomography (CT) and Planar Radiographs (X-ray) represent fundamentally distinct physical imaging mechanisms and **MUST NEVER receive identical normalization**:
> - **CT Scans** represent calibrated physical electron density (attenuation) expressed in **Hounsfield Units (HU)**. CT preprocessing requires linear slope/intercept calibration and skeletal **Bone Windowing** ($\text{WL} = +400\text{ HU}, \text{WW} = 1500\text{ HU}$).
> - **Planar Radiographs (X-rays)** represent 2D cumulative projection densities with uncalibrated sensor exposures. X-ray preprocessing requires **percentile intensity normalization** (0.5th to 99.5th percentiles) and optional **Contrast-Limited Adaptive Histogram Equalization (CLAHE)**.

```
                               ┌────────────────────────────────────────┐
                               │       Medical Image Input File         │
                               │   (DICOM / NIfTI / PNG / JPG / TIFF)   │
                               └───────────────────┬────────────────────┘
                                                   │
                                                   ▼
                               ┌────────────────────────────────────────┐
                               │      Unified Loader & Inspector        │
                               │   - SHA-256 Checksum Calculation       │
                               │   - Format & Quality Validation        │
                               │   - Corrupt / Dead File Rejection      │
                               │   - Spatial & Orientation Metadata     │
                               │   - HIPAA PHI De-identification        │
                               └───────────┬────────────────┬───────────┘
                                           │                │
                         Modality == "ct"  │                │  Modality == "xray" / "bone_scan"
                                           ▼                ▼
                 ┌────────────────────────────────┐   ┌────────────────────────────────┐
                 │       CT Bone Pipeline         │   │       X-ray / Planar Pipeline  │
                 │ 1. HU Conversion:              │   │ 1. Percentile Normalization:   │
                 │    HU = Pixel * Slope + Intercept│ │    Clip outliers (0.5% - 99.5%)│
                 │ 2. Bone Windowing:             │   │    Min-Max scale to [0.0, 1.0] │
                 │    WL: +400 HU, WW: 1500 HU    │   │ 2. Optional CLAHE:             │
                 │    Boundaries: [-350, +1150 HU]│   │    β = 2.0, Grid: 8 x 8        │
                 │    Scale to [0.0, 1.0]         │   │    Accentuates trabeculae      │
                 └────────────────┬───────────────┘   └────────────────┬───────────────┘
                                  │                                    │
                                  └─────────────────┬──────────────────┘
                                                    │
                                                    ▼
                               ┌────────────────────────────────────────┐
                               │         Spatial Standardization        │
                               │   - Aspect-Ratio Preserving Resize     │
                               │   - Symmetric Zero Padding             │
                               │   - Target Size: 224 x 224 / 512 x 512 │
                               │   - 3-Channel ImageNet Tensor [3,H,W]  │
                               │   - Full Provenance Manifest Logged    │
                               └────────────────────────────────────────┘
```

---

## 2. Supported Medical Image Formats

| Format | File Extensions | Parsing Library | Extracted Metadata | Spatial Tracking |
| :--- | :--- | :--- | :--- | :--- |
| **DICOM** | `.dcm`, `.dicom` | `pydicom` | Full header tags, RescaleSlope/Intercept, WindowCenter/Width, KVP | Voxel spacing `(dx, dy, dz)`, slice thickness, orientation cosines |
| **NIfTI** | `.nii`, `.nii.gz` | `nibabel` | Volume affine matrix, dimensions, data type | Voxel zooms `(dx, dy, dz)`, coordinate origin |
| **PNG** | `.png` | `Pillow` / `OpenCV` | Bit depth, color mode, dimensions | Grayscale 2D array, default spacing `(1.0, 1.0, 1.0)` |
| **JPEG** | `.jpg`, `.jpeg` | `Pillow` / `OpenCV` | Dimensions, color mode | Grayscale 2D array |
| **TIFF** | `.tif`, `.tiff` | `Pillow` / `OpenCV` | Resolution DPI, photometric interpretation | High dynamic range 16-bit / 8-bit support |

---

## 3. Detailed Modality Processing Pipelines

### 3.1 Computed Tomography (CT) Pipeline

1. **Hounsfield Unit (HU) Conversion**:
   Raw stored pixel values in CT DICOM files do not directly represent tissue density. They are converted using the modality tags:
   $$\text{HU} = (\text{PixelValue} \times \text{RescaleSlope}) + \text{RescaleIntercept}$$
   - Air: $\approx -1000\text{ HU}$
   - Fat: $\approx -120 \text{ to } -90\text{ HU}$
   - Water: $0\text{ HU}$
   - Muscle: $\approx +40\text{ HU}$
   - Cancellous Bone: $\approx +300 \text{ to } +700\text{ HU}$
   - Cortical Bone: $\approx +1000 \text{ to } +2000+\text{ HU}$

2. **Skeletal Bone Windowing**:
   Skeletal pathologies (metastatic lysis/sclerosis, Pagetic trabecular coarsening) require targeted bone windowing:
   - **Window Level ($\text{WL}$)**: $+400\text{ HU}$
   - **Window Width ($\text{WW}$)**: $1500\text{ HU}$
   - **Lower Cutoff**: $\text{WL} - \frac{\text{WW}}{2} = 400 - 750 = -350\text{ HU}$
   - **Upper Cutoff**: $\text{WL} + \frac{\text{WW}}{2} = 400 + 750 = +1150\text{ HU}$

   The linear transfer function is defined as:
   $$\text{HU}_{\text{clipped}} = \min(\max(\text{HU}, -350), 1150)$$
   $$I_{\text{normalized}} = \frac{\text{HU}_{\text{clipped}} - (-350)}{1500} \in [0.0, 1.0]$$

3. **Preservation of 3D Voxel Geometry**:
   CT slice thickness (`(0018,0050)`) and pixel spacing (`(0028,0030)`) are strictly preserved in the `SpatialMetadata` container for volumetric radiomic feature calculation.

---

### 3.2 Planar Radiography (X-ray) Pipeline

1. **Photometric Interpretation Normalization**:
   - Radiographs tagged with `MONOCHROME1` (where 0 is white and high values are dark) are automatically inverted to standard `MONOCHROME2` (0 is black and maximum values are white):
     $$I_{\text{inv}} = \max(I) - I$$

2. **Percentile Intensity Normalization**:
   Direct min-max scaling is vulnerable to unattenuated X-ray beam collimator edges and dead pixels. The pipeline computes the $0.5\text{th}$ and $99.5\text{th}$ percentiles ($p_{\text{low}}, p_{\text{high}}$):
   $$I_{\text{clipped}} = \min(\max(I, p_{\text{low}}), p_{\text{high}})$$
   $$I_{\text{scaled}} = \frac{I_{\text{clipped}} - p_{\text{low}}}{p_{\text{high}} - p_{\text{low}}} \in [0.0, 1.0]$$

3. **Contrast-Limited Adaptive Histogram Equalization (CLAHE)**:
   Planar radiographs of Osteogenesis Imperfecta (severe osteopenia) and Paget's disease (mixed sclerosis/lysis) benefit from local contrast enhancement:
   - **Clip Limit ($\beta$)**: $2.0$ (prevents noise over-amplification in uniform soft tissue)
   - **Tile Grid Size**: $8 \times 8$ contextual tiles

---

## 4. Spatial Standardization & CNN Input Transformation

1. **Aspect-Ratio Preserving Resizing**:
   Anatomical bone geometry (e.g. cortical width, trabecular angles) must never be stretched or distorted.
   Given an original dimension $(H, W)$ and target $(T_H, T_W)$:
   $$\text{scale} = \min\left(\frac{T_H}{H}, \frac{T_W}{W}\right)$$
   $$\text{new}_H = \text{round}(H \times \text{scale}), \quad \text{new}_W = \text{round}(W \times \text{scale})$$
   Symmetric zero padding is added along the shorter dimension:
   $$\text{pad}_{\text{top}} = \lfloor (T_H - \text{new}_H) / 2 \rfloor, \quad \text{pad}_{\text{bottom}} = T_H - \text{new}_H - \text{pad}_{\text{top}}$$
   $$\text{pad}_{\text{left}} = \lfloor (T_W - \text{new}_W) / 2 \rfloor, \quad \text{pad}_{\text{right}} = T_W - \text{new}_W - \text{pad}_{\text{left}}$$

2. **PyTorch CNN Tensor Formulation**:
   Single-channel grayscale images are duplicated across 3 channels $[3, 224, 224]$ and normalized using standard ImageNet statistics:
   $$Z_c = \frac{X_c - \mu_c}{\sigma_c}$$
   where $\mu = [0.485, 0.456, 0.406]$ and $\sigma = [0.229, 0.224, 0.225]$.

---

## 5. HIPAA Safe Harbor PHI De-identification

Before metadata are recorded in dataset manifests or logged to disk, `sanitize_dicom_metadata()` scrubs all 18 HIPAA identifiers:
- Redacts: `PatientName`, `PatientBirthDate`, `AccessionNumber`, `InstitutionName`, `ReferringPhysicianName`, `StudyDate`, `OperatorsName`.
- Generates a deterministic, irreversibly salted pseudo-anonymous identifier:
  $$\text{PseudoID} = \text{"PT\_ANON\_"} + \text{SHA256}(\text{Salt} + \text{RawID})[:12]$$
- Coarsens patient age into 10-year age groups (with nonagenarians aggregated into `"90+"`).
- Strictly preserves technical non-identifying parameters: `PixelSpacing`, `SliceThickness`, `KVP`, `Modality`.

---

## 6. Failure Handling & Resilience Rules

1. **Corrupt File Detection**:
   - Zero-byte files, truncated binary streams, or non-numeric values immediately raise `CorruptImageError`.
2. **Dead Image Detection**:
   - Arrays with zero dynamic range ($\max - \min \le 0$) are rejected as unreadable sensor failures.
3. **Non-Fatal Metadata Warnings**:
   - **Never Silently Discard Studies**: If non-critical tags (such as `PixelSpacing` or `SliceThickness`) are missing from older radiographs, the pipeline logs an explicit warning, assigns standardized fallback values (e.g., $1.0\text{ mm}$ isotropic), and processes the study.
4. **Orientation Validation**:
   - Validates that row and column direction cosines in `ImageOrientationPatient` are orthogonal ($|\vec{r} \cdot \vec{c}| \le 0.05$). Deviations log diagnostic warnings for clinician review.

---

## 7. Preprocessing Provenance Audit Record

Every processed image produces a reproducible JSON-serializable provenance record:

```json
{
  "timestamp_utc": "2026-11-14T22:56:40Z",
  "source_file": "ml_service/data/raw/metastasis/PT01_metastasis_01.png",
  "source_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "format": "png",
  "modality": "xray",
  "original_shape": [64, 80],
  "original_dtype": "uint8",
  "voxel_spacing": [1.0, 1.0, 1.0],
  "slice_thickness": 1.0,
  "modality_pipeline": {
    "pipeline": "xray_intensity_normalization",
    "lower_percentile": 0.5,
    "upper_percentile": 99.5,
    "clahe_applied": true,
    "clip_limit": 2.0
  },
  "target_size": [224, 224],
  "scale_factor": 2.8,
  "padding": [0, 0, 0, 0],
  "output_tensor_shape": [3, 224, 224]
}
```

---

## 8. Automated Testing & Verification

The preprocessing pipeline is verified by 10 comprehensive unit tests in `ml_service/tests/test_preprocessing_pipeline.py`:
- `test_ct_hu_conversion_and_bone_windowing`: Boundary check (-350 HU $\to$ 0.0, +400 HU $\to$ 0.5, +1150 HU $\to$ 1.0).
- `test_distinct_ct_and_xray_normalizations`: Verifies strict mathematical divergence between CT and X-ray pipelines.
- `test_xray_clahe_preprocessing`: Validates local contrast enhancement output in $[0.0, 1.0]$.
- `test_spatial_resizing_preserves_aspect_ratio`: Verifies symmetric zero padding and scale calculations.
- `test_to_torch_backbone_tensor`: Confirms $[3, 224, 224]$ tensor construction with ImageNet normalization.
- `test_phi_deidentification_sanitization`: Verifies HIPAA Safe Harbor redaction and salted pseudo-ID generation.
- `test_orientation_validation`: Tests direction cosine orthogonality validation.
- `test_corrupt_file_detection`: Confirms zero-byte, truncated, and dead image rejection.
- `test_standard_formats_loading`: Verifies loading of PNG, JPEG, and TIFF files.
- `test_end_to_end_preprocessed_study`: Validates complete pipeline execution and provenance record generation.

*(Note: In compliance with zero-dataset rules, all unit tests utilize synthetic NumPy test arrays clearly distinguished from clinical training data).*


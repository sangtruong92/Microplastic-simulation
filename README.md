# Microplastic Imaging Simulation in Water

A comprehensive Python simulation for modeling camera imaging of microplastic particles suspended in water, incorporating realistic optical physics including refraction, defocus blur, and noise.

## 📋 Overview

This project simulates how microplastic particles appear when photographed underwater, accounting for:
- **Optical refraction** at the water-air interface
- **Defocus blur** based on particle depth
- **Magnification effects** from camera lens properties
- **Realistic noise** (shot noise and read noise)
- **Fluorescence imaging** (in the advanced version)

## 📁 Files

### Main Notebooks
- **`Microplastic_Simulation_In_Walter.ipynb`** - Complete simulation with detailed explanations, mathematical formulas, and visualizations
- **`Python_Learning.ipynb`** - Learning notebook with code explanations and Python concepts

### Python Scripts
- **`mp_image_sim_v1 (1).py`** - Fluorescence microplastic imaging simulation (standalone script)

### Output Images
- `fluorescence_single_f*.png` - Single particle simulations at different focal lengths
- `focal_length_comparison.png` - Comparison across focal lengths
- `size_analysis.png` - Quantitative analysis of particle size measurements

## 🚀 Features

### 1. Realistic Physics Simulation
- **Refraction**: Light bending at water-air interface (Snell's Law)
- **Magnification**: Lens equation with depth-dependent scaling
- **Defocus blur**: Gaussian blur increasing with distance from focal plane
- **Intensity attenuation**: Out-of-focus particles appear dimmer

### 2. Comprehensive Visualizations
- Particle mask generation
- Refraction effect diagrams
- Magnification vs depth curves
- Defocus blur demonstrations
- Blur parameter sensitivity analysis
- Size conversion pipeline

### 3. Educational Content
- Step-by-step mathematical formulas
- Detailed code explanations
- Visual comparisons (exact vs approximate formulas)
- Python programming concepts (list comprehensions, f-strings, etc.)

## 🔧 Requirements

```bash
pip install numpy matplotlib scipy
```

## 💻 Usage

### Basic Simulation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Set parameters
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
FOCAL_LENGTH_MM = 6.0
NUM_PARTICLES = 20

# Run simulation
simulated_image, particles = simulate_microplastic_image(NUM_PARTICLES)

# Display result
plt.imshow(simulated_image, cmap='gray')
plt.show()
```

### Key Parameters

- **Camera Settings**
  - `IMAGE_WIDTH`, `IMAGE_HEIGHT`: Image resolution (pixels)
  - `PIXEL_SIZE_UM`: Pixel pitch (micrometers)

- **Optical System**
  - `FOCAL_LENGTH_MM`: Lens focal length (2.1-12 mm)
  - `WATER_DEPTH_MM`: Water container depth (mm)
  - `FOCAL_PLANE_DEPTH_MM`: Focus position in water

- **Particle Parameters**
  - `PARTICLE_SIZE_MIN_UM`, `PARTICLE_SIZE_MAX_UM`: Size range (50-1000 μm)
  - `NUM_PARTICLES`: Number of particles to simulate

## 📊 Key Formulas

### Refraction (Snell's Law)
```
apparent_depth = true_depth / n_water
```

### Magnification
```
M = f / (d_object - f)  [exact]
M ≈ f / d_object        [approximate, when d >> f]
```

### Size Conversion
```
size_pixels = (size_μm × M) / pixel_size_μm
```

### Defocus Blur
```
σ_blur = 0.5 + 3.0 × |depth - focal_plane|
```

### Intensity Attenuation
```
I_factor = 1 / (1 + Δz²)
```

## 🎓 Educational Use

This simulation is designed for:
- Understanding underwater imaging physics
- Validating particle detection algorithms
- Teaching optical principles
- Microplastic research visualization
- Python programming education

## 📈 Example Outputs

The simulation generates:
1. **Synthetic images** showing realistic particle appearances
2. **Quantitative analysis** comparing true vs measured sizes
3. **Visualization plots** explaining optical effects
4. **Time-lapse sequences** (for fluorescence mode)

## 🔬 Applications

- **Algorithm validation**: Test particle detection/sizing algorithms
- **System design**: Optimize camera/lens parameters
- **Research**: Understand imaging limitations
- **Education**: Teach optical physics and computer vision

## 📝 Notes

- Blur parameters (0.5, 3.0) are empirical and can be calibrated to match real systems
- Magnification uses simplified formula valid when object distance >> focal length
- Supports both grayscale and fluorescence imaging modes

## 👤 Author

Sang Truong

## 📄 License

This project is open source and available for educational and research purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

---

**Repository**: https://github.com/sangtruong92/Microplastic-simulation

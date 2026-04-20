"""
Fluorescence Microplastic Imaging Simulation
---------------------------------------------
Features:
- Black background
- Fluorescence color (e.g., green) with adjustable SBR
- Time‑dependent fluorescence growth (intensity increases over time)
- Refraction, defocus blur, and noise as before
- Supports single particles and multi‑particle scenes
- Outputs RGB images and time‑lapse sequences
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label
from scipy.signal import convolve2d
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# SYSTEM PARAMETERS (same as before)
# =============================================================================
CAMERA_RES = (3840, 2160)            # (width, height)
PIXEL_PITCH_UM = 1.45                # µm/pixel
WATER_THICKNESS_MM = 5.0
LENS_TO_WATER_SURFACE_MM = 50.0
N_AIR = 1.0
N_WATER = 1.33
FOCAL_LENGTHS_MM = [2.1, 2.8, 3.6, 6, 8, 12]
PARTICLE_SIZE_RANGE_UM = (50, 1000)
PARTICLE_ASPECT_RANGE = (0.7, 1.0)
FOCAL_PLANE_DEPTH_MM = 2.5
BASE_SIGMA_PX = 0.5
BLUR_COEFF_PX_PER_MM = 12.0
DEFOCUS_INTENSITY_HALF_MM = 1.0

# Noise & background
ENABLE_SHOT_NOISE = True
ENABLE_READ_NOISE = True
READ_NOISE_SIGMA = 2.0
BACKGROUND_GRADIENT = False          # disable to keep background truly black

# Fluorescence settings
FLUORESCENCE_COLOR = (0, 255, 0)     # pure green (R,G,B)
SIGNAL_TO_BACKGROUND_RATIO = 10.0    # peak intensity / background level
BACKGROUND_MEAN = 0.01               # baseline intensity (0-1 range)

# Time‑lapse settings (fluorescence growth)
ENABLE_GROWTH = False

GROWTH_MODE = "intensity"            # "intensity" or "size" (size growth not fully implemented here)
GROWTH_RATE = 0.3                    # per time step, exponential factor
MAX_INTENSITY_FACTOR = 5.0           # max fluorescence intensity relative to initial

# Random seed
RNG_SEED = 42
np.random.seed(RNG_SEED)

# =============================================================================
# OPTICAL FUNCTIONS (unchanged)
# =============================================================================
def compute_magnification(f_mm, depth_water_mm):
    f_m = f_mm * 1e-3
    depth_water_m = depth_water_mm * 1e-3
    apparent_depth = depth_water_m / N_WATER
    object_distance = LENS_TO_WATER_SURFACE_MM * 1e-3 + apparent_depth
    M = f_m / object_distance
    return M

def object_size_to_image_pixels(size_um, f_mm, depth_water_mm):
    M = compute_magnification(f_mm, depth_water_mm)
    image_size_m = size_um * 1e-6 * M
    pixels = image_size_m / (PIXEL_PITCH_UM * 1e-6)
    return pixels

def defocus_blur_sigma_px(depth_mm, f_mm):
    defocus = abs(depth_mm - FOCAL_PLANE_DEPTH_MM)
    sigma_px = np.sqrt(BASE_SIGMA_PX**2 + (BLUR_COEFF_PX_PER_MM * defocus)**2)
    return sigma_px

def defocus_intensity_factor(depth_mm):
    defocus = abs(depth_mm - FOCAL_PLANE_DEPTH_MM)
    factor = 1.0 / (1.0 + (defocus / DEFOCUS_INTENSITY_HALF_MM)**2)
    return factor

# =============================================================================
# PARTICLE SIMULATION (color + SBR + growth)
# =============================================================================
def generate_particle_mask(center, size_x_px, size_y_px, img_shape):
    h, w = img_shape
    yy, xx = np.ogrid[:h, :w]
    a = size_x_px / 2.0
    b = size_y_px / 2.0
    mask = ((xx - center[0])**2 / a**2 + (yy - center[1])**2 / b**2) <= 1
    return mask.astype(np.float64)

def apply_psf_to_particle(particle_mask, sigma_px, intensity_factor):
    blurred = gaussian_filter(particle_mask, sigma=sigma_px, mode='constant')
    blurred = blurred * intensity_factor
    return blurred

def add_shot_noise(image, peak_signal=255.0):
    if not ENABLE_SHOT_NOISE:
        return image
    scaled = image * 10000.0
    noisy = np.random.poisson(scaled).astype(np.float64) / 10000.0
    return noisy

def add_read_noise(image, sigma=READ_NOISE_SIGMA):
    if not ENABLE_READ_NOISE:
        return image
    noise = np.random.normal(0, sigma/255.0, image.shape)  # scaled to 0-1
    return image + noise

def fluorescence_growth_factor(time_step, max_time_steps=10):
    """
    Returns a multiplier for fluorescence intensity based on time.
    time_step: current frame index (0 = first frame)
    max_time_steps: total number of frames in time‑lapse
    """
    if not ENABLE_GROWTH or GROWTH_MODE != "intensity":
        return 1.0
    # Exponential approach to max intensity
    t_norm = time_step / max_time_steps  # 0 → 1
    factor = 1.0 + (MAX_INTENSITY_FACTOR - 1.0) * (1 - np.exp(-GROWTH_RATE * time_step))
    return min(factor, MAX_INTENSITY_FACTOR)

def render_rgb_from_grayscale(gray_img, color, sbr, background_level=BACKGROUND_MEAN):
    """
    Convert grayscale intensity image to RGB fluorescence image.
    gray_img: values in [0, 1] (peak = 1)
    color: (R,G,B) tuple (0-255)
    sbr: signal-to-background ratio (peak_intensity / background_level)
    background_level: baseline intensity (0-1)
    """
    # Scale particle intensity to achieve desired SBR
    # Background level is fixed; peak should be sbr * background_level
    desired_peak = sbr * background_level
    if np.max(gray_img) > 0:
        gray_scaled = gray_img / np.max(gray_img) * desired_peak
    else:
        gray_scaled = gray_img
    # Add background
    rgb_img = np.zeros((gray_img.shape[0], gray_img.shape[1], 3), dtype=np.float64)
    for c in range(3):
        rgb_img[:,:,c] = background_level  # uniform background
    # Add fluorescence in the specified color channel(s)
    # Usually fluorescence is in one channel (e.g., green), but we can blend
    # For pure green: set green channel = gray_scaled, others unchanged
    # Here we assume color is the fluorescence emission (e.g., (0,255,0) means only green)
    # Normalize color components to 0-1
    color_norm = np.array(color) / 255.0
    for c in range(3):
        rgb_img[:,:,c] += gray_scaled * color_norm[c]
    # Clip to [0,1]
    rgb_img = np.clip(rgb_img, 0, 1)
    return rgb_img

def simulate_single_particle_fluorescence(f_mm, particle_size_um, depth_mm, time_step=0, 
                                          max_time_steps=10, output_path=None):
    """
    Simulate fluorescence image of a single particle with time-dependent growth.
    Returns RGB image (0-255 uint8) and apparent size in pixels.
    """
    # Basic optical mapping
    apparent_size_px = object_size_to_image_pixels(particle_size_um, f_mm, depth_mm)
    size_x_px = size_y_px = apparent_size_px
    sigma_px = defocus_blur_sigma_px(depth_mm, f_mm)
    intensity_factor = defocus_intensity_factor(depth_mm)
    
    # Generate grayscale intensity map
    h, w = CAMERA_RES[1], CAMERA_RES[0]
    img_gray = np.zeros((h, w), dtype=np.float64)
    center = (w // 2, h // 2)
    mask = generate_particle_mask(center, size_x_px, size_y_px, img_gray.shape)
    blurred = apply_psf_to_particle(mask, sigma_px, intensity_factor)
    img_gray += blurred
    
    # Apply fluorescence growth
    growth = fluorescence_growth_factor(time_step, max_time_steps)
    img_gray = img_gray * growth
    
    # Normalize max to 1
    if np.max(img_gray) > 0:
        img_gray = img_gray / np.max(img_gray)
    
    # Convert to RGB with SBR control
    rgb_img = render_rgb_from_grayscale(img_gray, FLUORESCENCE_COLOR, 
                                        SIGNAL_TO_BACKGROUND_RATIO, BACKGROUND_MEAN)
    
    # Add noise
    rgb_img = add_shot_noise(rgb_img)
    rgb_img = add_read_noise(rgb_img)
    rgb_img = np.clip(rgb_img * 255, 0, 255).astype(np.uint8)
    
    # Save annotated image if requested
    if output_path:
        plt.figure(figsize=(12, 8))
        plt.imshow(rgb_img)
        title = (f"f = {f_mm} mm | size = {particle_size_um} µm | depth = {depth_mm} mm | "
                 f"t = {time_step} | SBR = {SIGNAL_TO_BACKGROUND_RATIO}")
        plt.title(title, fontsize=10)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return rgb_img, apparent_size_px

def simulate_multi_particle_fluorescence(f_mm, num_particles, time_step=0, max_time_steps=10,
                                         output_path=None):
    """
    Simulate multiple fluorescent particles with time‑dependent growth.
    Returns RGB image, list of particles, and measured sizes.
    """
    h, w = CAMERA_RES[1], CAMERA_RES[0]
    img_gray = np.zeros((h, w), dtype=np.float64)
    particles = []
    centers = []
    min_distance_px = 30
    max_attempts = 5000
    
    # Place particles without excessive overlap
    for i in range(num_particles):
        attempt = 0
        while attempt < max_attempts:
            x = np.random.randint(50, w - 50)
            y = np.random.randint(50, h - 50)
            if all(np.hypot(x - cx, y - cy) > min_distance_px for cx, cy in centers):
                centers.append((x, y))
                break
            attempt += 1
        if attempt == max_attempts:
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            centers.append((x, y))
    
    # Generate each particle
    for idx, (cx, cy) in enumerate(centers):
        size_um = np.random.uniform(*PARTICLE_SIZE_RANGE_UM)
        depth_mm = np.random.uniform(0, WATER_THICKNESS_MM)
        apparent_size_px = object_size_to_image_pixels(size_um, f_mm, depth_mm)
        aspect = np.random.uniform(*PARTICLE_ASPECT_RANGE)
        size_x_px = apparent_size_px
        size_y_px = apparent_size_px * aspect
        sigma_px = defocus_blur_sigma_px(depth_mm, f_mm)
        intensity_factor = defocus_intensity_factor(depth_mm)
        mask = generate_particle_mask((cx, cy), size_x_px, size_y_px, img_gray.shape)
        blurred = apply_psf_to_particle(mask, sigma_px, intensity_factor)
        img_gray += blurred
        particles.append({
            'center': (cx, cy),
            'true_size_um': size_um,
            'depth_mm': depth_mm,
            'true_apparent_px': apparent_size_px,
            'size_x_px': size_x_px,
            'size_y_px': size_y_px
        })
    
    # Apply fluorescence growth (same factor for all particles, could be particle‑specific)
    growth = fluorescence_growth_factor(time_step, max_time_steps)
    img_gray = img_gray * growth
    
    # Normalize
    if np.max(img_gray) > 0:
        img_gray = img_gray / np.max(img_gray)
    
    # Convert to RGB with SBR
    rgb_img = render_rgb_from_grayscale(img_gray, FLUORESCENCE_COLOR,
                                        SIGNAL_TO_BACKGROUND_RATIO, BACKGROUND_MEAN)
    rgb_img = add_shot_noise(rgb_img)
    rgb_img = add_read_noise(rgb_img)
    rgb_img = np.clip(rgb_img * 255, 0, 255).astype(np.uint8)
    
    # Measure particle sizes (simulate image analysis)
    measured = measure_particle_sizes(rgb_img, particles)
    
    if output_path:
        plt.figure(figsize=(12, 8))
        plt.imshow(rgb_img)
        plt.title(f"Multi‑particle, f = {f_mm} mm, t = {time_step}, SBR = {SIGNAL_TO_BACKGROUND_RATIO}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return rgb_img, particles, measured

def measure_particle_sizes(rgb_img, true_particles):
    """Estimate particle sizes from the fluorescence image (using intensity threshold)."""
    # Convert to grayscale for measurement
    if rgb_img.dtype == np.uint8:
        gray = rgb_img[:,:,1].astype(np.float64) / 255.0  # green channel
    else:
        gray = rgb_img[:,:,1]
    measured = []
    for p in true_particles:
        cx, cy = p['center']
        radius_guess = int(p['true_apparent_px'] // 2 + 10)
        x0 = max(0, cx - radius_guess)
        x1 = min(gray.shape[1], cx + radius_guess)
        y0 = max(0, cy - radius_guess)
        y1 = min(gray.shape[0], cy + radius_guess)
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            measured.append(0)
            continue
        local_max = np.max(crop)
        if local_max < 0.05:
            measured.append(0)
            continue
        threshold = 0.3 * local_max
        binary = crop > threshold
        labeled, num_features = label(binary)
        crop_h, crop_w = crop.shape
        center_crop = (crop_h//2, crop_w//2)
        label_at_center = labeled[center_crop] if labeled[center_crop] > 0 else None
        if label_at_center is None:
            sizes = np.bincount(labeled.ravel())
            if len(sizes) > 1:
                label_at_center = np.argmax(sizes[1:]) + 1
            else:
                measured.append(0)
                continue
        area = np.sum(labeled == label_at_center)
        diameter_px = 2 * np.sqrt(area / np.pi) if area > 0 else 0
        measured.append(diameter_px)
    return measured

# =============================================================================
# TIME‑LAPSE GENERATION (growing effect)
# =============================================================================
def generate_time_lapse(f_mm, particle_size_um, depth_mm, num_frames=10, output_dir="time_lapse"):
    """Create a sequence of images showing fluorescence growth over time."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    for t in range(num_frames):
        out_path = f"{output_dir}/frame_{t:03d}_f{f_mm}mm.png"
        img, _ = simulate_single_particle_fluorescence(f_mm, particle_size_um, depth_mm,
                                                        time_step=t, max_time_steps=num_frames,
                                                        output_path=out_path)
    print(f"Time‑lapse saved to {output_dir}/")

# =============================================================================
# QUANTITATIVE ANALYSIS
# =============================================================================

def generate_analysis_plot(all_data, output_path=None):
    """
    Create scatter plot: true size vs measured size (pixels), color-coded by depth.
    all_data: list of tuples (f_mm, true_sizes, measured_sizes, depths)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (f_mm, true_sizes, measured_sizes, depths) in enumerate(all_data):
        ax = axes[idx]
        sc = ax.scatter(true_sizes, measured_sizes, c=depths, cmap='viridis',
                        alpha=0.7, edgecolors='k', s=50)
        ax.set_xlabel('True particle size (µm)')
        ax.set_ylabel('Measured size (pixels)')
        ax.set_title(f'f = {f_mm} mm')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add linear fit
        if len(true_sizes) > 1:
            z = np.polyfit(true_sizes, measured_sizes, 1)
            p = np.poly1d(z)
            x_fit = np.linspace(min(true_sizes), max(true_sizes), 100)
            ax.plot(x_fit, p(x_fit), 'r--', linewidth=1.5,
                    label=f'fit: {z[0]:.2f} px/µm')
            ax.legend()
        
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Depth (mm)')
    
    plt.suptitle('Quantitative Analysis: True vs Measured Particle Size', fontsize=16)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()

# =============================================================================
# MAIN SIMULATION
# =============================================================================
def run_simulation():
    print("Fluorescence microplastic simulation")
    print(f"SBR = {SIGNAL_TO_BACKGROUND_RATIO}, growth enabled = {ENABLE_GROWTH}")
    
    # Single particle examples (with time = final frame)
    single_size = 500
    single_depth = 2.0
    for f_mm in FOCAL_LENGTHS_MM:
        out_path = f"fluorescence_single_f{f_mm}mm.png"
        simulate_single_particle_fluorescence(f_mm, single_size, single_depth,
                                              time_step=9, max_time_steps=10,  # final time
                                              output_path=out_path)
        print(f"Saved: {out_path}")
    
    # Multi‑particle at final time
    npsim = 50
    analysis_data = []
    for f_mm in FOCAL_LENGTHS_MM:
        out_path = f"fluorescence_multi_f{f_mm}mm.png"
        img, particles, measured_sizes = simulate_multi_particle_fluorescence(f_mm, num_particles=npsim,
                                             time_step=9, max_time_steps=10,
                                             output_path=out_path)
        # Collect data for analysis
        true_sizes = [p['true_size_um'] for p in particles]
        depths = [p['depth_mm'] for p in particles]
        analysis_data.append((f_mm, true_sizes, measured_sizes, depths))
        print(f"Saved: {out_path}")
    
    # Example time‑lapse (growing effect) for first focal length
    generate_time_lapse(FOCAL_LENGTHS_MM[0], single_size, single_depth, num_frames=8)
    
    # Quantitative analysis (same as before)
    # Generate analysis plot
    generate_analysis_plot(analysis_data, output_path="size_analysis.png")
    print("\nSimulation complete. Output files saved.")
    
    # (Bonus) Side-by-side comparison for one scenario
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    test_size = 300
    test_depth = 3.0
    for idx, f_mm in enumerate(FOCAL_LENGTHS_MM):
        img, _ = simulate_single_particle_fluorescence(f_mm, test_size, test_depth, output_path=None)
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(f'{f_mm} mm')
        axes[idx].axis('off')
    plt.suptitle(f'Comparison: Particle size {test_size} µm at depth {test_depth} mm', fontsize=14)
    plt.tight_layout()
    plt.savefig("focal_length_comparison.png", dpi=150)
    plt.show()
    # (We reuse the previous analysis but with measured data; omitted for brevity)
    print("Simulation complete.")

if __name__ == "__main__":
    run_simulation()

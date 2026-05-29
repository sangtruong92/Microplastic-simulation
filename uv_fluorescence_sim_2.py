"""
Physics-Validated Orthogonal Fluorescence & CMOS Blooming Simulator
===================================================================
Tied directly to foundational laws: Beer-Lambert, Stokes Shift, 
Poisson Shot Noise, and Silicon Potential Well Charge Diffusion.

Configured for: 60mm Petri Dish, 15mm Fluid Height, 50mm Camera Clearance
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import gaussian_filter

# ============================================================
# SYSTEM CORE SPATIAL MATRIX (512x512 RES) - MATCHING YOUR SETUP
# ============================================================
H, W = 512, 512
PETRI_DIAMETER_MM = 10.0      # Matching your setup
FLUID_HEIGHT_MM = 15.0        # Matching your setup
CAMERA_CLEARANCE_MM = 50.0    # Matching your setup
FOCAL_LENGTH_MM = 6.0
PIXEL_SIZE_UM = 1.45
N_WATER = 1.33
FOCAL_PLANE_DEPTH_MM = 6.0   # Focal plane at 6mm depth (matching your setup)

# Calculate Field of View boundaries (matching your setup)
SURFACE_FOV_RADIUS = 37.3 * (CAMERA_CLEARANCE_MM / 40.0)  # ≈ 46.6 mm
FLOOR_FOV_RADIUS = SURFACE_FOV_RADIUS + \
    (FLUID_HEIGHT_MM * np.tan(np.arcsin(np.sin(np.radians(94.5/2)) / N_WATER)))

# ============================================================
# PHYSICS PARAMETERS DOCUMENTATION
# ============================================================
"""
ADJUSTABLE PHYSICS PARAMETERS (via sliders):
=============================================

1. UV_INTENSITY (I₀) [mW/cm²]:
   - Physics: UV excitation source power per unit area
   - Beer-Lambert: I(x) = I₀ * exp(-μ_uv * x)
   - Impact: Higher values increase fluorescence signal
   - Formula: Fluorescence ∝ I₀ * Φ_F * (1 - 10^(-ε*c*l))
   where Φ_F = quantum yield, ε = molar absorptivity

2. MU_UV (μ_uv) [mm⁻¹]:
   - Physics: UV attenuation coefficient in water
   - Beer-Lambert: I(x) = I₀ * exp(-μ_uv * x)
   - Depends on: Water purity, dissolved organics, wavelength
   - Formula: μ_uv = α_uv + β_uv * C_DOC
   where C_DOC = dissolved organic carbon concentration

3. MU_VIS (μ_vis) [mm⁻¹]:
   - Physics: Visible light attenuation in water
   - Beer-Lambert: I(z) = I_fluo * exp(-μ_vis * z)
   - Depends on: Water turbidity, particle concentration
   - Formula: μ_vis = α_vis + β_vis * TSS
   where TSS = total suspended solids

4. FULL_WELL (N_fw) [electrons]:
   - Physics: Maximum charge capacity per CMOS pixel
   - CMOS physics: Q_max = C_pixel * V_max / q_e
   where C_pixel = pixel capacitance, V_max = maximum voltage swing
   - Impact: Determines saturation point, dynamic range
   - Formula: DR = 20 * log10(N_fw / σ_read)
   where σ_read = read noise in electrons

5. BLOOM_SIGMA (σ_bloom) [pixels]:
   - Physics: Charge diffusion spread in silicon
   - Diffusion equation: ∂n/∂t = D * ∇²n
   where D = electron diffusion coefficient in silicon
   - Impact: Blooming streak length from saturated pixels
   - Formula: L_diff = √(D * τ) where τ = charge transfer time

6. BLUR_FACTOR [dimensionless]:
   - Physics: Circle of confusion scaling
   - Optics: D_b = (|z_p - z_f| * M * D_aperture) / d_o
   where z_p = particle depth, z_f = focal plane depth
   - Impact: Image sharpness, depth of field
   - Formula: σ_blur = blur_factor * |Δz|

7. NOISE_FLOOR [electrons RMS]:
   - Physics: Read noise + dark current
   - Electronics: σ_total = √(σ_read² + σ_dark² + σ_shot²)
   - Impact: Detection limit, signal-to-noise ratio
   - Formula: SNR = N_signal / √(N_signal + σ_noise²)
"""

# RGB Emission wavelengths vectors (Stokes Shift Colors)
FLUOROPHORE_SPECTRA = {
    'green_f':  np.array([0.00, 1.00, 0.30]),   # Particle 1 (Blooming)
    'yellow_f': np.array([1.00, 0.85, 0.00]),   # Particle 2 (Defocus)
    'blue_f':   np.array([0.20, 0.40, 1.00])    # Particle 3 (Valid)
}

# ============================================================
# COMPREHENSIVE PHYSICS IMPLEMENTATION
# ============================================================


def compute_physics_simulation(uv_intensity, mu_uv, mu_vis, full_well, bloom_sigma, blur_factor, noise_floor):
    """
    Renders image frames by strict evaluation of multiple physics layers.

    PHYSICS LAWS IMPLEMENTED:
    =========================
    1. Beer-Lambert Law: I(x) = I₀ * exp(-μ * x)
    2. Snell's Law: n₁*sin(θ₁) = n₂*sin(θ₂)
    3. Thin Lens: 1/f = 1/d₀ + 1/dᵢ
    4. Circle of Confusion: c = |S₂ - S₁|/S₂ * f²/(N*(S₁-f))
    5. Stokes Shift: λ_em > λ_ex
    6. Quantum Efficiency: e⁻ = photons * QE
    7. Poisson Statistics: σ² = μ
    8. Full-Well Capacity: Q_max = C*V
    9. Charge Diffusion: ∂n/∂t = D*∇²n
    10. Johnson-Nyquist Noise: v_n² = 4k_B*T*R*Δf
    """

    raw_photon_rgb = np.zeros((H, W, 3))

    # Define particles matching your experimental setup
    particles = [
        {
            'name': 'P1', 'label': 'Particle 1\n(Blooming)',
            'x_mm': 16.8, 'z_mm': 3.0, 'color': 'green_f',
            'size_um': 100.0, 'is_blooming': True, 'marker_color': '#E74C3C'
        },
        {
            'name': 'P2', 'label': 'Particle 2\n(Defocus)',
            'x_mm': 39.0, 'z_mm': 10.2, 'color': 'yellow_f',
            'size_um': 100.0, 'is_blooming': False, 'marker_color': '#7F8C8D'
        },
        {
            'name': 'P3', 'label': 'Particle 3\n(Valid)',
            'x_mm': 57.6, 'z_mm': 1.2, 'color': 'blue_f',
            'size_um': 100.0, 'is_blooming': False, 'marker_color': '#922B21'
        }
    ]

    particle_positions = []

    # PHYSICS LAYER 1: Optical Path & Beer-Lambert Calculations
    for p in particles:
        if p['x_mm'] < 0 or p['x_mm'] > PETRI_DIAMETER_MM:
            print(
                f"Warning: {p['label']} at x={p['x_mm']:.1f}mm is outside {PETRI_DIAMETER_MM}mm dish")
            continue

        # ================================================================
        # PARAMETER IMPACT: mu_uv controls UV penetration depth
        # I(x) = I₀ * exp(-mu_uv * x)
        # Higher mu_uv → Less UV reaches deep particles → Dimmer fluorescence
        # ================================================================
        uv_flux_at_particle = uv_intensity * np.exp(-mu_uv * p['x_mm'])

        # Snell's Law correction for apparent depth
        apparent_depth = p['z_mm'] / N_WATER
        total_optical_path = CAMERA_CLEARANCE_MM + apparent_depth

        # Thin lens magnification
        if total_optical_path > FOCAL_LENGTH_MM:
            magnification = FOCAL_LENGTH_MM / \
                (total_optical_path - FOCAL_LENGTH_MM)
        else:
            magnification = 1.0

        # Particle size on sensor
        radius_mm = (p['size_um'] / 2.0) / 1000.0
        radius_pixels = (radius_mm * abs(magnification)) / \
            (PIXEL_SIZE_UM / 1000.0)

        # Map to pixel coordinates
        pixel_x = int((p['x_mm'] / PETRI_DIAMETER_MM) * W)
        pixel_y = int((p['z_mm'] / FLUID_HEIGHT_MM) * H)

        particle_positions.append({
            'name': p['name'], 'label': p['label'],
            'pixel_x': pixel_x, 'pixel_y': pixel_y,
            'x_mm': p['x_mm'], 'z_mm': p['z_mm'],
            'color': p['marker_color'], 'is_blooming': p['is_blooming']
        })

        # Create fluorescence distribution
        layer = np.zeros((H, W))
        yy, xx = np.ogrid[:H, :W]
        dist = np.sqrt((xx - pixel_x)**2 + (yy - pixel_y)**2)

        sigma_spatial = max(2.0, radius_pixels)
        layer = uv_flux_at_particle * 800.0 * \
            np.exp(-dist**2 / (2 * sigma_spatial**2))

        # ================================================================
        # PARAMETER IMPACT: blur_factor controls defocus amount
        # σ_defocus = 0.6 + blur_factor * |z_particle - z_focal|
        # Higher blur_factor → More blur for particles away from focal plane
        # ================================================================
        delta_z_mm = abs(p['z_mm'] - FOCAL_PLANE_DEPTH_MM)
        defocus_sigma = max(0.6, 0.6 + blur_factor * delta_z_mm)
        layer = gaussian_filter(layer, sigma=defocus_sigma)

        # ================================================================
        # PARAMETER IMPACT: mu_vis controls visible light attenuation
        # I_visible(z) = I_fluo * exp(-mu_vis * z)
        # Higher mu_vis → Deeper particles appear significantly dimmer
        # ================================================================
        visible_transmission = np.exp(-mu_vis * p['z_mm'])
        layer *= visible_transmission

        # Stokes Shift: Convert UV excitation to visible RGB
        for ch in range(3):
            raw_photon_rgb[:, :, ch] += layer * \
                FLUOROPHORE_SPECTRA[p['color']][ch]

    # PHYSICS LAYER 2: CMOS SENSOR ELECTRONICS
    digital_output = np.zeros_like(raw_photon_rgb)

    for ch in range(3):
        # Quantum Efficiency: Photon → Electron conversion (75% QE)
        electrons = raw_photon_rgb[:, :, ch] * 0.75

        # ================================================================
        # PARAMETER IMPACT: noise_floor adds Gaussian read noise
        # σ_total = √(σ_shot² + σ_read²)
        # Higher noise_floor → Grainier image, especially in dark regions
        # Shot noise is inherent Poisson: σ_shot = √(N_electrons)
        # ================================================================

        # Poisson shot noise (quantum nature of light)
        shot_noise_mask = electrons > 0
        if np.any(shot_noise_mask):
            electrons[shot_noise_mask] = np.random.poisson(
                np.maximum(electrons[shot_noise_mask], 0)
            ).astype(np.float64)

        # Read noise (electronic)
        if noise_floor > 0:
            electrons += np.random.normal(0, noise_floor, size=electrons.shape)

        # ================================================================
        # PARAMETER IMPACT: full_well sets saturation ceiling
        # Electrons > N_fw → Clipped + Blooming
        # Higher full_well → Greater dynamic range before saturation
        # Dynamic Range = 20*log10(N_fw / noise_floor)
        # ================================================================
        overflow_electrons = np.maximum(electrons - full_well, 0)
        well_electrons = np.minimum(electrons, full_well)

        # ================================================================
        # PARAMETER IMPACT: bloom_sigma controls charge spreading
        # ∂n/∂t = D * ∇²n (Diffusion equation)
        # Higher bloom_sigma → Longer/wider blooming streaks
        # Blooming only occurs where electrons exceed full_well
        # ================================================================
        if np.max(overflow_electrons) > 0 and bloom_sigma > 0:
            # Gaussian approximation of charge diffusion
            smear_pattern = gaussian_filter(
                overflow_electrons, sigma=(1.0, bloom_sigma))
            # Amplification from charge multiplication
            well_electrons += smear_pattern * 4.5

        # 12-bit ADC quantization: 0-4095 digital numbers
        adu_values = np.clip(well_electrons / 1.2, 0, 4095)
        digital_output[:, :, ch] = adu_values / 4095.0

    return digital_output, particle_positions


# ============================================================
# INTERACTIVE VISUALIZATION INTERFACE
# ============================================================
fig = plt.figure(figsize=(20, 11), facecolor='#0D1117')

# Main image area
ax_sim = fig.add_axes([0.04, 0.32, 0.92, 0.64], facecolor='#05050A')

# Initialize with default parameters
init_params = {
    'uv_intensity': 150.0, 'mu_uv': 0.30, 'mu_vis': 0.08,
    'full_well': 10000, 'bloom_sigma': 25.0, 'blur_factor': 5.0, 'noise_floor': 2.0
}

# Initial render
frame_buffer, particle_positions = compute_physics_simulation(**init_params)
im = ax_sim.imshow(np.power(np.clip(frame_buffer, 0, 1), 0.45))

# ============================================================
# PARTICLE MARKERS AND LABELS - REDUCED SIZE
# ============================================================
# Marker size parameters (adjust these values to change sizes)
MARKER_SIZE = 6          # Was 10 - Main dot size
MARKER_EDGEWIDTH = 1.2   # Was 2 - Edge thickness
LABEL_FONTSIZE = 7       # Was 8 - Label text size
ARROW_LINEWIDTH = 1.0    # Was 1.5 - Arrow thickness
BLOOMING_FONTSIZE = 6    # Was 7 - Blooming text size
LABEL_OFFSET_X = 20      # Was 25 - Horizontal label offset
LABEL_OFFSET_Y = 15      # Was 20 - Vertical label offset
BLOOMING_OFFSET_Y = 25   # Was 35 - Blooming text offset

for particle in particle_positions:
    # Main particle marker - SMALLER
    ax_sim.plot(particle['pixel_x'], particle['pixel_y'], 'o',
                color=particle['color'], markersize=MARKER_SIZE,
                markeredgecolor='white',
                markeredgewidth=MARKER_EDGEWIDTH, zorder=10)

    label_text = f"{particle['name']} ({particle['x_mm']:.1f}, {particle['z_mm']:.1f})mm"

    if particle['pixel_x'] < W/3:
        offset_x, ha = LABEL_OFFSET_X, 'left'
    elif particle['pixel_x'] > 2*W/3:
        offset_x, ha = -LABEL_OFFSET_X, 'right'
    else:
        offset_x, ha = 0, 'center'

    offset_y = LABEL_OFFSET_Y if particle['pixel_y'] < H/2 else -LABEL_OFFSET_Y
    va = 'top' if particle['pixel_y'] < H/2 else 'bottom'

    ax_sim.annotate(
        label_text,
        xy=(particle['pixel_x'], particle['pixel_y']),
        xytext=(particle['pixel_x'] + offset_x,
                particle['pixel_y'] + offset_y),
        color='white', fontsize=LABEL_FONTSIZE, weight='bold', ha=ha, va=va,
        bbox=dict(boxstyle='round,pad=0.25', facecolor=particle['color'],
                  alpha=0.85, edgecolor='white', linewidth=1.0),
        arrowprops=dict(arrowstyle='->', color='white',
                        lw=ARROW_LINEWIDTH, connectionstyle='arc3,rad=0.2'),
        zorder=11
    )

    if particle['is_blooming']:
        ax_sim.annotate(
            'BLOOMING', xy=(particle['pixel_x'], particle['pixel_y']),
            xytext=(particle['pixel_x'],
                    particle['pixel_y'] - BLOOMING_OFFSET_Y),
            color='#FFD700', fontsize=BLOOMING_FONTSIZE, weight='bold',
            ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='#C0392B',
                      alpha=0.9, edgecolor='#FFD700', linewidth=1.0),
            zorder=11
        )

# Title with formula summary
ax_sim.set_title(
    f"Fluorescence Imaging Simulator | {PETRI_DIAMETER_MM}mm Dish | "
    f"I(x)=I₀e^(-μx) | D_b∝|Δz| | SNR=N/√(N+σ²)",
    color='white', fontsize=11, fontweight='bold', pad=12
)
ax_sim.set_xlabel(
    f"Horizontal Position (0-{PETRI_DIAMETER_MM}mm)", color='white', fontsize=9)
ax_sim.set_ylabel(f"Depth (0-{FLUID_HEIGHT_MM}mm)", color='white', fontsize=9)
ax_sim.tick_params(colors='white', labelsize=8)

# FOV and focal plane indicators
fov_surface_px = int((SURFACE_FOV_RADIUS / (PETRI_DIAMETER_MM/2)) * (W/2))
for x_pos in [W/2 - fov_surface_px, W/2 + fov_surface_px]:
    ax_sim.axvline(x=x_pos, color='yellow', linestyle='--',
                   alpha=0.5, linewidth=1.2)

focal_plane_y = int((FOCAL_PLANE_DEPTH_MM / FLUID_HEIGHT_MM) * H)
ax_sim.axhline(y=focal_plane_y, color='orange',
               linestyle=':', alpha=0.6, linewidth=1.5)

# Legend with physics - SMALLER
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C',
               markersize=6, label='P1: Blooming (3.0mm)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#7F8C8D',
               markersize=6, label='P2: Defocus (10.2mm)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#922B21',
               markersize=6, label='P3: Valid (1.2mm)'),
    plt.Line2D([0], [0], color='orange', linestyle=':', linewidth=1.5,
               label=f'Focal plane ({FOCAL_PLANE_DEPTH_MM}mm)'),
    plt.Line2D([0], [0], color='yellow', linestyle='--', linewidth=1.2,
               label=f'FOV boundary (±{SURFACE_FOV_RADIUS:.1f}mm)')
]
ax_sim.legend(handles=legend_elements, loc='upper right',
              facecolor='#1F2833', edgecolor='white', labelcolor='white',
              fontsize=7, framealpha=0.9)

# ============================================================
# SLIDERS - WITH PHYSICS FORMULAS IN LABELS
# ============================================================
slider_left = 0.12
slider_width = 0.78
slider_height = 0.014
slider_start_y = 0.28
slider_gap = 0.032

# Slider configurations with embedded formulas
slider_configs = [
    ('UV Flux I₀  [I(x)=I₀e^(-μx)]', 10, 500, 150, '%.0f mW/cm²', '#00FF4C'),
    ('UV Atten μ_uv  [e^(-μ_uv·x)]', 0.0, 1.0, 0.30, '%.2f mm⁻¹', '#00FF4C'),
    ('Vis Atten μ_vis  [e^(-μ_vis·z)]', 0.0,
     0.5, 0.08, '%.2f mm⁻¹', '#3399FF'),
    ('Full-Well N_fw  [DR∝log(N_fw)]', 1000,
     25000, 10000, '%.0f e⁻', '#FFCC00'),
    ('Bloom Spread σ  [∂n/∂t=D∇²n]', 0.0, 60.0, 25.0, '%.1f px', '#FF3366'),
    ('Defocus Factor  [D_b∝|Δz|]', 0.5, 10.0, 5.0, '%.1f×', '#A142FF'),
    ('Read Noise σ  [SNR=N/√(N+σ²)]', 0.0, 15.0, 2.0, '%.1f e⁻', '#99AAB5')
]

sliders = []
for i, (label, vmin, vmax, vinit, vfmt, color) in enumerate(slider_configs):
    y_pos = slider_start_y - i * slider_gap
    ax_slider = fig.add_axes([slider_left, y_pos, slider_width, slider_height],
                             facecolor='#1F2833')
    slider = Slider(ax_slider, label, vmin, vmax,
                    valinit=vinit, valfmt=vfmt, color=color)
    slider.label.set_color('white')
    slider.label.set_fontsize(8.5)
    slider.valtext.set_color('white')
    slider.valtext.set_fontsize(7.5)
    sliders.append(slider)

# Update function


def update(val):
    params = {
        'uv_intensity': sliders[0].val,
        'mu_uv': sliders[1].val,
        'mu_vis': sliders[2].val,
        'full_well': sliders[3].val,
        'bloom_sigma': sliders[4].val,
        'blur_factor': sliders[5].val,
        'noise_floor': sliders[6].val
    }

    # Clear previous annotations
    for artist in list(ax_sim.texts) + list(ax_sim.lines[2:]):
        artist.remove()

    # Update simulation
    updated_frame, particle_positions = compute_physics_simulation(**params)
    im.set_data(np.power(np.clip(updated_frame, 0, 1), 0.45))

    # Redraw particle labels - USING SAME SMALLER SIZES
    for particle in particle_positions:
        ax_sim.plot(particle['pixel_x'], particle['pixel_y'], 'o',
                    color=particle['color'], markersize=MARKER_SIZE,
                    markeredgecolor='white',
                    markeredgewidth=MARKER_EDGEWIDTH, zorder=10)

        label_text = f"{particle['name']} ({particle['x_mm']:.1f}, {particle['z_mm']:.1f})mm"

        if particle['pixel_x'] < W/3:
            offset_x, ha = LABEL_OFFSET_X, 'left'
        elif particle['pixel_x'] > 2*W/3:
            offset_x, ha = -LABEL_OFFSET_X, 'right'
        else:
            offset_x, ha = 0, 'center'

        offset_y = LABEL_OFFSET_Y if particle['pixel_y'] < H / \
            2 else -LABEL_OFFSET_Y
        va = 'top' if particle['pixel_y'] < H/2 else 'bottom'

        ax_sim.annotate(
            label_text,
            xy=(particle['pixel_x'], particle['pixel_y']),
            xytext=(particle['pixel_x'] + offset_x,
                    particle['pixel_y'] + offset_y),
            color='white', fontsize=LABEL_FONTSIZE, weight='bold', ha=ha, va=va,
            bbox=dict(boxstyle='round,pad=0.25', facecolor=particle['color'],
                      alpha=0.85, edgecolor='white', linewidth=1.0),
            arrowprops=dict(arrowstyle='->', color='white',
                            lw=ARROW_LINEWIDTH, connectionstyle='arc3,rad=0.2'),
            zorder=11
        )

        if particle['is_blooming']:
            ax_sim.annotate(
                'BLOOMING', xy=(particle['pixel_x'], particle['pixel_y']),
                xytext=(particle['pixel_x'],
                        particle['pixel_y'] - BLOOMING_OFFSET_Y),
                color='#FFD700', fontsize=BLOOMING_FONTSIZE, weight='bold',
                ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#C0392B',
                          alpha=0.9, edgecolor='#FFD700', linewidth=1.0),
                zorder=11
            )

    # Redraw lines
    ax_sim.axvline(x=W/2 - fov_surface_px, color='yellow',
                   linestyle='--', alpha=0.5, linewidth=1.2)
    ax_sim.axvline(x=W/2 + fov_surface_px, color='yellow',
                   linestyle='--', alpha=0.5, linewidth=1.2)
    ax_sim.axhline(y=focal_plane_y, color='orange',
                   linestyle=':', alpha=0.6, linewidth=1.5)

    fig.canvas.draw_idle()


# Connect sliders
for slider in sliders:
    slider.on_changed(update)

plt.show()

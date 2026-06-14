"""
Physics-Validated Orthogonal Fluorescence & CMOS Blooming Simulator
===================================================================
Tied directly to foundational laws: Beer-Lambert, Stokes Shift, 
Poisson Shot Noise, and Silicon Potential Well Charge Diffusion.

Configured for: 60mm Petri Dish, 15mm Fluid Height, 20mm Camera Clearance
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import gaussian_filter

# ============================================================
# SYSTEM CORE SPATIAL MATRIX & NEW GEOMETRY PARAMETERS
# ============================================================
H, W = 512, 512
PETRI_DIAMETER_MM = 60.0      # Updated to 60mm setup
FLUID_HEIGHT_MM = 15.0
CAMERA_CLEARANCE_MM = 20.0    # Updated camera clearance distance
FOCAL_LENGTH_MM = 3.08        # Updated lens focal length
PIXEL_SIZE_UM = 1.55          # Updated pixel dimension
N_WATER = 1.33
FOCAL_PLANE_DEPTH_MM = 3.2    # Updated focal plane matching target depth

# Define particles matching your new experimental setup positions
particles = [
    {
        'name': 'P1', 'label': 'Particle 1\n(Blooming)',
        'x_mm': 16.8, 'z_mm': 5.0, 'color': 'green_f',
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

# Calculate Field of View boundaries using your updated equations
Particle_Size = particles[0]["size_um"]
particle_Depth = particles[0]["z_mm"]

SURFACE_FOV_RADIUS = (Particle_Size / PIXEL_SIZE_UM) * (FOCAL_LENGTH_MM /
                                                        (CAMERA_CLEARANCE_MM + particle_Depth / N_WATER - FOCAL_LENGTH_MM))
FLOOR_FOV_RADIUS = SURFACE_FOV_RADIUS + \
    (FLUID_HEIGHT_MM * np.tan(np.arcsin(np.sin(np.radians(88 / 2)) / N_WATER)))

print(f"Calculated Surface FOV Radius: {SURFACE_FOV_RADIUS:.4f} mm")
print(f"Calculated Floor FOV Radius: {FLOOR_FOV_RADIUS:.4f} mm")

# RGB Emission wavelengths vectors (Stokes Shift Colors)
FLUOROPHORE_SPECTRA = {
    'green_f':  np.array([0.00, 1.00, 0.30]),   # Particle 1
    'yellow_f': np.array([1.00, 0.85, 0.00]),   # Particle 2
    'blue_f':   np.array([0.20, 0.40, 1.00])    # Particle 3
}

# ============================================================
# COMPREHENSIVE PHYSICS IMPLEMENTATION
# ============================================================


def compute_physics_simulation(uv_intensity, mu_uv, mu_vis, full_well, bloom_sigma, blur_factor, noise_floor):
    """
    Renders image frames by strict evaluation of multiple physics layers.
    """
    raw_photon_rgb = np.zeros((H, W, 3))
    particle_positions = []

    # PHYSICS LAYER 1: Optical Path & Beer-Lambert Calculations
    for p in particles:
        if p['x_mm'] < 0 or p['x_mm'] > PETRI_DIAMETER_MM:
            continue

        # UV excitation attenuation via Beer-Lambert
        uv_flux_at_particle = uv_intensity * np.exp(-mu_uv * p['x_mm'])

        # Snell's Law correction for apparent depth matrix mapping
        apparent_depth = p['z_mm'] / N_WATER
        total_optical_path = CAMERA_CLEARANCE_MM + apparent_depth

        # Thin lens magnification factor
        if total_optical_path > FOCAL_LENGTH_MM:
            magnification = FOCAL_LENGTH_MM / \
                (total_optical_path - FOCAL_LENGTH_MM)
        else:
            magnification = 1.0

        # Projected particle size on active CMOS sensor surface
        radius_mm = (p['size_um'] / 2.0) / 1000.0
        radius_pixels = (radius_mm * abs(magnification)) / \
            (PIXEL_SIZE_UM / 1000.0)

        # Map spatial physical dimensions into system matrix coordinate frames
        pixel_x = int((p['x_mm'] / PETRI_DIAMETER_MM) * W)
        pixel_y = int((p['z_mm'] / FLUID_HEIGHT_MM) * H)

        particle_positions.append({
            'name': p['name'], 'label': p['label'],
            'pixel_x': pixel_x, 'pixel_y': pixel_y,
            'x_mm': p['x_mm'], 'z_mm': p['z_mm'],
            'color': p['marker_color'], 'is_blooming': p['is_blooming']
        })

        # Generate spatial fluorescence profile intensity distribution
        layer = np.zeros((H, W))
        yy, xx = np.ogrid[:H, :W]
        dist = np.sqrt((xx - pixel_x)**2 + (yy - pixel_y)**2)

        sigma_spatial = max(2.0, radius_pixels)
        layer = uv_flux_at_particle * 800.0 * \
            np.exp(-dist**2 / (2 * sigma_spatial**2))

        # Defocus calculation based on separation from target focal plane
        delta_z_mm = abs(p['z_mm'] - FOCAL_PLANE_DEPTH_MM)
        defocus_sigma = max(0.6, 0.6 + blur_factor * delta_z_mm)
        layer = gaussian_filter(layer, sigma=defocus_sigma)

        # Visible light transmission loss through media depth
        visible_transmission = np.exp(-mu_vis * p['z_mm'])
        layer *= visible_transmission

        # Stokes Shift: Map mono generation values across RGB spectra parameters
        for ch in range(3):
            raw_photon_rgb[:, :, ch] += layer * \
                FLUOROPHORE_SPECTRA[p['color']][ch]

    # PHYSICS LAYER 2: CMOS SENSOR ELECTRONICS SIMULATION
    digital_output = np.zeros_like(raw_photon_rgb)

    for ch in range(3):
        # Quantum Efficiency conversion scaling
        electrons = raw_photon_rgb[:, :, ch] * 0.75

        # Poisson shot noise modeling
        shot_noise_mask = electrons > 0
        if np.any(shot_noise_mask):
            electrons[shot_noise_mask] = np.random.poisson(
                np.maximum(electrons[shot_noise_mask], 0)
            ).astype(np.float64)

        # Electronic Gaussian read noise injection
        if noise_floor > 0:
            electrons += np.random.normal(0, noise_floor, size=electrons.shape)

        # Full-Well capacity constraint saturation tracking
        overflow_electrons = np.maximum(electrons - full_well, 0)
        well_electrons = np.minimum(electrons, full_well)

        # Silicon charge diffusion calculation modeling pixel bleeding
        if np.max(overflow_electrons) > 0 and bloom_sigma > 0:
            smear_pattern = gaussian_filter(
                overflow_electrons, sigma=(1.0, bloom_sigma))
            well_electrons += smear_pattern * 4.5

        # 12-bit ADC digitization scaling matrix conversion
        adu_values = np.clip(well_electrons / 1.2, 0, 4095)
        digital_output[:, :, ch] = adu_values / 4095.0

    return digital_output, particle_positions


# ============================================================
# INTERACTIVE VISUALIZATION INTERFACE CONFIGURATION
# ============================================================
fig = plt.figure(figsize=(20, 11), facecolor='#0D1117')
ax_sim = fig.add_axes([0.04, 0.32, 0.92, 0.64], facecolor='#05050A')

# Active simulation parameters
init_params = {
    'uv_intensity': 150.0, 'mu_uv': 0.12, 'mu_vis': 0.05,
    'full_well': 8000, 'bloom_sigma': 30.0, 'blur_factor': 6.0, 'noise_floor': 2.5
}

frame_buffer, particle_positions = compute_physics_simulation(**init_params)
im = ax_sim.imshow(np.power(np.clip(frame_buffer, 0, 1), 0.45))

# Interface scale tuning parameters
MARKER_SIZE = 6
MARKER_EDGEWIDTH = 1.2
LABEL_FONTSIZE = 7
ARROW_LINEWIDTH = 1.0
BLOOMING_FONTSIZE = 6
LABEL_OFFSET_X = 20
LABEL_OFFSET_Y = 15
BLOOMING_OFFSET_Y = 25


def draw_visual_annotations(positions):
    for particle in positions:
        ax_sim.plot(particle['pixel_x'], particle['pixel_y'], 'o',
                    color=particle['color'], markersize=MARKER_SIZE,
                    markeredgecolor='white', markeredgewidth=MARKER_EDGEWIDTH, zorder=10)

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
            label_text, xy=(particle['pixel_x'], particle['pixel_y']),
            xytext=(particle['pixel_x'] + offset_x,
                    particle['pixel_y'] + offset_y),
            color='white', fontsize=LABEL_FONTSIZE, weight='bold', ha=ha, va=va,
            bbox=dict(boxstyle='round,pad=0.25',
                      facecolor=particle['color'], alpha=0.85, edgecolor='white', linewidth=1.0),
            arrowprops=dict(arrowstyle='->', color='white',
                            lw=ARROW_LINEWIDTH, connectionstyle='arc3,rad=0.2'),
            zorder=11
        )

        if particle['is_blooming']:
            ax_sim.annotate(
                'BLOOMING', xy=(particle['pixel_x'], particle['pixel_y']),
                xytext=(particle['pixel_x'],
                        particle['pixel_y'] - BLOOMING_OFFSET_Y),
                color='#FFD700', fontsize=BLOOMING_FONTSIZE, weight='bold', ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#C0392B',
                          alpha=0.9, edgecolor='#FFD700', linewidth=1.0),
                zorder=11
            )


draw_visual_annotations(particle_positions)

# Dynamic Titles mapping current configuration parameters
ax_sim.set_title(
    f"Fluorescence Imaging Simulator | {PETRI_DIAMETER_MM:.0f}mm Dish | Setup Layout: Clearance={CAMERA_CLEARANCE_MM}mm, f={FOCAL_LENGTH_MM}mm\n"
    f"Calculated Surface FOV Radius: {SURFACE_FOV_RADIUS:.2f}mm | Floor FOV Radius: {FLOOR_FOV_RADIUS:.2f}mm",
    color='white', fontsize=11, fontweight='bold', pad=12
)
ax_sim.set_xlabel(
    f"Horizontal Position (0-{PETRI_DIAMETER_MM:.0f}mm)", color='white', fontsize=9)
ax_sim.set_ylabel(
    f"Fluid Depth (0-{FLUID_HEIGHT_MM:.0f}mm)", color='white', fontsize=9)
ax_sim.tick_params(colors='white', labelsize=8)

# Map mathematical FOV bounds into relative canvas columns
fov_surface_px = int((SURFACE_FOV_RADIUS / (PETRI_DIAMETER_MM / 2)) * (W / 2))
for x_pos in [W/2 - fov_surface_px, W/2 + fov_surface_px]:
    if 0 <= x_pos <= W:
        ax_sim.axvline(x=x_pos, color='yellow', linestyle='--',
                       alpha=0.5, linewidth=1.2)

focal_plane_y = int((FOCAL_PLANE_DEPTH_MM / FLUID_HEIGHT_MM) * H)
ax_sim.axhline(y=focal_plane_y, color='orange',
               linestyle=':', alpha=0.6, linewidth=1.5)

# Update legend entries with new depth data properties
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C',
               markersize=6, label='P1: Blooming (5.0mm Depth)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#7F8C8D',
               markersize=6, label='P2: Defocus (10.2mm Depth)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#922B21',
               markersize=6, label='P3: Valid (1.2mm Depth)'),
    plt.Line2D([0], [0], color='orange', linestyle=':', linewidth=1.5,
               label=f'Focal Plane Target ({FOCAL_PLANE_DEPTH_MM:.1f}mm)'),
    plt.Line2D([0], [0], color='yellow', linestyle='--', linewidth=1.2,
               label=f'Surface FOV Boundary (±{SURFACE_FOV_RADIUS:.1f}mm)')
]
ax_sim.legend(handles=legend_elements, loc='upper right', facecolor='#1F2833',
              edgecolor='white', labelcolor='white', fontsize=7, framealpha=0.9)

# ============================================================
# SLIDERS CAPTURE & INTERACTION PIPELINE
# ============================================================
slider_left, slider_width, slider_height = 0.12, 0.78, 0.014
slider_start_y, slider_gap = 0.28, 0.032

slider_configs = [
    ('UV Flux I₀  [I(x)=I₀e^(-μx)]', 10, 500,
     init_params['uv_intensity'], '%.0f mW/cm²', '#00FF4C'),
    ('UV Atten μ_uv  [e^(-μ_uv·x)]', 0.0, 1.0,
     init_params['mu_uv'], '%.2f mm⁻¹', '#00FF4C'),
    ('Vis Atten μ_vis  [e^(-μ_vis·z)]', 0.0, 0.5,
     init_params['mu_vis'], '%.2f mm⁻¹', '#3399FF'),
    ('Full-Well N_fw  [DR∝log(N_fw)]', 1000, 25000,
     init_params['full_well'], '%.0f e⁻', '#FFCC00'),
    ('Bloom Spread σ  [∂n/∂t=D∇²n]', 0.0, 60.0,
     init_params['bloom_sigma'], '%.1f px', '#FF3366'),
    ('Defocus Factor  [D_b∝|Δz|]', 0.5, 10.0,
     init_params['blur_factor'], '%.1f×', '#A142FF'),
    ('Read Noise σ  [SNR=N/√(N+σ²)]', 0.0, 15.0,
     init_params['noise_floor'], '%.1f e⁻', '#99AAB5')
]

sliders = []
for i, (label, vmin, vmax, vinit, vfmt, color) in enumerate(slider_configs):
    y_pos = slider_start_y - i * slider_gap
    ax_slider = fig.add_axes(
        [slider_left, y_pos, slider_width, slider_height], facecolor='#1F2833')
    slider = Slider(ax_slider, label, vmin, vmax,
                    valinit=vinit, valfmt=vfmt, color=color)
    slider.label.set_color('white')
    slider.label.set_fontsize(8.5)
    slider.valtext.set_color('white')
    slider.valtext.set_fontsize(7.5)
    sliders.append(slider)


def update(val):
    params = {
        'uv_intensity': sliders[0].val, 'mu_uv': sliders[1].val, 'mu_vis': sliders[2].val,
        'full_well': int(sliders[3].val), 'bloom_sigma': sliders[4].val, 'blur_factor': sliders[5].val,
        'noise_floor': sliders[6].val
    }

    # Clear previous frame vector graphics artists safely
    for artist in list(ax_sim.texts) + list(ax_sim.lines[2:]):
        artist.remove()

    updated_frame, post_positions = compute_physics_simulation(**params)
    im.set_data(np.power(np.clip(updated_frame, 0, 1), 0.45))

    # Redraw standard annotations
    draw_visual_annotations(post_positions)

    # Re-apply spatial references lines onto updated context frame
    if 0 <= (W/2 - fov_surface_px) <= W:
        ax_sim.axvline(x=W/2 - fov_surface_px, color='yellow',
                       linestyle='--', alpha=0.5, linewidth=1.2)
        ax_sim.axvline(x=W/2 + fov_surface_px, color='yellow',
                       linestyle='--', alpha=0.5, linewidth=1.2)
    ax_sim.axhline(y=focal_plane_y, color='orange',
                   linestyle=':', alpha=0.6, linewidth=1.5)

    fig.canvas.draw_idle()


for slider in sliders:
    slider.on_changed(update)

plt.show()

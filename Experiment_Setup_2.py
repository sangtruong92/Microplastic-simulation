import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ============================================================
# ADJUSTABLE SYSTEM CONFIGURATION PARAMETERS
# ============================================================

# PETRI DISH DIMENSIONS
# ---------------------
PETRI_DIAMETER_MM = 10.0
# Formula: V_fluid = π*(D/2)²*H
# For D=10mm, H=15mm → V = π*5²*15 = 1,178 mm³ ≈ 1.18 mL
FLUID_HEIGHT_MM = 15.0

# CAMERA CLEARANCE (d₀)
# ---------------------
CAMERA_CLEARANCE_MM = 50.0  # Distance from lens to water surface
# Formula: Magnification M = f/(d₀ + z/n_water - f)
# Larger d₀ → Smaller M → Wider FOV but lower resolution
# FOV_surface ∝ d₀  (linear relationship)
# FOV_floor = FOV_surface + H*tan(θ_water)

# FOCAL DEPTH (y_f)
# -----------------
FOCAL_DEPTH_MM = 6.0
# Formula: Circle of Confusion D_b = (|y_p - y_f| * M * D_aperture) / d_effective
# where d_effective = d₀ + y_f/n_water
# Particles at y_f appear sharp; blur increases with |y_p - y_f|

# ============================================================
# CAMERA GAIN & EXPOSURE PARAMETERS (NEW)
# ============================================================

# ANALOG GAIN (ISO equivalent)
# ----------------------------
ANALOG_GAIN_DB = 0.0  # Analog gain in dB (0 = base gain)
# Formula: G_linear = 10^(Gain_dB / 20)
# Signal_out = Signal_in * G_linear
# Noise_out = sqrt((Noise_in * G_linear)² + Noise_amplifier²)
# SNR decreases with gain: SNR_out = SNR_in / sqrt(1 + (N_amp/(G*N_in))²)

# DIGITAL GAIN (brightness multiplier)
# ------------------------------------
DIGITAL_GAIN = 1.0  # Digital gain multiplier (1.0 = no digital gain)
# Formula: ADU_final = ADU_original * Digital_Gain
# Digital gain amplifies both signal AND noise equally
# Does NOT improve SNR: SNR_digital = SNR_original

# EXPOSURE TIME
# -------------
EXPOSURE_TIME_MS = 100.0  # Exposure time in milliseconds
# Formula: Signal = Irradiance * QE * A_pixel * t_exp / (h*c/λ)
# SNR improves with sqrt(t_exp): SNR ∝ sqrt(t_exp)
# Motion blur risk: blur_pixels = v_particle * t_exp / pixel_size

# ============================================================
# OPTICAL PARAMETERS (NEW)
# ============================================================

# APERTURE (f-number)
# -------------------
F_NUMBER = 2.8  # f/2.8 aperture
FOCAL_LENGTH_MM = 6.0
APERTURE_DIAMETER_MM = FOCAL_LENGTH_MM / F_NUMBER  # D = f/N ≈ 2.14mm
# Formula: Light collection ∝ (D/f)² = 1/N²
# f/2.8 collects 4x more light than f/5.6
# Diffraction limit: d_diff = 2.44 * λ * N (≈ 3.7µm at f/2.8 for 550nm)

# QUANTUM EFFICIENCY
# ------------------
QUANTUM_EFFICIENCY = 0.75  # 75% QE (typical CMOS)
# Formula: N_electrons = N_photons * QE
# Sony IMX577: ~75% at 530nm peak

# FULL-WELL CAPACITY
# ------------------
FULL_WELL_ELECTRONS = 10000  # electrons per pixel
# Formula: DR_dB = 20 * log10(FWC / ReadNoise)
# DR = 20 * log10(10000 / 2) ≈ 74 dB

# READ NOISE
# ----------
READ_NOISE_ELECTRONS = 2.0  # electrons RMS
# Formula: SNR_max = FWC / ReadNoise = 10000/2 = 5000:1

# REFRACTIVE INDICES
# ------------------
N_WATER = 1.33  # Affects: apparent depth = real_depth/n_water
N_AIR = 1.0     # Affects: refraction angle via Snell's Law
# Snell's Law: n₁sin(θ₁) = n₂sin(θ₂)
# Critical angle for water→air: θ_c = arcsin(1/1.33) ≈ 48.8°

# ============================================================
# GAIN PHYSICS DOCUMENTATION
# ============================================================
"""
CAMERA GAIN & EXPOSURE PHYSICS:

1. ANALOG GAIN (ISO):
   - Amplifies signal BEFORE ADC conversion
   - Formula: V_out = V_in * 10^(Gain_dB/20)
   - Reduces read noise contribution at high gain
   - Dynamic range decreases with gain: DR_effective = DR_base - Gain_dB

2. DIGITAL GAIN:
   - Multiplies ADU values AFTER ADC conversion
   - Formula: ADU_out = ADU_in * Digital_Gain
   - Does NOT improve signal-to-noise ratio
   - Can cause posterization (gaps in histogram)

3. EXPOSURE TIME:
   - Signal accumulation: S = ∫(I(t) * QE) dt ≈ I_avg * QE * t_exp
   - Dark current: N_dark = I_dark * t_exp (increases with temperature)
   - Shot noise: σ_shot = sqrt(S * QE)
   - Total SNR: SNR = (S * QE) / sqrt(S * QE + σ_read² + I_dark * t_exp)

4. SIGNAL CHAIN WITH GAIN:
   Photons → QE → Electrons → Analog_Gain → ADC → Digital_Gain → Output
   
5. EFFECTIVE SYSTEM GAIN:
   G_eff = Analog_Gain * Digital_Gain * (t_exp / t_ref)
"""

# PARTICLE POSITIONS (x, y) in mm from dish center
# ------------------------------------------------
PARTICLES = [
    {"name": "Particle 1", "pos": (-12.0, -3.0),
     "status_in": "Blooming", "color": "#E74C3C"},

    {"name": "Particle 2", "pos": (8.0, -10.0),
     "status_in": "Defocus", "color": "#7F8C8D"},

    {"name": "Particle 3", "pos": (-20.0, -12.0),
     "status_in": "Valid", "color": "#922B21"}
]

# ============================================================
# MATH FUNCTIONS
# ============================================================


def solve_snell_xs(xp, yp, xl, yl, n_water=N_WATER, n_air=N_AIR):
    low = min(xp, xl) - 20.0
    high = max(xp, xl) + 20.0
    for _ in range(100):
        xs = (low + high) / 2.0
        sin_water = (xs - xp) / np.sqrt((xs - xp)**2 + yp**2)
        sin_air = (xl - xs) / np.sqrt((xl - xs)**2 + yl**2)

        val = n_water * sin_water - n_air * sin_air
        if abs(val) < 1e-6:
            return xs
        if val < 0:
            low = xs
        else:
            high = xs
    return (low + high) / 2.0


def calculate_fov_boundaries():
    surface_coverage = 37.3 * (CAMERA_CLEARANCE_MM / 40.0)
    floor_coverage = surface_coverage + \
        (FLUID_HEIGHT_MM * np.tan(np.arcsin(np.sin(np.radians(94.5/2)) / N_WATER)))
    return surface_coverage, floor_coverage

# ============================================================
# GRAPHICS GENERATOR (LEFT SETUP, RIGHT TEXT & MATH PANELS)
# ============================================================


def draw_dynamic_setup():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Main Diagram Title Text
    ax.text(0.5, 8.5, "Wide-Field Flood UV Fluorescence Fluid Imaging System Layout",
            fontsize=15, weight='bold', color='#2C3E50')

    # Setup positioning
    SETUP_CENTER_X = 4.25
    water_surface_y = 3.8
    dish_bottom_y = water_surface_y - (FLUID_HEIGHT_MM * 0.18)
    water_height = water_surface_y - dish_bottom_y

    def to_canvas(x_mm, y_mm):
        cx = SETUP_CENTER_X + x_mm * (3.5 / 60.0)
        if y_mm >= 0:
            cy = water_surface_y + y_mm * (2.4 / CAMERA_CLEARANCE_MM)
        else:
            cy = water_surface_y + y_mm * (water_height / FLUID_HEIGHT_MM)
        return cx, cy

    dish_half = PETRI_DIAMETER_MM / 2.0
    dish_wall_l, _ = to_canvas(-dish_half, 0)
    dish_wall_r, _ = to_canvas(dish_half, 0)
    dish_display_width = dish_wall_r - dish_wall_l

    # Draw Petri Dish & Fluid Column
    petri_dish = patches.Rectangle((dish_wall_l, dish_bottom_y), dish_display_width, water_height,
                                   linewidth=2.5, edgecolor='#2C3E50', facecolor='none', zorder=2)
    ax.add_patch(petri_dish)

    water = patches.Rectangle((dish_wall_l, dish_bottom_y), dish_display_width, water_height,
                              facecolor='#EBF5FB', alpha=0.9, zorder=1)
    ax.add_patch(water)

    ax.plot([dish_wall_l, dish_wall_r], [water_surface_y,
            water_surface_y], color='#2980B9', lw=3, zorder=3)
    ax.text(dish_wall_l + 0.1, water_surface_y + 0.15,
            f"Air-Water Interface (n = {N_WATER})", fontsize=9, color='#1F618D', weight='bold')

    # Snell's Law label at interface
    ax.text(dish_wall_r - 0.3, water_surface_y + 0.15,
            r"$n_{air}\sin\theta_{air}=n_{water}\sin\theta_{water}$",
            fontsize=7, color='#1F618D', ha='right', style='italic')

    # Draw Focal Plane
    _, cf_y = to_canvas(0, -FOCAL_DEPTH_MM)
    ax.plot([dish_wall_l, dish_wall_r], [cf_y, cf_y],
            color='#E67E22', lw=1.5, linestyle=':', zorder=3)
    ax.text(dish_wall_l - 0.2, cf_y,
            f"Focal Plane Depth ($y_f$) = -{FOCAL_DEPTH_MM} mm",
            fontsize=8, color='#D35400', va='center', ha='right')

    # Hardware Rendering (Imaging Sensor) with gain info
    cam_x, cam_y = to_canvas(0, CAMERA_CLEARANCE_MM)
    camera = patches.Rectangle((cam_x - 0.8, cam_y + 1.0), 1.6,
                               1.1, facecolor='#212F3D', edgecolor='black', zorder=4)
    ax.add_patch(camera)
    ax.text(cam_x, cam_y + 1.55, f"SENSOR PLANE\n512 x 512 Pixels\n"
            f"Gain: {ANALOG_GAIN_DB:.0f}dB | Exp: {EXPOSURE_TIME_MS:.0f}ms",
            color='white', fontsize=8, ha='center', va='center', weight='bold')

    lens = patches.Rectangle((cam_x - 0.5, cam_y), 1.0, 1.0,
                             facecolor='#4D5656', edgecolor='black', zorder=4)
    ax.add_patch(lens)
    ax.text(cam_x, cam_y + 0.5, f"LENS ASSEMBLY\nf/{F_NUMBER}, f={FOCAL_LENGTH_MM}mm\n"
            r"$D_{ap}$=" + f"{APERTURE_DIAMETER_MM:.1f}mm",
            color='white', fontsize=7, ha='center', va='center')

    # Wide-Field Emitter Array
    uv_source = patches.Rectangle((dish_wall_l, cam_y + 0.8), dish_display_width, 0.4,
                                  facecolor='#4A235A', edgecolor='#6C3483', zorder=4)
    ax.add_patch(uv_source)
    ax.text(SETUP_CENTER_X, cam_y + 1.0, "WIDE-FIELD UNIFORM UV OVERHEAD EMITTER ARRAY", color='white',
            fontsize=8, ha='center', va='center', weight='bold')

    # UV Wide Illumination Flood Cone
    uv_flood_pts = np.array([[dish_wall_l, cam_y + 0.8], [dish_wall_l - 0.3, dish_bottom_y],
                             [dish_wall_r + 0.3, dish_bottom_y], [dish_wall_r, cam_y + 0.8]])
    uv_flood = patches.Polygon(
        uv_flood_pts, facecolor='#9B59B6', alpha=0.10, zorder=1)
    ax.add_patch(uv_flood)
    ax.text(dish_wall_l + 0.15, dish_bottom_y + 0.15, "Full-Volume Flood UV Excitation Field",
            color='#6C3483', fontsize=8, weight='bold', style='italic')

    # Calculate Field of View limits
    surf_cov, floor_cov = calculate_fov_boundaries()
    fs_l_x, fs_l_y = to_canvas(-surf_cov, 0)
    fs_r_x, fs_r_y = to_canvas(surf_cov, 0)
    fb_l_x, fb_l_y = to_canvas(-floor_cov, -FLUID_HEIGHT_MM)
    fb_r_x, fb_r_y = to_canvas(floor_cov, -FLUID_HEIGHT_MM)

    # Visualizing the FOV Cone
    fov_poly_pts = np.array([[cam_x, cam_y], [fs_l_x, fs_l_y], [
                            fb_l_x, fb_l_y], [fb_r_x, fb_r_y], [fs_r_x, fs_r_y]])
    fov_cone = patches.Polygon(
        fov_poly_pts, facecolor='#F4D03F', alpha=0.07, zorder=1)
    ax.add_patch(fov_cone)

    ax.plot([cam_x, fs_l_x], [cam_y, fs_l_y],
            color='#F39C12', lw=1.2, linestyle='-.')
    ax.plot([cam_x, fs_r_x], [cam_y, fs_r_y],
            color='#F39C12', lw=1.2, linestyle='-.')
    ax.plot([fs_l_x, fb_l_x], [fs_l_y, fb_l_y],
            color='#D35400', lw=1.2, linestyle='-.')
    ax.plot([fs_r_x, fb_r_x], [fs_r_y, fb_r_y],
            color='#D35400', lw=1.2, linestyle='-.')

    # Hardware Dimension Lines
    if cam_y > water_surface_y + 0.5:
        ax.annotate('', xy=(dish_wall_r + 0.3, water_surface_y), xytext=(dish_wall_r + 0.3, cam_y),
                    arrowprops=dict(arrowstyle='<->', color='#7F8C8D', lw=1.2))
        ax.text(dish_wall_r + 0.4, (cam_y + water_surface_y)/2,
                f"Clearance\n$d_0 = {CAMERA_CLEARANCE_MM}$ mm",
                color='#7F8C8D', fontsize=8, ha='left', va='center', weight='bold')

    ax.annotate('', xy=(dish_wall_l - 0.3, dish_bottom_y), xytext=(dish_wall_l - 0.3, water_surface_y),
                arrowprops=dict(arrowstyle='<->', color='#2980B9', lw=1.2))
    ax.text(dish_wall_l - 0.4, (water_surface_y + dish_bottom_y)/2,
            f"Fluid Height\n$H = {FLUID_HEIGHT_MM}$ mm",
            color='#2980B9', fontsize=8, ha='right', va='center', weight='bold')

    # Process Particles
    particle_logs = []
    aperture_l_mm, aperture_r_mm = -4.0, 4.0
    ap_l_cx, ap_in_cy = to_canvas(aperture_l_mm, CAMERA_CLEARANCE_MM)
    ap_r_cx, _ = to_canvas(aperture_r_mm, CAMERA_CLEARANCE_MM)

    for idx, p in enumerate(PARTICLES):
        px, py = p["pos"]
        cx, cy = to_canvas(px, py)

        depth_fraction = abs(py) / FLUID_HEIGHT_MM
        allowed_radius_at_depth = surf_cov + \
            (floor_cov - surf_cov) * depth_fraction
        is_outside_fov = abs(px) > allowed_radius_at_depth
        coord_text = f"({px:+.1f}, {py:+.1f}) mm"

        if is_outside_fov:
            ax.scatter(cx, cy, color='#922B21', s=70,
                       edgecolor='black', zorder=6)
            avoid_box = dict(boxstyle='square,pad=0.2',
                             facecolor='#FADBD8', edgecolor='#E74C3C', lw=1)
            ax.text(cx, cy + 0.3, f"OUT\n{coord_text}\n[Outside FOV]", color='#C0392B',
                    fontsize=8, weight='bold', ha='center', va='bottom', bbox=avoid_box)

            xs_p_l = solve_snell_xs(px, py, aperture_l_mm, CAMERA_CLEARANCE_MM)
            s_l_cx, s_l_cy = to_canvas(xs_p_l, 0)
            ax.plot([cx, s_l_cx], [cy, s_l_cy], color='#95A5A6',
                    lw=1.0, linestyle='--', alpha=0.5)
            ax.plot([s_l_cx, cam_x + 0.4], [s_l_cy, ap_in_cy],
                    color='#C0392B', lw=1.1, linestyle=':', alpha=0.6)
            particle_logs.append(
                f"  • {p['name']} [{px}mm, {py}mm]: OUTSIDE FOV")
        else:
            if p["status_in"] == "Blooming":
                ax.scatter(cx, cy, color='#E74C3C',
                           s=750, alpha=0.15, zorder=5)
                ax.scatter(cx, cy, color='#E74C3C',
                           s=300, alpha=0.45, zorder=5)
                ax.plot([cx - 0.4, cx + 0.4], [cy, cy], color='#E74C3C',
                        lw=5, alpha=0.6, solid_capstyle='round', zorder=5)
                ax.scatter(cx, cy, color='#FFFFFF', s=50,
                           edgecolor='#C0392B', linewidth=1.2, zorder=6)
                ax.text(cx, cy - 0.3, f"{p['name']}\n{coord_text}\n[Saturated Blooming]",
                        fontsize=8, ha='center', va='top', weight='bold')

                ax.annotate('Leakage\n' + r'$S \cdot \eta_e \geq N_{fw}$', xy=(cx, cy), xytext=(cx - 1.1, cy - 1.2),
                            arrowprops=dict(facecolor='#C0392B',
                                            shrink=0.08, width=0.8, headwidth=4),
                            fontsize=7, color='#C0392B', weight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='#FADBD8', alpha=0.8))
            else:
                ax.scatter(cx, cy, color=p["color"], s=220,
                           edgecolor='black', alpha=0.5, zorder=6)
                ax.scatter(cx, cy, color='black', s=12, zorder=6)
                ax.text(cx + 0.2, cy, f"{p['name']}\n{coord_text}\n[Heavy Defocus Blur]",
                        fontsize=8, va='center', weight='bold')

                ax.annotate(r'$D_b \propto \Delta y \cdot NA$', xy=(cx, cy), xytext=(cx + 0.8, cy - 0.8),
                            arrowprops=dict(facecolor='#7F8C8D',
                                            shrink=0.08, width=0.8, headwidth=4),
                            fontsize=7, color='#34495E', weight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='#EAEDED', alpha=0.8))

            xs_l_mm = solve_snell_xs(
                px, py, aperture_l_mm, CAMERA_CLEARANCE_MM)
            xs_r_mm = solve_snell_xs(
                px, py, aperture_r_mm, CAMERA_CLEARANCE_MM)
            surf_l_cx, surf_l_cy = to_canvas(xs_l_mm, 0)
            surf_r_cx, surf_r_cy = to_canvas(xs_r_mm, 0)

            cone_pts = np.array([[cx, cy], [surf_l_cx, surf_l_cy], [ap_l_cx, ap_in_cy], [
                                ap_r_cx, ap_in_cy], [surf_r_cx, surf_r_cy]])
            ax.add_patch(patches.Polygon(
                cone_pts, facecolor='#2ECC71', alpha=0.11, zorder=2))
            ax.plot([cx, surf_l_cx], [cy, surf_l_cy], color='#27AE60',
                    lw=1.0, linestyle='--', alpha=0.6, zorder=3)
            ax.plot([cx, surf_r_cx], [cy, surf_r_cy], color='#27AE60',
                    lw=1.0, linestyle='--', alpha=0.6, zorder=3)
            particle_logs.append(
                f"  • {p['name']} [{px}mm, {py}mm]: FOV Inside -> {p['status_in']}")

    # ============================================================
    # INFORMATION PANELS ON THE RIGHT HAND SIDE
    # ============================================================
    RIGHT_PANEL_X = 9.4

    # Calculate derived values for display
    analog_gain_linear = 10**(ANALOG_GAIN_DB / 20)
    effective_dr = 20 * \
        np.log10(FULL_WELL_ELECTRONS / READ_NOISE_ELECTRONS) - ANALOG_GAIN_DB
    light_collection = 1.0 / (F_NUMBER**2)  # Relative to f/1.0

    # Panel 1: System Parameters & Captured Metrics
    box_style = dict(boxstyle='round,pad=0.6',
                     facecolor='#FDFEFE', edgecolor='#BDC3C7')
    table_text = (
        "EXPERIMENTAL SETUP PARAMETERS\n\n"
        f"1. Petri Dish:\n"
        f"  • D = {PETRI_DIAMETER_MM} mm | H = {FLUID_HEIGHT_MM} mm\n"
        f"  • Volume = {np.pi*(PETRI_DIAMETER_MM/2)**2*FLUID_HEIGHT_MM/1000:.1f} mL\n\n"
        f"2. Optical Configuration:\n"
        f"  • Clearance: d₀ = {CAMERA_CLEARANCE_MM} mm\n"
        f"  • Lens: f = {FOCAL_LENGTH_MM} mm, f/{F_NUMBER}\n"
        f"  • Aperture: D_ap = {APERTURE_DIAMETER_MM:.2f} mm\n"
        f"  • Mag: M ≈ {FOCAL_LENGTH_MM/(CAMERA_CLEARANCE_MM + FOCAL_DEPTH_MM/N_WATER - FOCAL_LENGTH_MM):.3f}\n\n"
        f"3. Camera Settings:\n"
        f"  • Analog Gain: {ANALOG_GAIN_DB:.0f} dB ({analog_gain_linear:.2f}×)\n"
        f"  • Digital Gain: {DIGITAL_GAIN:.1f}×\n"
        f"  • Exposure: {EXPOSURE_TIME_MS:.0f} ms\n"
        f"  • QE: {QUANTUM_EFFICIENCY*100:.0f}% | FWC: {FULL_WELL_ELECTRONS:,} e⁻\n"
        f"  • Read Noise: {READ_NOISE_ELECTRONS:.1f} e⁻ RMS\n"
        f"  • Eff. DR: {effective_dr:.1f} dB\n\n"
        f"4. Light Collection:\n"
        f"  • Relative: {light_collection:.3f} (vs f/1.0)\n"
        f"  • FOV Surface: ±{surf_cov:.1f} mm\n"
        f"  • FOV Floor: ±{floor_cov:.1f} mm"
    )
    ax.text(RIGHT_PANEL_X, 8.1, table_text, fontsize=9,
            bbox=box_style, va='top', linespacing=1.2)

    # Panel 2: Camera Gain & Signal Chain Physics
    formula_box = dict(boxstyle='round,pad=0.6',
                       facecolor='#FBFCFC', edgecolor='#34495E', lw=1.5)
    optical_formulas = (
        "CAMERA GAIN & SIGNAL CHAIN PHYSICS\n\n"
        "1. Analog Gain (Voltage Amplification):\n"
        r"   $G_{linear} = 10^{Gain_{dB}/20}$" + "\n"
        f"   Current: {ANALOG_GAIN_DB:.0f} dB = {analog_gain_linear:.2f}×\n\n"
        "2. Exposure Integration:\n"
        r"   $S = \int_0^{t_{exp}} I(t) \cdot QE \cdot dt$" + "\n"
        r"   $SNR \propto \sqrt{t_{exp}}$ (shot noise limited)\n\n"
        "3. Total Signal Chain:\n"
        r"   $ADU = clip(G_{analog} \cdot G_{dig} \cdot S / k, 0, 4095)$" + "\n"
        "   where k = conversion gain [e⁻/ADU]\n\n"
        "4. Effective Dynamic Range:\n"
        r"   $DR_{eff} = 20\log_{10}(\frac{FWC}{N_{read}}) - Gain_{dB}$" + "\n"
        f"   Current: {effective_dr:.1f} dB\n\n"
        "5. Noise Model:\n"
        r"   $\sigma_{total} = \sqrt{\sigma_{shot}^2 + (\sigma_{read} \cdot G)^2}$"
    )
    ax.text(RIGHT_PANEL_X, 5.0, optical_formulas, fontsize=9,
            bbox=formula_box, va='top', color='#2C3E50')

    # Panel 3: Saturated Blooming Under Flood UV Illumination
    blooming_box = dict(boxstyle='round,pad=0.6',
                        facecolor='#FDF2E9', edgecolor='#E67E22', lw=1.5)
    blooming_formulas = (
        "FLOOD-LIT SENSOR BLOOMING MATHEMATICS\n\n"
        "1. Wide-Field Integrated Photon Signal (S):\n"
        r"   $S = (I_0 \cdot \sigma_{fluor}) \cdot \Omega_{coll} \cdot t_{exposure}$" + "\n"
        "   where $I_0$ = Full Overhead Array UV Source Irradiance\n\n"
        "2. Full-Well Over-Exposure Limit:\n"
        r"   Condition for Bloom Bleed: $S \cdot \eta_e \geq N_{fw}$" + "\n\n"
        "3. Lateral Charge Saturation Transfer:\n"
        r"   $I_{bleed} = -D_e \cdot \frac{\partial^2 Q}{\partial x^2} \longrightarrow$ Horizontal Blooming Trace"
    )
    ax.text(RIGHT_PANEL_X, 2.5, blooming_formulas, fontsize=9,
            bbox=blooming_box, va='top', color='#7E5109')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    draw_dynamic_setup()

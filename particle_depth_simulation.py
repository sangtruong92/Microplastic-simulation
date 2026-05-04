"""
Underwater Particle Depth Simulation
=====================================
Simulates how a particle appears in a camera image
as it moves from near the water surface (5mm) to deep water (40mm).

Physics:
  - Refraction: apparent_depth = real_depth / N_WATER
  - Object distance: d_o = CAMERA_DISTANCE_MM + apparent_depth
  - Magnification (exact): M = f / (d_o - f)
  - Image size: size_px = (real_size_um * M) / pixel_size_um
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ===========================================================
# 0. OUTPUT DIRECTORY (works on any OS)
# ===========================================================
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Outputs will be saved to: {OUTPUT_DIR}")

# ===========================================================
# 1. CONSTANTS
# ===========================================================
N_WATER             = 1.33      # Refractive index of water
FOCAL_LENGTH_MM     = 6.0       # Camera focal length (mm)
PIXEL_SIZE_UM       = 3.45      # Sensor pixel size (micrometers)
CAMERA_DISTANCE_MM  = 50.0      # Distance from camera lens to water surface (mm)
PARTICLE_SIZE_UM    = 100.0     # Real particle diameter (micrometers)
IMAGE_SHAPE         = (256, 256) # (height, width) in pixels

DEPTH_START_MM      = 5.0       # Shallowest depth (mm)
DEPTH_END_MM        = 40.0      # Deepest depth (mm)
DEPTH_STEP_MM       = 5.0       # Step size (mm)

# ===========================================================
# 2. CORE FUNCTIONS
# ===========================================================

def apparent_depth(real_depth_mm):
    """
    Apply Snell's law refraction to compute apparent depth.
    Formula: apparent_depth = real_depth / n_water
    Water bends light so the particle appears shallower than it is.
    """
    return real_depth_mm / N_WATER


def compute_magnification(depth_mm,
                           focal_length_mm=FOCAL_LENGTH_MM,
                           camera_dist_mm=CAMERA_DISTANCE_MM):
    """
    Exact thin-lens magnification accounting for refraction.

    Steps:
      1. Convert real depth to apparent depth (refraction)
      2. d_o = camera_dist + apparent_depth  (total object distance)
      3. M  = f / (d_o - f)                 (thin-lens magnification)
    """
    d_o = camera_dist_mm + apparent_depth(depth_mm)
    M   = focal_length_mm / (d_o - focal_length_mm)
    return M


def micrometers_to_pixels(size_um, depth_mm,
                           focal_length_mm=FOCAL_LENGTH_MM,
                           pixel_size_um=PIXEL_SIZE_UM,
                           camera_dist_mm=CAMERA_DISTANCE_MM):
    """
    Convert real particle diameter (µm) to apparent size in pixels.

    Steps:
      1. Get magnification M at this depth
      2. image_size_um = real_size_um * M
      3. size_px = image_size_um / pixel_size_um

    Returns (size_px, M)
    """
    M             = compute_magnification(depth_mm, focal_length_mm, camera_dist_mm)
    image_size_um = size_um * M
    size_px       = image_size_um / pixel_size_um
    return size_px, M


def create_particle(center_x, center_y, size_px, image_shape):
    """
    Generate a binary circular mask representing a particle.

    Parameters
    ----------
    center_x, center_y : float  — particle centre in pixels
    size_px            : float  — particle diameter in pixels
    image_shape        : tuple  — (height, width)

    Returns
    -------
    mask : np.ndarray (float), shape = image_shape
           1.0 inside the circle, 0.0 outside
    """
    h, w   = image_shape
    y, x   = np.ogrid[:h, :w]
    radius = size_px / 2.0
    dist   = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask   = (dist <= radius).astype(float)
    return mask


# ===========================================================
# 3. DEPTH SCAN  (5 mm to 40 mm, step 5 mm)
# ===========================================================

depth_range = np.arange(DEPTH_START_MM, DEPTH_END_MM + 1e-9, DEPTH_STEP_MM)

results = []
print(f"\n{'Depth (mm)':<12} {'App.Depth (mm)':<17} {'Magnif.':<12} {'Size (px)'}")
print("-" * 56)

for depth_mm in depth_range:
    size_px, M = micrometers_to_pixels(PARTICLE_SIZE_UM, depth_mm)
    app_d      = apparent_depth(depth_mm)

    cx   = IMAGE_SHAPE[1] // 2
    cy   = IMAGE_SHAPE[0] // 2
    mask = create_particle(cx, cy, size_px, IMAGE_SHAPE)

    results.append({
        "depth_mm"      : depth_mm,
        "apparent_mm"   : app_d,
        "magnification" : M,
        "size_px"       : size_px,
        "mask"          : mask,
    })

    print(f"{depth_mm:<12.1f} {app_d:<17.3f} {M:<12.6f} {size_px:.4f}")


# ===========================================================
# 4. FIGURE 1 — Particle Images at Each Depth
# ===========================================================

n    = len(results)
cols = 4
rows = int(np.ceil(n / cols))

fig1, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
axes = axes.flatten()

for i, r in enumerate(results):
    ax = axes[i]
    ax.imshow(r["mask"], cmap="inferno", vmin=0, vmax=1)
    ax.set_title(
        f"Depth: {r['depth_mm']:.0f} mm\n"
        f"Size : {r['size_px']:.2f} px\n"
        f"M    = {r['magnification']:.5f}",
        fontsize=8
    )
    ax.axis("off")

    # Dashed cyan circle overlay showing exact particle boundary
    cx     = IMAGE_SHAPE[1] // 2
    cy     = IMAGE_SHAPE[0] // 2
    circle = plt.Circle((cx, cy), r["size_px"] / 2,
                         color="cyan", fill=False,
                         linewidth=1.2, linestyle="--")
    ax.add_patch(circle)

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

fig1.suptitle(
    f"Particle (O {PARTICLE_SIZE_UM:.0f} um) -- Depth "
    f"{DEPTH_START_MM:.0f} to {DEPTH_END_MM:.0f} mm",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "particle_grid.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Saved: particle_grid.png")


# ===========================================================
# 5. FIGURE 2 — Analysis Plots (3 subplots)
# ===========================================================

depths = [r["depth_mm"]      for r in results]
sizes  = [r["size_px"]       for r in results]
mags   = [r["magnification"] for r in results]
app_ds = [r["apparent_mm"]   for r in results]

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

# --- (a) Apparent depth vs real depth ---
ax = axes2[0]
ax.plot(depths, app_ds, "go-", linewidth=2, markersize=7)
ax.set_xlabel("Real Depth in Water (mm)", fontsize=11)
ax.set_ylabel("Apparent Depth (mm)", fontsize=11)
ax.set_title("Apparent Depth vs Real Depth", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)
for d, a in zip(depths, app_ds):
    ax.annotate(f"{a:.1f}", (d, a), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8)

# --- (b) Magnification vs depth ---
ax = axes2[1]
ax.plot(depths, mags, "bo-", linewidth=2, markersize=7)
ax.set_xlabel("Real Depth in Water (mm)", fontsize=11)
ax.set_ylabel("Magnification", fontsize=11)
ax.set_title("Magnification vs Depth", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)
for d, m in zip(depths, mags):
    ax.annotate(f"{m:.4f}", (d, m), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=7)

# --- (c) Apparent pixel size vs depth ---
ax = axes2[2]
ax.plot(depths, sizes, "rs-", linewidth=2, markersize=7)
ax.set_xlabel("Real Depth in Water (mm)", fontsize=11)
ax.set_ylabel("Apparent Particle Size (px)", fontsize=11)
ax.set_title("Particle Size (px) vs Depth", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)
for d, s in zip(depths, sizes):
    ax.annotate(f"{s:.2f}px", (d, s), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8)

fig2.suptitle("Depth Scan Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "depth_analysis.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Saved: depth_analysis.png")


# ===========================================================
# 6. FIGURE 3 — Exact vs Approximate Magnification Error
# ===========================================================

depth_fine   = np.linspace(DEPTH_START_MM, DEPTH_END_MM, 200)
M_exact_arr  = []
M_approx_arr = []
error_arr    = []

for d in depth_fine:
    d_o = CAMERA_DISTANCE_MM + apparent_depth(d)
    M_e = FOCAL_LENGTH_MM / (d_o - FOCAL_LENGTH_MM)   # exact thin-lens
    M_a = FOCAL_LENGTH_MM / d_o                        # approximate (d_o >> f)
    M_exact_arr.append(M_e)
    M_approx_arr.append(M_a)
    error_arr.append(abs(M_e - M_a) / M_e * 100)

fig3, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))

ax_l.plot(depth_fine, M_exact_arr,  "b-",  linewidth=2.5,
          label="Exact: M = f / (d_o - f)")
ax_l.plot(depth_fine, M_approx_arr, "r--", linewidth=2.5,
          label="Approx: M = f / d_o")
ax_l.fill_between(depth_fine, M_exact_arr, M_approx_arr,
                  alpha=0.2, color="orange")
ax_l.set_xlabel("Depth (mm)", fontsize=11)
ax_l.set_ylabel("Magnification", fontsize=11)
ax_l.set_title("Exact vs Approximate Magnification", fontsize=12, fontweight="bold")
ax_l.legend(fontsize=10)
ax_l.grid(True, alpha=0.3)

ax_r.plot(depth_fine, error_arr, color="purple", linewidth=2.5)
ax_r.fill_between(depth_fine, 0, error_arr, alpha=0.25, color="purple")
ax_r.axhline(10, color="red", linestyle="--", linewidth=1.5,
             label="10% error threshold")
ax_r.set_xlabel("Depth (mm)", fontsize=11)
ax_r.set_ylabel("Error (%)", fontsize=11)
ax_r.set_title("Approximation Error (%)", fontsize=12, fontweight="bold")
ax_r.legend(fontsize=10)
ax_r.grid(True, alpha=0.3)

avg_err = np.mean(error_arr)
max_err = np.max(error_arr)
ax_r.text(0.05, 0.95,
          f"Avg error: {avg_err:.3f}%\nMax error: {max_err:.3f}%",
          transform=ax_r.transAxes, fontsize=9,
          verticalalignment="top",
          bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

fig3.suptitle("Formula Accuracy Check", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "formula_error.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Saved: formula_error.png")


# ===========================================================
# 7. SUMMARY
# ===========================================================
print("\n" + "="*56)
print("SIMULATION SUMMARY")
print("="*56)
print(f"  Particle diameter   : {PARTICLE_SIZE_UM} um")
print(f"  Focal length        : {FOCAL_LENGTH_MM} mm")
print(f"  Pixel size          : {PIXEL_SIZE_UM} um")
print(f"  Camera distance     : {CAMERA_DISTANCE_MM} mm")
print(f"  Depth range         : {DEPTH_START_MM} - {DEPTH_END_MM} mm "
      f"(step {DEPTH_STEP_MM} mm)")
print(f"  Refractive index    : {N_WATER}")
print(f"  Formula avg error   : {avg_err:.4f}%")
print(f"  Formula max error   : {max_err:.4f}%")
print(f"  Output folder       : {OUTPUT_DIR}")
print("="*56)

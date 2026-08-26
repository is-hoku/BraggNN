"""
Configuration Roofline Model — BraggNN conv2 layer on Gemmini
================================================================
Plots the theoretical configuration roofline (Eq. 2: concurrent,
Eq. 3: sequential, from "The Configuration Wall", ASPLOS'26) and
marks conv2's operating point on it.

All values here are THEORETICAL (no hardware/simulator measurements):
  - P_Peak    = 512 ops/cycle       (16x16 systolic array, 1 MAC = 2 ops)
  - BW_Config = 16/9 bytes/cycle    (paper's Eq. from Gemmini RoCC config,
                                      derived as 16 bytes / (2 instr * 3 cyc/instr) --
                                      actually 16/(3*3) per the paper's own footnote 4
                                      approximation: 3 RoCC-related instructions,
                                      3 cycles/instruction)
  - conv2 config bytes/call = 77    (Table 1 field-width sum: 4 addresses (64b) +
                                      I/J/K sizes (16b x3) + pad_I/J/K (16b x3) +
                                      4 strides (64b) + act (6b) + transpose (1b x2),
                                      rounded to whole 16-byte RoCC register writes)
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Theoretical constants (paper-derived, Section 4 & 4.6)
# ---------------------------------------------------------------
P_PEAK    = 512.0            # ops/cycle
BW_CONFIG = 16.0 / (3.0 * 3.0)   # bytes/cycle  (theoretical estimate, paper Eq. in §4.6)

# ---------------------------------------------------------------
# conv2 kernel data (BraggNN, from braggnn_tune.c / braggnn.h)
#   9x9x64 input -> 3x3x64 kernel -> 32 output channels -> 7x7x32 output
# ---------------------------------------------------------------
CONV2_OPS       = 1_806_336          # 2 * 3*3 * 64*32 * 7*7
CONV2_CFG_BYTES = 112.0               # Table-1-derived config bytes per macro-op call
CONV2_I_OC      = CONV2_OPS / CONV2_CFG_BYTES   # ops/byte

# ---------------------------------------------------------------
# Roofline formulas
# ---------------------------------------------------------------
def p_sequential(i_oc):
    """Eq. 3: sequential configuration roofline (Gemmini is sequential)."""
    return 1.0 / (1.0 / P_PEAK + 1.0 / (BW_CONFIG * i_oc))

# ---------------------------------------------------------------
# Build the roofline curves over a wide I_OC range (log-spaced)
# ---------------------------------------------------------------
i_oc_range = np.logspace(0, 6, 500)   # 1 to 1e6 ops/byte
p_seq  = p_sequential(i_oc_range)

# conv2's own attainable performance under each model
conv2_p_seq  = p_sequential(CONV2_I_OC)

# Knee point: where BW_CONFIG * I_OC == P_PEAK
knee_i_oc = P_PEAK / BW_CONFIG

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(i_oc_range, p_seq, color='black', linewidth=1, linestyle='-')

# Peak performance ceiling (dashed horizontal reference)
ax.axhline(P_PEAK, color='gray', linewidth=1, linestyle=':', alpha=0.7)
#ax.text(1.3, P_PEAK * 1.03, r'$P_{Peak}=512$ ops/cycle', color='gray', fontsize=9)

# Knee point marker
#ax.plot(knee_i_oc, P_PEAK, 'o', color='black', markersize=6)
#ax.annotate('knee point', xy=(knee_i_oc, P_PEAK),
#            xytext=(knee_i_oc * 1.3, P_PEAK * 0.75),
#            fontsize=9, color='dimgray',
#            arrowprops=dict(arrowstyle='->', color='dimgray', lw=0.8))

# conv2 operating point
ax.plot(CONV2_I_OC, conv2_p_seq, marker='o', color='red', markersize=8,
        zorder=5, label=f'conv2 (Seq.), $P_{{Attainable}}$={conv2_p_seq:.1f}')

#ax.annotate(
#    f'conv2\n$I_{{OC}}$={CONV2_I_OC:,.0f} ops/byte',
#    xy=(CONV2_I_OC, conv2_p_seq),
#    xytext=(CONV2_I_OC * 0.15, conv2_p_seq * 0.55),
#    fontsize=9, color='blue',
#    arrowprops=dict(arrowstyle='->', color='blue', lw=0.8)
#)

# Axes formatting (log-log, matching paper's Figure 4/12 style)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$I_{OC}$ (ops/byte)', fontsize=12)
ax.set_ylabel(r'$P$ (ops/cycle)', fontsize=12)
#ax.set_title('Configuration Roofline — BraggNN conv2 on Gemmini (theoretical)',
#             fontsize=12)
ax.set_xlim(1, 1e6)
ax.set_ylim(1, 1000)
ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.6)
#ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig('./conv2_roofline.png', dpi=150)
print('Saved plot to ./conv2_roofline.png')

# ---------------------------------------------------------------
# Print numeric summary
# ---------------------------------------------------------------
print()
print('=== conv2 Configuration Roofline Summary (theoretical) ===')
print(f'P_Peak                = {P_PEAK} ops/cycle')
print(f'BW_Config             = {BW_CONFIG:.6f} bytes/cycle')
print(f'conv2 ops             = {CONV2_OPS:,}')
print(f'conv2 config bytes    = {CONV2_CFG_BYTES:.0f} bytes/call')
print(f'conv2 I_OC            = {CONV2_I_OC:,.4f} ops/byte')
print(f'BW_Config x I_OC      = {BW_CONFIG * CONV2_I_OC:,.4f} ops/cycle')
print(f'P_Attainable (Seq.)   = {conv2_p_seq:.4f} ops/cycle')
print(f'Knee point I_OC       = {knee_i_oc:.4f} ops/byte')


# CONV1

CONV1_OPS       = 93_312
CONV1_CFG_BYTES = 112.0               # Table-1-derived config bytes per macro-op call
CONV1_I_OC      = CONV1_OPS / CONV1_CFG_BYTES   # ops/byte

# ---------------------------------------------------------------
# Roofline formulas
# ---------------------------------------------------------------
def p_sequential(i_oc):
    """Eq. 3: sequential configuration roofline (Gemmini is sequential)."""
    return 1.0 / (1.0 / P_PEAK + 1.0 / (BW_CONFIG * i_oc))

# ---------------------------------------------------------------
# Build the roofline curves over a wide I_OC range (log-spaced)
# ---------------------------------------------------------------
i_oc_range = np.logspace(0, 6, 500)   # 1 to 1e6 ops/byte
p_seq  = p_sequential(i_oc_range)

# conv2's own attainable performance under each model
conv1_p_seq  = p_sequential(CONV1_I_OC)

# Knee point: where BW_CONFIG * I_OC == P_PEAK
knee_i_oc = P_PEAK / BW_CONFIG

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(i_oc_range, p_seq, color='black', linewidth=1, linestyle='-')

# Peak performance ceiling (dashed horizontal reference)
ax.axhline(P_PEAK, color='gray', linewidth=1, linestyle=':', alpha=0.7)
#ax.text(1.3, P_PEAK * 1.03, r'$P_{Peak}=512$ ops/cycle', color='gray', fontsize=9)

# Knee point marker
#ax.plot(knee_i_oc, P_PEAK, 'o', color='black', markersize=6)
#ax.annotate('knee point', xy=(knee_i_oc, P_PEAK),
#            xytext=(knee_i_oc * 1.3, P_PEAK * 0.75),
#            fontsize=9, color='dimgray',
#            arrowprops=dict(arrowstyle='->', color='dimgray', lw=0.8))

# conv2 operating point
ax.plot(CONV1_I_OC, conv1_p_seq, marker='o', color='red', markersize=8,
        zorder=5, label=f'conv1 (Seq.), $P_{{Attainable}}$={conv1_p_seq:.1f}')

#ax.annotate(
#    f'conv2\n$I_{{OC}}$={CONV2_I_OC:,.0f} ops/byte',
#    xy=(CONV2_I_OC, conv2_p_seq),
#    xytext=(CONV2_I_OC * 0.15, conv2_p_seq * 0.55),
#    fontsize=9, color='blue',
#    arrowprops=dict(arrowstyle='->', color='blue', lw=0.8)
#)

# Axes formatting (log-log, matching paper's Figure 4/12 style)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$I_{OC}$ (ops/byte)', fontsize=12)
ax.set_ylabel(r'$P$ (ops/cycle)', fontsize=12)
#ax.set_title('Configuration Roofline — BraggNN conv2 on Gemmini (theoretical)',
#             fontsize=12)
ax.set_xlim(1, 1e6)
ax.set_ylim(1, 1000)
ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.6)
#ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig('./conv1_roofline.png', dpi=150)
print('Saved plot to ./conv1_roofline.png')

# ---------------------------------------------------------------
# Print numeric summary
# ---------------------------------------------------------------
print()
print('=== conv1 Configuration Roofline Summary (theoretical) ===')
print(f'P_Peak                = {P_PEAK} ops/cycle')
print(f'BW_Config             = {BW_CONFIG:.6f} bytes/cycle')
print(f'conv2 ops             = {CONV1_OPS:,}')
print(f'conv2 config bytes    = {CONV1_CFG_BYTES:.0f} bytes/call')
print(f'conv2 I_OC            = {CONV1_I_OC:,.4f} ops/byte')
print(f'BW_Config x I_OC      = {BW_CONFIG * CONV1_I_OC:,.4f} ops/cycle')
print(f'P_Attainable (Seq.)   = {conv1_p_seq:.4f} ops/cycle')
print(f'Knee point I_OC       = {knee_i_oc:.4f} ops/byte')


# FC4
FC4_OPS       = 16
FC4_CFG_BYTES = 96.0               # Table-1-derived config bytes per macro-op call
FC4_I_OC      = FC4_OPS / FC4_CFG_BYTES   # ops/byte

# ---------------------------------------------------------------
# Roofline formulas
# ---------------------------------------------------------------
def p_sequential(i_oc):
    return 1.0 / (1.0 / P_PEAK + 1.0 / (BW_CONFIG * i_oc))

# ---------------------------------------------------------------
# Build the roofline curves over a wide I_OC range (log-spaced)
# ---------------------------------------------------------------
i_oc_range = np.logspace(0, 6, 500)   # 1 to 1e6 ops/byte
p_seq  = p_sequential(i_oc_range)

# conv2's own attainable performance under each model
fc4_p_seq  = p_sequential(FC4_I_OC)

# Knee point: where BW_CONFIG * I_OC == P_PEAK
knee_i_oc = P_PEAK / BW_CONFIG

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(i_oc_range, p_seq, color='black', linewidth=1, linestyle='-')

# Peak performance ceiling (dashed horizontal reference)
ax.axhline(P_PEAK, color='gray', linewidth=1, linestyle=':', alpha=0.7)
#ax.text(1.3, P_PEAK * 1.03, r'$P_{Peak}=512$ ops/cycle', color='gray', fontsize=9)

# Knee point marker
#ax.plot(knee_i_oc, P_PEAK, 'o', color='black', markersize=6)
#ax.annotate('knee point', xy=(knee_i_oc, P_PEAK),
#            xytext=(knee_i_oc * 1.3, P_PEAK * 0.75),
#            fontsize=9, color='dimgray',
#            arrowprops=dict(arrowstyle='->', color='dimgray', lw=0.8))

# conv2 operating point
ax.plot(FC4_I_OC, fc4_p_seq, marker='o', color='red', markersize=8,
        zorder=5, label=f'fc4 (Seq.), $P_{{Attainable}}$={fc4_p_seq:.1f}')

#ax.annotate(
#    f'conv2\n$I_{{OC}}$={CONV2_I_OC:,.0f} ops/byte',
#    xy=(CONV2_I_OC, conv2_p_seq),
#    xytext=(CONV2_I_OC * 0.15, conv2_p_seq * 0.55),
#    fontsize=9, color='blue',
#    arrowprops=dict(arrowstyle='->', color='blue', lw=0.8)
#)

# Axes formatting (log-log, matching paper's Figure 4/12 style)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$I_{OC}$ (ops/byte)', fontsize=12)
ax.set_ylabel(r'$P$ (ops/cycle)', fontsize=12)
#ax.set_title('Configuration Roofline — BraggNN conv2 on Gemmini (theoretical)',
#             fontsize=12)
ax.set_xlim(1, 1e6)
ax.set_ylim(1, 1000)
ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.6)
#ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig('./fc4_roofline.png', dpi=150)
print('Saved plot to ./fc4_roofline.png')

# ---------------------------------------------------------------
# Print numeric summary
# ---------------------------------------------------------------
print()
print('=== fc4 Configuration Roofline Summary (theoretical) ===')
print(f'P_Peak                = {P_PEAK} ops/cycle')
print(f'BW_Config             = {BW_CONFIG:.6f} bytes/cycle')
print(f'conv2 ops             = {FC4_OPS:,}')
print(f'conv2 config bytes    = {FC4_CFG_BYTES:.0f} bytes/call')
print(f'conv2 I_OC            = {FC4_I_OC:,.4f} ops/byte')
print(f'BW_Config x I_OC      = {BW_CONFIG * FC4_I_OC:,.4f} ops/cycle')
print(f'P_Attainable (Seq.)   = {fc4_p_seq:.4f} ops/cycle')
print(f'Knee point I_OC       = {knee_i_oc:.4f} ops/byte')

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains Jupyter notebooks for analyzing Philips Vereos PET scanner list mode data. All notebooks are designed to run in **Google Colab**, not locally. The project uses AI-assisted iterative development — notebooks document the user's prompts and AI-generated code side by side.

## Running Notebooks

These notebooks run exclusively in Google Colab:
- Open any notebook via the "Open in Colab" badge at the top of each notebook
- Data files are uploaded interactively via `files.upload()` or mounted from Google Drive
- No local build or test commands apply

## Repository Structure

Two types of input data files are used:

**Binary `.list` files** — raw Vereos PET scanner list mode acquisition files (big-endian binary):
- `3OCLK_2bd6.list`, `Center_Emiss_4794.list` — small sample files checked into the repo

**ListPrint `.txt` files** — human-readable output from the Philips `listPrint` utility:
- `src_at_bottom_24934.txt`, `3OCLK_0_1000.txt`, `listPrint_Randoms2min_250k.txt`
- Each event line format: `N: PROMPT: xa(n) xb(n) tof(n) za(n) zb(n) lea(n) leb(n)` or same with `DELAY:`

**Notebooks by purpose:**
- `MJS_AnthropicAI_PET_List_dataframe_20250525.ipynb` — Parse listPrint .txt → pandas DataFrame, histograms, circular ring plots (Claude Sonnet 4)
- `MJS_OpenAI_04_mini_List_file_Headers_20250525.ipynb` — Parse binary .list file headers (o4-mini)
- `MJS_Gemini_TOF_List_data_20250525.ipynb` — Parse binary .list event data, signed TOF decoding (Gemini)
- `MJS_AI_PET_Ring_Circular_Visualization_revD/revD2/revG/revH_*.ipynb` — Iterative improvements to circular polar ring visualization
- `MJS_Sonnet4_Randoms_list_data_*.ipynb` — 3D heatmaps and TOF analysis of random coincidences (latest work)

## Domain Knowledge

### Vereos PET Ring Geometry
| Parameter | Value |
|-----------|-------|
| Axial FOV | 164 mm |
| Ring Diameter | 764 mm |
| Crystal Size | 4 × 4 × 19 mm |
| Total detector elements | 23,040 |
| Timing Resolution | 320 ps |
| Coincidence Window | 256 FOV: 2.0 ns; 576 FOV: 4.0 ns; 676 FOV: 4.6 ns |
| Ring circumference | 764 mm × π ≈ 2,400 mm → 4.17 mm between crystal centers |
| Reconstruction diameter | 764 mm + 19 mm = 783 mm |

- **Ring layout**: 18 modules × 4 tiles across × 8 pixels/tile = **576 crystals** around circumference (IDs 0–575)
- **Axial layout**: 5 tiles deep × 8 pixels/tile = **40 crystals** in axial direction (za, zb values 0–39)
- Crystal 0 is at **3 o'clock (0°)**; crystal IDs increase **clockwise**
- Angular position: `theta = crystal_id × 2π / 576`
- Module number: `module = crystal_id // 32`
- Light travels **299.8 mm per nanosecond**; max 576 FOV LOR length ≈ 764 mm → ≈ 2,548 ps flight time

### LOR Data Fields
| Field | Description |
|-------|-------------|
| `xa`, `xb` | Transaxial crystal pair (circumference); should be ~180° apart for true events |
| `za`, `zb` | Axial crystal position for the same two detectors |
| `tof` | Signed Time of Flight bin (negative = xa receives photon first) |
| `PROMPT` | True coincidence event |
| `DELAY` | Random coincidence event (delayed window) |

### Coincidence Event Types
| Type | Definition |
|------|------------|
| **Singles** | Individual gamma rays detected by any element, regardless of coincidence |
| **Prompt** | All events within the timing window — includes true + random coincidences; measured Coincidence Rate = Prompt Rate |
| **True** | Two 511 keV gammas from the same annihilation, detected by opposing detectors within the coincidence window |
| **Random** | Two gammas from unrelated annihilations detected within the coincidence window (DELAY events in list data) |
| **Scatter** | One or both gammas have undergone Compton scattering before detection, causing mispositioning |

### Key Physics
- For a **centered source**: `xb ≈ (xa + 288) % 576` (exact 180°, i.e., 288 crystals = half ring)
- For an **offset source**: LOR pairs are non-diametric; shorter LORs are more frequent near the source
- TOF is negative when the source is farther from xb (xb receives the photon later)
- `tofTstampScale` ≈ 19.53 ps/bin; multiply TOF bin by this to get picoseconds
- Coincidence half-window (`c_h_w_b`) limits max TOF bin for valid coincidences (e.g., 119 bins × 19.5 ps ≈ 2,320 ps)

### Parsing listPrint .txt Files
```python
pattern = r'xa\((\d+)\) xb\((\d+)\) tof\(([-]?\d+)\) za\((\d+)\) zb\((\d+)\)'
# Filter lines: ('PROMPT:' in line or 'DELAY:' in line) and 'xa(' in line
```

### Binary `.list` File Structure (big-endian)
| Offset | Size | Content |
|--------|------|---------|
| 0 | 512 B | Listview header (`magicNumber='Xtal'`, `listType=16`) |
| 512 | 8704 B | Main header (17 records × 512 B) |
| 9216 | 512 B | Sub header |
| **9744** | remainder | **List data** (event and control word pairs) |

Key header fields:
- `tofTstampScale` (float): offset `0x3C` in main header → ps per TOF bin
- `c_h_w_b` (uint16): offset `0x07D0` in main header → coincidence half-window bins

### Binary Event Word Format (each event = two 32-bit words)
**Word 1** (bits 31:30 = `00`):
- Bit 29: delay flag (1=delayed, 0=prompt)
- Bit 27: TOF sign bit
- Bits 26:20: TOF magnitude (0–127)
- Bits 19:10: xb (10 bits)
- Bits 9:0: xa (10 bits)

**Word 2** (bits 31:30 = `01`):
- Bits 29:23: zb (7 bits)
- Bits 22:16: za (7 bits)
- Bits 15:8: leb (local energy B)
- Bits 7:0: lea (local energy A)

**Control words**: bits 31:30 = `10` (word 1) or `11` (word 2); word 1 contains cwt (bits 29:27) and cwinfo (bits 26:0).

Signed TOF: `signed_tof = tof_mag if sign_bit == 0 else -tof_mag`

## Standard Libraries Used
`pandas`, `numpy`, `matplotlib` (including polar/3D), `plotly`, `re`, `struct`, `google.colab.files`

## Circular Plot Coordinate Convention (settled in revG/revH)
```python
# Crystal ID → polar angle (radians), clockwise from 3 o'clock
theta = crystal_id * 2 * np.pi / 576
ax.set_theta_direction(-1)   # clockwise
ax.set_theta_offset(0)       # 0 rad = 3 o'clock (Module 0)
ax.set_thetalim(0, 2*np.pi)  # ensure full 360°
```
Module boundary lines every 32 crystals; label key modules (0, 3, 6, 9, 12, 15).

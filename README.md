# bnct-response-matrix-model
Supplementary material for "An improved response matrix derivation model for fast calculation of neutron moderation spectra"
# Supplementary Material: An improved response matrix derivation model for fast calculation of neutron moderation spectra

## File descriptions

- `input_spectrum_2e7.csv`: 300-group absolute neutron counts (sum = 2e7), used as the source spectrum for validation simulations.
- `gen_base_matrices.py`: OpenMC script to generate transmission matrix A and reflection matrix B for AlF₃ moderator at thicknesses 1 cm and 5 cm (monoenergetic scan, 300 energy groups, 2e5 particles per group). Outputs: `A1.csv`, `B1.csv`, `A5.csv`, `B5.csv`.
- `validate_spectra.py`: OpenMC script to simulate exit spectra for thicknesses 1–25 cm using the continuous input spectrum (2e7 particles). Outputs: `ALF_{d}cm.csv` for each thickness.
- `optimize_X_multi_material.py`: Python script that:
  - Loads base matrices and reference spectra
  - Optimizes the 10-parameter block‑diagonal correction matrix X using differential evolution
  - Evaluates the corrected and uncorrected models
  - Generates comparison plots (EPS vector format) for AlF₃, MgF₂, and BiF₃ (black: MC reference, green: uncorrected, red: corrected)
- `requirements.txt` (optional): List of Python dependencies.

## Dependencies

- OpenMC (for Monte Carlo simulations)
- Python 3.12+
- NumPy, SciPy, pandas, matplotlib
- scikit-learn (for R² calculation)

## Usage

1. Run `gen_base_matrices.py` to generate base matrices A1, B1, A5, B5 for AlF₃.
2. Run `validate_spectra.py` to generate reference spectra for thicknesses 1–25 cm.
3. Run `optimize_X_multi_material.py` and select the material (1: AlF₃, 2: MgF₂, 3: BiF₃). The script will:
   - Optimize X for the selected material
   - Save response matrices, predicted spectra, and summary tables (R², NMSE)
   - Generate EPS comparison plots

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


# 补充材料：用于快算中子慢化能谱的改进响应矩阵推导模型

## 文件说明

- `input_spectrum_2e7.csv`：300 群绝对中子计数（总和为 2e7），用作验证模拟的源能谱。
- `gen_base_matrices.py`：OpenMC 脚本，用于生成 AlF₃ 慢化体在厚度 1 cm 和 5 cm 时的透射矩阵 A 和反射矩阵 B（单能扫描，300 能群，每能群 2e5 粒子）。输出文件：`A1.csv`, `B1.csv`, `A5.csv`, `B5.csv`。
- `validate_spectra.py`：OpenMC 脚本，使用连续输入能谱（2e7 粒子）模拟厚度 1–25 cm 的出射能谱。输出文件：各厚度的 `ALF_{d}cm.csv`。
- `optimize_X_multi_material.py`：Python 脚本，功能包括：
  - 加载基矩阵和参考能谱
  - 使用差分进化算法优化 10 参数的分块对角校正矩阵 X
  - 评估校正模型与未校正模型的精度
  - 生成 AlF₃、MgF₂、BiF₃ 三种材料的对比图（EPS 矢量格式，黑线：MC 参考，绿线：未校正，红线：校正）
- `requirements.txt`（可选）：Python 依赖包列表。

## 依赖环境

- OpenMC（用于蒙特卡洛模拟）
- Python 3.12+
- NumPy、SciPy、pandas、matplotlib
- scikit-learn（用于计算 R²）

## 使用说明

1. 运行 `gen_base_matrices.py` 生成 AlF₃ 的基矩阵 A1、B1、A5、B5。
2. 运行 `validate_spectra.py` 生成厚度 1–25 cm 的参考能谱。
3. 运行 `optimize_X_multi_material.py`，根据提示选择材料（1: AlF₃, 2: MgF₂, 3: BiF₃）。脚本将：
   - 对所选材料优化校正矩阵 X
   - 保存响应矩阵、预测能谱和评估表格（R²、NMSE）
   - 生成 EPS 格式的对比图
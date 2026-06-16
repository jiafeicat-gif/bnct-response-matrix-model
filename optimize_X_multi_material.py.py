"""
优化调整矩阵 X（分块对角形式）- 二阶递推 + 层间修正（等权重版）
支持多种材料：AlF₃, MGF, BIF
用户通过输入选择材料，自动加载对应路径的基矩阵和参考能谱
增加功能：输出三条线对比图（蒙卡参考、无校正、有校正）的 EPS 矢量图
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import os
import matplotlib.pyplot as plt
import time

# ====================== 材料配置 ======================
MATERIALS = {
    "1": {
        "name": "AlF₃",
        "base_dir": r"D:/材料库/ALF",
        "output_dir": r"D:/ALF/反射/优化结果_二阶_层间X_DE_等权重",
        "output_dir_1cm": r"D:/ALF/反射/优化结果_1cm基",
    },
    "2": {
        "name": "MGF",
        "base_dir": r"D:/材料库/MGF",
        "output_dir": r"D:/MGF/反射/优化结果_二阶_层间X_DE_等权重",
        "output_dir_1cm": r"D:/MGF/反射/优化结果_1cm基",
    },
    "3": {
        "name": "BIF",
        "base_dir": r"D:/材料库/BIF",
        "output_dir": r"D:/BIF/反射/优化结果_二阶_层间X_DE_等权重",
        "output_dir_1cm": r"D:/BIF/反射/优化结果_1cm基",
    }
}

INPUT_FILE = r"D:/ALF/优化调整矩阵/实际能谱.csv"

BOUNDS = [(0.7, 1.3) for _ in range(10)]
POPSIZE = 30
MAXITER = 1000
SEED = 42
TOL = 1e-6

def read_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.lower().endswith('.csv'):
        return pd.read_csv(file_path, header=None).values
    else:
        return pd.read_excel(file_path, header=None).values

def build_X(params):
    X = np.zeros((300, 300))
    for i in range(10):
        X[i*30:(i+1)*30, i*30:(i+1)*30] = np.eye(30) * params[i]
    return X

def compute_Rd(A, B, X, d, base_thickness):
    if d == base_thickness:
        return A.copy()
    steps = (d - base_thickness) // base_thickness
    T = A.copy()
    R = B.copy()
    for _ in range(steps):
        T_next = (T @ X @ A
                  + T @ X @ B @ X @ R @ X @ A
                  + T @ X @ B @ X @ R @ X @ B @ X @ R @ X @ A)
        R_next = (R
                  + T @ X @ B @ X @ T
                  + T @ X @ B @ X @ R @ X @ B @ X @ T)
        T, R = T_next, R_next
    return T

def compute_R2(S_in, Rd, S_out_meas):
    S_calc = S_in @ Rd
    ss_res = np.sum((S_out_meas - S_calc)**2)
    ss_tot = np.sum((S_out_meas - np.mean(S_out_meas))**2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1 - ss_res / ss_tot

def objective(params, A, B, S_in, S_out_list, thicknesses, base_thickness):
    X = build_X(params)
    r2_list = []
    for d, meas in zip(thicknesses, S_out_list):
        Rd = compute_Rd(A, B, X, d, base_thickness)
        r2 = compute_R2(S_in, Rd, meas)
        r2_list.append(r2)
    avg_r2 = np.mean(r2_list)
    return -avg_r2

def optimize_and_evaluate(A, B, S_in, S_out_list, thicknesses, base_thickness, output_dir, label, material_name):
    """通用优化评估函数，并绘制对比图（EPS矢量图）"""
    print(f"\n========== 优化 {label} ==========")
    result = differential_evolution(
        objective,
        BOUNDS,
        args=(A, B, S_in, S_out_list, thicknesses, base_thickness),
        maxiter=MAXITER,
        popsize=POPSIZE,
        tol=TOL,
        seed=SEED,
        disp=True
    )
    opt_params = result.x
    X_opt = build_X(opt_params)
    print(f"最优 {label} 参数: {opt_params}")
    print(f"最佳平均 R²: {-result.fun:.6f}")

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame([opt_params]).to_csv(os.path.join(output_dir, f"optimized_X_{label}_params.csv"), index=False)
    pd.DataFrame(X_opt).to_csv(os.path.join(output_dir, f"optimized_X_{label}.csv"), index=False, header=False)

    results = []
    I = np.eye(300)  # 单位矩阵，用于无校正模型

    for d, meas in zip(thicknesses, S_out_list):
        # 有校正
        Rd_corr = compute_Rd(A, B, X_opt, d, base_thickness)
        S_calc_corr = (S_in @ Rd_corr).flatten()
        # 无校正 (X = I)
        Rd_no_corr = compute_Rd(A, B, I, d, base_thickness)
        S_calc_no_corr = (S_in @ Rd_no_corr).flatten()

        # 评估指标
        r2 = compute_R2(S_in, Rd_corr, meas)
        total_meas = np.sum(meas)
        total_calc = np.sum(S_calc_corr)
        rel_diff = (total_calc - total_meas) / total_meas * 100
        results.append([d, r2, total_meas, total_calc, rel_diff])
        print(f"Thickness {d} cm: R² = {r2:.6f}, Total meas={total_meas:.2e}, calc={total_calc:.2e}, diff={rel_diff:+.2f}%")

        # 保存响应矩阵和预测谱（有校正）
        pd.DataFrame(Rd_corr).to_csv(os.path.join(output_dir, f"Rd_{d}cm_{label}.csv"), index=False, header=False)
        pd.DataFrame([S_calc_corr]).to_csv(os.path.join(output_dir, f"pred_{d}cm_{label}.csv"), index=False, header=False)

        # ========== 绘制三条线对比图（EPS） ==========
        meas_flat = meas.flatten()
        bins = np.arange(1, len(meas_flat) + 1)  # 能群索引

        plt.figure(figsize=(10, 6))
        plt.plot(bins, meas_flat, 'k-', linewidth=1.5, label='Monte Carlo reference')
        plt.plot(bins, S_calc_no_corr, 'g--', linewidth=1.5, label='Uncorrected model (X=I)')
        plt.plot(bins, S_calc_corr, 'r-', linewidth=1.5, label='Corrected model (optimized X)')
        plt.xlabel('Energy group index')
        plt.ylabel('Neutron counts (absolute)')
        plt.title(f'{material_name}, thickness = {d} cm')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # 保存为 EPS 矢量图
        eps_file = os.path.join(output_dir, f"{material_name}_{d}cm_comparison.eps")
        plt.savefig(eps_file, format='eps', bbox_inches='tight')
        plt.close()
        print(f"已保存对比图: {eps_file}")

    df = pd.DataFrame(results, columns=["Thickness(cm)","R²","Total_meas","Total_calc","Rel_diff(%)"])
    df.to_csv(os.path.join(output_dir, f"summary_{label}.csv"), index=False)
    return X_opt, results

def get_file_paths(base_dir, abbr):
    paths = {
        "A5": os.path.join(base_dir, f"{abbr}_A5.csv"),
        "B5": os.path.join(base_dir, f"{abbr}_B5.csv"),
        "A1": os.path.join(base_dir, f"{abbr}_A1.csv"),
        "B1": os.path.join(base_dir, f"{abbr}_B1.csv"),
    }
    thick_5_25 = [5,10,15,20,25]
    thick_1_5 = [1,2,3,4,5]
    for d in thick_5_25:
        paths[f"ref_{d}"] = os.path.join(base_dir, f"{abbr}_{d}cm.csv")
    for d in thick_1_5:
        if d not in paths:
            paths[f"ref_{d}"] = os.path.join(base_dir, f"{abbr}_{d}cm.csv")
    return paths

def main():
    print("请选择材料：")
    print("1. AlF₃")
    print("2. MGF")
    print("3. BIF")
    choice = input("输入编号 (1/2/3): ").strip()
    if choice not in MATERIALS:
        print("无效选择，退出")
        return

    mat = MATERIALS[choice]
    base_dir = mat["base_dir"]
    # 获取材料名称（用于图标题）
    material_name = mat["name"]
    # 材料缩写（假设文件命名与名称对应，去掉下标）
    abbr = material_name.replace("₃", "").upper()  # 例如 ALF, MGF, BIF

    paths = get_file_paths(base_dir, abbr)

    required = ["A5", "B5", "A1", "B1"] + [f"ref_{d}" for d in [5,10,15,20,25,1,2,3,4]]
    missing = [p for p in required if not os.path.exists(paths.get(p, ""))]
    if missing:
        print(f"缺少文件: {missing}")
        print("请确保以下文件存在：")
        for m in missing:
            print(f"  {paths.get(m)}")
        return

    S_in = read_file(INPUT_FILE)
    if S_in.shape == (300,):
        S_in = S_in.reshape(1, -1)
    elif S_in.shape == (300, 1):
        S_in = S_in.T
    elif S_in.shape != (1, 300):
        raise ValueError(f"输入能谱形状应为 (1,300)，实际为 {S_in.shape}")

    # 5cm 基
    A5 = read_file(paths["A5"])
    B5 = read_file(paths["B5"])
    thick5 = [5,10,15,20,25]
    S_out5 = [read_file(paths[f"ref_{d}"]).reshape(1, -1) for d in thick5]
    output_dir_5cm = mat["output_dir"]
    optimize_and_evaluate(A5, B5, S_in, S_out5, thick5, 5, output_dir_5cm, "5cm_base", material_name)

    # 1cm 基
    A1 = read_file(paths["A1"])
    B1 = read_file(paths["B1"])
    thick1 = [1,2,3,4,5]
    S_out1 = [read_file(paths[f"ref_{d}"]).reshape(1, -1) for d in thick1]
    output_dir_1cm = mat["output_dir_1cm"]
    optimize_and_evaluate(A1, B1, S_in, S_out1, thick1, 1, output_dir_1cm, "1cm_base", material_name)

    print(f"\n所有结果已保存到 {output_dir_5cm} 和 {output_dir_1cm}")
    print("对比图已以 EPS 格式保存。")

if __name__ == "__main__":
    main()
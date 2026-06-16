import openmc
import numpy as np
import os
import csv
import warnings
warnings.filterwarnings('ignore')

# ======================== 全局配置 ========================
INPUT_SPECTRUM_PATH = "/mnt/d/ALF/实际值/能谱降级/实际能谱_降级2e7.csv"
# 厚度列表（单位：cm）
THICKNESS_LIST = [1, 2, 3, 4, 5, 10, 15, 20, 25]

# ======================== 材料定义 ========================
MATERIALS_DB = [
    {"name": "氟化铝", "abbr": "ALF", "density": 2.88,  "elements": [('Al', 1.0), ('F', 3.0)], "sab": None},
    {"name": "石墨",   "abbr": "C",   "density": 2.0,   "elements": [('C', 1.0)],               "sab": "c_Graphite"},
    {"name": "氟化镁", "abbr": "MGF", "density": 3.18,  "elements": [('Mg', 1.0), ('F', 2.0)],   "sab": None},
    {"name": "聚乙烯", "abbr": "PE",  "density": 0.93,  "elements": [('C', 1.0), ('H', 2.0)],    "sab": "c_H_in_CH2"},
    {"name": "铅",     "abbr": "PB",  "density": 11.35, "elements": [('Pb', 1.0)],               "sab": None},
    {"name": "氟化铋", "abbr": "BIF", "density": 8.3,   "elements": [('Bi', 1.0), ('F', 3.0)],   "sab": None},
    {"name": "水",     "abbr": "H2O", "density": 1.0,   "elements": [('H', 2.0), ('O', 1.0)],    "sab": "c_H_in_H2O"},
]

# ======================== 能量网格 ========================
ENERGY_MIN = 1e-3   # eV
ENERGY_MAX = 1e7    # eV
ENERGY_POINT_COUNT = 301
LOG_E = np.linspace(np.log10(ENERGY_MIN), np.log10(ENERGY_MAX), ENERGY_POINT_COUNT)
ENERGY_BINS = 10 ** LOG_E          # 301个边界
ENERGY_FILTER_BINS = ENERGY_BINS   # tally用的能量过滤
SOURCE_ENERGIES = ENERGY_BINS[1:]  # 300个源能量点（使用每个bin的上边界）

# ======================== 材料构建函数 ========================
def create_materials(mat_info):
    mats_lib = {}
    # 真空
    vacuum = openmc.Material(name='Vacuum')
    vacuum.set_density('g/cm3', 1e-10)
    vacuum.add_nuclide('H1', 1e-20)
    mats_lib['真空'] = vacuum

    # 慢化材料
    mat = openmc.Material(name=mat_info["name"])
    mat.set_density("g/cm3", mat_info["density"])
    for elem, ratio in mat_info["elements"]:
        if elem in ('H', 'H2', 'H1'):
            mat.add_nuclide('H1', ratio)
        else:
            mat.add_element(elem, ratio)
    if mat_info.get("sab"):
        try:
            mat.add_s_alpha_beta(mat_info["sab"])
        except Exception as e:
            print(f"⚠️  {mat_info['name']} 的 S(α,β) 加载失败: {e}")
    mats_lib['慢化材料'] = mat

    # 铅屏蔽
    lead = openmc.Material(name='Lead')
    lead.add_element('Pb', 1, 'wo')
    lead.set_density('g/cm3', 11.35)
    mats_lib['铅'] = lead

    # 空气
    air = openmc.Material(name='Air')
    air.add_element('N', 0.755, 'wo')
    air.add_element('O', 0.232, 'wo')
    air.add_element('Ar', 0.013, 'wo')
    air.set_density('g/cm3', 0.001)
    mats_lib['空气'] = air

    return list(mats_lib.values()), mats_lib

# ======================== 几何构建 ========================
def create_geometry(materials_dict, thickness):
    main_mat = materials_dict['慢化材料']
    lead_mat = materials_dict['铅']
    air_mat = materials_dict['空气']
    geometry = openmc.Geometry()

    x_min = openmc.XPlane(x0=0.0)
    x_max = openmc.XPlane(x0=thickness)

    y_inner_min = openmc.YPlane(y0=-25.0)
    y_inner_max = openmc.YPlane(y0=25.0)
    z_inner_min = openmc.ZPlane(z0=-25.0)
    z_inner_max = openmc.ZPlane(z0=25.0)
    y_outer_min = openmc.YPlane(y0=-55.0)
    y_outer_max = openmc.YPlane(y0=55.0)
    z_outer_min = openmc.ZPlane(z0=-55.0)
    z_outer_max = openmc.ZPlane(z0=55.0)
    vacuum_x_min = openmc.XPlane(x0=-100.0, boundary_type='vacuum')
    vacuum_x_max = openmc.XPlane(x0=100.0, boundary_type='vacuum')
    vacuum_y_min = openmc.YPlane(y0=-100.0, boundary_type='vacuum')
    vacuum_y_max = openmc.YPlane(y0=100.0, boundary_type='vacuum')
    vacuum_z_min = openmc.ZPlane(z0=-100.0, boundary_type='vacuum')
    vacuum_z_max = openmc.ZPlane(z0=100.0, boundary_type='vacuum')

    inner_region = +x_min & -x_max & +y_inner_min & -y_inner_max & +z_inner_min & -z_inner_max
    lead_region = (+x_min & -x_max & +y_outer_min & -y_outer_max & +z_outer_min & -z_outer_max) & ~inner_region
    vacuum_region = (+vacuum_x_min & -vacuum_x_max & +vacuum_y_min & -vacuum_y_max & +vacuum_z_min & -vacuum_z_max) & ~lead_region & ~inner_region

    inner_cell = openmc.Cell(name="inner_moderator", fill=main_mat, region=inner_region)
    outer_cell = openmc.Cell(name="outer_lead", fill=lead_mat, region=lead_region)
    vacuum_cell = openmc.Cell(name="vacuum", fill=air_mat, region=vacuum_region)

    root_universe = openmc.Universe(cells=[inner_cell, outer_cell, vacuum_cell])
    geometry.root_universe = root_universe
    return geometry, x_max

# ======================== 源定义 ========================
def create_spectrum_source():
    """读取输入能谱，返回混合源和总中子数"""
    if not os.path.exists(INPUT_SPECTRUM_PATH):
        raise FileNotFoundError(f"输入能谱文件不存在：{INPUT_SPECTRUM_PATH}")

    # 读取CSV，假设只有一行300个数值
    spectrum_abs = np.loadtxt(INPUT_SPECTRUM_PATH, delimiter=',').flatten()
    if len(spectrum_abs) != 300:
        raise ValueError(f"输入能谱长度应为300，实际为{len(spectrum_abs)}")

    total_source = spectrum_abs.sum()
    prob = spectrum_abs / total_source

    source = openmc.Source()
    source.space = openmc.stats.Box((-1, -25, -25), (-1, 25, 25))
    source.energy = openmc.stats.Discrete(x=SOURCE_ENERGIES, p=prob)
    source.angle = openmc.stats.Monodirectional((1.0, 0.0, 0.0))
    source.particle = 'neutron'

    return source, total_source

# ======================== 计数卡 ========================
def create_tallies(exit_surface):
    surface_filter = openmc.SurfaceFilter(exit_surface)
    energy_filter = openmc.EnergyFilter(ENERGY_FILTER_BINS)
    neutron_filter = openmc.ParticleFilter("neutron")

    tally = openmc.Tally(name="outgoing_spectrum")
    tally.filters = [surface_filter, energy_filter, neutron_filter]
    tally.scores = ["current"]
    return openmc.Tallies([tally])

# ======================== 文件清理 ========================
def clean_simulation_files():
    clean_files = [
        "materials.xml", "geometry.xml", "settings.xml", "tallies.xml",
        "summary.h5", "fort.14", "fort.60", "fort.70", "fort.80", "fort.90",
        "statepoint.1.h5", "model.xml"
    ]
    for f in clean_files:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception as e:
                print(f"⚠️  清理文件 {f} 失败: {e}")

# ======================== 交互菜单 ========================
def select_material():
    print("\n" + "=" * 50)
    print("请选择慢化材料：")
    for i, mat in enumerate(MATERIALS_DB, 1):
        print(f"  {i}. {mat['name']} ({mat['abbr']})")
    print("=" * 50)
    while True:
        try:
            choice = int(input("输入编号 (1-7): "))
            if 1 <= choice <= len(MATERIALS_DB):
                return MATERIALS_DB[choice - 1]
        except ValueError:
            pass
        print("无效输入，请重新选择。")

# ======================== 单次模拟 ========================
def run_simulation(mat_info, thickness):
    abbr = mat_info["abbr"]
    thick_str = str(thickness)
    output_filename = f"{abbr}_{thick_str}cm.csv"

    print(f"\n▶ 厚度：{thickness} cm")
    print(f"  输出文件：{output_filename}")

    # 准备材料、几何、源
    materials_list, materials_dict = create_materials(mat_info)
    geometry, exit_surface = create_geometry(materials_dict, thickness)

    # 创建源（会计算总中子数）
    source, total_source = create_spectrum_source()
    total_particles = int(round(total_source))
    print(f"  输入能谱总中子数：{total_source:.0f}，模拟粒子数：{total_particles}")

    # 构建模型
    model = openmc.Model()
    model.materials = materials_list
    model.geometry = geometry
    model.settings.run_mode = "fixed source"
    model.settings.particles = total_particles
    model.settings.batches = 1
    model.settings.inactive = 0
    model.settings.source = source
    model.settings.photon_transport = False
    model.tallies = create_tallies(exit_surface)

    # 清理旧文件
    clean_simulation_files()

    # 运行
    print("  模拟运行中...", end=" ")
    model.run(output=False)
    print("完成")

    # 提取结果
    sp = openmc.StatePoint("statepoint.1.h5")
    tally = sp.get_tally(name="outgoing_spectrum")
    flux_per_source = tally.mean.flatten()  # 每个源中子的平均电流
    sp.close()

    # 转换为绝对中子数
    flux_absolute = flux_per_source * total_particles

    # 保存为1行300列
    np.savetxt(output_filename, [flux_absolute], delimiter=',', fmt='%.6e')
    print(f"  ✅ 已保存：{output_filename} (shape: 1x300)")
    print(f"  输出总中子数：{flux_absolute.sum():.0f}")

# ======================== 主函数 ========================
def main():
    print("=" * 60)
    print("   BNCT 输出能谱模拟（交互式选择材料）")
    print(f"   输入能谱：{INPUT_SPECTRUM_PATH}")
    print("=" * 60)

    mat_info = select_material()
    print(f"\n▶ 当前材料：{mat_info['name']} ({mat_info['abbr']})")
    print(f"▶ 将依次模拟 {len(THICKNESS_LIST)} 种厚度")

    for thick in THICKNESS_LIST:
        try:
            run_simulation(mat_info, thick)
        except Exception as e:
            print(f"  ❌ 厚度 {thick}cm 失败：{e}")
            continue

    print(f"\n🎉 {mat_info['name']} 全部模拟完成！")

if __name__ == "__main__":
    main()
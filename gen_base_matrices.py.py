import openmc
import numpy as np
import os
import csv
import warnings
warnings.filterwarnings('ignore')

# ======================== 配置类 ========================
class BNCTConfig:
    PARTICLES = 100000
    BATCHES = 2
    INACTIVE_BATCHES = 0

    ENERGY_MIN = 1e-3
    ENERGY_MAX = 1e7
    ENERGY_POINT_COUNT = 301
    LOG_E = np.linspace(np.log10(ENERGY_MIN), np.log10(ENERGY_MAX), ENERGY_POINT_COUNT)
    ENERGY_BINS = 10 ** LOG_E
    ENERGY_FILTER_BINS = ENERGY_BINS
    SOURCE_ENERGIES = ENERGY_BINS[1:]

# ======================== 材料库 ========================
MATERIALS_DB = [
    {"name": "氟化铝", "abbr": "ALF", "density": 2.88,  "elements": [('Al', 1.0), ('F', 3.0)], "sab": None},
    {"name": "石墨",   "abbr": "C",   "density": 2.0,   "elements": [('C', 1.0)],               "sab": "c_Graphite"},
    {"name": "氟化镁", "abbr": "MGF", "density": 3.18,  "elements": [('Mg', 1.0), ('F', 2.0)],   "sab": None},
    {"name": "聚乙烯", "abbr": "PE",  "density": 0.93,  "elements": [('C', 1.0), ('H', 2.0)],    "sab": "c_H_in_CH2"},
    {"name": "铅",     "abbr": "PB",  "density": 11.35, "elements": [('Pb', 1.0)],               "sab": None},
    {"name": "氟化铋", "abbr": "BIF", "density": 8.3,   "elements": [('Bi', 1.0), ('F', 3.0)],   "sab": None},
    {"name": "水",     "abbr": "H2O", "density": 1.0,   "elements": [('H', 2.0), ('O', 1.0)],    "sab": "c_H_in_H2O"},
]

def create_materials(mat_info):
    mats_lib = {}
    vacuum = openmc.Material(name='Vacuum')
    vacuum.set_density('g/cm3', 1e-10)
    vacuum.add_nuclide('H1', 1e-20)
    mats_lib['真空'] = vacuum

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

    lead = openmc.Material(name='Lead')
    lead.add_element('Pb', 1, 'wo')
    lead.set_density('g/cm3', 11.35)
    mats_lib['铅'] = lead

    air = openmc.Material(name='Air')
    air.add_element('N', 0.755, 'wo')
    air.add_element('O', 0.232, 'wo')
    air.add_element('Ar', 0.013, 'wo')
    air.set_density('g/cm3', 0.001)
    mats_lib['空气'] = air

    return list(mats_lib.values()), mats_lib

# ======================== 几何 ========================
def create_geometry(materials_dict, thickness):
    main_mat = materials_dict['慢化材料']
    lead_mat = materials_dict['铅']
    air_mat = materials_dict['空气']
    geometry = openmc.Geometry()

    x_entry = openmc.XPlane(x0=0.0, boundary_type='transmission')
    x_exit = openmc.XPlane(x0=thickness)

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

    inner_region = +x_entry & -x_exit & +y_inner_min & -y_inner_max & +z_inner_min & -z_inner_max
    lead_region = (+x_entry & -x_exit & +y_outer_min & -y_outer_max & +z_outer_min & -z_outer_max) & ~inner_region
    vacuum_region = (+vacuum_x_min & -vacuum_x_max & +vacuum_y_min & -vacuum_y_max & +vacuum_z_min & -vacuum_z_max) & ~lead_region & ~inner_region

    inner_cell = openmc.Cell(name="inner_moderator", fill=main_mat, region=inner_region)
    outer_cell = openmc.Cell(name="outer_lead", fill=lead_mat, region=lead_region)
    vacuum_cell = openmc.Cell(name="vacuum", fill=air_mat, region=vacuum_region)

    root_universe = openmc.Universe(cells=[inner_cell, outer_cell, vacuum_cell])
    geometry.root_universe = root_universe
    return geometry, x_entry, x_exit

# ======================== 源 ========================
def create_source(energy_eV):
    source = openmc.Source()
    source.space = openmc.stats.Box((-1, -25, -25), (-1, 25, 25))
    source.energy = openmc.stats.Discrete(x=[energy_eV], p=[1.0])
    source.angle = openmc.stats.Monodirectional((1.0, 0.0, 0.0))
    source.particle = 'neutron'
    return source

# ======================== 计数卡 ========================
def create_tallies(entry_surface, exit_surface):
    energy_filter = openmc.EnergyFilter(BNCTConfig.ENERGY_FILTER_BINS)
    neutron_filter = openmc.ParticleFilter("neutron")

    tally_reflect = openmc.Tally(name="Reflect_Tally")
    tally_reflect.filters = [openmc.SurfaceFilter(entry_surface), energy_filter, neutron_filter]
    tally_reflect.scores = ["current"]

    tally_transmit = openmc.Tally(name="Transmit_Tally")
    tally_transmit.filters = [openmc.SurfaceFilter(exit_surface), energy_filter, neutron_filter]
    tally_transmit.scores = ["current"]

    return openmc.Tallies([tally_reflect, tally_transmit])

# ======================== 工具函数 ========================
def clean_simulation_files():
    clean_files = [
        "materials.xml", "geometry.xml", "settings.xml", "tallies.xml",
        "summary.h5", "fort.14", "fort.60", "fort.70", "fort.80", "fort.90",
        f"statepoint.{BNCTConfig.BATCHES}.h5", "model.xml"
    ]
    for f in clean_files:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception as e:
                print(f"⚠️  清理文件 {f} 失败: {e}")

def extract_rows():
    sp = openmc.StatePoint(f"statepoint.{BNCTConfig.BATCHES}.h5")
    tally_ref = sp.get_tally(name="Reflect_Tally")
    current_ref = tally_ref.mean.flatten()
    reflect_prob = np.where(current_ref < 0, -current_ref, 0.0)

    tally_trans = sp.get_tally(name="Transmit_Tally")
    current_trans = tally_trans.mean.flatten()
    transmit_prob = np.where(current_trans > 0, current_trans, 0.0)

    sp.close()
    return reflect_prob.tolist(), transmit_prob.tolist()

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

# ======================== 单厚度模拟 ========================
def run_thickness(mat_info, thickness):
    thick_str = str(int(thickness))
    file_a = f"{mat_info['abbr']}_A{thick_str}.csv"
    file_b = f"{mat_info['abbr']}_B{thick_str}.csv"

    print(f"\n{'='*50}")
    print(f"  厚度 = {thickness} cm  ({thick_str})")
    print(f"{'='*50}")

    materials_list, materials_dict = create_materials(mat_info)
    geometry, entry_surf, exit_surf = create_geometry(materials_dict, thickness)
    tallies = create_tallies(entry_surf, exit_surf)

    matrix_a = []
    matrix_b = []

    for i, energy_eV in enumerate(BNCTConfig.SOURCE_ENERGIES, 1):
        disp = f"{energy_eV:.3e} eV" if energy_eV < 1e6 else f"{energy_eV/1e6:.3f} MeV"
        print(f"  [{i:03d}/300] {disp}", end=" ")
        clean_simulation_files()

        try:
            source = create_source(energy_eV)
            model = openmc.Model()
            model.materials = materials_list
            model.geometry = geometry
            model.settings.run_mode = "fixed source"
            model.settings.particles = BNCTConfig.PARTICLES
            model.settings.batches = BNCTConfig.BATCHES
            model.settings.inactive = BNCTConfig.INACTIVE_BATCHES
            model.settings.source = source
            model.settings.statepoint = {"batches": [BNCTConfig.BATCHES]}
            model.tallies = tallies
            model.settings.photon_transport = False

            model.export_to_xml()
            model.run()

            ref_row, trans_row = extract_rows()
            matrix_b.append(ref_row)
            matrix_a.append(trans_row)
            print("✓")
        except Exception as e:
            print(f"✗ 失败: {e}")
            matrix_b.append([0.0]*300)
            matrix_a.append([0.0]*300)

    for fname, mat in [(file_a, matrix_a), (file_b, matrix_b)]:
        with open(fname, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(mat)
        print(f"  ✅ 已保存: {fname} ({len(mat)}×{len(mat[0])})")

# ======================== 主程序 ========================
def main():
    print("=" * 60)
    print("   BNCT 响应矩阵模拟器 —— 同时获取穿透矩阵(A)和反射矩阵(B)")
    print("   厚度：1cm + 5cm | 粒子数：100,000 | 批次：2")
    print("=" * 60)

    mat_info = select_material()
    print(f"\n▶ 当前材料：{mat_info['name']} ({mat_info['abbr']})")
    print(f"▶ 将依次模拟 1cm 和 5cm 两种厚度")

    run_thickness(mat_info, 1.0)
    run_thickness(mat_info, 5.0)

    print(f"\n{'='*60}")
    print(f"🎉 {mat_info['name']} 全部完成！")
    print(f"   生成文件：{mat_info['abbr']}_A1.csv / {mat_info['abbr']}_B1.csv")
    print(f"            {mat_info['abbr']}_A5.csv / {mat_info['abbr']}_B5.csv")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
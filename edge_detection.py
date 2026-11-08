import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os
import random
import time
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 固定种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ================= 1. 边缘检测算子 =================
def _to_uint8(grad):
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)
    return grad.astype(np.uint8)

def edge_sobel(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return _to_uint8(cv2.magnitude(gx, gy))

def edge_prewitt(gray):
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    return _to_uint8(cv2.magnitude(cv2.filter2D(gray, cv2.CV_32F, kx), cv2.filter2D(gray, cv2.CV_32F, ky)))

def edge_roberts(gray):
    kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
    ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    return _to_uint8(cv2.magnitude(cv2.filter2D(gray, cv2.CV_32F, kx), cv2.filter2D(gray, cv2.CV_32F, ky)))

def edge_laplacian(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return _to_uint8(np.abs(lap))

def edge_log(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    return _to_uint8(np.abs(lap))

def edge_canny_low(gray):
    return cv2.Canny(gray, 20, 60)

def edge_canny_high(gray):
    return cv2.Canny(gray, 100, 200)

# ================= 2. 定量指标计算 =================
def get_edge_stats(edge_img, is_binary=False):
    thr = 0 if is_binary else 50
    edge_px = (edge_img > thr).sum()
    density = edge_px / edge_img.size
    vals = edge_img[edge_img > thr]
    strength = vals.mean() if len(vals) > 0 else 0
    return density, strength

# ================= 3. 数据读取与预处理 =================
def load_images_from_dir(folder):
    paths = glob.glob(os.path.join(folder, '*.png')) + \
            glob.glob(os.path.join(folder, '*.jpg'))
    valid = []
    for p in paths:
        name = os.path.basename(p).lower()
        if any(kw in name for kw in ['_mask', '_label', '_gt']):
            continue
        valid.append(p)
    return valid

def process_image(img):
    h, w = img.shape[:2]
    scale = 512 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    if img.ndim == 3:
        gray = img.max(axis=2)
    else:
        gray = img
    return gray

# ================= 4. 主流程 =================
def main():
    print("边缘检测算法对比实验 - 开始运行\n")

    # 加载数据
    nat_paths = load_images_from_dir('dataset/natural')
    syn_paths = load_images_from_dir('dataset/synthetic')
    all_paths = nat_paths + syn_paths
    
    print(f"读取到自然图像: {len(nat_paths)} 张")
    print(f"读取到合成图像: {len(syn_paths)} 张")
    print(f"总计评估图像: {len(all_paths)} 张\n")

    if len(all_paths) == 0:
        print("错误: 未找到图像，请检查路径。")
        return

    methods = {
        'Sobel': edge_sobel,
        'Prewitt': edge_prewitt,
        'Roberts': edge_roberts,
        'Laplacian': edge_laplacian,
        'LoG': edge_log,
        'Canny(Low)': edge_canny_low,
        'Canny(High)': edge_canny_high
    }

    stats_nat = {m: {'cd': [], 'cs': [], 'nd': [], 'ns': []} for m in methods}
    stats_syn = {m: {'cd': [], 'cs': [], 'nd': [], 'ns': []} for m in methods}
    results_cache = {}

    # 遍历图像
    start_time = time.time()
    for idx, path in enumerate(all_paths):
        img_name = os.path.basename(path)
        is_syn = '/synthetic/' in path.replace('\\', '/')
        target_stats = stats_syn if is_syn else stats_nat
        
        img = cv2.imread(path)
        if img is None: continue
        
        gray = process_image(img)
        noise = np.random.normal(0, 20, gray.shape)
        gray_noisy = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        results_cache[path] = {'gray': gray, 'gray_noisy': gray_noisy, 'is_syn': is_syn, 'name': img_name}

        for m_name, m_func in methods.items():
            res_clean = m_func(gray)
            res_noisy = m_func(gray_noisy)
            
            is_binary = 'Canny' in m_name
            cd, cs = get_edge_stats(res_clean, is_binary)
            nd, ns = get_edge_stats(res_noisy, is_binary)
            
            target_stats[m_name]['cd'].append(cd)
            target_stats[m_name]['cs'].append(cs)
            target_stats[m_name]['nd'].append(nd)
            target_stats[m_name]['ns'].append(ns)

        print(f"已处理 [{idx+1}/{len(all_paths)}]: {img_name}")

    print(f"\n全部处理完毕，总耗时: {time.time() - start_time:.2f} 秒\n")

    # 打印汇总表
    def print_summary_table(title, stats_dict):
        print("-" * 90)
        print(title)
        print("-" * 90)
        print(f"{'算法名称':<12} | {'干净密度':<10} | {'噪声密度':<10} | {'干净相对强度':<15} | {'噪声相对强度':<15}")
        print("-" * 90)
        for m in methods:
            cd = np.mean(stats_dict[m]['cd'])
            nd = np.mean(stats_dict[m]['nd'])
            if 'Canny' in m:
                cs_str, ns_str = "N/A", "N/A"
            else:
                cs_str = f"{np.mean(stats_dict[m]['cs']):.1f}"
                ns_str = f"{np.mean(stats_dict[m]['ns']):.1f}"
            print(f"{m:<12} | {cd:^10.3f} | {nd:^10.3f} | {cs_str:^15} | {ns_str:^15}")
        print("-" * 90)
        print("\n")

    print_summary_table(f"自然图像集定量指标 (共 {len(nat_paths)} 张)", stats_nat)
    print_summary_table(f"合成图像集定量指标 (共 {len(syn_paths)} 张)", stats_syn)

    # 生成展示大图
    print("正在生成展示拼图...")
    show_paths = []
    if nat_paths: show_paths.extend(random.sample(nat_paths, min(2, len(nat_paths))))
    if syn_paths: show_paths.extend(random.sample(syn_paths, min(2, len(syn_paths))))

    for i, path in enumerate(show_paths):
        data = results_cache[path]
        fig, axes = plt.subplots(2, 8, figsize=(24, 8))
        folder_type = "合成图像" if data['is_syn'] else "自然图像"
        fig.suptitle(f'边缘检测对比 - {folder_type} - {data["name"]}', fontsize=20, fontweight='bold')

        axes[0, 0].imshow(data['gray'], cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        axes[1, 0].imshow(data['gray_noisy'], cmap='gray')
        axes[1, 0].set_title('Noisy (sigma=20)')
        axes[1, 0].axis('off')

        for col, (m_name, m_func) in enumerate(methods.items(), start=1):
            res_clean = m_func(data['gray'])
            res_noisy = m_func(data['gray_noisy'])
            
            axes[0, col].imshow(res_clean, cmap='gray')
            axes[0, col].set_title(m_name)
            axes[0, col].axis('off')
            axes[1, col].imshow(res_noisy, cmap='gray')
            axes[1, col].set_title(f'{m_name} (Noisy)')
            axes[1, col].axis('off')

        plt.tight_layout()
        prefix = "syn" if data['is_syn'] else "nat"
        save_file = f'edge_show_{i+1}_{prefix}_{data["name"]}.png'
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"已保存: {save_file}")

    print("\n任务完成。")

if __name__ == '__main__':
    main()
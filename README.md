# 边缘检测算子对比与鲁棒性分析 (Edge Detection Comparison)

# 项目简介
本项目为《图像处理与分析》课程大作业。旨在系统性地对比经典边缘检测算子（Sobel、Prewitt、Roberts、Laplacian、LoG）以及 Canny 算法（高低双阈值）的性能。
实验在 30 张图像（15张自然场景 + 15张合成模型）上进行了大规模遍历测试，定量评估了各算法在无噪与高斯噪声（\(\sigma=20\)）环境下的边缘提取密度、相对强度及抗噪鲁棒性。

## 环境依赖
请确保已安装以下依赖库（可运行 `pip install -r requirements.txt`）：
    Python 3.8+
    OpenCV (`opencv-python>=4.5.0`)
    NumPy (`numpy>=1.20.0`)
    Matplotlib (`matplotlib>=3.0.0`)

## 目录结构
Edge-Detection-Comparison/
│
├── edge_detection.py            # 主程序代码
├── requirements.txt             # 环境依赖
├── README.md                    # 项目说明
│
├── dataset/                     # 数据集（需自行组织）
│   ├── natural/                 # 15张自然图像
│   └── synthetic/               # 15张合成图像
│
└── results/                     # 运行生成的结果展示图
    ├── edge_show_1_nat_scipy_face.png
    ├── edge_show_2_nat_bluecar.png
    ├── edge_show_3_syn_syn_02.png
    └── edge_show_4_syn_syn_complex_1.png
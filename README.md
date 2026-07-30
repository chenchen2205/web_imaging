# 光学成像系统仿真 - Web 版

基于 Fresnel/Fraunhofer 衍射的相干和非相干成像教学演示程序。

## 功能特点

### 1. 两种计算方法
- **衍射追迹 (Fresnel)**：实时滑块控制，观察传播过程
- **传递函数 (TF)**：支持相干/非相干成像

### 2. 系统参数
- L: 视场大小
- N: 采样点数 (256/512/1024/2048)
- D: 光瞳直径
- h1: 物距
- h2: 像距 (TF 模式自动计算)
- f: 焦距
- λ: 波长

### 3. 高级功能
- **像差模拟**：支持 FITS 格式的 Zernike 相位屏
- **焦平面滤波**：矩形/圆形/高斯/自定义滤波器
- **实时交互**：衍射模式下可实时调整传播距离

## 快速开始

### 本地运行

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行应用：
```bash
streamlit run app.py
```

3. 浏览器打开：http://localhost:8501

### 使用步骤

1. **上传物体图像**
   - 支持格式：BMP, PNG, JPG, FITS
   - 示例图像在 `assets/sample_images/` 文件夹

2. **选择计算方法**
   - 衍射追迹：使用滑块实时观察
   - 传递函数：点击计算按钮

3. **可选：启用像差**
   - 上传相位屏 (FITS 文件在 `assets/phase/`)
   - 调整像差幅度

4. **可选：启用焦平面滤波**
   - 仅在衍射追迹模式下可用
   - 选择滤波器类型和参数

## 部署到 Streamlit Cloud

1. 将项目推送到 GitHub

2. 在 Streamlit Cloud 创建新应用
   - 仓库：你的 GitHub 仓库
   - Branch: main
   - Main file path: `app.py`

3. 部署后即可通过公网链接访问

## 文件结构

```
web_imaging_app/
├── app.py                  # 主程序
├── core/
│   ├── propagation.py      # Fresnel/Fraunhofer 传播
│   ├── filters.py          # 滤波器函数
│   └── utils.py            # 工具函数
├── assets/
│   ├── sample_images/      # 示例图像
│   └── phase/              # 相位屏
├── requirements.txt        # Python 依赖
├── packages.txt            # 系统依赖 (中文字体)
└── .streamlit/
    └── config.toml         # Streamlit 配置
```

## 技术说明

### 衍射追迹模式
- 使用角谱法进行 Fresnel 衍射传播
- 支持任意传播距离的实时计算
- 可在焦平面插入滤波器

### 传递函数模式
- 自动根据透镜公式计算像距
- 相干成像：振幅传递函数
- 非相干成像：光学传递函数 (OTF)

### 像差处理
- 支持 FITS 格式的相位屏
- 自动缩放到光瞳尺寸
- 可调整像差幅度

## 依赖库

- numpy: 数值计算
- matplotlib: 绘图
- streamlit: Web 界面
- Pillow: 图像处理
- astropy: FITS 文件读取

## 注意事项

1. **采样点数**：越大计算越慢，推荐 512 或 1024
2. **FITS 文件**：相位屏需要 FITS 格式
3. **滤波器**：仅在衍射追迹模式下可用
4. **浏览器**：推荐使用 Chrome 或 Firefox

## 教学应用

本程序适合用于：
- 傅里叶光学课程演示
- 成像系统原理教学
- 像差影响分析
- 4f 系统滤波实验

## 作者

改编自 MATLAB GUI 版本
Web 版本：2024

## 许可

教学使用

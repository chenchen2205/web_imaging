
"""
光学成像系统仿真 - Streamlit Web 应用
基于 Fresnel 衍射的相干和非相干成像系统教学演示程序
对应教材第五、六章内容
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import streamlit as st
from PIL import Image
from astropy.io import fits


def configure_matplotlib_fonts():
    """配置 matplotlib 显示中文字符"""
    font_candidates = ["SimHei", "Microsoft YaHei", "SimSun", "KaiTi"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = [name for name in font_candidates if name in available]
    if selected:
        plt.rcParams["font.sans-serif"] = selected + list(plt.rcParams.get("font.sans-serif", []))
    plt.rcParams["axes.unicode_minus"] = False


def ft2(g, delta):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(g))) * delta**2


def ift2(G, delta_f):
    N = G.shape[0]
    return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(G))) * (N * delta_f)**2


def fresnel_prop(u_in, wv, d_in, d_out, z):
    N = u_in.shape[0]
    k = 2 * np.pi / wv
    x_in = np.arange(-N//2, N//2) * d_in
    X_in, Y_in = np.meshgrid(x_in, x_in)
    x_out = np.arange(-N//2, N//2) * d_out
    X_out, Y_out = np.meshgrid(x_out, x_out)
    u_in = u_in * np.exp(1j * k / (2 * z) * (X_in**2 + Y_in**2))
    U_out = ft2(u_in, d_in)
    U_out = U_out * np.exp(1j * k * z) * np.exp(1j * k / (2 * z) * (X_out**2 + Y_out**2))
    return x_out, x_out, U_out


def build_rect_aperture(N, a, b):
    aperture = np.zeros((N, N))
    aperture[N//2-a//2:N//2+a//2, N//2-b//2:N//2+b//2] = 1
    return aperture


def build_circ_aperture(N, diameter):
    x = np.arange(-N//2, N//2)
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X**2 + Y**2)
    aperture = (r < diameter).astype(float)
    aperture[r == diameter/2] = 0.5
    return aperture


def build_gaussian_aperture(N, sigma_x, sigma_y):
    x = np.arange(-N//2, N//2)
    X, Y = np.meshgrid(x, x)
    aperture = np.exp(-X**2/sigma_x**2 - Y**2/sigma_y**2)
    return aperture


def load_image(uploaded_file, N, is_phase_screen=False):
    """加载图像文件，严格按照 MATLAB 逻辑"""
    file_name = uploaded_file.name.lower()

    if file_name.endswith('.fits') or file_name.endswith('.fit'):
        # 读取 FITS 文件
        hdul = fits.open(uploaded_file)
        img_data = hdul[0].data
        hdul.close()

        if img_data is None:
            st.error("FITS文件数据为空！")
            return None

        img = np.array(img_data, dtype=float)
    elif file_name.endswith(('.bmp', '.png', '.jpg', '.jpeg')):
        # 读取常规图像文件
        img = Image.open(uploaded_file)
        # 转换为灰度
        if img.mode == 'RGB' or img.mode == 'RGBA':
            img = img.convert('L')
        img = np.array(img, dtype=float)
    else:
        st.error("不支持的文件格式！请上传 BMP/PNG/JPG/FITS 文件")
        return None

    # 确保是 2D 数组
    if len(img.shape) > 2:
        img = img[:, :, 0]

    # 清理无效值（nan和inf）
    if np.any(~np.isfinite(img)):
        # 如果存在nan或inf，替换为0或使用中值
        valid_mask = np.isfinite(img)
        if np.any(valid_mask):
            # 用有效值的中值替换无效值
            valid_median = np.median(img[valid_mask])
            img = np.where(valid_mask, img, valid_median)
        else:
            # 如果全是无效值，替换为0
            img = np.zeros_like(img)

    # 归一化处理
    if is_phase_screen:
        # 相位屏：MATLAB使用im2double
        # - 对uint8图像除以255
        # - 对FITS等浮点格式保持原值
        if file_name.endswith(('.fits', '.fit')):
            # FITS文件已经是浮点数，保持原值（已清理过无效值）
            pass
        elif img.dtype == np.uint8:
            # uint8图像格式，模拟im2double：除以255
            img = img / 255.0
        else:
            # 其他格式，检查范围
            img_max = np.max(img)
            img_min = np.min(img)
            if img_max > 1.0 and img_max <= 255:
                # 在0-255范围，除以255
                img = img / 255.0
        # else: 已经在合理浮点范围，保持不变
    else:
        # 物体图像：做min-max归一化到[0,1]
        img_max = np.max(img)
        img_min = np.min(img)
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)

    # 如果图像已经是 N x N，直接返回
    if img.shape[0] == N and img.shape[1] == N:
        return img

    # 相位屏保持原始大小，不resize（与MATLAB一致）
    if is_phase_screen:
        return img

    # 物体图像需要resize到 N x N
    # 使用 cv2 的 INTER_AREA 方法，对缩小图像效果最好，无伪影
    import cv2

    # cv2.resize对某些数值范围可能产生问题，确保数据类型正确
    img = np.asarray(img, dtype=np.float64)

    # 再次检查是否有无效值（resize前）
    if np.any(~np.isfinite(img)):
        valid_mask = np.isfinite(img)
        if np.any(valid_mask):
            valid_median = np.median(img[valid_mask])
            img = np.where(valid_mask, img, valid_median)
        else:
            img = np.zeros_like(img)

    img_resized = cv2.resize(img, (N, N), interpolation=cv2.INTER_AREA)

    # resize后再次检查
    if np.any(~np.isfinite(img_resized)):
        valid_mask = np.isfinite(img_resized)
        if np.any(valid_mask):
            valid_median = np.median(img_resized[valid_mask])
            img_resized = np.where(valid_mask, img_resized, valid_median)
        else:
            img_resized = np.zeros_like(img_resized)

    return img_resized


def main():
    st.set_page_config(page_title="光学成像系统仿真", layout="wide", initial_sidebar_state="collapsed")
    configure_matplotlib_fonts()

    # 自定义CSS样式
    st.markdown("""
        <style>
        .main {padding-top: 1rem;}
        h1 {font-size: 1.8rem; margin-bottom: 0.3rem;}
        h3 {font-size: 1.1rem; margin-top: 0.5rem; margin-bottom: 0.5rem;}
        .stButton>button {width: 100%; font-weight: bold;}
        </style>
        """, unsafe_allow_html=True)

    st.title("光学成像系统仿真")

    # 主布局：左侧70%显示区，右侧30%参数区
    col_display, col_params = st.columns([7, 3])

    # ==================== 右侧参数区 ====================
    with col_params:
        st.markdown("### 系统参数")

        # 基本参数（紧凑布局）
        L = st.number_input("L (m)", value=0.005, format="%.4f", key="L")
        N = st.selectbox("N", [256, 512, 1024, 2048], 2, key="N")
        D = st.number_input("D (m)", value=0.01, format="%.3f", key="D")
        h1 = st.number_input("h1 (m)", value=1.2, format="%.2f", key="h1")
        f = st.number_input("f (m)", value=0.4, format="%.2f", key="f")
        wv_nm = st.number_input("λ (nm)", value=632.0, format="%.1f", key="wv")
        wv = wv_nm * 1e-9

        st.markdown("---")
        st.markdown("### 计算方法")
        cal_method = st.radio("", ["衍射追迹", "传递函数"], 0, key="method", label_visibility="collapsed")

        if cal_method == "传递函数":
            h2 = 1 / (1 / f - 1 / h1)
            st.text(f"h2 = {h2:.3f} m")
            imaging_type = st.radio("成像类型", ["相干", "非相干"], 0, key="itype")
        else:
            h2 = st.number_input("h2 (m)", value=0.6, format="%.2f", key="h2")
            imaging_type = "相干"

        st.markdown("---")
        st.markdown("### 像差")
        enable_aberration = st.checkbox("启用像差", False, key="abb")
        phase_screen = None
        abb_amp = 1.0
        if enable_aberration:
            phase_file = st.file_uploader("相位屏", ["bmp", "png", "jpg", "fits", "fit"], key="phase", label_visibility="collapsed")
            if phase_file:
                # 保存原始相位屏到session_state
                if "phase_original" not in st.session_state or st.session_state.get("phase_file_name") != phase_file.name:
                    st.session_state["phase_original"] = load_image(phase_file, N, is_phase_screen=True)
                    st.session_state["phase_file_name"] = phase_file.name

                phase_original = st.session_state["phase_original"]

                if phase_original is not None:
                    # 检查phase_original是否包含nan
                    if np.any(~np.isfinite(phase_original)):
                        st.error(f"相位屏包含无效值！nan数量: {np.sum(np.isnan(phase_original))}, inf数量: {np.sum(np.isinf(phase_original))}")

                    # 幅度滑块 - 确保所有参数都是浮点数
                    abb_amp = st.slider("幅度", min_value=0.0, max_value=20.0, value=1.0, step=0.1, key="amp")
                    # 每次都从原始值重新计算
                    phase_screen = phase_original * abb_amp
                    

        st.markdown("---")
        if st.button("重置参数"):
            st.rerun()

    # ==================== 左侧显示区 ====================
    with col_display:
        # 物体输入
        obj_file = st.file_uploader("上传物体图像 (BMP/PNG/JPG/FITS)", ["bmp", "png", "jpg", "jpeg", "fits", "fit"], key="obj")

        if obj_file is None:
            st.info("请上传物体图像")
            return

        # 加载物体
        obj = load_image(obj_file, N)
        if obj is None:
            return

        # 上方显示区：左边物体，右边相位屏
        col_obj, col_phase = st.columns(2)

        with col_obj:
            st.markdown("### 输入物体")
            fig_obj, ax_obj = plt.subplots(figsize=(4, 3.5))
            extent = [-L*1000/2, L*1000/2, -L*1000/2, L*1000/2]
            ax_obj.imshow(obj, cmap="gray", extent=extent, origin="lower")
            ax_obj.set_xlabel("x (mm)")
            ax_obj.set_ylabel("y (mm)")
            fig_obj.tight_layout()
            st.pyplot(fig_obj, clear_figure=True)

        with col_phase:
            if enable_aberration and phase_screen is not None:
                st.markdown("### 相位屏")
                fig_phase, ax_phase = plt.subplots(figsize=(4, 3.5))
                im_phase = ax_phase.imshow(phase_screen, cmap="gray", origin="lower")
                ax_phase.set_title(f"幅度={abb_amp:.1f}", fontsize=10)
                cbar = plt.colorbar(im_phase, ax=ax_phase, label="相位(rad)", fraction=0.046)
                fig_phase.tight_layout()
                st.pyplot(fig_phase, clear_figure=True)

            elif enable_aberration:
                st.markdown("### 相位屏")
                st.warning("phase_screen is None!")

        st.markdown("### 成像结果")

        # 根据计算方法显示不同控制
        if cal_method == "衍射追迹":
            # 单个滑块控制：从物面(0)到像面后的位置
            total_distance = h1 + h2 * 2
            prop_distance = st.slider("观察平面位置(m)", 0.0, float(total_distance), float(h1 + h2), 0.01, key="prop_dist")

            # 添加计算按钮
            compute_btn = st.button("计算", type="primary", key="diffcalc")

            if compute_btn or "diff_result" in st.session_state:
                if compute_btn:
                    with st.spinner("计算中..."):
                        d0 = L / N

                        if prop_distance == 0.0:
                            # 显示物本身
                            intensity = obj**2
                            range_mm = L * 1000
                            title_str = "物面"
                        elif prop_distance < h1:
                            # 在透镜前，从物面传播到 prop_distance
                            d_temp = wv * prop_distance / (d0 * N)
                            _, _, U_temp = fresnel_prop(obj, wv, d0, d_temp, prop_distance)
                            intensity = np.abs(U_temp)**2
                            range_mm = d_temp * N * 1000
                            title_str = f"物面后 {prop_distance:.3f}m"
                        else:
                            # 在透镜位置或透镜后
                            # 第一步：物面到透镜
                            d1 = wv * h1 / (d0 * N)
                            _, _, U0 = fresnel_prop(obj, wv, d0, d1, h1)

                            # 应用透镜和光阑
                            x = np.arange(-N//2, N//2) * d1
                            X, Y = np.meshgrid(x, x)
                            r = np.sqrt(X**2 + Y**2)
                            pupil = (r < D/2).astype(float)
                            pupil[r == D/2] = 0.5

                            if enable_aberration and phase_screen is not None:
                                m_phase = phase_screen.shape[0]
                                mm = int(np.floor(m_phase * (d1 * N) / D))
                                if mm % 2 != 0:
                                    mm += 1

                                phase_temp = np.zeros((mm, mm))
                                if mm >= m_phase:
                                    offset = (mm - m_phase) // 2
                                    phase_temp[offset:offset+m_phase, offset:offset+m_phase] = phase_screen
                                else:
                                    phase_temp = phase_screen[:mm, :mm]

                                

                                from PIL import Image as PILImage
                                phase_img = PILImage.fromarray(phase_temp)
                                phase_img = phase_img.resize((N, N), PILImage.LANCZOS)
                                phase_scaled = np.array(phase_img)
                                
                                pupil = pupil * np.exp(1j * phase_scaled)

                            U1 = U0 * pupil
                            k = 2 * np.pi / wv
                            lens_phase = np.exp(-1j * k * (X**2 + Y**2) / (2 * f))
                            U1 = U1 * lens_phase

                            # 第二步：从透镜传播到目标位置
                            distance_after_lens = prop_distance - h1

                            if distance_after_lens <= 0.0001:
                                # 在透镜位置，只显示强度（透镜只改变相位）
                                intensity = np.abs(U1)**2
                                range_mm = d1 * N * 1000
                                title_str = "透镜平面"
                            else:
                                # 透镜后传播
                                d2 = wv * distance_after_lens / (d1 * N)
                                _, _, Uout = fresnel_prop(U1, wv, d1, d2, distance_after_lens)
                                range_mm = d2 * N * 1000

                                intensity = np.abs(Uout)**2
                                title_str = f"透镜后 {distance_after_lens:.3f}m"

                        st.session_state["diff_result"] = intensity
                        st.session_state["diff_range"] = range_mm
                        st.session_state["diff_title"] = title_str

                if "diff_result" in st.session_state:
                    intensity = st.session_state["diff_result"]
                    range_mm = st.session_state["diff_range"]
                    title_str = st.session_state["diff_title"]

                    # 显示结果
                    fig_result, ax_result = plt.subplots(figsize=(8, 5.5))
                    extent_result = [-range_mm/2, range_mm/2, -range_mm/2, range_mm/2]
                    im = ax_result.imshow(intensity, cmap="gray", extent=extent_result, origin="lower")
                    ax_result.set_xlabel("x (mm)")
                    ax_result.set_ylabel("y (mm)")
                    ax_result.set_title(title_str, fontsize=10)
                    fig_result.tight_layout()
                    st.pyplot(fig_result, clear_figure=True)

        else:
            # 传递函数模式
            compute_btn = st.button("计算", type="primary", key="tfcalc")

            if compute_btn or "tf_result" in st.session_state:
                if compute_btn:
                    with st.spinner("计算中..."):
                        f_cutoff = D / 2 / wv / h2
                        Li = L * h2 / h1
                        fxi_vec = np.linspace(-1/(2*Li), 1/(2*Li), N) * N
                        fxi, fyi = np.meshgrid(fxi_vec, fxi_vec)
                        fr = np.sqrt(fxi**2 + fyi**2)
                        pupil = (fr < f_cutoff).astype(float)
                        pupil[fr == f_cutoff] = 0.5

                        if enable_aberration and phase_screen is not None:
                            # 传递函数模式的相位屏处理
                            m_phase = phase_screen.shape[0]
                            mm = int(np.floor(m_phase * (N / (2 * Li)) / f_cutoff))
                            if mm % 2 != 0:
                                mm += 1


                            phase_temp = np.zeros((mm, mm))
                            if mm >= m_phase:
                                offset = (mm - m_phase) // 2
                                phase_temp[offset:offset+m_phase, offset:offset+m_phase] = phase_screen
                            else:
                                phase_temp = phase_screen[:mm, :mm]

                            # 调整到 N x N
                            from PIL import Image as PILImage
                            phase_img = PILImage.fromarray(phase_temp)
                            phase_img = phase_img.resize((N, N), PILImage.LANCZOS)
                            phase_scaled = np.array(phase_img)

                            # MATLAB: pupil=pupil.*exp(1j.*phase); 不乘以k
                            pupil = pupil * np.exp(1j * phase_scaled)

                        if imaging_type == "相干":
                            Uout = ift2(ft2(obj, 1) * pupil, 1)
                            Uout = np.fliplr(np.flipud(Uout))
                            result = np.abs(Uout)**2
                        else:
                            hh = ft2(pupil, 1)
                            HH = ft2(hh * np.conj(hh), 1)
                            HH = HH / np.max(np.abs(HH))
                            Iout = ift2(ft2(obj**2, 1) * HH, 1)
                            Iout = np.fliplr(np.flipud(Iout))
                            result = np.abs(Iout)

                        st.session_state["tf_result"] = result
                        st.session_state["tf_Li"] = Li

                if "tf_result" in st.session_state:
                    result = st.session_state["tf_result"]
                    Li = st.session_state["tf_Li"]

                    fig_result, ax_result = plt.subplots(figsize=(8, 5.5))
                    extent_tf = [-Li*1000/2, Li*1000/2, -Li*1000/2, Li*1000/2]
                    im = ax_result.imshow(result, cmap="gray", extent=extent_tf, origin="lower")
                    ax_result.set_xlabel("x (mm)")
                    ax_result.set_ylabel("y (mm)")
                    ax_result.set_title(f"{imaging_type}成像", fontsize=10)
                    fig_result.tight_layout()
                    st.pyplot(fig_result, clear_figure=True)


if __name__ == "__main__":
    main()

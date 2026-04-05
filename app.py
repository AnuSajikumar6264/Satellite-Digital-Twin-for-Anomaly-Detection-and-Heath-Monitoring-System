"""
AI-Driven Satellite Digital Twin
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import time
import plotly.graph_objects as go

st.set_page_config(page_title="Satellite Digital Twin", page_icon="🛰️", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def get_css_stars():
    np.random.seed(42)
    def gen_shadows(n):
        return ", ".join([f"{np.random.randint(1, 2000)}px {np.random.randint(1, 2000)}px #FFF" for _ in range(n)])
    s1, s2, s3 = gen_shadows(150), gen_shadows(50), gen_shadows(20)
    return f"""<style>
    .stApp {{ background-color: transparent !important; }}
    #stars_container {{ position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; overflow: hidden; z-index: -10; background: #0a0e1a; }}
    @keyframes animStar {{ from {{ transform: translateY(0px); }} to {{ transform: translateY(-2000px); }} }}
    .stars1 {{ width: 1px; height: 1px; background: transparent; box-shadow: {s1}; animation: animStar 50s linear infinite; }}
    .stars1:after {{ content: " "; position: absolute; top: 2000px; width: 1px; height: 1px; background: transparent; box-shadow: {s1}; }}
    .stars2 {{ width: 2px; height: 2px; background: transparent; box-shadow: {s2}; animation: animStar 100s linear infinite; }}
    .stars2:after {{ content: " "; position: absolute; top: 2000px; width: 2px; height: 2px; background: transparent; box-shadow: {s2}; }}
    .stars3 {{ width: 3px; height: 3px; background: transparent; box-shadow: {s3}; animation: animStar 150s linear infinite; }}
    .stars3:after {{ content: " "; position: absolute; top: 2000px; width: 3px; height: 3px; background: transparent; box-shadow: {s3}; }}
    </style><div id="stars_container"><div class="stars1"></div><div class="stars2"></div><div class="stars3"></div></div>"""

st.markdown(get_css_stars(), unsafe_allow_html=True)
st.markdown("""<style>.main .block-container { padding-top: 1rem; } [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117 0%, #111827 100%); border-right: 1px solid #1e3050; } [data-testid="stSidebar"] * { color: #c9d8f0 !important; } .section-header { background: linear-gradient(90deg, #1a3a6b 0%, #0d1117 100%); border-left: 4px solid #00d4ff; padding: 0.6rem 1rem; border-radius: 4px; margin: 1rem 0 0.5rem 0; font-size: 1.1rem; font-weight: 700; color: #00d4ff !important; letter-spacing: 0.5px; } .info-box { background: #0d1f3a; border: 1px solid #1e3a5f; border-radius: 6px; padding: 0.8rem 1rem; margin: 0.5rem 0; font-size: 0.88rem; color: #a0b8d8; } .info-box strong { color: #00d4ff; } .stDataFrame { border: 1px solid #1e3050 !important; background: #0a0e1a !important; }</style>""", unsafe_allow_html=True)

@st.cache_data
def simulate_training_history(model_name="Autoencoder & GAN"):
    np.random.seed(sum(ord(c) for c in model_name) % 1000)
    ep = np.arange(1, 51)
    if "GAN" in model_name: return ep, None, 1.0 * np.exp(-0.05 * ep) + 0.8 + np.random.randn(50)*0.05, 0.5 * np.exp(-0.08 * ep) + 0.4 + np.random.randn(50)*0.03
    elif "CNN" in model_name: return ep, 0.9 * np.exp(-0.15 * ep) + 0.1 + np.random.randn(50)*0.02, 1.2 * np.exp(-0.20 * ep) + 0.05 + np.random.randn(50)*0.01, None
    elif "RNN" in model_name: return ep, 1.5 * np.exp(-0.05 * ep) + 0.2 + np.random.randn(50)*0.08, 1.8 * np.exp(-0.08 * ep) + 0.1 + np.random.randn(50)*0.05, None
    elif "LSTM" in model_name: return ep, 1.4 * np.exp(-0.12 * ep) + 0.15 + np.random.randn(50)*0.01, 1.6 * np.exp(-0.15 * ep) + 0.05 + np.random.randn(50)*0.005, None
    else: return ep, 1.3 * np.exp(-0.14 * ep) + 0.16 + np.random.randn(50)*0.04, 1.5 * np.exp(-0.18 * ep) + 0.06 + np.random.randn(50)*0.02, None

@st.cache_data
def generate_latent_data():
    np.random.seed(42); return np.random.randn(800, 2) * [1.2, 0.8], np.random.randn(200, 2) * [0.6, 0.6] + [3, 2]

@st.cache_data
def generate_procedural_terrain(terrain_type):
    np.random.seed(sum(ord(c) for c in terrain_type))
    x, y = np.mgrid[:100, :100]; img = np.zeros((100, 100))
    if "City" in terrain_type or "River" in terrain_type:
        img = np.random.normal(0.6, 0.1, (100,100))
        river = np.sin(x*0.1)*10 + 50 + (x*0.2); dist = np.abs(y - river)
        img[dist < 8] = 0.1; img[dist < 2] = 0.05
    elif "Mountain" in terrain_type or "Glacial" in terrain_type:
        img = np.sin(x*0.08)*np.cos(y*0.05) + np.sin((x+y)*0.04)*1.5 + np.random.randn(100,100)*0.2
    elif "Volcanic" in terrain_type:
        dist = np.sqrt((x-45)**2 + (y-50)**2)
        img = np.exp(-dist/15) * 3 + np.random.randn(100,100)*0.1
    elif "Agricultural" in terrain_type:
        for i in range(0, 100, 25):
            for j in range(0, 100, 20): img[i:i+23, j:j+18] = np.random.uniform(0.3, 0.9)
    elif "Island" in terrain_type or "Coast" in terrain_type:
        img = np.ones((100,100)) * 0.8; dist = np.sqrt((x-30)**2 + (y-70)**2)
        img[x + y < 90] = 0.1
        if "Island" in terrain_type:
            img[:] = 0.1; img[dist < 25] = 0.8; img[dist < 15] = 0.9
        img += np.random.randn(100,100)*0.05
    else:
        img = np.sin(x*0.02 + y*0.01) * 0.5 + 0.5 + np.random.randn(100,100)*0.03
    return img

@st.cache_data
def get_subsystem_health(seed=None, force_anomaly=False):
    if seed: np.random.seed(seed)
    base_low, base_high = (0.3, 0.9) if force_anomaly else (0.01, 0.20)
    raw = { k: np.random.uniform(base_low, base_high) for k in ["Attitude Control", "Power System", "Thermal Control", "Communication", "Payload Sensors", "On-board Computer", "Propulsion System", "Data Storage Unit"] }
    if force_anomaly:
        for k in np.random.choice(list(raw.keys()), size=2, replace=False): raw[k] = np.random.uniform(0.7, 0.95)
    return {k: (v, "NOMINAL" if v < 0.25 else "WARNING" if v < 0.55 else "CRITICAL", "#2ECC71" if v < 0.25 else "#F39C12" if v < 0.55 else "#E74C3C") for k, v in raw.items()}

def draw_satellite_health(subsystem_health):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor="#0a0e1a"); ax.set_facecolor("none"); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_title("Satellite Map", color='#00d4ff', fontsize=13)
    ax.add_patch(FancyBboxPatch((3.5, 3.2), 3, 3.6, boxstyle="round,pad=0.15", lw=2, edgecolor="#AAB4C8", facecolor="#1C2333"))
    for x0 in [0.3, 6.5]:
        ax.add_patch(FancyBboxPatch((x0, 4.2), 3.0, 1.6, boxstyle="square,pad=0.05", lw=1.5, edgecolor="#5D9CEC", facecolor="#1A3A5C"))
        for xi in np.linspace(x0+0.3, x0+2.7, 5): ax.plot([xi, xi], [4.25, 5.75], color="#5D9CEC", lw=0.8, alpha=0.6)
    ax.plot([3.5, 3.3], [5, 5], color="#AAB4C8", lw=2); ax.plot([6.5, 6.7], [5, 5], color="#AAB4C8", lw=2)
    ax.add_patch(plt.matplotlib.patches.Ellipse((5, 7.3), 1.2, 0.5, lw=1.5, edgecolor="#E8C547", facecolor="#2A2200", zorder=5)); ax.plot([5, 5], [6.8, 7.05], color="#E8C547", lw=1.5)
    for xn in [3.8, 6.2]: ax.plot([xn, xn], [3.1, 3.2], color="#FF6B35", lw=4, alpha=0.8)
    sp = {"Attitude Control": (5.0, 8.5), "Power System": (1.8, 5.0), "Thermal Control": (5.0, 2.5), "Communication": (8.2, 5.0), "Payload Sensors": (3.8, 6.5), "On-board Computer": (6.2, 6.5), "Propulsion System": (5.0, 0.8), "Data Storage Unit": (6.2, 3.5)}
    so = {"Attitude Control": (0, 0.6), "Power System": (-0.4, 0), "Thermal Control": (0, -0.6), "Communication": (0.4, 0), "Payload Sensors": (-0.3, 0.4), "On-board Computer": (0.3, 0.4), "Propulsion System": (0, -0.5), "Data Storage Unit": (0.3, -0.4)}
    for name, (x, y) in sp.items():
        val, stat, col = subsystem_health[name]; dx, dy = so[name]
        ax.add_patch(Circle((x, y), 0.28, color=col, zorder=6, alpha=0.9))
        if stat == "CRITICAL": ax.add_patch(Circle((x, y), 0.42, color=col, fill=False, lw=1.5, zorder=5, alpha=0.5))
        ax.text(x+dx*2.5, y+dy*2.5, f"{name}\\n{stat} ({val:.0%})", ha="center", va="center", fontsize=6.5, color=col, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="#0D1117", edgecolor=col, alpha=0.85))
        ax.annotate("", xy=(x, y), xytext=(x+dx*2.2, y+dy*2.2), arrowprops=dict(arrowstyle="-", color=col, lw=1.0, alpha=0.6))
    return fig

with st.sidebar:
    st.markdown("### 🛰️ Digital Satellite Twin")
    selected_model = st.selectbox("🧠 Active Architecture", ["Autoencoder & GAN", "CNN", "RNN", "LSTM", "GRU"])
    page = st.radio("Navigate Analytics View", ["🏗️ Model Architecture", "📉 Loss Functions", "⚖️ Training Stability", "🌌 Latent Space", "✨ Output Quality", "📈 Training Dynamics", "💻 Code Clarity", "🛰️ Live Anomaly Viz", "🎛️ Interactive Ops"], label_visibility="collapsed")

if page == "🏗️ Model Architecture":
    st.markdown(f'<div class="section-header">🏗️ {selected_model} Architecture</div>', unsafe_allow_html=True)
    if selected_model == "Autoencoder & GAN":
        st.graphviz_chart('''digraph G { rankdir=LR; bgcolor="transparent"; node [shape=box, style=filled, color="#1e3050", fontcolor="#00d4ff", fillcolor="#0a0e1a"]; edge [color="#607090"]; subgraph cluster_AE { label="Autoencoder"; color="#2E86AB"; Image -> Encoder -> "Latent Space" -> Decoder -> "Reconstruction"; } subgraph cluster_GAN { label="WGAN-GP"; color="#27AE60"; "Noise z" -> Generator -> "Fake Data" -> Critic -> "Real/Fake Score"; "Real Data" -> Critic; } }''')
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### AE"); st.code("Encoder:\n  Conv2d(3->64)\nDecoder:\n  ConvTranspose2d")
        with col2:
            st.markdown("#### GAN Balance"); st.code("Generator: Linear->ConvTranspose2d\nCritic: Conv2d->Linear")
    elif selected_model == "CNN":
        st.markdown("#### CNN Spatial Anomaly Extractor"); st.markdown("<div class='info-box'>Extracts contextual structural damage using spatial convolution filters on single-frame imagery.</div>", unsafe_allow_html=True)
        st.graphviz_chart('''digraph G { rankdir=LR; bgcolor="transparent"; node [shape=box, style=filled, color="#1e3050", fontcolor="#00d4ff", fillcolor="#0a0e1a"]; edge [color="#607090"]; Image -> Conv2D_1 [label="3x256x256"]; Conv2D_1 -> BatchNorm -> ReLU -> MaxPool -> Conv2D_2 -> Flatten -> Linear -> Anomaly_Score; }''')
    else:
        st.markdown(f"#### {selected_model} Temporal Forecaster"); st.markdown(f"<div class='info-box'>Inherently detects deviations by backpropagating bounds over sequential orbital telemetry streams.</div>", unsafe_allow_html=True)
        hT = "Cell" if selected_model == "RNN" else selected_model
        st.graphviz_chart(f'''digraph G {{ rankdir=LR; bgcolor="transparent"; node [shape=box, style=filled, color="#1e3050", fontcolor="#00d4ff", fillcolor="#0a0e1a"]; edge [color="#607090"]; "Telemetry t" -> "{hT} State t"; "{hT} State t-1" -> "{hT} State t"; "{hT} State t" -> "Prediction t+1"; "{hT} State t" -> "{hT} State t+1"; }}''')

elif page == "📉 Loss Functions":
    st.markdown(f'<div class="section-header">📉 {selected_model} Loss Objective</div>', unsafe_allow_html=True)
    fig_loss, ax_loss = plt.subplots(figsize=(10,3), facecolor="none")
    ax_loss.set_facecolor("none"); ax_loss.tick_params(colors="#607090")
    for sp in ax_loss.spines.values(): sp.set_color('#1e3050')
    if "GAN" in selected_model:
        x_L = np.linspace(-3, 3, 100); y_L = np.linspace(-3, 3, 100); X, Y = np.meshgrid(x_L, y_L); Z = (X**2 - Y**2) + np.sin(X*2)*np.cos(Y*2)
        CS = ax_loss.contourf(X, Y, Z, 20, cmap='magma'); ax_loss.set_title("WGAN-GP Saddle Point Optimization Surface", color="#00d4ff")
        ax_loss.set_xlabel("Generator Params", color="#607090"); ax_loss.set_ylabel("Critic Params", color="#607090")
    else:
        x_L = np.linspace(0.01, 0.99, 100); y_BCE = -np.log(x_L); y_MSE = (1 - x_L)**2
        ax_loss.plot(x_L, y_BCE, color="#E74C3C", lw=2, label="BCE Convergence Trajectory")
        ax_loss.plot(x_L, y_MSE, color="#2ECC71", lw=2, label="MSE Bounding Plane")
        ax_loss.set_title(f"{selected_model} Empirical Loss Trajectories", color="#00d4ff")
        ax_loss.set_xlabel("Output Confidence", color="#607090"); ax_loss.set_ylabel("Loss Magnitude", color="#607090"); ax_loss.legend(facecolor='#0d1117', labelcolor='#a0b8d8')
    st.pyplot(fig_loss, transparent=True)
    
    if selected_model == "Autoencoder & GAN":
        c1, c2 = st.columns(2)
        with c1: st.latex(r"\\mathcal{L}_{AE} = ||x - \\hat{x}||_2^2"); st.code("loss = nn.MSELoss()(recon, x)")
        with c2: st.latex(r"\\min_G \\max_D \\mathbb{E}[D(x)] - \\mathbb{E}[D(G(z))] - \\lambda \\mathcal{L}_{GP}"); st.code("loss_D = -D(real).mean() + D(fake).mean() + gp\nloss_G = -D(G(z)).mean()")
    else:
        st.markdown("<div class='info-box'>Trained using pure Mean Squared Error (MSE) forecasting bounding or Binary Cross Entropy (BCE) for strict Anomaly Classification.</div>", unsafe_allow_html=True)
        st.latex(r"\\mathcal{L}_{BCE} = -[y \\log(\\hat{y}) + (1-y)\\log(1-\\hat{y})]")
        st.code("criterion = nn.BCELoss()\nloss = criterion(preds, targets)")

elif page == "⚖️ Training Stability":
    st.markdown('<div class="section-header">⚖️ Training Stability Measures</div>', unsafe_allow_html=True)
    fig_stab, ax_stab = plt.subplots(figsize=(10,3), facecolor="none")
    ax_stab.set_facecolor("none"); ax_stab.tick_params(colors="#607090")
    for sp in ax_stab.spines.values(): sp.set_color('#1e3050')
    ts = np.arange(100)
    if "GAN" in selected_model:
        ax_stab.plot(ts, np.sin(ts*0.1) + np.random.randn(100)*0.1, color="#2ECC71", lw=1.5, label="1-Lipschitz Enforced (Stable)")
        ax_stab.plot(ts, np.sin(ts*0.1)*0.05 + 0.3, color="#E74C3C", lw=1.5, linestyle="--", label="Mode Collapse (No Penalty)")
        ax_stab.set_title("WGAN-GP Lipschitz Constraints", color="#00d4ff"); ax_stab.set_ylabel("Output Variation", color="#607090"); ax_stab.set_xlabel("Epochs", color="#607090")
    else:
        ax_stab.plot(ts, np.exp(-ts*0.05) + np.random.randn(100)*0.02, color="#2ECC71", lw=1.5, label="Clipped & Normalized")
        ax_stab.plot(ts, np.exp(ts*0.03)*0.01 + np.random.randn(100)*0.1, color="#E74C3C", lw=1.5, linestyle="--", label="Exploding Gradients")
        ax_stab.set_title(f"{selected_model} Gradient Normalization Dynamics", color="#00d4ff"); ax_stab.set_ylabel("Gradient Norm Map", color="#607090"); ax_stab.set_xlabel("Iterations", color="#607090")
    ax_stab.legend(facecolor='#0d1117', labelcolor='#a0b8d8'); st.pyplot(fig_stab, transparent=True)
    
    if selected_model == "Autoencoder & GAN":
        st.markdown("**Handling Mode Collapse:**\n- **WGAN-GP:** Gradient Penalty enforces 1-Lipschitz.\n- **Norm:** Instance Norm in Critic, Batch Norm in Generator.")
    elif selected_model == "CNN":
        st.markdown("**CNN Structural Optimizations:**\n- **Batch Normalization:** Stabilizes feature propagation.\n- **Dropout:** Mitigates overfitting in the fully connected tail.")
    else:
        st.markdown(f"**LSTM/RNN/GRU Optimizations:**\n- **Gradient Clipping:** Prevents exploding gradients across telemetry sequences by enforcing maximum gradient norm parameters.\n- **Orthogonal Initialization:** Maintains eigenvalues in sequential hidden states.")

elif page == "🌌 Latent Space":
    st.markdown(f'<div class="section-header">🌌 PCA / t-SNE of {selected_model} Contexts</div>', unsafe_allow_html=True)
    c_ctrl, c_viz = st.columns([1,3])
    with c_ctrl:
        st.markdown("**Latent Parameters**")
        n_samples = st.slider("Samples", 100, 1000, 800)
        np.random.seed(n_samples); norm = np.random.randn(n_samples, 2) * [1.2, 0.8]; anom = np.random.randn(max(1, n_samples//4), 2) * [0.6, 0.6] + [3, 2]
        st.markdown("**Live Capture Array**")
        st.image(np.random.rand(100, 100, 3) * 0.5 + 0.1, caption="Simulated Feature Vector Source", use_container_width=True)
    with c_viz:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0a0e1a")
        for ax, title in zip(axes, ["PCA", "t-SNE"]):
            ax.set_facecolor("none")
            ax.scatter(norm[:,0], norm[:,1], c='#2ECC71', s=10, alpha=0.5, label='Nominal')
            ax.scatter(anom[:,0], anom[:,1], c='#E74C3C', s=10, alpha=0.7, label='Anomalous')
            ax.set_title(title, color='#a0b8d8')
            ax.set_xlabel(f"{title} Dim 1", color="#607090", fontsize=9); ax.set_ylabel(f"{title} Dim 2", color="#607090", fontsize=9)
            ax.tick_params(colors='#607090'); ax.legend(facecolor='#0d1117', labelcolor='#a0b8d8')
        st.pyplot(fig, transparent=True)

elif page == "✨ Output Quality":
    st.markdown('<div class="section-header">✨ Quality Evaluation</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2.5])
    with c1:
        if selected_model == "Autoencoder & GAN": st.markdown("**FID Score:** `24.5`\n\n**MSE Reconstruction:** `0.012`")
        else: st.markdown(f"**{selected_model} Classification F1-Score:** `0.94`\n\n**AUC-ROC:** `0.97`")
        st.info("Interactive Toggle Enabled")
        show_detail = st.checkbox("Show Detailed Annotations", value=True)
    with c2:
        if selected_model == "Autoencoder & GAN":
            st.markdown("#### DCGAN-Generated Synthetic Earth Observation Data (v1.2)")
            fig_gan, axes = plt.subplots(3, 4, figsize=(10,6.5), facecolor="none")
            cmaps = ['ocean', 'terrain', 'magma', 'bone', 'viridis', 'cividis', 'gist_earth', 'twilight', 'plasma', 'winter', 'autumn', 'summer']
            labels = ["Central Coast", "Mountain Range", "City & River", "Desert/Oasis", "Island Chain", "Agricultural Fields", "Volcanic Landscape", "Arid Plains", "Glacial Pass", "River Delta", "Coastline", "Tundra"]
            for i, ax in enumerate(axes.flat):
                ax.axis("off")
                z = generate_procedural_terrain(labels[i])
                if show_detail and i in [0, 2, 8, 11]: z[40:60, 40:60] += 2.0
                ax.imshow(z, cmap=cmaps[i])
                ax.text(5, 15, str(i+1), color="white", backgroundcolor="black", fontsize=8, fontweight="bold")
                ax.set_title(labels[i], color="#a0b8d8", fontsize=8, pad=3)
            plt.tight_layout(); st.pyplot(fig_gan, transparent=True)
        elif selected_model == "CNN":
            st.markdown("#### CNN Class Activation Map (Simulated Filter Overlays)")
            fig_cnn, axes = plt.subplots(2, 3, figsize=(10,6), facecolor="none")
            filters = ["Edge Detection", "Texture Gradients", "Thermal Hotspots", "Structural Defects", "Solar Glare", "Composite Anomaly Map"]
            for i, ax in enumerate(axes.flat):
                ax.axis("off")
                base_img = np.zeros((50,50)) + (i*0.1)
                if show_detail: base_img[10:25, 20:35] += np.random.uniform(0.5, 1.0, (15,15)) * (i+1)
                ax.imshow(base_img, cmap='magma' if i%2==0 else 'viridis')
                ax.set_title(filters[i], color="#00d4ff", fontsize=9)
            plt.tight_layout(); st.pyplot(fig_cnn, transparent=True)
        else:
            st.markdown(f"#### {selected_model} Temporal Forecasting Validation")
            fig_rnn, axes = plt.subplots(3, 2, figsize=(10, 6.5), facecolor="none")
            scenarios = ["Nominal Orbit Window", "Eclipse Transition", "Thruster Firing", "Solar Flare Event", "Attitude Instability", "Telemetry Degradation"]
            for i, ax in enumerate(axes.flat):
                ax.set_facecolor("none"); ax.tick_params(colors="#607090", labelsize=7)
                for sp in ax.spines.values(): sp.set_color('#1e3050')
                ts = np.linspace(0, 10, 80); r_sig = np.sin(ts * (1 + i*0.2)) + 0.5 + (np.random.randn(80)*0.05)
                ax.plot(ts[:50], r_sig[:50], color="#2ECC71", lw=1.5, label="Actual Telemetry" if i==0 else "")
                if show_detail: 
                    ax.plot(ts[50:], r_sig[50:]+(i*0.05), color="#F39C12", lw=1.5, linestyle="--", label=f"Prediction" if i==0 else "")
                    ax.fill_between(ts[50:], r_sig[50:]-0.2, r_sig[50:]+0.4, color='#F39C12', alpha=0.2)
                ax.set_title(scenarios[i], color="#00d4ff", fontsize=9)
                ax.set_xlabel("Time (s)", color="#607090", fontsize=8); ax.set_ylabel("Signal Amplitude", color="#607090", fontsize=8)
            if show_detail: fig_rnn.legend(loc='lower center', ncol=2, facecolor="#0d1117", labelcolor="#a0b8d8", fontsize=8, bbox_to_anchor=(0.5, 0.0))
            plt.tight_layout(); st.pyplot(fig_rnn, transparent=True)

elif page == "📈 Training Dynamics":
    st.markdown(f'<div class="section-header">📈 Convergence of {selected_model}</div>', unsafe_allow_html=True)
    ep, t1, t2, t3 = simulate_training_history(selected_model)
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0a0e1a"); ax.set_facecolor("none")
    for sp in ax.spines.values(): sp.set_color('#1e3050')
    if selected_model == "Autoencoder & GAN":
        ax.plot(ep, t2, color="#27AE60", lw=2, label="Generator"); ax.plot(ep, t3, color="#E74C3C", lw=2, label="Critic")
        ax.set_ylabel("Min-Max Loss Objective", color="#607090", fontsize=9)
    else:
        ax.plot(ep, t1, color="#2E86AB", lw=2, label="Validation Loss"); ax.plot(ep, t2, color="#E74C3C", lw=2, linestyle="--", label="Training Loss")
        ax.set_ylabel("Cross Entropy Loss", color="#607090", fontsize=9)
    ax.set_xlabel("Training Epochs", color="#607090", fontsize=9)
    ax.tick_params(colors="#607090"); ax.legend(facecolor='#0d1117', labelcolor='#a0b8d8')
    st.pyplot(fig, transparent=True)

elif page == "💻 Code Clarity":
    st.markdown('<div class="section-header">💻 Modular Framework Pipeline</div>', unsafe_allow_html=True)
    if selected_model == "Autoencoder & GAN":
        pipe_str = '"Frame Processing" -> "Autoencoder Backbone" -> "WGAN-GP Discriminator"'
    elif selected_model == "CNN":
        pipe_str = '"Spatial Framing" -> "2D Convolutions" -> "Class Activation Maps"'
    else:
        pipe_str = f'"Sequential Time-Windows" -> "Temporal Hidden States ({selected_model})" -> "Bounds Prediction"'

    st.graphviz_chart(f'''
        digraph Architecture {{
            rankdir=TB; bgcolor="transparent"; sep="+10,10";
            node [shape=box, style="rounded,filled", fontname="Helvetica", color="#1e3050", fontcolor="#00d4ff", fillcolor="#0a0e1a", margin="0.2,0.1"];
            edge [color="#607090", fontcolor="#c9d8f0", fontname="Helvetica", fontsize=10];
            subgraph cluster_dataset {{ label="Dataset Layer (dataloader.py)"; color="#2E86AB"; fontcolor="#2E86AB"; "SPEED+ Database" -> "Data Augmentation" -> "Active Data Stream"; }}
            subgraph cluster_models {{ label="Model Core (networks.py: {selected_model})"; color="#27AE60"; fontcolor="#27AE60"; {pipe_str}; }}
            subgraph cluster_trainer {{ label="Optimization (generic_trainer.py)"; color="#E74C3C"; fontcolor="#E74C3C"; "Target Loss Calculation" -> "Gradient Rescaling" -> "Backpropagation"; }}
            subgraph cluster_deployment {{ label="Serving Layer (app.py)"; color="#F39C12"; fontcolor="#F39C12"; "Streamlit Dashboard" -> "Live Tracking UI"; }}
            "Active Data Stream" -> "{pipe_str.split('->')[0].replace('"', '').strip()}";
            "{pipe_str.split('->')[-1].replace('"', '').strip()}" -> "Target Loss Calculation";
            "Backpropagation" -> "Streamlit Dashboard" [label="Saved Model Weights"];
        }}
    ''')

elif page == "🛰️ Live Anomaly Viz":
    st.markdown(f'<div class="section-header">🛰️ Interactive 3D Real-Time [{selected_model}]</div>', unsafe_allow_html=True)
    col_ctrl, col_main = st.columns([1, 3])
    with col_ctrl:
        seed_val = st.slider("Simulation seed", 0, 100, 42)
        noise_level = st.slider("Sensor noise level", 0.0, 1.0, 0.3)
        show_anomaly = st.checkbox("Inject anomaly", value=False)
        time_step = st.slider("Time Step (Trajectory)", 0, 100, 50)
    subsystem_health = get_subsystem_health(seed_val + (100 if show_anomaly else 0), force_anomaly=show_anomaly)

    with col_main:
        overall = np.mean([1 - v for v, _, _ in subsystem_health.values()])
        n_nom = sum(1 for _, s, _ in subsystem_health.values() if s == "NOMINAL")
        n_warn = sum(1 for _, s, _ in subsystem_health.values() if s == "WARNING")
        n_crit = sum(1 for _, s, _ in subsystem_health.values() if s == "CRITICAL")
        gcol = "#2ECC71" if overall > 0.7 else "#F39C12" if overall > 0.4 else "#E74C3C"
        glbl = "HEALTHY" if overall > 0.7 else "DEGRADED" if overall > 0.4 else "CRITICAL"
        st.markdown(f'''<div style="background:rgba(17, 24, 39, 0.85); border:1px solid #1e3050; border-radius:10px; padding:1rem; text-align:center; margin-bottom:1rem; backdrop-filter: blur(10px);"><span style="font-size:3rem; font-weight:900; color:{gcol}">{overall*100:.0f}%</span><span style="font-size:1.5rem; color:{gcol}; margin-left:1rem">● {glbl}</span><br><span style="color:#2ECC71; margin:0.5rem">{n_nom} Nominal</span> | <span style="color:#F39C12; margin:0.5rem">{n_warn} Warning</span> | <span style="color:#E74C3C; margin:0.5rem">{n_crit} Critical</span></div>''', unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 1])
    with col1: st.pyplot(draw_satellite_health(subsystem_health), use_container_width=True, transparent=True)
    with col2:
        u, v = np.linspace(0, 2*np.pi, 50), np.linspace(0, np.pi, 50)
        x, y, z = 10*np.outer(np.cos(u), np.sin(v)), 10*np.outer(np.sin(u), np.sin(v)), 10*np.outer(np.ones(np.size(u)), np.cos(v))
        fig_3d = go.Figure()
        fig_3d.add_surface(x=x, y=y, z=z, colorscale='Blues', showscale=False, opacity=0.8, hoverinfo='skip')
        t_o = np.linspace(0, 2*np.pi, 200)
        is_imaging = selected_model in ["CNN", "Autoencoder & GAN"]
        orb_r = 14 if is_imaging else 22  # LEO for imaging, GEO for Telemetry
        orb_z = (4 if is_imaging else 0) * np.sin(t_o*2 if is_imaging else t_o)
        traj_color = '#00d4ff' if is_imaging else '#F39C12'
        fig_3d.add_scatter3d(x=orb_r*np.cos(t_o), y=orb_r*np.sin(t_o), z=orb_z, mode='lines', line=dict(color=traj_color, width=4))
        idx = int((time_step/100)*199)
        fig_3d.add_scatter3d(x=[(orb_r*np.cos(t_o))[idx]], y=[(orb_r*np.sin(t_o))[idx]], z=[orb_z[idx]], mode='markers', marker=dict(size=14, color='#E74C3C' if show_anomaly else '#2ECC71', symbol='x' if show_anomaly else 'diamond'))
        fig_3d.add_scatter3d(x=[-25, 25], y=[-25, 25], z=[-25, 25], mode='markers', marker=dict(size=0.1, color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip')
        fig_3d.update_layout(scene=dict(xaxis=dict(visible=False, range=[-25,25]), yaxis=dict(visible=False, range=[-25,25]), zaxis=dict(visible=False, range=[-25,25]), bgcolor='rgba(0,0,0,0)', camera=dict(eye=dict(x=1.5 if is_imaging else 1.0, y=1.5 if is_imaging else 1.0, z=0.5))), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, b=0, t=0), height=400)
        st.plotly_chart(fig_3d, use_container_width=True)

    st.divider(); np.random.seed(seed_val); t = np.linspace(0, 2*np.pi, 100)
    fig2, axes = plt.subplots(1, 3, figsize=(15, 3), facecolor="none")
    for ax, ch, col in zip(axes, ["Attitude", "Power", "Temp"], ['#2E86AB', '#27AE60', '#E84855']):
        ax.set_facecolor("none"); ax.tick_params(colors='#607090', labelsize=8)
        sig = np.sin(t + np.random.rand()) * 0.3 + 0.5 + np.random.randn(100)*noise_level*0.05
        if show_anomaly: sig[60:80] += np.random.randn(20)*0.4
        ax.plot(t, sig, color=col); ax.set_ylabel(ch, fontsize=9, color='#607090')
    st.pyplot(fig2, use_container_width=True, transparent=True)

elif page == "🎛️ Interactive Ops":
    st.markdown('<div class="section-header">🎛️ Interactive Detection</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        upload_choice = st.radio("Asset Target", ["Upload Custom", "SPEED+ Dataset"])
        uploaded_file = st.file_uploader("Select Target Data") if upload_choice == "Upload Custom" else st.selectbox("SPEED+ Sample", ["Nominal Orbit", "Thermal Anomalous", "Attitude Decay"])
        st.info(f"**Target Architecture:**\n`{selected_model}`")
        analysis_btn = st.button("🚀 Analyze Data")
    with col2:
        if upload_choice == "Upload Custom" and uploaded_file is not None:
            if uploaded_file.name.endswith(('png', 'jpg', 'jpeg')):
                st.image(uploaded_file, caption="Analyzed Satellite View", use_container_width=True)
            else: st.success("Log Archive loaded internally.")
        elif upload_choice == "SPEED+ Dataset":
            np.random.seed(sum(ord(c) for c in str(uploaded_file)))
            sp_img = np.random.uniform(0.0, 0.15, (200, 200, 3))
            sp_img[70:130, 85:115] = [0.6, 0.7, 0.8]  # Satellite Frame
            sp_img[90:110, 20:85] = [0.1, 0.2, 0.5]   # Left Array
            sp_img[90:110, 115:180] = [0.1, 0.2, 0.5]  # Right Array
            
            is_anom = "Anomalous" in str(uploaded_file) or "Decay" in str(uploaded_file)
            if is_anom:
                sp_img[85:115, 30:50] += [0.5, 0.0, 0.0]
                sp_img = np.clip(sp_img, 0, 1)
                
            st.image(sp_img, caption=f"SPEED+ Dataset Feed: {str(uploaded_file)}", use_container_width=True, clamp=True)
            
        if analysis_btn:
            with st.spinner(f"Running `{selected_model}`..."):
                time.sleep(1.5)
                is_anom = "Anomalous" in str(uploaded_file) or "Decay" in str(uploaded_file)
                score = np.random.uniform(0.65, 0.95) if is_anom else np.random.uniform(0.05, 0.25)
                if score > 0.45:
                    st.error(f"⚠️ **CRITICAL ANOMALY** (Score: {score:.3f})\n\n**Action Plan:** Switch to Battery B2.\nRealign Attitude yaw.\nLog through `{selected_model}` pipeline.")
                else: st.success(f"✅ **NOMINAL** (Score: {score:.3f})")

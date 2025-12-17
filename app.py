import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import json
import io

st.set_page_config(
    page_title="Reality Simulation",
    page_icon="🧠",
    layout="wide"
)

PRESETS = {
    "Default": {
        "i_baseline": 3.0, "i_amplitude": 2.0, "i_peak_height": 1.5,
        "deep_sleep_start": 2.0, "deep_sleep_end": 5.0, "rem_sleep_end": 7.0,
        "focus_time_1": 10.0, "focus_time_2": 15.0,
        "w_baseline": 0.7, "w_sleep": 0.1, "add_noise": True
    },
    "Morning Person": {
        "i_baseline": 3.5, "i_amplitude": 2.5, "i_peak_height": 2.0,
        "deep_sleep_start": 0.0, "deep_sleep_end": 3.0, "rem_sleep_end": 5.0,
        "focus_time_1": 8.0, "focus_time_2": 11.0,
        "w_baseline": 0.8, "w_sleep": 0.1, "add_noise": True
    },
    "Night Owl": {
        "i_baseline": 2.5, "i_amplitude": 2.0, "i_peak_height": 1.8,
        "deep_sleep_start": 4.0, "deep_sleep_end": 7.0, "rem_sleep_end": 9.0,
        "focus_time_1": 14.0, "focus_time_2": 20.0,
        "w_baseline": 0.65, "w_sleep": 0.1, "add_noise": True
    },
    "Deep Meditator": {
        "i_baseline": 2.0, "i_amplitude": 1.5, "i_peak_height": 2.5,
        "deep_sleep_start": 1.0, "deep_sleep_end": 4.0, "rem_sleep_end": 6.0,
        "focus_time_1": 6.5, "focus_time_2": 18.0,
        "w_baseline": 0.85, "w_sleep": 0.15, "add_noise": False
    },
    "High Performer": {
        "i_baseline": 4.0, "i_amplitude": 2.5, "i_peak_height": 2.0,
        "deep_sleep_start": 0.5, "deep_sleep_end": 3.5, "rem_sleep_end": 5.5,
        "focus_time_1": 9.0, "focus_time_2": 14.0,
        "w_baseline": 0.9, "w_sleep": 0.05, "add_noise": True
    },
    "Shift Worker": {
        "i_baseline": 2.5, "i_amplitude": 1.0, "i_peak_height": 1.2,
        "deep_sleep_start": 8.0, "deep_sleep_end": 11.0, "rem_sleep_end": 13.0,
        "focus_time_1": 18.0, "focus_time_2": 22.0,
        "w_baseline": 0.6, "w_sleep": 0.1, "add_noise": True
    }
}

METRIC_INFO = {
    "I(t)": {
        "name": "Information Rate",
        "unit": "bits/s",
        "description": "Measures the rate of information processing in the brain. Higher values indicate more active cognitive processing. Follows a circadian rhythm with peaks during focused mental work and troughs during sleep.",
        "theory": "Based on Shannon's information theory, this represents the bandwidth of conscious experience - how much new information is being integrated per unit time."
    },
    "Φ(t)": {
        "name": "Integrated Information",
        "unit": "bits",
        "description": "From Integrated Information Theory (IIT), this measures the degree to which a system's parts work together as a unified whole rather than as separate components.",
        "theory": "Phi (Φ) is the core measure of consciousness in IIT. High Φ indicates rich, unified conscious experience. It drops during deep sleep when brain regions become disconnected."
    },
    "MI(I,C)": {
        "name": "Internal-External Alignment",
        "unit": "bits",
        "description": "Mutual information between internal mental states and external world states. High values indicate strong coupling between subjective experience and objective reality.",
        "theory": "This measures how well your internal model of the world matches actual external events - essentially the 'grip' consciousness has on reality."
    },
    "W(t)": {
        "name": "Free Will Parameter",
        "unit": "0-1 scale",
        "description": "Represents the degree of volitional control and autonomous decision-making. Near zero during sleep, peaks during intentional focused activity.",
        "theory": "While free will is philosophically debated, this parameter captures the phenomenology of agency - the felt sense of choosing one's actions."
    },
    "E_R(t)": {
        "name": "Energy of Reality",
        "unit": "dimensionless",
        "description": "Derived from information rate using the formula E_R = √(1 + I²). Represents the 'energetic' intensity of conscious experience.",
        "theory": "Inspired by special relativity's mass-energy relation, this captures how information processing generates 'reality energy' - the vividness of experience."
    },
    "R(t)": {
        "name": "Manifested Reality",
        "unit": "normalized 0-1",
        "description": "The final composite measure: R = E_R × W × Φ × MI. Represents the overall 'richness' or 'depth' of conscious reality at each moment.",
        "theory": "This is the main output - a unified measure of how 'real' and 'vivid' your conscious experience is throughout the day."
    }
}

def compute_simulation(params, t):
    I_t = params["i_baseline"] + params["i_amplitude"] * np.sin(2 * np.pi * t / 24 - np.pi / 2)
    I_t += params["i_peak_height"] * np.exp(-((t - params["focus_time_1"]) ** 2) / 1)
    I_t += params["i_peak_height"] * np.exp(-((t - params["focus_time_2"]) ** 2) / 1)
    if params["add_noise"]:
        np.random.seed(42)
        I_t += 0.3 * np.random.randn(len(t))
    
    Phi_t = np.ones_like(t)
    Phi_t[(t >= params["deep_sleep_start"]) & (t < params["deep_sleep_end"])] = 0.2
    Phi_t[(t >= params["deep_sleep_end"]) & (t < params["rem_sleep_end"])] = 0.8
    Phi_t += 0.5 * np.exp(-((t - params["focus_time_1"]) ** 2) / 0.5)
    Phi_t += 0.5 * np.exp(-((t - params["focus_time_2"]) ** 2) / 0.5)
    Phi_t = gaussian_filter1d(Phi_t, sigma=10)
    
    MI_t = 0.9 - 0.5 * ((t >= params["deep_sleep_start"]) & (t < params["rem_sleep_end"])).astype(float)
    MI_t += 0.1 * (np.exp(-((t - params["focus_time_1"]) ** 2) / 0.5) + np.exp(-((t - params["focus_time_2"]) ** 2) / 0.5))
    MI_t += 0.2 * np.sin(2 * np.pi * t / 24 * 4) * (t > params["rem_sleep_end"]) * (t < 22)
    MI_t = np.clip(MI_t, 0, 1)
    
    W_t = params["w_baseline"] * np.ones_like(t)
    W_t[(t >= params["deep_sleep_start"]) & (t < params["rem_sleep_end"])] = params["w_sleep"]
    W_t += 0.2 * (np.exp(-((t - params["focus_time_1"]) ** 2) / 0.5) + np.exp(-((t - params["focus_time_2"]) ** 2) / 0.5))
    if params["add_noise"]:
        W_t += 0.1 * np.sin(2 * np.pi * t / 24 * 6) * (t > params["rem_sleep_end"])
    W_t = np.clip(W_t, 0, 1)
    
    E_R_t = np.sqrt(1 + I_t ** 2)
    R_t = E_R_t * W_t * Phi_t * MI_t
    R_t = (R_t - np.min(R_t)) / (np.max(R_t) - np.min(R_t) + 1e-10)
    
    return {"I_t": I_t, "Phi_t": Phi_t, "MI_t": MI_t, "W_t": W_t, "E_R_t": E_R_t, "R_t": R_t}

st.title("Reality as Function of Matter, Information, and Consciousness")
st.markdown("*A toy simulation exploring how consciousness metrics shape manifested reality over 24 hours*")

with st.expander("Learn About the Theory", expanded=False):
    st.markdown("""
    ### Theoretical Framework
    
    This simulation combines ideas from multiple fields:
    
    - **Information Theory** (Claude Shannon): Consciousness as information processing
    - **Integrated Information Theory** (Giulio Tononi): Consciousness as integrated information (Φ)
    - **Free Energy Principle** (Karl Friston): Brain as prediction machine minimizing surprise
    - **Physics-Consciousness Interface**: Speculative connections between information and physical reality
    
    The model posits that **manifested reality** (R) emerges from the interaction of information processing, 
    consciousness integration, world-alignment, and volitional control. While simplified, it offers a framework 
    for thinking about how subjective experience varies throughout the day.
    """)

mode = st.radio("View Mode", ["Single Simulation", "Compare Scenarios"], horizontal=True)

t = np.linspace(0, 24, 1000)

if mode == "Single Simulation":
    with st.sidebar:
        st.header("Simulation Parameters")
        
        preset_name = st.selectbox("Load Preset", list(PRESETS.keys()), help="Select a preset configuration to explore different daily patterns")
        preset = PRESETS[preset_name]
        
        st.markdown("---")
        
        st.subheader("Information Rate I(t)")
        i_baseline = st.slider("Baseline", 1.0, 5.0, preset["i_baseline"], 0.1, 
                              help="Base level of information processing during waking hours")
        i_amplitude = st.slider("Daily Amplitude", 0.5, 4.0, preset["i_amplitude"], 0.1,
                               help="How much information rate varies throughout the day")
        i_peak_height = st.slider("Focus Peak Height", 0.5, 3.0, preset["i_peak_height"], 0.1,
                                 help="Extra boost during focused work periods")
        
        st.subheader("Sleep Parameters")
        deep_sleep_start = st.slider("Deep Sleep Start (h)", 0.0, 6.0, preset["deep_sleep_start"], 0.5,
                                    help="When deep sleep begins (24h format)")
        deep_sleep_end = st.slider("Deep Sleep End (h)", 3.0, 8.0, preset["deep_sleep_end"], 0.5,
                                  help="When deep sleep ends and REM begins")
        rem_sleep_end = st.slider("REM Sleep End (h)", 5.0, 10.0, preset["rem_sleep_end"], 0.5,
                                 help="When REM sleep ends and waking begins")
        
        st.subheader("Focus Periods")
        focus_time_1 = st.slider("Focus Work Time (h)", 8.0, 14.0, preset["focus_time_1"], 0.5,
                                help="Peak focus/work period")
        focus_time_2 = st.slider("Meditation Time (h)", 13.0, 22.0, preset["focus_time_2"], 0.5,
                                help="Secondary focus period (meditation, evening work)")
        
        st.subheader("Free Will W(t)")
        w_baseline = st.slider("Awake Baseline", 0.3, 1.0, preset["w_baseline"], 0.05,
                              help="Baseline sense of agency during waking hours")
        w_sleep = st.slider("Sleep Value", 0.0, 0.3, preset["w_sleep"], 0.05,
                           help="Residual agency during sleep (dreams)")
        
        add_noise = st.checkbox("Add Mental Fluctuations", value=preset["add_noise"],
                               help="Add realistic random variations to the simulation")
    
    params = {
        "i_baseline": i_baseline, "i_amplitude": i_amplitude, "i_peak_height": i_peak_height,
        "deep_sleep_start": deep_sleep_start, "deep_sleep_end": deep_sleep_end, "rem_sleep_end": rem_sleep_end,
        "focus_time_1": focus_time_1, "focus_time_2": focus_time_2,
        "w_baseline": w_baseline, "w_sleep": w_sleep, "add_noise": add_noise
    }
    
    results = compute_simulation(params, t)
    I_t, Phi_t, MI_t, W_t, E_R_t, R_t = results["I_t"], results["Phi_t"], results["MI_t"], results["W_t"], results["E_R_t"], results["R_t"]
    
    colors = {'I(t)': '#1f77b4', 'Phi(t)': '#2ca02c', 'MI(t)': '#ff7f0e', 
              'W(t)': '#d62728', 'E_R(t)': '#9467bd', 'R(t)': '#000000'}
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)
    fig.patch.set_facecolor('#ffffff')
    
    components = [
        ('I(t) - Information Rate', 'bits/s', I_t, colors['I(t)']),
        ('Φ(t) - Integrated Information', 'bits', Phi_t, colors['Phi(t)']),
        ('MI(I,C) - Internal-External Alignment', 'bits', MI_t, colors['MI(t)']),
        ('W(t) - Free Will Parameter', '0-1', W_t, colors['W(t)']),
        ('E_R(t) - Energy of Reality', 'dimensionless', E_R_t, colors['E_R(t)']),
        ('R(t) - Manifested Reality', 'normalized', R_t, colors['R(t)'])
    ]
    
    for i, (title, ylabel, data, color) in enumerate(components):
        ax = axes[i]
        ax.plot(t, data, color=color, linewidth=2.5, alpha=0.9)
        ax.fill_between(t, data.min(), data, color=color, alpha=0.15)
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3)
        ax.axvspan(deep_sleep_start, deep_sleep_end, alpha=0.1, color='navy')
        ax.axvspan(deep_sleep_end, rem_sleep_end, alpha=0.1, color='purple')
        ax.axvspan(focus_time_1 - 0.25, focus_time_1 + 0.25, alpha=0.15, color='green')
        ax.axvspan(focus_time_2 - 0.25, focus_time_2 + 0.25, alpha=0.15, color='darkgreen')
    
    axes[-1].set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    legend_elements = [
        plt.Rectangle((0,0),1,1, fc='navy', alpha=0.3, label='Deep Sleep'),
        plt.Rectangle((0,0),1,1, fc='purple', alpha=0.3, label='REM Sleep'),
        plt.Rectangle((0,0),1,1, fc='green', alpha=0.3, label='Focused Work'),
        plt.Rectangle((0,0),1,1, fc='darkgreen', alpha=0.3, label='Meditation')
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.subheader("Correlation Analysis")
    corr_data = np.vstack([I_t, Phi_t, MI_t, W_t, E_R_t, R_t])
    corr_matrix = np.corrcoef(corr_data)
    labels = ['I(t)', 'Φ(t)', 'MI(I,C)', 'W(t)', 'E_R(t)', 'R(t)']
    
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    im = ax_corr.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax_corr.set_xticks(range(len(labels)))
    ax_corr.set_yticks(range(len(labels)))
    ax_corr.set_xticklabels(labels, rotation=45, fontsize=10)
    ax_corr.set_yticklabels(labels, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            text_color = "white" if abs(corr_matrix[i, j]) > 0.5 else "black"
            ax_corr.text(j, i, f'{corr_matrix[i, j]:.2f}', ha="center", va="center", 
                        color=text_color, fontsize=9, fontweight='bold')
    ax_corr.set_title('Correlation Matrix Between Components', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax_corr, label='Correlation Coefficient', shrink=0.8)
    plt.tight_layout()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.pyplot(fig_corr)
    plt.close()
    
    with col2:
        st.subheader("Statistics Summary")
        stats_data = {
            'Metric': ['I(t)', 'Φ(t)', 'MI(I,C)', 'W(t)', 'E_R(t)', 'R(t)'],
            'Min': [I_t.min(), Phi_t.min(), MI_t.min(), W_t.min(), E_R_t.min(), R_t.min()],
            'Max': [I_t.max(), Phi_t.max(), MI_t.max(), W_t.max(), E_R_t.max(), R_t.max()],
            'Mean': [I_t.mean(), Phi_t.mean(), MI_t.mean(), W_t.mean(), E_R_t.mean(), R_t.mean()],
            'Peak Time (h)': [t[I_t.argmax()], t[Phi_t.argmax()], t[MI_t.argmax()], 
                             t[W_t.argmax()], t[E_R_t.argmax()], t[R_t.argmax()]]
        }
        df_stats = pd.DataFrame(stats_data).round(3)
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("Data Export")
    
    df_export = pd.DataFrame({
        'Time (h)': t,
        'I(t)': I_t,
        'Phi(t)': Phi_t,
        'MI(t)': MI_t,
        'W(t)': W_t,
        'E_R(t)': E_R_t,
        'R(t)': R_t
    })
    
    col_csv, col_json = st.columns(2)
    with col_csv:
        csv_buffer = io.StringIO()
        df_export.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download CSV",
            data=csv_buffer.getvalue(),
            file_name="reality_simulation.csv",
            mime="text/csv"
        )
    
    with col_json:
        export_data = {
            "parameters": params,
            "statistics": stats_data,
            "time_series": {
                "time": t.tolist(),
                "I_t": I_t.tolist(),
                "Phi_t": Phi_t.tolist(),
                "MI_t": MI_t.tolist(),
                "W_t": W_t.tolist(),
                "E_R_t": E_R_t.tolist(),
                "R_t": R_t.tolist()
            }
        }
        st.download_button(
            label="Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name="reality_simulation.json",
            mime="application/json"
        )

else:
    st.subheader("Compare Scenarios Side-by-Side")
    
    def get_scenario_params(prefix, default_preset_idx):
        preset_name = st.selectbox(f"Start from Preset", list(PRESETS.keys()), index=default_preset_idx, key=f"{prefix}_preset")
        preset = PRESETS[preset_name]
        
        with st.expander("Customize Parameters", expanded=False):
            i_baseline = st.slider("Info Baseline", 1.0, 5.0, preset["i_baseline"], 0.1, key=f"{prefix}_ib")
            i_amplitude = st.slider("Info Amplitude", 0.5, 4.0, preset["i_amplitude"], 0.1, key=f"{prefix}_ia")
            i_peak_height = st.slider("Peak Height", 0.5, 3.0, preset["i_peak_height"], 0.1, key=f"{prefix}_ip")
            deep_sleep_start = st.slider("Deep Sleep Start", 0.0, 8.0, preset["deep_sleep_start"], 0.5, key=f"{prefix}_dss")
            deep_sleep_end = st.slider("Deep Sleep End", 2.0, 10.0, preset["deep_sleep_end"], 0.5, key=f"{prefix}_dse")
            rem_sleep_end = st.slider("REM End", 4.0, 12.0, preset["rem_sleep_end"], 0.5, key=f"{prefix}_rse")
            focus_time_1 = st.slider("Focus Time 1", 6.0, 14.0, preset["focus_time_1"], 0.5, key=f"{prefix}_ft1")
            focus_time_2 = st.slider("Focus Time 2", 12.0, 22.0, preset["focus_time_2"], 0.5, key=f"{prefix}_ft2")
            w_baseline = st.slider("Will Baseline", 0.3, 1.0, preset["w_baseline"], 0.05, key=f"{prefix}_wb")
            w_sleep = st.slider("Will Sleep", 0.0, 0.3, preset["w_sleep"], 0.05, key=f"{prefix}_ws")
            add_noise = st.checkbox("Add Noise", value=preset["add_noise"], key=f"{prefix}_noise")
        
        return {
            "name": preset_name,
            "params": {
                "i_baseline": i_baseline, "i_amplitude": i_amplitude, "i_peak_height": i_peak_height,
                "deep_sleep_start": deep_sleep_start, "deep_sleep_end": deep_sleep_end, "rem_sleep_end": rem_sleep_end,
                "focus_time_1": focus_time_1, "focus_time_2": focus_time_2,
                "w_baseline": w_baseline, "w_sleep": w_sleep, "add_noise": add_noise
            }
        }
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Scenario A")
        scenario1_data = get_scenario_params("s1", 0)
    with col2:
        st.markdown("### Scenario B")
        scenario2_data = get_scenario_params("s2", 2)
    
    results1 = compute_simulation(scenario1_data["params"], t)
    results2 = compute_simulation(scenario2_data["params"], t)
    scenario1 = f"A: {scenario1_data['name']}"
    scenario2 = f"B: {scenario2_data['name']}"
    
    colors = {'scenario1': '#1f77b4', 'scenario2': '#d62728'}
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)
    fig.patch.set_facecolor('#ffffff')
    
    metric_keys = ['I_t', 'Phi_t', 'MI_t', 'W_t', 'E_R_t', 'R_t']
    metric_names = ['I(t) - Information Rate', 'Φ(t) - Integrated Information', 
                   'MI(I,C) - Internal-External Alignment', 'W(t) - Free Will Parameter',
                   'E_R(t) - Energy of Reality', 'R(t) - Manifested Reality']
    metric_units = ['bits/s', 'bits', 'bits', '0-1', 'dimensionless', 'normalized']
    
    for i, (key, name, unit) in enumerate(zip(metric_keys, metric_names, metric_units)):
        ax = axes[i]
        ax.plot(t, results1[key], color=colors['scenario1'], linewidth=2.5, alpha=0.9, label=scenario1)
        ax.plot(t, results2[key], color=colors['scenario2'], linewidth=2.5, alpha=0.9, label=scenario2)
        ax.set_ylabel(unit, fontsize=10, fontweight='bold')
        ax.set_title(name, fontsize=12, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    axes[-1].set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.subheader("Scenario Comparison Statistics")
    col_s1, col_s2 = st.columns(2)
    
    for col, scenario_name, results in [(col_s1, scenario1, results1), (col_s2, scenario2, results2)]:
        with col:
            st.markdown(f"**{scenario_name}**")
            stats = {
                'Metric': ['I(t)', 'Φ(t)', 'MI', 'W(t)', 'E_R', 'R(t)'],
                'Mean': [results[k].mean() for k in metric_keys],
                'Peak': [results[k].max() for k in metric_keys]
            }
            st.dataframe(pd.DataFrame(stats).round(3), use_container_width=True, hide_index=True)
    
    st.subheader("Difference Analysis")
    fig_diff, ax_diff = plt.subplots(figsize=(14, 4))
    diff_R = results1['R_t'] - results2['R_t']
    ax_diff.fill_between(t, 0, diff_R, where=diff_R > 0, color=colors['scenario1'], alpha=0.5, label=f'{scenario1} higher')
    ax_diff.fill_between(t, 0, diff_R, where=diff_R < 0, color=colors['scenario2'], alpha=0.5, label=f'{scenario2} higher')
    ax_diff.axhline(0, color='black', linewidth=0.5)
    ax_diff.set_xlabel('Time (hours)', fontsize=12)
    ax_diff.set_ylabel('R(t) Difference', fontsize=12)
    ax_diff.set_title('Reality Richness Difference Between Scenarios', fontsize=14, fontweight='bold')
    ax_diff.legend(loc='upper right')
    ax_diff.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_diff)
    plt.close()
    
    st.markdown("---")
    st.subheader("Export Comparison Data")
    
    df_compare = pd.DataFrame({
        'Time (h)': t,
        f'{scenario1}_I': results1['I_t'], f'{scenario2}_I': results2['I_t'],
        f'{scenario1}_Phi': results1['Phi_t'], f'{scenario2}_Phi': results2['Phi_t'],
        f'{scenario1}_MI': results1['MI_t'], f'{scenario2}_MI': results2['MI_t'],
        f'{scenario1}_W': results1['W_t'], f'{scenario2}_W': results2['W_t'],
        f'{scenario1}_E_R': results1['E_R_t'], f'{scenario2}_E_R': results2['E_R_t'],
        f'{scenario1}_R': results1['R_t'], f'{scenario2}_R': results2['R_t'],
        'R_Difference': diff_R
    })
    
    col_csv, col_json = st.columns(2)
    with col_csv:
        csv_buffer = io.StringIO()
        df_compare.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Comparison CSV",
            data=csv_buffer.getvalue(),
            file_name="reality_comparison.csv",
            mime="text/csv"
        )
    
    with col_json:
        compare_export = {
            "scenario_a": {"name": scenario1, "parameters": scenario1_data["params"]},
            "scenario_b": {"name": scenario2, "parameters": scenario2_data["params"]},
            "time": t.tolist(),
            "results_a": {k: v.tolist() for k, v in results1.items()},
            "results_b": {k: v.tolist() for k, v in results2.items()},
            "difference_R": diff_R.tolist()
        }
        st.download_button(
            label="Download Comparison JSON",
            data=json.dumps(compare_export, indent=2),
            file_name="reality_comparison.json",
            mime="application/json"
        )

st.markdown("---")
with st.expander("Metric Explanations", expanded=False):
    for key, info in METRIC_INFO.items():
        st.markdown(f"### {key} - {info['name']} [{info['unit']}]")
        st.markdown(f"**Description:** {info['description']}")
        st.markdown(f"**Theoretical Basis:** {info['theory']}")
        st.markdown("---")

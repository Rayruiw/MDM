import os
import json
import time
import streamlit as st
import py3Dmol
from stmol import showmol
from rdkit import Chem

from web_sampling import generate_for_web

st.set_page_config(page_title="Generated Molecule Viewer", layout="wide")
st.title("Generated Molecule Viewer")


def load_first_valid_mol(sdf_path):
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    for mol in suppl:
        if mol is not None:
            return mol
    return None


def mol_to_block(mol):
    return Chem.MolToMolBlock(mol)


def build_viewer(mol_block, style="stick"):
    viewer = py3Dmol.view(width=900, height=650)
    viewer.addModel(mol_block, "mol")

    if style == "stick":
        viewer.setStyle({"stick": {}})
    elif style == "sphere":
        viewer.setStyle({"sphere": {"scale": 0.28}, "stick": {"radius": 0.12}})
    elif style == "line":
        viewer.setStyle({"line": {}})
    elif style == "ball_stick":
        viewer.setStyle({"stick": {"radius": 0.18}, "sphere": {"scale": 0.25}})

    viewer.zoomTo()
    viewer.setBackgroundColor("white")
    return viewer


with st.sidebar:
    st.header("Sampling Settings")
    ckpt_path = st.text_input(
        "Checkpoint Path",
        "logs/qm9_full_ddpm_2losses_2026_03_26__16_57_55/checkpoints/450.pt"
    )
    num_samples = st.slider("Number of Molecules", min_value=1, max_value=100, value=20, step=1)
    batch_size = st.slider("Batch Size", min_value=1, max_value=50, value=20, step=1)
    device = st.selectbox("Device", ["cuda", "cpu"], index=0)
    sampling_type = st.selectbox("Sampling Type", ["generalized", "ddpm_noisy", "ld"], index=0)

    clip = st.number_input("clip", value=3.0)
    eta = st.number_input("eta", value=1.0)
    w_global_pos = st.number_input("w_global_pos", value=1.0)
    w_local_pos = st.number_input("w_local_pos", value=1.5)
    w_global_node = st.number_input("w_global_node", value=0.4)
    w_local_node = st.number_input("w_local_node", value=0.8)

    add_hs = st.checkbox("Add Hydrogens", value=True)
    optimize = st.checkbox("Optimize Structure", value=True)

    run_name = st.text_input("Run Name", f"web_run_{int(time.time())}")
    generate_btn = st.button("Generate Molecules", use_container_width=True)


if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "molecules" not in st.session_state:
    st.session_state.molecules = []
if "output_dir" not in st.session_state:
    st.session_state.output_dir = None


if generate_btn:
    with st.spinner("Generating molecules..."):
        metrics, molecules, output_dir = generate_for_web(
            ckpt_path=ckpt_path,
            num_samples=num_samples,
            batch_size=batch_size,
            device=device,
            sampling_type=sampling_type,
            clip=float(clip),
            eta=float(eta),
            w_global_pos=float(w_global_pos),
            w_local_pos=float(w_local_pos),
            w_global_node=float(w_global_node),
            w_local_node=float(w_local_node),
            add_hs=add_hs,
            optimize=optimize,
            output_root="./web_results",
            run_name=run_name,
        )
        st.session_state.metrics = metrics
        st.session_state.molecules = molecules
        st.session_state.output_dir = output_dir


metrics = st.session_state.metrics
molecules = st.session_state.molecules

if metrics is not None:
    st.subheader("Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Validity", f"{metrics['validity']:.4f}")
    c2.metric("Stability", f"{metrics['stability']:.4f}")
    c3.metric("Uniqueness", f"{metrics['uniqueness']:.4f}")
    c4.metric("Novelty", f"{metrics['novelty']:.4f}")

    st.caption(f"Saved to: {st.session_state.output_dir}")

    if molecules:
        st.divider()
        style = st.selectbox("Render Style", ["stick", "sphere", "line", "ball_stick"], index=0)

        options = [
            f"[{m['id']:03d}] {m['file_name']} | atoms={m['num_atoms']} | smiles={m['smiles'][:60]}"
            for m in molecules
        ]
        selected_idx = st.selectbox(
            "Choose Molecule",
            range(len(options)),
            format_func=lambda i: options[i]
        )

        selected = molecules[selected_idx]
        mol = load_first_valid_mol(selected["sdf_path"])

        if mol is not None:
            col1, col2 = st.columns([2.3, 1])

            with col1:
                viewer = build_viewer(mol_to_block(mol), style=style)
                showmol(viewer, height=650, width=900)
                st.caption("支持鼠标拖动旋转，滚轮缩放。")

            with col2:
                st.subheader("Molecule Info")
                st.write(f"**ID:** {selected['id']}")
                st.write(f"**File:** {selected['file_name']}")
                st.write(f"**Atoms:** {selected['num_atoms']}")
                st.write(f"**Bonds:** {selected['num_bonds']}")
                st.write(f"**SMILES:** `{selected['smiles']}`")
                st.write(f"**Status:** {selected['status']}")
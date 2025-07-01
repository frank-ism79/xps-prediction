import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from dscribe.descriptors import SOAP
import tempfile
import os
import py3Dmol
from scipy.ndimage import gaussian_filter1d
from ase.visualize.plot import plot_atoms
import streamlit.components.v1 as components
# Carica il modello ML da file locale una sola volta
@st.cache_resource
def load_embedded_model():
    with open("C1s-v1.sav", "rb") as f:
        return pickle.load(f)

# Configuration of SOAP descriptors
def configure_soap():
    return SOAP(
        species=["H", "C", "O", "N", "F", "Cl", "Br", "I", "S"],
        r_cut=8.0,
        n_max=8,
        l_max=4,
        sigma=0.05,
        periodic=False,
        rbf='gto',
        average="off"
    )

# Generate descriptors of C atoms
def extract_features(molecule, soap, symbol="C"):
    all_descriptors = soap.create(molecule)
    indices = [i for i, atom in enumerate(molecule) if atom.symbol == symbol]
    descriptors = all_descriptors[indices]
    return descriptors, indices

# BE prediction
def predict_binding_energies(model, descriptors):
    return model.predict(descriptors)
# Show molecule
def show_3d_molecule(xyz_string):
    view = py3Dmol.view(width=500, height=400)
    view.addModel(xyz_string, 'xyz')
    view.setStyle({'stick': {}})
    view.zoomTo()

    lines = xyz_string.strip().splitlines()
    atoms = lines[2:]  # Salta le prime due righe XYZ
    for i, atom in enumerate(atoms):
        parts = atom.split()
        if len(parts) >= 4:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            if parts[0]=="C":
                view.addLabel(str(i), {"position": {"x": x, "y": y, "z": z}, "backgroundColor": "white", "fontColor": "black", "fontsize" : "3"})

    components.html(view._make_html(), height=420)

# -----------------------------  STREAMLIT INTERFACE -----------------------------
st.image("logo-cnr-ism-icsc.png",width=1000) # use_container_width=True)
st.set_page_config(page_title="XPS C1s Predictor", layout="centered")
st.title("🔬 C1s BEs for isolated molecules")
st.markdown("""
Once the molecule is uploaded in XYZ format, the geometrical coordinates are converted into rotationally invariant descriptors.The app uses a pre-trained machine learning model to predict C1s binding energies for molecular systems containing elements ranging from N, O, and S to halogens from F to I. """)

xyz_file = st.file_uploader("📁 Upload molecular geometry(.xyz)", type=["xyz"])

if xyz_file:
    #xyz_content = xyz_file.read().decode("utf-8").strip()
    #print(xyz_content)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xyz") as temp_xyz:
        temp_xyz.write(xyz_file.read())
        temp_xyz_path = temp_xyz.name

    try:
        molecule = read(temp_xyz_path)
        model = load_embedded_model()
        soap = configure_soap()

        descriptors, indices = extract_features(molecule, soap, symbol="C")
        predictions = predict_binding_energies(model, descriptors)

        st.success("✅ Prediction completed")

        st.subheader("📊 Simulated XPS spectra")

        # Parametri DOS
        energy_range = np.linspace(min(predictions) - 2, max(predictions) + 2, 1000)
        dos = np.zeros_like(energy_range)
        sigma = 0.3  # broadening in eV

        for e in predictions:
            dos += np.exp(-0.5 * ((energy_range - e) / sigma) ** 2)
        dos /= (sigma * np.sqrt(2 * np.pi))  # normalizzazione Gaussiana

        fig, ax = plt.subplots()
        ax.plot(energy_range, dos, color='navy')
        for i in range(len(predictions)):
            ax.plot([predictions[i],predictions[i]],[0,0.5])
        ax.fill_between(energy_range, dos, color='skyblue', alpha=0.5)
        ax.set_xlabel("BE (eV)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title("Simulated XPS spectra")
        ax.invert_xaxis()  # Per XPS, scala energetica decrescente
        st.pyplot(fig)
        col1, col2 = st.columns([1, 2]) 
        with(col1):
            for idx, energy in zip(indices, predictions):
                st.write(f"C #{idx}: Predicted BE = {energy:.2f} eV")
        with(col2):
            #st.subheader("🧬Atoms index")
            xyz_file.seek(0)
            xyz_content = xyz_file.read().decode("utf-8").strip()
            print(xyz_content)
            show_3d_molecule(xyz_content)


    except Exception as e:
        st.error(f"Errore durante l'elaborazione: {str(e)}")
    finally:
        os.unlink(temp_xyz_path)
#else:
    #st.info("📄 Carica un file .xyz per iniziare la predizione.")
url = "https://doi.org/10.1063/5.0272583"
st.write("References: \n F.Porcelli, F.Filippone, E.Colasante and G.Mattioli,**Photoemission spectroscopy of organic molecules using plane wave/pseudopotential density functional theory and machine learning: A comprehensive and predictive computational protocol for isolated molecules, molecular aggregates, and organic thin films** J. Chem. Phys. 162, 244101 (**2025**) [link](%s)"  % url)

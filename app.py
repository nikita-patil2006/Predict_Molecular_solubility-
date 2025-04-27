import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from sklearn.model_selection import GridSearchCV

if "loaded" not in st.session_state:
    st.session_state.loaded = False

# Custom CSS for loading animation, atomic background, and transition effects
loading_css = """
<style>
/* Atomic background with animated atoms and bonds */
@keyframes rotateBackground {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
body {
    background-color: #1a1a2e;
    font-family: Arial, sans-serif;
    margin: 0;
    overflow: hidden;
    height: 100%;
}
.container {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 100vh;
    text-align: center;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    z-index: 1; /* Ensure it is on top of the background */
    opacity: 0;
    animation: fadeIn 2s forwards;
}
.loadingWrapper {
    display: flex;
    align-items: center;
    margin-bottom: 2rem;
}
.loadingText {
    font-size: 2rem;
    color: #e0e0e0;
    margin-right: 1rem;
    animation: fadeInText 2s ease-in-out;
}
.loadingCircle {
    width: 20px;
    height: 20px;
    border: 3px solid #3282b8;
    border-top: 3px solid #e0e0e0;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
@keyframes fadeIn {
    0% { opacity: 0; }
    100% { opacity: 1; }
}
@keyframes fadeInText {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInPage {
    0% { opacity: 0; }
    100% { opacity: 1; }
}

/* Atomic background with animated circles representing atoms */
@keyframes pulse {
    0% {
        transform: scale(1);
        opacity: 0.6;
    }
    50% {
        transform: scale(1.2);
        opacity: 0.8;
    }
    100% {
        transform: scale(1);
        opacity: 0.6;
    }
}
.atomic-background {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;  /* Make sure background is behind the content */
    background: radial-gradient(circle, rgba(0,0,0,0.1) 10%, transparent 10%),
                radial-gradient(circle, rgba(0,0,0,0.1) 10%, transparent 10%);
    background-size: 150px 150px;
    animation: rotateBackground 60s infinite linear;
}

.atomic-background .atom {
    position: absolute;
    border-radius: 50%;
    background-color: #3282b8;
    opacity: 0.6;
    animation: pulse 2s infinite ease-in-out;
}

.page-content {
    opacity: 0;
    animation: fadeInPage 2s 1s forwards;
}
</style>
"""
# Initialize session state for app loading and navigation
if "loaded" not in st.session_state:
    st.session_state.loaded = False
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Home"  # Initialize to "Home" by default
# Display the loading screen if the app is not yet loaded
if not st.session_state.loaded:
    st.markdown(loading_css, unsafe_allow_html=True)
    
    # Show loading animation
    st.markdown('<div class="container"><div class="loadingWrapper">'
                '<div class="loadingText">Loading Molecular Solubility App...</div>'
                '<div class="loadingCircle"></div></div></div>', unsafe_allow_html=True)
    
    # Wait for 4 seconds
    time.sleep(4)
    
    # Set loaded state to True and rerun the app
    st.session_state.loaded = True
    st.rerun()
# Set Streamlit page configuration
st.set_page_config(page_title="Molecular Solubility Predictor", layout="wide")

# Function to load custom CSS
def load_css(css_path):
    """Loads custom CSS from an external file."""
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ CSS file not found: {css_path}. Styles may not be applied.")

st.markdown(
    """
    <div class="orbit-container orbit1">
        <div class="atom atom1"></div>
    </div>
    <div class="orbit-container orbit2">
        <div class="atom atom2"></div>
    </div>
    <div class="orbit-container orbit3">
        <div class="atom atom3"></div>
    </div>
    <div class="orbit-container orbit4">
        <div class="atom atom4"></div>
    </div>
    <div class="orbit-container orbit5">
        <div class="atom atom5"></div>
    </div>

    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #0A192F;
        overflow: hidden;
    }

    .atom {
        position: absolute;
        width: 15px;
        height: 15px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(173,216,230,0.8) 0%, rgba(173,216,230,0) 70%);
    }

    .orbit-container {
        position: fixed;
        width: 200px;
        height: 200px;
        border-radius: 50%;
    }

    .orbit1 { top: 15%; left: 20%; animation: rotate1 6s linear infinite; }
    .orbit2 { top: 15%; left: 70%; animation: rotate2 8s linear infinite; }
    .orbit3 { top: 50%; left: 50%; animation: rotate3 7s linear infinite; }
    .orbit4 { top: 75%; left: 25%; animation: rotate4 5s linear infinite; }
    .orbit5 { top: 75%; left: 75%; animation: rotate5 9s linear infinite; }

    .atom1 { animation: orbit1 4s linear infinite; }
    .atom2 { animation: orbit2 6s linear infinite; }
    .atom3 { animation: orbit3 8s linear infinite; }
    .atom4 { animation: orbit4 5s linear infinite; }
    .atom5 { animation: orbit5 7s linear infinite; }

    @keyframes rotate1 {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    @keyframes rotate2 {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(-360deg); }
    }
    @keyframes rotate3 {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    @keyframes rotate4 {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(-360deg); }
    }
    @keyframes rotate5 {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    @keyframes orbit1 {
        0% { transform: rotate(0deg) translateX(60px) rotate(0deg); }
        100% { transform: rotate(360deg) translateX(60px) rotate(-360deg); }
    }
    @keyframes orbit2 {
        0% { transform: rotate(0deg) translateX(80px) rotate(0deg); }
        100% { transform: rotate(360deg) translateX(80px) rotate(-360deg); }
    }
    @keyframes orbit3 {
        0% { transform: rotate(0deg) translateX(100px) rotate(0deg); }
        100% { transform: rotate(360deg) translateX(100px) rotate(-360deg); }
    }
    @keyframes orbit4 {
        0% { transform: rotate(0deg) translateX(120px) rotate(0deg); }
        100% { transform: rotate(360deg) translateX(120px) rotate(-360deg); }
    }
    @keyframes orbit5 {
        0% { transform: rotate(0deg) translateX(140px) rotate(0deg); }
        100% { transform: rotate(360deg) translateX(140px) rotate(-360deg); }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load CSS file (Ensure the correct path)
css_path = "style.css"
load_css(css_path)

# Function to compute aromatic proportion safely
def AromaticProportion(m):
    """Calculate the proportion of aromatic atoms in a molecule. Prevents division by zero."""
    if not m:
        return 0.0  # Return 0 for invalid molecules
    aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
    AromaticAtom = sum(1 for i in aromatic_atoms if i)
    HeavyAtom = Descriptors.HeavyAtomCount(m)
    return AromaticAtom / HeavyAtom if HeavyAtom > 0 else 0.0  # Avoid division by zero

# Function to generate molecular descriptors
def generate(smiles):
    moldata = [Chem.MolFromSmiles(elem) for elem in smiles]
    baseData, valid_smiles = [], []

    for smi, mol in zip(smiles, moldata):
        if mol:
            row = [
                Descriptors.MolLogP(mol),
                Descriptors.MolWt(mol),
                Descriptors.NumRotatableBonds(mol),
                AromaticProportion(mol)
            ]
            baseData.append(row)
            valid_smiles.append(smi)
        else:
            st.warning(f"⚠️ Invalid SMILES string skipped: {smi}")

    return pd.DataFrame(baseData, columns=["MolLogP", "Molecular Weight", "NumRotatableBonds", "AromaticProportion"]), valid_smiles

# Function to load dataset and train the model
@st.cache_data(show_spinner=False, persist=False)
# Function to categorize solubility based on logS
def solubility_judgment(logS):
    if logS > 0:
        return "Highly soluble"
    elif 0 > logS > -1:
        return "Moderately soluble"
    elif -1 > logS > -2:
        return "Low solubility"
    elif -2 > logS > -3:
        return "Poorly soluble"
    else:
        return "Practically insoluble"
    
def load_data_and_train_model():
    dataset_filepath = "augmented_data.csv"  # Update this path as needed
    if not os.path.exists(dataset_filepath):
        st.error(f"Dataset file '{dataset_filepath}' not found.")
        st.stop()

    # Load dataset
    dataset = pd.read_csv(dataset_filepath)
    dataset.rename(columns={"MolWt": "Molecular Weight"}, inplace=True)

    # Define Features & Target
    y = dataset['logS']
    X = dataset.drop(columns=['logS'])

    # ✅ Split dataset: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # ✅ Random Forest with Regularization
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=15,          # Prevents overfitting
        min_samples_split=4,   # Minimum samples required to split
        max_features='sqrt',   # Random feature selection
        random_state=42,
        verbose=0
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # ✅ Performance Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

    val_r2 = r2_score(y_val, y_pred_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    return model, X.columns, (train_r2, train_rmse, val_r2, val_rmse, test_r2, test_rmse)


# Load pre-trained model
model, model_features, metrics = load_data_and_train_model()

# Sidebar Navigation
st.sidebar.header("📌 Navigation")
if st.sidebar.button("Home"):
    st.session_state.selected_page = "Home"
if st.sidebar.button("Meet Our Team"):
    st.session_state.selected_page = "Meet Our Team"
if st.sidebar.button("Model Training & Performance"):
    st.session_state.selected_page = "Model Training & Performance"
st.sidebar.header("Input Method")
if st.sidebar.button("SMILES"):
    st.session_state.selected_page = "SMILES"
if st.sidebar.button("Manual Descriptors"):
    st.session_state.selected_page = "Manual Input"

# Retrieve the selected page from session state
selected_page = st.session_state.selected_page

# Display Content Based on Selection
if selected_page == "Home":
    st.title("🏠 Home")
    st.write("""
    # 🔬Welcome to the Molecular Solubility Predictor 

    ## What is This App?  
    This application allows you to **predict the solubility of molecules** using advanced AI models.  
    Whether you're a **chemist, researcher, or data scientist**, this tool helps you analyze molecular solubility quickly and accurately.  

    ## How Does It Work?  
    - Input a **SMILES string** (Simplified Molecular Input Line Entry System).  
    - Our AI model predicts the **logS value (solubility)** of the molecule.  
    - The molecule is classified as **highly soluble, moderately soluble, or poorly soluble** based on its logS value.  
    - Visualize the molecule with an interactive structure viewer.  

    ## Why Use This App?  
    ✅ **AI-Powered Predictions:** Leverages cutting-edge **deep learning** for accurate solubility predictions.  
    ✅ **Easy-to-Use Interface:** Simply enter a SMILES string or manually input molecular descriptors.  
    ✅ **Visual & Interactive:** Get molecule structures, solubility categories, and performance insights.  
    ✅ **Customizable & Expandable:** Designed for researchers, students, and professionals in cheminformatics.  

    ## Try It Now! 🚀  
    🔹 **Enter a SMILES string** below to predict solubility instantly.  
    🔹 **Explore model training & performance** in the sidebar.  
    🔹 **Meet our team** and check out our GitHub profiles.  
    """)

elif selected_page == "Meet Our Team":
    st.title("Meet Our Team")
    
    # Define team members
    team_members = [
        {
            "name": "John Theroth",
            "github": "https://github.com/John-skech",
            "profile_pic": "https://github.com/John-skech.png"  
        },
        {
            "name": "Niya Johnson",
            "github": "https://github.com/niyajohnsonk",
            "profile_pic": "https://github.com/niyajohnsonk.png" 
        },
        {
            "name": "Nikita Patil",
            "github": "https://github.com/nikita-patil2006",
            "profile_pic": "https://github.com/nikita-patil2006.png"  
        },
        {
            "name": "Allen Peter",
            "github": "https://github.com/allen1407",
            "profile_pic": "https://github.com/allen1407.png"
        }
    ]

    # Use columns to display GitHub profile pictures
    cols = st.columns(4)  # Create 4 columns for 4 team members

    for idx, member in enumerate(team_members):
        with cols[idx]:
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <a href="{member['github']}" target="_blank">
                        <img src="{member['profile_pic']}" 
                             alt="{member['name']}" 
                             style="
                                 width: 200px;
                                 height: 200px;
                                 border-radius: 50%;
                                 object-fit: cover;
                                 border: 2px solid #333;
                                 margin-bottom: 10px;
                             ">
                    </a>
                    <h6 style="margin-top: 10px; margin-bottom: 5px;">{member['name']}</h6>
                </div>
                """,
                unsafe_allow_html=True
            )
elif selected_page == "Model Training & Performance":
    st.title("📊 Model Training & Performance")
    train_r2, train_rmse, test_r2, test_rmse, val_r2, val_rmse = metrics
    st.write(f"✅ **Training R²**: {train_r2:.4f}")
    st.write(f"✅ **Training RMSE**: {train_rmse:.4f}")
    st.write(f"✅ **Testing R²**: {test_r2:.4f}")
    st.write(f"✅ **Testing RMSE**: {test_rmse:.4f}")
    st.write(f"✅ **Validation R²**: {val_r2:.4f}")
    st.write(f"✅ **Validation RMSE**: {val_rmse:.4f}")
    metrics_df = pd.DataFrame({
        "Metric": ["Training R²", "Training RMSE", "Testing R²", "Testing RMSE", "Validation R²", "Validation RMSE"],
        "Value": [train_r2, train_rmse, test_r2, test_rmse, val_r2, val_rmse]
    })
elif selected_page == "SMILES":
    st.header("SMILES Input")
    SMILES_input = st.text_area("Enter SMILES (one per line):", placeholder="Example: CCO (ethanol)\nCC(=O)O (acetic acid)")
    SMILES = SMILES_input.strip().split('\n') if SMILES_input else []

    if st.button("🔮 Predict Solubility", use_container_width=True):
        valid_mols = []
        valid_SMILES = []

        for smi in SMILES:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                valid_mols.append(mol)
                valid_SMILES.append(smi)
            else:
                st.error(f"⚠️ Invalid SMILES: {smi}")

        if valid_mols:
            st.header("Molecular Structures and Atom Color Legend")
            cols = st.columns([2, 3])
            with cols[0]:
                img = Draw.MolsToGridImage(valid_mols, molsPerRow=3, subImgSize=(200, 200))
                st.image(img, use_container_width=True)
            with cols[1]:
                st.markdown("""
                    **Atom Color Legend in SMILES:**
                    - ⚫️ **Carbon (C)**
                    - ⚪ **Hydrogen (H)**
                    - 🔴 **Oxygen (O)**
                    - 🔵 **Nitrogen (N)**
                    - 🟡 **Sulfur (S)**
                    - 🟠 **Phosphorus (P)**
                    - 🟢 **Halogens (F, Cl, Br, I)**
                """)

            X, valid_SMILES = generate(valid_SMILES)

            if not X.empty:
                st.header("Computed Molecular Descriptors")
                st.write(X)

                st.header("Predicted LogS Values")
                prediction = model.predict(X)
                results = pd.DataFrame({
                    "SMILES": valid_SMILES,
                    "Predicted LogS": prediction,
                    "Solubility Judgment": [solubility_judgment(p) for p in prediction]
                })
                st.write(results)

        elif SMILES:
            st.error("⚠️ No valid SMILES entered.")
            
elif selected_page == "Manual Input":
    st.header("Manual Descriptor Input")
    st.write("Provide molecular descriptor values for prediction.")

    descriptor_ranges = {
        "Molecular Weight": (0.0, 5000.0),
        "NumRotatableBonds": (0.0, 50.0),
        "AromaticProportion": (0.0, 1.0),
        "MolLogP": (-10.0, 15.0),
    }

    cols = st.columns(3)
    input_data = {}
    invalid_inputs = []

    for i, feature in enumerate(model_features):
        min_val, max_val = descriptor_ranges.get(feature, (None, None))
        placeholder = f"Range: {min_val} - {max_val}" if min_val is not None else "Enter value"

        with cols[i % 3]:
            value = st.text_input(feature, placeholder=placeholder)

            try:
                if value:
                    num_value = float(value)
                    if min_val is not None and not (min_val <= num_value <= max_val):
                        invalid_inputs.append(f"⚠️ {feature} must be between {min_val} and {max_val}.")
                    else:
                        input_data[feature] = num_value
                else:
                    input_data[feature] = None
            except ValueError:
                invalid_inputs.append(f"⚠️ {feature} must be a valid number.")

    if invalid_inputs:
        for msg in invalid_inputs:
            st.warning(msg)

    if st.button("🔮 Predict Solubility", use_container_width=True):
        if any(v is None for v in input_data.values()) or invalid_inputs:
            st.error("⚠️ Please enter valid molecular descriptor values within the specified ranges.")
        else:
            with st.spinner("🔄 Calculating solubility..."):
                predicted_solubility = model.predict(pd.DataFrame([input_data], columns=model_features))[0]
                judgment = solubility_judgment(predicted_solubility)
            st.success(f"✅ **Predicted Solubility (logS): {predicted_solubility:.4f}**")
            st.success(f"✅ **Solubility Judgment: {judgment}**")



st.markdown("---")

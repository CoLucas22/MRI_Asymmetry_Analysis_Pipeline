# 🧠 MRI Asymmetry Analysis

Project for **Magnetic Resonance Imaging (MRI)** analysis focusing on asymmetry measures from medical images.

---

## 📂 Repository Structure

```
.
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── data_example/           # Example dataset (MRI + CSV)
│   ├── MRIs/               # Raw MRI images (DICOM)
│   ├── CSV_files/          # Intermediate CSV files
│   ├── data_aggregated.csv # Aggregated example data
│   └── Runs_dataset.csv    # Dataset metadata
├── results/                # Generated outputs
│   └── figures/            # Exported figures
└── scripts/                # Analysis scripts
    ├── preprocess.py       # MRI preprocessing
    ├── extract_features.py # Feature extraction
    ├── visualize.py        # Visualization 
    └── utils.py            # Utility functions
```

---

## ⚙️ Installation

Create a Python environment (recommended: `conda` or `venv`) and install dependencies:

```bash
# 1. Create environment
python -m venv venv_mri

# 2. Activate it
source venv_mri/bin/activate


# 3. Upgrae pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### 3. Generate Results Visualization
```bash
python scripts/visualize.py --input data_example/MRIs/Patient_1/MRI_1/export_00062.DCM
```

---

## 📦 Main Dependencies
- `numpy`, `scipy`, `pandas`
- `matplotlib`, `seaborn`
- `opencv-python`
- `SimpleITK`

(Full list available in `requirements.txt`)

---

## 📝 Notes
- Example MRI files (`.DCM`) are provided in `data_example/MRIs/`.  
- Scripts accept command-line arguments (`--input`, `--output`).  
- Outputs (features, figures) are stored in `results/`.  

---

## 👤 Author
Developed by **Corentin Lucas**, PhD Student in INRIA Rennes, Dyliss Team, for medical imaging analysis and research of microbiome biomarkers related to Crohn's disease.

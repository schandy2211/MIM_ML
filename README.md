# 🧠 MIM-ML: Predicting NMR Shifts of Nucleic Acids using ML + Quantum Fragments

MIM-ML is a Random Forest-based machine learning model that predicts **¹H and ¹³C NMR chemical shifts** of nucleic acids using quantum chemical fragment-based descriptors.

It combines **structural and electronic features** derived from low-cost semi-empirical (GFN2-xTB) calculations, trained on high-accuracy DFT chemical shifts computed via the **Molecules-in-Molecules (MIM)** fragmentation scheme.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8ef9c252-f907-408f-b5b8-bd0c1a3405c9" width="400"/>
</p>

---

## 🔬 Why This Matters

Traditional quantum NMR predictions are computationally expensive for biomolecules.  
MIM-ML enables:
- **Rapid predictions** for large nucleic acids
- **Accuracy comparable to full DFT**, with orders of magnitude less cost
- **Integration of electronic + structural features** from quantum fragments

---

## 📊 Dataset & Training Details

- **Training set**: 2080 ¹H and 1780 ¹³C chemical shifts across 10 nucleic acids  
- **Reference method**: [MIM-DFT + microsolvation](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.2c00967)  
- **Descriptors**: GFN2-xTB-based electronic properties + structural features  
- **Model**: Random Forest (scikit-learn)

---


## 🚀 How to Use
git clone https://github.com/schandy2211/mim-ml.git


# Requirements
- python(version>=3.6)
* numpy
+ Biopython
- scikit-learn

# Cite as 
[Chandy, S. K.; Raghavachari, K. MIM-ML: A Novel Quantum Chemical Fragment-Based Random Forest Model for Accurate Prediction of NMR Chemical Shifts of Nucleic Acids. Journal of Chemical Theory and Computation 2023, 19 (19), 6632-6642.](https://pubs.acs.org/doi/10.1021/acs.jctc.3c00563)

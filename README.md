# **MIM-ML: A Novel Quantum Chemical Fragment-Based Random Forest Model for Accurate Prediction of NMR Chemical Shifts of Nucleic Acids**

MIM-ML is a random forest machine learning (ML) model for the prediction of 1H and 13C NMR chemical shifts of nucleic acids. Our ML model is trained entirely on reproducing computed chemical shifts obtained previously on [10 nucleic acids](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.2c00967) using a [Molecules-in-Molecules](https://pubs.acs.org/doi/full/10.1021/ct200033b)(MIM) fragment-based density functional theory (DFT) protocol including microsolvation effects. Our ML model includes structural descriptors as well as electronic descriptors from an inexpensive low-level semi-empirical calculation (GFN2-xTB) and trained on a relatively small number of DFT chemical shifts (2080 1H chemical shifts and 1780 13C chemical shifts on the 10 nucleic acids). 
![image](https://github.com/user-attachments/assets/8ef9c252-f907-408f-b5b8-bd0c1a3405c9)


# Requirements
- python(version>=3.6)
* numpy
+ Biopython
- scikit-learn

# Cite as 
[Chandy, S. K.; Raghavachari, K. MIM-ML: A Novel Quantum Chemical Fragment-Based Random Forest Model for Accurate Prediction of NMR Chemical Shifts of Nucleic Acids. Journal of Chemical Theory and Computation 2023, 19 (19), 6632-6642.](https://pubs.acs.org/doi/10.1021/acs.jctc.3c00563)

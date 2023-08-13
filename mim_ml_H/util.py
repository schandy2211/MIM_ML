from Bio.PDB import PDBParser, NeighborSearch
#from Bio.PDB.DSSP import DSSP
import numpy as np
#from Bio.PDB.Polypeptide import get_atom_mass
from Bio.PDB.PDBExceptions import PDBException, PDBConstructionException
from Bio.PDB.Polypeptide import aa1, aa3, is_aa, three_to_one, one_to_three, \
    CaPPBuilder, PPBuilder, \
    PPBuilder
#from Bio.PDB.Atom import Atom.element.mass
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import sys
from Bio.SeqUtils import molecular_weight
from Bio.Data import IUPACData
from periodictable import elements
import warnings
from Bio import BiopythonWarning

warnings.simplefilter("ignore", BiopythonWarning)

def get_molecular_weight(symbol):
    # lookup atomic weight of element in periodic table
    if symbol.startswith('H'):
        return 1.008
    elif symbol.startswith('C'):
        return 12.011
    elif symbol.startswith('N'):
        return 14.007
    elif symbol.startswith('O'):
        return 15.999
    elif symbol.startswith('S'):
        return 32.066
    elif symbol.startswith('P'):
        return 30.970
    else:
        return 0
def get_en(symbol):
    # lookup atomic weight of element in periodic table
    if symbol.startswith('H'):
        return 2.20
    elif symbol.startswith('C'):
        return 2.55
    elif symbol.startswith('N'):
        return 3.04
    elif symbol.startswith('O'):
        return 3.44
    elif symbol.startswith('P'):
        return 2.19
    else:
        return 0
def get_atomic_number(element_symbol):
    """
    Given an element symbol, return its atomic number.
    """
    element = elements.symbol(element_symbol)
    return element.number

def extract_feature_vector(pdb_file, chem_shift_file, csv_file, csv_file1):
    pdb_id = pdb_file.split('/')[-1].split('.')[0] # extract PDB ID from filename
    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure(pdb_file.split(".")[0], pdb_file)
 #   dssp = DSSP(structure[0], pdb_file)
    ns = NeighborSearch(list(structure.get_atoms()))
    feature_vector = {}
    with open(chem_shift_file) as f:
        chem_shifts = f.readlines()
    chem_shifts = [float(x.strip()) for x in chem_shifts]
    import csv

    # Read csv file and extract point1 values
    rad = []
    SASA = []
    hbond = []
    with open(csv_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rad.append(float(row['rad/Å']))
            SASA.append(float(row['SASA/Å²']))
            hbond.append(float(row['H-bond']))
   
    # Read csv file and extract point1 values
    CN = [] #covCN,q,C6AA,α(0)
    q = []
    AA = []
    Alpha = []
    with open(csv_file1) as csvfile1:
        reader = csv.reader(csvfile1)
        next(reader) 
        for row in reader:
            CN.append(float(row[3]))
            q.append(float(row[4]))
            AA.append(float(row[5]))
            Alpha.append(float(row[6]))
    i = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    neighbors = ns.search(atom.get_coord(), 2.0) # distance cutoff of 2.0 angstrom
                    neighbor_names = []
                    neighbor_resnames = []
                    neighbor_resids = []
                    for neighbor in neighbors:
                        neighbor_names.append(neighbor.get_name())
                        neighbor_resnames.append(neighbor.get_parent().get_resname())
                        neighbor_resids.append(neighbor.get_parent().get_id()[1])
                    feature_vector[structure,atom] = {
                        'pdb_id': pdb_id, # add PDB ID to dictionary
                        'name': atom.get_name(),
                        'rad': rad[i],
                        'SASA': SASA[i],
                        'hbond': hbond[i],
                        'CN': CN[i],
                        'q': q[i],
                        'AA': AA[i],
                        'Alpha': Alpha[i],
                        'resname': residue.get_resname(),
                        'chem_shift': chem_shifts[i],
                        'neighbor_names': neighbor_names,
                        'neighbor_resnames': neighbor_resnames,
                        'mw' : get_molecular_weight(atom.element),
                        'an' : get_atomic_number(atom.element),
                        'en' : get_en(atom.element)
                    }
                    i += 1
    return feature_vector

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py pdb_file chem_shift_file")
        sys.exit()
    pdb_file = sys.argv[1]
    chem_shift_file = sys.argv[2]
    feature_vector = extract_feature_vector(pdb_file, chem_shift_file)
    print(feature_vector)
    print(len(feature_vector))




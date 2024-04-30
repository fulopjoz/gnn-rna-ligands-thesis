import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
import numpy as np

class DiversityPicker:
    def __init__(self, sdf_path, num_to_pick, radius=3, nBits=2048):
        self.sdf_path = sdf_path
        self.num_to_pick = num_to_pick
        self.radius = radius
        self.nBits = nBits
        self.df = pd.DataFrame()
    
    @staticmethod    
    def load_sdf_to_dataframe(sdf_path):  # Corrected: Removed 'self', use 'sdf_path'
        suppl = Chem.SDMolSupplier(sdf_path)  # Use 'sdf_path' directly
        rows = []
        for mol in suppl:
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                rna = mol.GetProp('rna') if mol.HasProp('rna') else 'NA'
                rows.append({'mol': mol, 'SMILES': smiles, 'rna': rna})
        return pd.DataFrame(rows)
    
    def add_fingerprints_to_dataframe(self):
        self.df['fingerprints'] = self.df['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, self.radius, self.nBits))
        
    def diversity_picking(self):
        fps = list(self.df['fingerprints'])
        len_fps = len(fps)
        mmp = MaxMinPicker()
        
        def distij(i, j, fps=fps):
            return 1 - DataStructs.TanimotoSimilarity(fps[i], fps[j])
        
        picks = mmp.LazyPick(distij, len_fps, self.num_to_pick, [])
        
        self.picked_df = self.df.iloc[picks].reset_index(drop=True)

    def save_to_sdf(self, output_sdf_path):
        writer = Chem.SDWriter(output_sdf_path)
        for i, row in self.picked_df.iterrows():
            writer.write(row['mol'])
        writer.close()
        
    def run(self):
        # Adjusted to use the static method correctly
        self.df = DiversityPicker.load_sdf_to_dataframe(self.sdf_path)
        self.add_fingerprints_to_dataframe()
        self.diversity_picking()

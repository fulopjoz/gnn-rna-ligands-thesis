#!/usr/bin/env python3
# Author: Jozef Fulop
# Institution: UCT in Prague

from rdkit import Chem
from rdkit.Chem import AllChem, SDWriter
from chembl_structure_pipeline import standardizer
import pandas as pd
import logging
import sys
import os

# Setup logging
logging.basicConfig(filename='molecule_processing.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


class SuppressStdOutput:
    """Context manager to redirect stdout and stderr to the log file."""

    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.log_file = open('molecule_processing.log', 'a')
        sys.stdout = self.log_file
        sys.stderr = self.log_file
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.log_file.close()


class MoleculeProcessor:
    def __init__(
            self,
            sdf_paths,
            output_format='smi',
            deduplicated_filename='deduplicated',
            duplicates_filename='duplicates'):
        self.sdf_paths = sdf_paths  # This is now expected to be a list of strings
        self.output_format = output_format
        self.deduplicated_filename = deduplicated_filename
        self.duplicates_filename = duplicates_filename
        self.initial_counts = {}
        self.all_molecules_with_source = []
        self.deduplicated_molecules_with_source = []
        self.duplicates = []

    def run(self):
        self.process_and_combine_datasets()
        self.deduplicate_and_final_check()
        self.save_results()
        self.log_final_analysis()

    def load_molecules_from_sdf(self, sdf_path):
        logging.info(f"Loading molecules from {sdf_path}...")
        if not os.path.exists(sdf_path):
            logging.error(f"File not found: {sdf_path}")
            return []
        with SuppressStdOutput():
            suppl = Chem.SDMolSupplier(sdf_path)
            molecules = [(mol, sdf_path) for mol in suppl if mol is not None]
        logging.info(f"Loaded {len(molecules)} molecules from {sdf_path}")
        return molecules

    def standardize_and_desalt(self, mol_with_source):
        mol, source = mol_with_source
        if mol:
            with SuppressStdOutput():
                std_mol = standardizer.standardize_mol(mol)
                parent_mol, _ = standardizer.get_parent_mol(std_mol)
            if parent_mol and parent_mol.GetNumAtoms() > 0:
                return parent_mol, source
        return None, source

    def convert_to_2D(self, mol_with_source):
        """Convert molecule to 2D coordinates."""
        mol, source = mol_with_source
        if mol is not None:
            AllChem.Compute2DCoords(mol)
        return mol, source

    def process_and_combine_datasets(self):
        """Process and combine datasets from multiple SDF paths."""
        for path in self.sdf_paths:  # Assuming sdf_paths is a list of strings
            logging.info(f"\nProcessing dataset: {path}")
            molecules_with_source = self.load_molecules_from_sdf(path)
            self.initial_counts[path] = len(molecules_with_source)

            # Convert molecules to 2D, then standardize and desalt
            # Correction: Ensure that each method directly returns a tuple of
            # (mol, source) and filter out None values correctly
            molecules_with_source = [self.convert_to_2D(
                mol_with_source) for mol_with_source in molecules_with_source 
                                     if mol_with_source[0] is not None]
            
            molecules_with_source = [self.standardize_and_desalt(
                mol_with_source) for mol_with_source in molecules_with_source 
                                     if mol_with_source[0] is not None]

            self.all_molecules_with_source.extend(molecules_with_source)

    @staticmethod
    def deduplicate_and_final_check(df):
        df['SMILES'] = df['mol'].apply(lambda x: Chem.MolToSmiles(x, canonical=True) if x else '')
        df = df[~df['SMILES'].str.contains('\\.', regex=False)]
        duplicate_mask = df.duplicated('SMILES', keep=False)
        duplicates = list(df[duplicate_mask][['mol', 'source']].itertuples(index=False, name=None))
        deduplicated_molecules_with_source = list(df[~duplicate_mask][['mol', 'source']].itertuples(index=False, name=None))
        
        # Assuming you have a way to log or you might print instead
        print(f"Final deduplicated dataset contains {len(deduplicated_molecules_with_source)} unique molecules.")
        print(f"Duplicates removed: {len(duplicates)}")
        
        # Depending on what you want to return, adjust accordingly
        return deduplicated_molecules_with_source

    def save_smiles_to_txt_with_source(self, molecules_with_source, filename):
        with open(filename, 'w') as file:
            for mol, source in molecules_with_source:
                if mol:
                    smiles = Chem.MolToSmiles(mol, canonical=True)
                    file.write(f"{smiles}\t{source}\n")

    def save_as_sdf(self, molecules_with_source, filename):
        """Save molecules in SDF format, including the source filename (without extension) as a property."""
        logging.debug(f"Starting to save molecules to SDF file: {filename}")
        directory = os.path.dirname(filename)

        if directory:
            logging.debug(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)
        else:
            logging.debug(
                "No directory specified, using current working directory.")

        writer = SDWriter(filename)
        count = 0
        for mol, source in molecules_with_source:
            if mol:
                count += 1
                source_filename = os.path.splitext(
                    os.path.basename(str(source)))[0]
                logging.debug(
                    f"Processing molecule #{count} from source: {source_filename}")
                mol.SetProp("Source", source_filename)
                writer.write(mol)
            else:
                logging.warning(
                    f"Skipping invalid molecule from source: {source}")

        writer.close()
        logging.info(f"Finished writing {count} molecules to {filename}.")

    def save_results(self):
        dedup_file = f'{self.deduplicated_filename}.{self.output_format}'
        dup_file = f'{self.duplicates_filename}.{self.output_format}'

        # Ensure output directories are created
        for path in [dedup_file, dup_file]:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        if self.output_format == 'sdf':
            self.save_as_sdf(
                self.deduplicated_molecules_with_source,
                dedup_file)
            self.save_as_sdf(self.duplicates, dup_file)
        elif self.output_format == 'smi':
            self.save_smiles_to_txt_with_source(
                self.deduplicated_molecules_with_source, dedup_file)
            self.save_smiles_to_txt_with_source(self.duplicates, dup_file)
        else:
            logging.error("Unsupported output format specified.")

    def log_final_analysis(self):
        total_initial = sum(self.initial_counts.values())
        total_final = len(
            self.deduplicated_molecules_with_source) + len(self.duplicates)
        logging.info("\nFinal Analysis:")
        logging.info(f"Total molecules initially: {total_initial}")
        logging.info(f"Total unique molecules after processing: {total_final}")
        for path in self.sdf_paths:
            final_count = len(
                [m for m in self.deduplicated_molecules_with_source if m[1] == path])
            logging.info(
                f"{path}: Initial = {self.initial_counts[path]}, Final = {final_count}")
        logging.info(f"Duplicates removed: {len(self.duplicates)}")

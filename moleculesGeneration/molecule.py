import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, AllChem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.AtomPairs import Pairs
import requests
import json


class Molecule():

    def __init__(self, smiles=None, name=None):
        if all([smiles is None, name is None]):
            raise Exception("Wrong input")
        if smiles is not None:
            try:
                self.mol = Chem.MolFromSmiles(smiles)
                if self.mol is None: raise Exception("Was not able to process a molecule from smile: " + str(smiles))
            except:
                raise Exception("Was not able to process a molecule from smile: " + str(smiles))
            self.original_smile = smiles
        else:
            self.name = name
            try:
                self.original_smile = self.get_smiles_from_name_pubchem()
            except:
                raise Exception("Was not able to process a molecule from name: " + str(name))
            try:
                self.mol = Chem.MolFromSmiles(self.original_smile)
            except:
                raise Exception("Was not able to process a molecule from name-->smile: " + str(smiles))
        self.properties = {}
        self.have_properties = False

    def compute_all_properties(self):
        if self.have_properties: return self.properties
        self.logp()
        self.MolMR()
        self.numHDonors()
        self.numHAcceptors()
        self.numAtoms()
        self.exactMolWt()
        self.get_morgan_fp()
        self.rdkit_fp()
        self.atom_pair_fp()
        self.numRadicalElectrons()
        self.numValenceElectrons()
        self.TPSA()
        self.numAtoms()
        self.numBonds()
        self.HeavyAtomCount()
        self.HeavyAtomMolWt()
        self.RingCount()
        self.FractionCSP3()
        self.NHOHCount()
        self.NOCount()
        self.NumHeteroatoms()
        self.NumRotatableBonds()
        self.NumAmideBonds()
        self.NumAromaticRings()
        self.NumSaturatedRings()
        self.NumAliphaticRings()
        self.NumAromaticHeterocycles()
        self.NumAliphaticHeterocycles()
        self.NumSaturatedHeterocycles()
        self.NumAromaticCarbocycles()
        self.NumSaturatedCarbocycles()
        self.NumAliphaticCarbocycles()
        self.BalabanJindex()
        self.BertzCT()
        self.have_properties = True
        return self.properties

    def smiles(self):
        if 'Smiles' not in self.properties:
            self.properties['Smiles'] = Chem.MolToSmiles(self.mol)
        return self.properties['Smiles']

    def name(self):
        return

    def logp(self):
        '''Wildman-Crippen LogP (partition coefficient) value
        Wildman and G. M. Crippen JCICS 39 868-873 (1999) '''

        if 'MolLogP' not in self.properties:
            self.properties['MolLogP'] = Descriptors.MolLogP(self.mol)
        return self.properties['MolLogP']

    def MolMR(self):
        '''Wildman-Crippen MR (molar refractivity) value
        Wildman and G. M. Crippen JCICS 39 868-873 (1999) '''

        if 'MolMR' not in self.properties:
            self.properties['MolMR'] = Descriptors.MolMR(self.mol)
        return self.properties['MolMR']

    def numHDonors(self):
        if 'NumHDonors' not in self.properties:
            self.properties['NumHDonors'] = Descriptors.NumHDonors(self.mol)
        return self.properties['NumHDonors']

    def numHAcceptors(self):
        if 'NumHAcceptors' not in self.properties:
            self.properties['NumHAcceptors'] = Descriptors.NumHAcceptors(self.mol)
        return self.properties['NumHAcceptors']

    def exactMolWt(self):
        if 'ExactMolWt' not in self.properties:
            self.properties['ExactMolWt'] = Descriptors.ExactMolWt(self.mol)
        return self.properties['ExactMolWt']

    def numAtoms(self):
        if 'NumAtoms' not in self.properties:
            self.properties['NumAtoms'] = self.mol.GetNumAtoms()
        return self.properties['NumAtoms']

    def numRadicalElectrons(self):
        if 'NumRadicalElectrons' not in self.properties:
            self.properties['NumRadicalElectrons'] = Descriptors.NumRadicalElectrons(self.mol)
        return self.properties['NumRadicalElectrons']

    def numValenceElectrons(self):
        if 'NumValenceElectrons' not in self.properties:
            self.properties['NumValenceElectrons'] = Descriptors.NumValenceElectrons(self.mol)
        return self.properties['NumValenceElectrons']

    def TPSA(self):
        # topological polar surface area (TPSA)
        if 'TPSA' not in self.properties:
            self.properties['TPSA'] = Descriptors.NumRadicalElectrons(self.mol)
        return self.properties['TPSA']

    def numBonds(self):
        if 'GetNumBonds' not in self.properties:
            self.properties['GetNumBonds'] = self.mol.GetNumBonds()
        return self.properties['GetNumBonds']

    def HeavyAtomCount(self):
        if 'HeavyAtomCount' not in self.properties:
            self.properties['HeavyAtomCount'] = Descriptors.HeavyAtomCount(self.mol)
        return self.properties['HeavyAtomCount']

    def HeavyAtomMolWt(self):
        if 'HeavyAtomMolWt' not in self.properties:
            self.properties['HeavyAtomMolWt'] = Descriptors.HeavyAtomMolWt(self.mol)
        return self.properties['HeavyAtomMolWt']

    def RingCount(self):
        '''returns the fraction of C atoms that are SP3 hybridized '''
        if 'RingCount' not in self.properties:
            self.properties['RingCount'] = Descriptors.RingCount(self.mol)
        return self.properties['RingCount']

    def FractionCSP3(self):
        if 'FractionCSP3' not in self.properties:
            self.properties['FractionCSP3'] = Descriptors.FractionCSP3(self.mol)
        return self.properties['FractionCSP3']

    def NHOHCount(self):
        if 'NHOHCount' not in self.properties:
            self.properties['NHOHCount'] = Descriptors.NHOHCount(self.mol)
        return self.properties['NHOHCount']

    def NOCount(self):
        if 'NOCount' not in self.properties:
            self.properties['NOCount'] = Descriptors.NOCount(self.mol)
        return self.properties['NOCount']

    def NumHeteroatoms(self):
        if 'NumHeteroatoms' not in self.properties:
            self.properties['NumHeteroatoms'] = Descriptors.NumHeteroatoms(self.mol)
        return self.properties['NumHeteroatoms']

    def NumRotatableBonds(self):
        if 'NumRotatableBonds' not in self.properties:
            self.properties['NumRotatableBonds'] = Descriptors.NumRotatableBonds(self.mol)
        return self.properties['NumRotatableBonds']

    def NumAmideBonds(self):
        if 'NumAmideBonds' not in self.properties:
            self.properties['NumAmideBonds'] = rdMolDescriptors.CalcNumAmideBonds(self.mol)
        return self.properties['NumAmideBonds']

    def NumAromaticRings(self):
        if 'NumAromaticRings' not in self.properties:
            self.properties['NumAromaticRings'] = Descriptors.NumAromaticRings(self.mol)
        return self.properties['NumAromaticRings']

    def NumSaturatedRings(self):
        if 'NumSaturatedRings' not in self.properties:
            self.properties['NumSaturatedRings'] = Descriptors.NumSaturatedRings(self.mol)
        return self.properties['NumSaturatedRings']

    def NumAliphaticRings(self):
        if 'NumAliphaticRings' not in self.properties:
            self.properties['NumAliphaticRings'] = Descriptors.NumAliphaticRings(self.mol)
        return self.properties['NumAliphaticRings']

    def NumAromaticHeterocycles(self):
        if 'NumAromaticHeterocycles' not in self.properties:
            self.properties['NumAromaticHeterocycles'] = Descriptors.NumAromaticHeterocycles(self.mol)
        return self.properties['NumAromaticHeterocycles']

    def NumSaturatedHeterocycles(self):
        if 'NumSaturatedHeterocycles' not in self.properties:
            self.properties['NumSaturatedHeterocycles'] = Descriptors.NumSaturatedHeterocycles(self.mol)
        return self.properties['NumSaturatedHeterocycles']

    def NumAliphaticHeterocycles(self):
        if 'NumAliphaticHeterocycles' not in self.properties:
            self.properties['NumAliphaticHeterocycles'] = Descriptors.NumAliphaticHeterocycles(self.mol)
        return self.properties['NumAliphaticHeterocycles']

    def NumAromaticCarbocycles(self):
        if 'NumAromaticCarbocycles' not in self.properties:
            self.properties['NumAromaticCarbocycles'] = Descriptors.NumAromaticCarbocycles(self.mol)
        return self.properties['NumAromaticCarbocycles']

    def NumSaturatedCarbocycles(self):
        if 'NumSaturatedCarbocycles' not in self.properties:
            self.properties['NumSaturatedCarbocycles'] = Descriptors.NumSaturatedCarbocycles(self.mol)
        return self.properties['NumSaturatedCarbocycles']

    def NumAliphaticCarbocycles(self):
        if 'NumAliphaticCarbocycles' not in self.properties:
            self.properties['NumAliphaticCarbocycles'] = Descriptors.NumAliphaticCarbocycles(self.mol)
        return self.properties['NumAliphaticCarbocycles']

    def ComputeGasteigerCharges(self):
        '''Marsili-Gasteiger Partial Charges'''
        if 'GasteigerCharges' not in self.properties:
            Descriptors.rdPartialCharges.ComputeGasteigerCharges(self.mol)
            self.properties['MinGasteigerCharges'], self.properties['MaxGasteigerCharges'] = self.mol._chargeDescriptors
        return self.properties['MinGasteigerCharges'], self.properties['MaxGasteigerCharges']

    def BalabanJindex(self):
        ''' Balabans J index'''
        if 'BalabanJindex' not in self.properties:
            self.properties['BalabanJindex'] = Descriptors.BalabanJ(self.mol)
        return self.properties['BalabanJindex']

    def BertzCT(self):
        ''' A topological index meant to quantify complexity of molecules.'''
        '''From S. H. Bertz, J. Am. Chem. Soc., vol 103, 3599-3601 (1981'''
        if 'BertzCT' not in self.properties:
            self.properties['BertzCT'] = Descriptors.BertzCT(self.mol)
        return self.properties['BertzCT']

    def generate2D(self, location='images/'):
        AllChem.Compute2DCoords(self.mol)
        Draw.MolToFile(self.mol, fileName=location + 'mol_' + str(self.smiles()) + '.png')

    def sub_strcture_search(self, sub):
        return self.mol.GetSubstructMatches(sub)

    def replace_sub_structure(self, old, new, replaceAll=True):
        try:
            new_mol = AllChem.ReplaceSubstructs(self.mol, old.mol, new.mol, replaceAll=replaceAll)
            new_mol_smile = Chem.MolToSmiles(new_mol[0])
            new_mol = Molecule(new_mol_smile)
        except:
            print("Recplacement Failed")
            return None
        return new_mol

    def get_morgan_fp(self, nbits=2048):
        if 'MorganFP' not in self.properties:
            fp = AllChem.GetMorganFingerprintAsBitVect(self.mol, 2, nBits=nbits)
            #fp = AllChem.GetMorganFingerprint(self.mol, 2)
            self.properties['MorganFP'] = fp
        return self.properties['MorganFP']

    def rdkit_fp(self):
        if 'rdkitFP' not in self.properties:
            fp = FingerprintMols.FingerprintMol(self.mol)
            prototype_fp = np.zeros(len(fp), np.int32)
            DataStructs.ConvertToNumpyArray(fp, prototype_fp)
            self.properties['rdkitFP'] = fp
            self.properties['rdkitFP_binary'] = prototype_fp

    def atom_pair_fp(self):
        if 'AtomPairFP' not in self.properties:
            fp = Pairs.GetAtomPairFingerprint(self.mol)
            self.properties['AtomPairFP'] = fp
            #self.properties['AtomPairFP_vector'] = list(fp)

    def number_of_lipinski_violations(self):
        number_of_violations = sum((self.logp() > 5, self.numHDonors() > 5, self.exactMolWt() > 500, self.numHAcceptors() > 10))
        return number_of_violations

    def get_full_record_by_name_pubchem(self):
        #TODO: Fix compound cid access
        compound_cid = self.get_cid_from_name_pubchem(self.name)
        return self.get_full_record_from_cid_pubchem(compound_cid)

    def get_full_record_from_cid_pubchem(self):
        js = requests.get('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/' + str(self.cid) + '/json')
        return json.loads(js.content)

    def get_cid_from_name_pubchem(self):
        temp = requests.get('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/' + self.name +'/cids/json')
        js = json.loads(temp.content)
        return js['IdentifierList']['CID'][0]

    def get_synonyms(self):
        if 'synonyms' not in self.properties:
            js = json.loads(requests.get('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/' + str(self.cid) + '/synonyms/json').content)
            self.properties['synonyms'] = js['InformationList']['Information'][0]['Synonym']
        return self.properties['synonyms']

    def get_smiles_from_name_pubchem(self):
        self.cid = self.get_cid_from_name_pubchem()
        js = json.loads(requests.get('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/' + str(self.cid) +'/property/CanonicalSMILES/json').content)
        return js['PropertyTable']['Properties'][0]['CanonicalSMILES']


"""
Loading Lassos from PDB and LassoProt. Also contains an iterator through all lassos.
https://www.rcsb.org/
https://lassoprot.cent.uw.edu.pl/
"""

import math
import numpy as np
import os
import re
import requests
import wget
from ast import literal_eval
from bs4 import BeautifulSoup
from collections import defaultdict, Counter
from pathlib import Path
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from google.cloud import storage

DATA_FOLDER = Path("data")
PDB_FOLDER = DATA_FOLDER / "pdb"
CIF_FOLDER = DATA_FOLDER / "cif"
LASSOPROT_FOLDER = DATA_FOLDER / "lassoprot"
ALPHALASSO_FOLDER = DATA_FOLDER / "alphalasso"

# create folders if they do not exists
PDB_FOLDER.mkdir(parents=True, exist_ok=True)
CIF_FOLDER.mkdir(parents=True, exist_ok=True)
LASSOPROT_FOLDER.mkdir(parents=True, exist_ok=True)
ALPHALASSO_FOLDER.mkdir(parents=True, exist_ok=True)



def _interpolate_nans(arr):
    """ if there are NaN's inside 3D array arr, this funtion interpolates the NaNs with the neighbour values"""
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]
    def interpolate_1D(row):
        nans, x= nan_helper(row)
        row[nans]= np.interp(x(nans), x(~nans), row[~nans])
    for row in arr.T:
        interpolate_1D(row)

class Lasso(dict):
    """ class containing a lasso"""
    def __init__(self, pdb, chain, ndxes, loop, Ntail, Ctail, lassoprot_data=None):
        self.pdb = pdb.upper()
        self.chain = chain
        self.ndxes = ndxes
        self.bridge = ndxes[1:3]
        self.endpoints = [ndxes[0], ndxes[3]]
        self.id = '{}_{} {:d}-{:d}'.format(self.pdb, self.chain, *self.bridge)
        self.loop = loop[0] # xzy
        self.loop_atoms = loop[1]
        self.loop_missing = loop[2]
        self.Ntail = Ntail[0]  # xzy
        self.Ntail_atoms = Ntail[1]
        self.Ntail_missing = Ntail[2]
        self.Ctail = Ctail[0]  # xzy
        self.Ctail_atoms = Ctail[1]
        self.Ctail_missing = Ctail[2]
        self.lassoprot_data = lassoprot_data
        self.intersections = self.xyz_intersections()
        self.n_deep_xyz = self.intersections["n-deep-xyz"]
        self.c_deep_xyz = self.intersections["c-deep-xyz"]

    def is_insane(self):
        return bool(self.loop_missing + self.Ntail_missing + self.Ctail_missing)

    def get_coords(self):
        return self.Ntail, np.vstack([self.loop, (self.loop[0] + self.loop[-1]) / 2]), self.Ctail

    def get_atom_index(self, tail, aminoacid_number, atom_type="CA"):
        """
        Returns the index of the xyz coordinate of a specific atom in the tail
        Args:
            tail:  "C" or "N"
            aminoacid_number:  ac number, e.g. 100
            atom_type: "N", "C", or "CA" (alpha-carbon)
        Returns: index of the element in either Ctail or NTail, depending on tail parameter
        """
        if tail.upper() == "C":
            return self.Ctail_atoms.index((aminoacid_number, atom_type))
        if tail.upper() == "N":
            return self.Ntail_atoms.index((aminoacid_number, atom_type))
        raise KeyError

    def xyz_intersections(self):
        debug = False
        lasso = self
        if debug: print("intersections", lasso.lassoprot_data)
        if debug: print("n tail atoms", lasso.Ntail_atoms)
        if debug: print("c tail atoms", lasso.Ctail_atoms)
        result = dict()

        result["n-shallow"] = tuple(
            lasso.Ntail_atoms.index((abs(ac), "CA")) for ac in lasso.lassoprot_data["n-shallow"])
        result["n-deep"] = tuple(lasso.Ntail_atoms.index((abs(ac), "CA")) for ac in lasso.lassoprot_data["n-deep"])
        result["c-shallow"] = tuple(
            lasso.Ctail_atoms.index((abs(ac), "CA")) for ac in lasso.lassoprot_data["c-shallow"])
        result["c-deep"] = tuple(lasso.Ctail_atoms.index((abs(ac), "CA")) for ac in lasso.lassoprot_data["c-deep"])

        tailN, loop, tailC = lasso.get_coords()
        result["n-shallow-xyz"] = [tailN[i] for i in result["n-shallow"]]
        result["n-deep-xyz"] = [tailN[i] for i in result["n-deep"]]
        result["c-shallow-xyz"] = [tailC[i] for i in result["c-shallow"]]
        result["c-deep-xyz"] = [tailC[i] for i in result["c-deep"]]
        if debug: print(result)
        return result

    def get_missing(self):
        if self.loop_missing:
            print('Missing loop ndxes: {}'.format(','.join([ndx for ndx, atom in self.loop_missing])))
        if self.Ntail_missing:
            print('Missing Ntail ndxes: {}'.format(','.join([ndx for ndx, atom in self.Ntail_missing])))
        if self.Ctail_missing:
            print('Missing Ctail ndxes: {}'.format(','.join([ndx for ndx, atom in self.Ctail_missing])))
        return self.Ntail_missing[::-1] + self.loop_missing + self.Ctail_missing

    def __str__(self):
        nice_dict = " ".join([str(key) + "=" + str(val)
                              for key, val in self.lassoprot_data.items() if val != tuple() and key != "range"])
        return '{}: {}'.format(self.id, nice_dict)

    def __repr__(self):
        return 'Lasso class object {}'.format(str(self))


class LassoExtractor:
    """class for extracting all lassos from a protein (PDB) """
    def __init__(self, pdb):
        #print('self.__init__() beg')
        #self.pdbs_folder = PDB_FOLDER  # global var
        self.pdb = pdb
        self.coords, self.cys_ndxes, self.chains = self.pdb2coords()
        self.lassoprot_data = dict(self.get_lassoprot_info())

        # print("LASSO")
        # print(self.lassoprot_data)
        # print(self.lassoprot_data.keys())
        # print(self.lassoprot_data.values())

        #print('self.find_bridges()')
        self.bridges = self.find_bridges()
        #print('self.get_lassos()')
        self.lassos = self.get_lassos()
        #print('self.__init__() end')

    def get_lassos(self):
        lassos = defaultdict(list)
        #print("BRIDGES", self.bridges)
        for chain, bridge in self.bridges:
            chain_endpoints = min(self.coords[chain].keys()), max(self.coords[chain].keys())
            expected_loop, expected_Ntail, expected_Ctail = self.get_expected_atoms(chain, chain_endpoints, bridge)
            loop, Ntail, Ctail, cut_N, cut_C = self.collect_lasso_coords(chain, expected_loop, expected_Ntail, expected_Ctail)
            _interpolate_nans(loop[0])
            _interpolate_nans(Ntail[0])
            _interpolate_nans(Ctail[0])
            loop_coords,  loop_atoms,  loop_missing  = loop
            Ntail_coords, Ntail_atoms, Ntail_missing = Ntail
            Ctail_coords, Ctail_atoms, Ctail_missing = Ctail
            #print("missing", loop_missing, Ntail_missing, Ctail_missing)
            # ndxes: (first_residue, first_loop_residue, last_loop_residue, last_residue)
            #        therefore indexes are growing from left to right
            ndxes = (chain_endpoints[0]+cut_N//3, bridge[0], bridge[1], chain_endpoints[1]-cut_C//3)

            #DEBUG = False
            DEBUG = True
            if DEBUG:
                if loop_missing:
                    print('Loop {:d}-{:d} of {}_{} is insane!'.format(ndxes[1], ndxes[2], self.pdb, chain))
                if Ntail_missing:
                    print('N tail {:d}-{:d} of {}_{} is insane!'.format(ndxes[0], ndxes[1], self.pdb, chain))
                if Ctail_missing:
                    print('C tail {:d}-{:d} of {}_{} is insane!'.format(ndxes[2], ndxes[3], self.pdb, chain))
            loop = loop_coords, loop_atoms, loop_missing
            Ntail = Ntail_coords, Ntail_atoms, Ntail_missing
            Ctail = Ctail_coords, Ctail_atoms, Ctail_missing
            # Bostjan added21.mar2024
            try:
                LASSO = Lasso(self.pdb, chain, ndxes, loop, Ntail, Ctail,
                                           self.lassoprot_data[chain][bridge])
                lassos[chain].append(LASSO)
            except:
                print(f"Lasso {self.pdb}, {chain}, bridge={bridge} not found (skipping)")
        return lassos

    def get_expected_atoms(self, chain, chain_endpoints, bridge):
        chain_beg, chain_end = chain_endpoints
        ndx1, ndx2 = bridge
        # loop_atoms
        expected_loop_atoms = [(ndx1,'SG'), (ndx1,'CB'), (ndx1,'CA'), (ndx1,'C')]
        for ndx in range(ndx1+1, ndx2):
            for atom in ['N','CA','C']:
                expected_loop_atoms.append((ndx,atom))
        expected_loop_atoms += [(ndx2,'N'), (ndx2,'CA'), (ndx2,'CB'), (ndx2,'SG')]
        # N_tail_atoms
        expected_Ntail_atoms = [(ndx1,'N')]
        for ndx in range(ndx1-1, chain_beg-1, -1):
            for atom in ['C','CA','N']:
                expected_Ntail_atoms.append((ndx,atom))
        expected_Ctail_atoms = [(ndx2,'C')]
        for ndx in range(ndx2+1, chain_end+1):
            for atom in ['N','CA','C']:
                expected_Ctail_atoms.append((ndx,atom))
        return expected_loop_atoms, expected_Ntail_atoms, expected_Ctail_atoms

    def collect_lasso_coords(self, chain, expected_loop_atoms, expected_Ntail_atoms, expected_Ctail_atoms):
        loop_coords  = np.array([self.coords[chain][ndx][atom] for ndx,atom in expected_loop_atoms])
        Ntail_coords = np.array([self.coords[chain][ndx][atom] for ndx,atom in expected_Ntail_atoms])
        Ctail_coords = np.array([self.coords[chain][ndx][atom] for ndx,atom in expected_Ctail_atoms])
        Ntail_coords, cut_N = self.cut_missing_tail_end(Ntail_coords)
        Ctail_coords, cut_C = self.cut_missing_tail_end(Ctail_coords)
        if cut_N:
            expected_Ntail_atoms = expected_Ntail_atoms[:-cut_N]
        if cut_C:
            expected_Ctail_atoms = expected_Ctail_atoms[:-cut_C]
        loop_atoms, loop_missing = self.insanity_check(loop_coords, expected_loop_atoms)
        Ntail_atoms, Ntail_missing = self.insanity_check(Ntail_coords, expected_Ntail_atoms)
        Ctail_atoms, Ctail_missing = self.insanity_check(Ctail_coords, expected_Ctail_atoms)
        loop  = (loop_coords,  loop_atoms,  loop_missing)
        Ntail = (Ntail_coords, Ntail_atoms, Ntail_missing)
        Ctail = (Ctail_coords, Ctail_atoms, Ctail_missing)
        return loop, Ntail, Ctail, cut_N, cut_C

    @staticmethod
    def cut_missing_tail_end(coords):
        cut = 0
        while np.isnan(coords[-1,0]):
            coords = coords[:-1]
            cut += 1
        return coords, cut

    @staticmethod
    def insanity_check(coords, expected):
        # checking if there are any breaks
        missing = []
        isnan = np.isnan(coords[:,0])
        if any(isnan):
            for k in np.argwhere(isnan)[::-1]:
                ndx = k[0]
                missing.append(expected.pop(ndx))
        return expected, missing[::-1]

    def download_pdb(self):
        try:
            print('self.pdb.upper(): ', self.pdb.upper())
            url = 'https://files.rcsb.org/download/{}.pdb'.format(self.pdb.upper())
            print("Downloading from", url)
            wget.download(url, out=str(PDB_FOLDER), bar=None)
        except:
            print(f"[[ problems with {self.pdb.upper()} ]]")

    def pdb2coords(self):
        # if locally file is not available, then download
        file_path = PDB_FOLDER / (self.pdb.upper() + ".pdb")
        if not os.path.isfile(file_path):
            print("pdb2coords()/download_pdb()")
            self.download_pdb()
        chain_atoms = ["CA","C","N"]
        bridge_atoms = ["CB","SG"]
        chains = set([])
        #coords = {chain:{res_ndx:{atom:(x,y,z)}}}ls

        coords = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: tuple([np.nan]*3))))
        #cysteines = {chain:[*res_ndx]}}
        cysteines = defaultdict(list)
        with open(file_path, 'r') as f:
            for line in f.readlines():
                if line[:4] == 'ATOM':
                    atom = line[13:16].strip()
                    insertion_symbol = line[16]
                    resname = line[17:20].strip()
                    chain = line[21]
                    res_ndx = int(line[22:26])
                    if not(insertion_symbol in [' ', 'A']):
                        continue
                    if atom in chain_atoms:
                        pass
                    elif resname == 'CYS' and atom in bridge_atoms:
                        chains.add(chain)
                        if atom == 'SG':
                            cysteines[chain].append(res_ndx)
                    else:
                        continue
                    x = line[30:38]
                    y = line[38:46]
                    z = line[46:54]
                    xyz = tuple([float(k) for k in (x,y,z)])
                    coords[chain][res_ndx][atom] = xyz
                elif line[:6] == 'ENDMDL':
                    break
        return coords, cysteines, sorted(chains)

    def find_bridges(self, cutoff=3):
        # cutoff 3 means that we treat CYS1-CYS2 as a bridge if their sulfur atoms (SG1, SG2)
        # are no more than 3 angstroms apart. same cutoff is used in PDB
        bridges = []
        for chain in self.chains:
            for i, cys1_ndx in enumerate(self.cys_ndxes[chain][:-1]):
                coords1 = self.coords[chain][cys1_ndx]['SG']
                for cys2_ndx in self.cys_ndxes[chain][i+1:]:
                    coords2 = self.coords[chain][cys2_ndx]['SG']
                    if self.calc_dist(coords1, coords2) <= 3.:
                        bridge = (cys1_ndx, cys2_ndx)
                        bridges.append((chain, bridge))
        return bridges

    @staticmethod
    def calc_dist(coords1, coords2):
        x1,y1,z1 = coords1
        x2,y2,z2 = coords2
        dist = math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        return dist

    def get_lassoprot_info(self):
        all_data = defaultdict()
        pdb_id = self.pdb.upper()
        for chain in self.chains:
            #file_path = 'lassoprot_data/{}.txt'.format(self.pdb.lower())
            file_path = LASSOPROT_FOLDER / '{}_{}.dat'.format(self.pdb.lower(), chain)
            if os.path.isfile(file_path):
                # with open(file_path, 'r') as f:
                #     print(literal_eval(f.read()))

                with open(file_path, 'r') as f:
                    loaded = literal_eval(f.read().strip())
                    # print(loaded)
                data = defaultdict(dict) # !!!! not string Boštjan
                for k,v in loaded.items():
                    data[k] = v
            else:
                data = self.download_lasso_info(pdb_id, chain)
                with open(file_path, 'w') as f:
                    f.write(str(dict(data)))

            # print(data)
            all_data[chain] = data

        return all_data

    @staticmethod
    def download_lasso_info(pdb, chain):
        " download intersections with the minimal surface directly from the html of the lassoprot website"


        def numbers(s, k=""):
            # keep only numbers and hypen and chars in k
            return re.sub(rf'[^0-9-{k}]', '', str(s))
        pdb_id = pdb.upper()
        chain = chain.upper()
        url = f"https://lassoprot.cent.uw.edu.pl/view/{pdb_id}/{chain}/"
#            print("Url", url)
        r = requests.get(url)
        soup = BeautifulSoup(r.content, "html.parser")
        #print(soup)


        data = []  # store lasso information
        all_data = dict()
        #all_data = defaultdict(dict)
        # for x in soup.find_all:
        #     print(str(x)[:10])
        # exit()


        for table in soup.find_all('table', attrs={"class": "table table-hover table-condensed"}):
            # Find all rows within the table
            for row in table.find_all('tr')[1:]:
                # Find all columns (cells) within the row
                data = dict()
                col = list(row.find_all(['td', 'th']))

                data["range"] = tuple(int(x) for x in numbers(col[6]).split("-"))
                bridge = data["range"]
                data["symbol"] = numbers(col[4],"LSNC").replace('--',"-")
                try:
                    data["area"] = int(numbers(col[12]))
                except:
                    data["area"] = 0

                data["n-deep"] = col[8].find("div", attrs={"class": "toggleShallow withoutShallow"}).get_text(separator=' ')
                if len(data["n-deep"]) > 0:
                    data["n-deep"] = tuple(int(x) for x in data["n-deep"].split(","))
                else: data["n-deep"] = tuple()
                data["n-shallow"] = tuple()
                for elem in col[8].find_all("span", attrs={"class": "shallow"}):
                    shallow = elem.get_text(separator=' ')
                    data["n-shallow"] = data["n-shallow"] + (int(numbers(shallow)),)

                data["c-deep"] = col[9].find("div", attrs={"class": "toggleShallow withoutShallow"}).get_text(separator=' ')
                if len(data["c-deep"]) > 0:
                    data["c-deep"] = tuple(int(x) for x in data["c-deep"].split(","))
                else: data["c-deep"] = tuple()
                data["c-shallow"] = tuple()
                for elem in col[9].find_all("span", attrs={"class": "shallow"}):
                    shallow = elem.get_text(separator=' ')
                    data["c-shallow"] = data["c-shallow"] + (int(numbers(shallow)),)

                # data["n"] = numbers(col[8])
                # data["c"] = numbers(col[9])
                #print(data)
                all_data[bridge] = data
        return all_data

class LassoExtractorAF(LassoExtractor):
    def __init__(self, uniprot_id):
        #self.pdbs_folder = PDB_FOLDER  # global var
        self.pdb = uniprot_id # IT IS uniprot_id, but easier not to change name of self.pdb
        self.coords, self.cys_ndxes, self.chains = self.uniprot_id2coords()
        self.lassoprot_data = dict(self.get_alphalasso_info())

        # print("LASSO")
        # print(self.alphalasso_data)
        # print(self.alphalasso_data.keys())
        # print(self.alphalasso_data.values())

        self.bridges = self.find_bridges()
        self.lassos = self.get_lassos()

    def download_AF(self, file_path):
        try:
            print(self.pdb.upper())
            self.download_v4_from_gcloud(self.pdb.upper(), file_path)
        except:
            print(f"[[ problems with {self.pdb.upper()} ]]")

    def uniprot_id2coords(self):
        # if locally file is not available, then download


        print("uniprot_id2coords")
        file_path_dummy = CIF_FOLDER / ('AF-' + self.pdb.upper() + "-F1-model_v4_test.cif")

        print("opening.")
        f = open(file_path_dummy, 'w')
        print("Writings to ", file_path_dummy)
        f.write("Hello")
        f.close()


        file_path = CIF_FOLDER / ('AF-' + self.pdb.upper() + "-F1-model_v4.cif")
        print("File path:", file_path)
        if not os.path.isfile(file_path):
            print("A")
            self.download_AF(file_path)
            print("B")
        # if still not available, return 0
        print("C")
        if not os.path.isfile(file_path):
            print("X")
            print(f"[[ {self.pdb.upper()} not available ]]")
            raise FileNotFoundError
        print("Y")

        chain_atoms = ["CA","C","N"]
        bridge_atoms = ["CB","SG"]
        chains = set([])
        #coords = {chain:{res_ndx:{atom:(x,y,z)}}}
        coords = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: tuple([np.nan]*3))))
        #cysteines = {chain:[*res_ndx]}}
        cysteines = defaultdict(list)

        cifdict = MMCIF2Dict(file_path)
        chains = cifdict['_entity_poly.pdbx_strand_id']
        model_vec = cifdict['_atom_site.pdbx_PDB_model_num']
        atom_vec = cifdict['_atom_site.label_atom_id']
        resnames_vec = cifdict['_atom_site.label_comp_id']
        n_vec = cifdict['_atom_site.label_seq_id']
        x_vec = cifdict['_atom_site.Cartn_x']
        y_vec = cifdict['_atom_site.Cartn_y']
        z_vec = cifdict['_atom_site.Cartn_z']
        chain_vec = cifdict['_atom_site.label_asym_id']
        used_model = min([int(x) for x in set(model_vec)])
        nxyz = {}
        for atom,resname,n,x,y,z,chain,model in zip(atom_vec,resnames_vec,n_vec,x_vec,y_vec,z_vec,chain_vec,model_vec):
            if chain in chains and n[0] in '123456789' and int(model)==used_model:
                n = int(n)
                if atom in chain_atoms:
                    xyz = tuple([float(k) for k in (x,y,z)])
                    coords[chain][n][atom] = xyz
                elif resname == 'CYS' and atom in bridge_atoms:
                    xyz = tuple([float(k) for k in (x,y,z)])
                    coords[chain][n][atom] = xyz
                    if atom == 'SG':
                        cysteines[chain].append(n)
        return coords, cysteines, sorted(chains)

    def get_alphalasso_info(self):
        all_data = defaultdict()
        pdb_id = self.pdb.upper()
        chain = "1"
        #file_path = 'lassoprot_data/{}.txt'.format(self.pdb.lower())
        file_path = ALPHALASSO_FOLDER / '{}_{}.dat'.format(self.pdb.upper(), chain)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                loaded = literal_eval(f.read().strip())
                # print(loaded)
            data = defaultdict(dict) # !!!! not string Boštjan
            for k,v in loaded.items():
                data[k] = v
        else:
            data = self.download_lasso_info(pdb_id)
            with open(file_path, 'w') as f:
                f.write(str(dict(data)))
            # print(data)
            all_data[chain] = data
        return all_data

    @staticmethod
    def download_v4_from_gcloud(uniprot_id, file_path):
        """Downloads a blob from the bucket."""
        # The ID of your GCS bucket
        print("a")
        bucket_name = "public-datasets-deepmind-alphafold-v4"

        # The ID of your GCS object
        source_blob_name = f"AF-{uniprot_id}-F1-model_v4.cif"

        print("b")

        # The path to which the file should be downloaded
        destination_file_name = file_path
        print("c")

        from google.auth import default

        creds, project = default()
        print("Project:", project)

        storage_client = storage.Client(credentials=creds, project=project)
        print("Client created")

        #storage_client = storage.Client()
        print("d")

        bucket = storage_client.bucket(bucket_name)

        print("e")

        # Construct a client side representation of a blob.
        # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
        # any content from Google Cloud Storage. As we don't need additional data,
        # using `Bucket.blob` is preferred here.
        blob = bucket.blob(source_blob_name)

        print("f", file_path)

        blob.download_to_filename(file_path)

        print("g")

        print("Downloaded storage object {} from bucket {} to local file {}.".format(
              source_blob_name, bucket_name, file_path))

    @staticmethod
    def download_lasso_info(uniprot_id):
        " download intersections with the minimal surface directly from the html of the lassoprot website"
        def numbers(s, k=""):
            # keep only numbers and hypen and chars in k
            return re.sub(rf'[^0-9-{k}]', '', str(s))
        uniprot_id = uniprot_id.upper()
        chain = "1"
        AF_version = "4"
        url = f"https://alphalasso.cent.uw.edu.pl/view/{uniprot_id}/{chain}/{AF_version}"
#            print("Url", url)
        r = requests.get(url)
        soup = BeautifulSoup(r.content, "html.parser")
        #print(soup)

        data = []  # store lasso information
        all_data = dict()
        #all_data = defaultdict(dict)
        # for x in soup.find_all:
        #     print(str(x)[:10])
        # exit()

        for table in soup.find_all('table', attrs={"class": "table table-hover table-condensed text-center"}):
            # Find all rows within the table
            for row in table.find_all('tr')[1:]:
                # Find all columns (cells) within the row
                data = dict()
                col = list(row.find_all(['td', 'th']))
                data["range"] = tuple(int(x) for x in numbers(col[10]).split("-"))
                bridge = data["range"]
                data["symbol"] = numbers(col[5].find('span', attrs={"lassoSymbol"}),"LSNC").replace('--',"-")
                try:
                    data["area"] = int(numbers(col[16]))
                except:
                    data["area"] = 0
                data["n-deep"] = col[12].find("span", attrs={"class": "toggleShallow withoutShallow"}).get_text(separator=' ')
                if len(data["n-deep"]) > 0:
                    data["n-deep"] = tuple(int(x) for x in data["n-deep"].split(","))
                else: data["n-deep"] = tuple()
                data["n-shallow"] = tuple()
                for elem in col[12].find_all("span", attrs={"class": "shallow"}):
                    shallow = elem.get_text(separator=' ')
                    data["n-shallow"] = data["n-shallow"] + (int(numbers(shallow)),)

                data["c-deep"] = col[13].find("span", attrs={"class": "toggleShallow withoutShallow"}).get_text(separator=' ')
                if len(data["c-deep"]) > 0:
                    data["c-deep"] = tuple(int(x) for x in data["c-deep"].split(","))
                else: data["c-deep"] = tuple()
                data["c-shallow"] = tuple()
                for elem in col[13].find_all("span", attrs={"class": "shallow"}):
                    shallow = elem.get_text(separator=' ')
                    data["c-shallow"] = data["c-shallow"] + (int(numbers(shallow)),)

                # data["n"] = numbers(col[8])
                # data["c"] = numbers(col[9])
                #print(data)
                all_data[bridge] = data
        return all_data

def _get_lasso_dict(lasso):
    tailN, loop, tailC = lasso.get_coords()
    tail_xyz_n, tail_atoms_n, shallow_index_n, deep_index_n, deep_xyz_n, deep_str_n = _get_terminus_data(lasso, "N")
    tail_xyz_c, tail_atoms_c, shallow_index_c, deep_index_c, deep_xyz_c, deep_str_c = _get_terminus_data(lasso, "C")

    return {
        "pdb": lasso.pdb,
        "chain": lasso.chain,
        "bridge": lasso.bridge,
        "symbol": lasso.lassoprot_data["symbol"],
        "id": lasso.pdb + lasso.chain + " " + str(lasso.bridge) + " " + lasso.lassoprot_data["symbol"],
        "xyz": {"n": tailN, "loop": loop, "c": tailC},
        "shallow_n": shallow_index_n,
        "deep_n": deep_index_n,
        "deep_xyz_n": deep_xyz_n,
        "deep_str_n": deep_str_n,
        "shallow_c": shallow_index_c,
        "deep_c": deep_index_c,
        "deep_xyz_c": deep_xyz_c,
        "deep_str_c": deep_str_c,
    }

def all_lasso_iterator(max_lassos=None, include_trivial=False, include_shallow=False, exclude_pdbs = None, pdb_starts_with=None):
    """iterates over all lassos, stops at max_lassos if given.
    """
    #download pdb/chains
    max_lassos = max_lassos or 10000000

    exclude_pdbs = ["4BH4", "8AVB", "3RME", "3WHE"]
    exclude_pdbs = exclude_pdbs or set()
    exclude_pdbs = {s.upper() for s in exclude_pdbs}

    # download lasso.txt from lassoprot
    if not os.path.exists(DATA_FOLDER / "lasso.txt"):  # TODO: should overwrite if new data arrives to lassoprot
        wget.download('https://lassoprot.cent.uw.edu.pl/lasso.txt', out=str(DATA_FOLDER / "lasso.txt"))
        print("File lasso.txt downloaded from LassoProt")

    with open(DATA_FOLDER / "lasso.txt", "r") as f:
        pdb_chain = [tuple(l.strip().split(" ")) for l in f.readlines()]

    counter = 0
    for pdb_id, chain in pdb_chain:
        if pdb_starts_with is not None and not pdb_id.startswith(pdb_starts_with):
            continue
        if pdb_id.upper() in exclude_pdbs:
            continue

        #print("****** PDB CHAIN", pdb_id, chain, [lasso.lassoprot_data["symbol"] for lasso in LassoExtractor(pdb_id).lassos[chain]])
        for lasso in LassoExtractor(pdb_id).lassos[chain]:
            # sometimes there is an empty lasso data ?!
            if not bool(lasso.lassoprot_data):
                continue
            deep_intersections = lasso.lassoprot_data["c-deep"] + lasso.lassoprot_data["n-deep"]
            shallow_intersections = lasso.lassoprot_data["c-shallow"] + lasso.lassoprot_data["n-shallow"]

            if not include_trivial and len(deep_intersections + shallow_intersections) == 0:
                continue
            if not include_shallow and len(deep_intersections) == 0 and len(shallow_intersections) != 0:
                continue
            if (counter := counter + 1) > max_lassos: return

            yield _get_lasso_dict(lasso)

def get_lasso(pdb, chain=None, index=None):
    if chain is None:
        chain = "A"
    if index is None:
        index = 0
    lassos = LassoExtractor(pdb).lassos[chain]  # choose PDB and chain
    lasso = lassos[index]  # choose loop
    return _get_lasso_dict(lasso)

def _all_lasso_iterator_alphalasso(max_lassos=None, include_trivial=False, include_shallow=False, exclude_uniprot_ids = None):
    chain = 1 # all structures in alphalasso have only on chain: "1"
    min_plddt = 70 # minimal required plddt quality of structure

    #download pdb/chains
    max_lassos = max_lassos or 10000000

    exclude_uniprot_ids = exclude_uniprot_ids or set()
    exclude_uniprot_ids = {s.upper() for s in exclude_uniprot_ids}

    if not os.path.exists(DATA_FOLDER / "lassoAF.txt"):  # TODO: should overwrite if new data arrives to lassoprot
        print("Query sent")
        api_query = f"browse?field=pLDDT_chain&val=>{min_plddt}&conj=NOT&field=Lasso_type&val=L0&raw=1&result_cols=Uniprot;pLDDT_chain;Bridge;Lasso_type"
        full_query = f''
        wget.download(f'https://alphalasso.cent.uw.edu.pl/{api_query}', out=str(DATA_FOLDER / "lassoAF.txt"))
        print("Downloaded")

    with open(DATA_FOLDER / "lassoAF.txt", "r") as f:
        f.readline()
        proteins = [line.strip().split('\t') for line in f.readlines()]

    counter = 0
    for uniprot_id, plddt, bridge, lassotype in proteins:
        if '-' in uniprot_id:
            uniprot_id = uniprot_id.split('-')[0]
        if uniprot_id.upper() in exclude_uniprot_ids:
            continue
        # we are interested only in SS bridges
        if bridge != "SS":
            continue
        if float(plddt) < min_plddt:
            continue

        try:
            extractor_object = LassoExtractorAF(uniprot_id)
        except FileNotFoundError:
            continue

        for lasso in extractor_object.lassos[chain]:
            # sometimes there is an empty lasso data ?!
            if not bool(lasso.alphalasso_data):
                continue
            deep_intersections = lasso.lassoprot_data["c-deep"] + lasso.lassoprot_data["n-deep"]
            shallow_intersections = lasso.lassoprot_data["c-shallow"] + lasso.lassoprot_data["n-shallow"]

            if not include_trivial and len(deep_intersections + shallow_intersections) == 0:
                continue
            if not include_shallow and len(deep_intersections) == 0 and len(shallow_intersections) != 0:
                continue

            if (counter := counter + 1) > max_lassos: return
            yield lasso

def _get_terminus_data(lasso, terminus):
    """Gets the N or C tail data for lasso
    Args:
        lasso: lasso object
        terminus: terminus/tail letter "N" or "C"
    Returns: a tuple of:
    - coordinates of the tail,
    - atom names of the tails,
    - indexes of deep intersections of the tail with minimal surface,
    - deep intersections of the tail with minimal surface,
    - string containing readable intersection info (atoms)
    """
    if terminus.upper()[0] == "N":
        deep_string = " ".join([str(lasso.Ntail_atoms[abs(i)]) for i in lasso.intersections["n-deep"]])
        return (lasso.get_coords()[0],
                lasso.Ntail_atoms,
                list(lasso.intersections["n-shallow"]),
                list(lasso.intersections["n-deep"]),
                np.array(lasso.n_deep_xyz),
                deep_string)

    if terminus.upper()[0] == "C":
        deep_string = " ".join([str(lasso.Ctail_atoms[abs(i)]) for i in lasso.intersections["c-deep"]])
        return (lasso.get_coords()[2],
                lasso.Ctail_atoms,
                list(lasso.intersections["c-shallow"]),
                list(lasso.intersections["c-deep"]),
                np.array(lasso.c_deep_xyz),
                deep_string)

    raise ValueError(f"Unknown terminus {terminus}")


def print_lasso(d):
    print("[Lasso]", d["pdb"], d["chain"], d["bridge"], d["symbol"])
    print("    xyz:", len(d["xyz"]["n"]), len(d["xyz"]["loop"]), len(d["xyz"]["c"]))
    print(" deep N:", d["deep_n"], "shallow:", d["shallow_n"])
    print(" deep C:", d["deep_c"], "shallow:", d["shallow_c"])


if __name__ == '__main__':
    # print first 3 lassos from lassoprot
    #for i, r in enumerate(all_lasso_iterator(22, include_trivial=False)):
    #    print_lasso(r)
    # print first 3 lassos from alphalasso
    for i, lasso in enumerate(_all_lasso_iterator_alphalasso(3)):
        print(f"Lasso #{i}", lasso.pdb, lasso.chain, lasso.bridge, lasso.lassoprot_data["symbol"])

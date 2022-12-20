import numpy as np 
from pdb2sql import interface
import h5py 

def generate_interface_center(mol_data):
    """Yield a the indices of a new center draw from a line
       between two contact residues

    Args:
        mol_data (tuple): hdf5 file name, molecule name
    """

    def get_grid_index(point, grid_points):
        """Get grid indices of a point from its xyz coordinates

        Args:
            point (np.ndarray): xyz coordinates of the point
            grid_points (tuple): (xgrid, ygrid, zgrid)

        Returns:
            list: indices of the point in the grid
        """
        index = []
        for pt_coord, grid_coord in zip(point, grid_points):
            index.append(np.argmin(np.abs(grid_coord - pt_coord)))
        return index

    def get_next_point(db, res):
        """generate the xyz coordinate of a random center

        Args:
            db (pdb2sql.interface): an interface instance created from the molecule 
            res (dict): a dictionar of interface residue obtained via .get_contact_residues()

        Returns:
            np.ndarray: xyz coordinate of the new center
        """
        resA, resB = res[chains[0]], res[chains[1]]
        nresA, nresB = len(resA), len(resB)

        rA = resA[np.random.randint(0, nresA)]
        rB = resB[np.random.randint(0, nresB)] 
        
        posA = np.array(db.get('x,y,z',chainID=rA[0], resSeq=rA[1])).mean(axis=0)
        posB = np.array(db.get('x,y,z',chainID=rB[0], resSeq=rB[1])).mean(axis=0)
        return posA + np.random.rand(3)*(posB-posA)
    
    # get the hdf5 filename and molecule name
    filename, molname = mol_data
    if isinstance(filename, (tuple, list)):
        filename = filename[0]

    if isinstance(molname, (tuple, list)):
        molname = molname[0]

    # get data from the hdf5 file
    with h5py.File(filename,'r') as f5:
        mol = f5[molname]['complex'][()]
        gridx = f5[molname]['grid_points']['x'][()]
        gridy = f5[molname]['grid_points']['y'][()]
        gridz = f5[molname]['grid_points']['z'][()]

    # assemble grid data
    grid_points = (gridx, gridy, gridz)

    # create the interfance and identify contact residues
    db = interface(mol)
    chains = db.get_chains()
    res = db.get_contact_residues(chain1=chains[0], chain2=chains[1])
     
    # get the first center
    xyz_center = get_next_point(db, res)
    yield get_grid_index(xyz_center, grid_points)

    # get all other centers
    while True:
        xyz_center = get_next_point(db, res)
        yield get_grid_index(xyz_center, grid_points)

if __name__ == "__main__":
    h5 = 'one_sample.hdf5'
    mol = 'BA_105966'
    mol_data = (h5, mol)
    gen = generate_interface_center(mol_data)
    (i,j,k) = next(gen)
import h5py
from torch_geometric.data import Data, DataLoader
from Source.params import *
from torch_geometric.transforms import Compose, RandomRotate

random_rotate = Compose([
    RandomRotate(degrees=180, axis=0),
    RandomRotate(degrees=180, axis=1),
    RandomRotate(degrees=180, axis=2),
])

# Import h5py file to construct the general dataset
# See for explanation of different field in Illustris files: https://www.tng-project.org/data/docs/specifications/#sec2a
def general_tab(path):

    # Read hdf5 file
    f = h5py.File(path, 'r')

    SubhaloPos = f["Subhalo/SubhaloPos"][:]
    SubhaloMassType = f["Subhalo/SubhaloMassType"][:,4]
    SubhaloLenType = f["Subhalo/SubhaloLenType"][:,4]
    SubhaloHalfmassRadType = f["Subhalo/SubhaloHalfmassRadType"][:,4]
    SubhaloVel = f["Subhalo/SubhaloVel"][:]
    SubhaloVel = np.sqrt(np.sum(SubhaloVel**2., 1))

    #StarMetal = f["Subhalo/SubhaloStarMetallicity"][:]
    #SubhaloSFR = f["Subhalo/SubhaloSFR"][:]
    #SubhaloVelDisp  = f["Subhalo/SubhaloVelDisp"][:]

    HaloID = np.array(f["Subhalo/SubhaloGrNr"][:], dtype=np.int32)
    HaloMass = np.log10(f["Group/GroupMass"][:])
    #HaloMass1 = f["Group/Group_M_TopHat200"][:]
    #HaloMass = np.log10(HaloMass[HaloMass1>0.])
    GroupPos = f["Group/GroupPos"][:]
    #GroupPos = GroupPos[HaloMass1>0.]
    f.close()

    tab = np.column_stack((HaloID, SubhaloPos, SubhaloMassType, SubhaloLenType, SubhaloHalfmassRadType, SubhaloVel))#, StarMetal, SubhaloSFR, SubhaloVelDisp))

    tab = tab[tab[:,4]>0.]  # restrict to subhalos with galaxies
    tab = tab[tab[:,5]>10]  # more or less equivalent to the condition above
    tab[:,4] = np.log10(tab[:,4])
    tab[:,5] = np.log10(tab[:,5])

    """halolist = np.array(np.unique(tab[:,0]), dtype=np.int32)
    for ind in halolist:
        tab = tab[tab[:,0]==ind]
        tab[:,1:4]-
        GroupPos[ind]"""

    return tab, HaloMass, GroupPos

# Normalize an array
def normalize(array):

    mean, std = array.mean(axis=0), array.std(axis=0)
    newarray = (array - mean)/std

    #min, max = array.min(axis=0), array.max(axis=0)
    #newarray = (array - min)/(max - min)

    return newarray

"""# Creates particles with random positions and masses
def get_graph(tab, haloindex, HaloMass):

    tab_halo = tab[tab[:,0]==haloindex]
    #tab_halo[:,1:4] = tab_halo[:,1:4] - HaloPos[haloindex]
    #tab_halo[:,1:] = normalize(tab_halo[:,1:])
    data = Data(x=torch.tensor(tab_halo[:,1:], dtype=torch.float32) , y=torch.tensor(HaloMass, dtype=torch.float32) )

    return data"""

# Split training and validation sets
def split_datasets(dataset):

    random.shuffle(dataset)

    num_train = len(dataset)
    split_valid = int(np.floor(valid_size * num_train))
    split_test = split_valid + int(np.floor(test_size * num_train))

    train_dataset = dataset[split_test:]
    valid_dataset = dataset[:split_valid]
    test_dataset = dataset[split_valid:split_test]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader

# Load data and create the dataset
def create_dataset():
    hist=[]
    hmasses = []
    shmasses = []
    dataset = []
    subs = 0
    symbreg_err = []
    tottrue, totsym = [], []

    for sim in range(n_sims):
        path = simpath + "{:03d}.hdf5".format(sim)
        tab, HaloMass, HaloPos = general_tab(path)
        #HaloMass = normalize(HaloMass)
        halolist = np.array(np.unique(tab[:,0]), dtype=np.int32)
        #print(tab.shape, tab[0])
        #subs+=tab.shape[0]

        # Write the subhalo positions as the relative position to the host halo
        for ind in halolist:
            tab[tab[:,0]==ind][:,1:4] += -HaloPos[ind]

        # Normalize columns (except ID)
        tab[:,1:], HaloMass = normalize(tab[:,1:]), normalize(HaloMass)

        for ind in halolist:
            tab_halo = tab[tab[:,0]==ind][:,1:]

            graph = Data(x=torch.tensor(tab_halo, dtype=torch.float32), pos=torch.tensor(tab_halo[:,:3], dtype=torch.float32), y=torch.tensor(HaloMass[ind], dtype=torch.float32))
            num_subhalos = graph.x.shape[0]
            #if (num_subhalos>=submin):
            subs+=num_subhalos
            hist.append(num_subhalos)
            hmasses.append(np.array(graph.y))
            shmasses.append( np.sum(np.array(graph.x[:,3])) )
            dataset.append(graph)

            #ins = graph.x[:,4].detach().numpy()
            #symbreg_err.append( np.abs(total_fit(ins) - graph.y)/graph.y )
            #tottrue.append(graph.y)
            #totsym.append(total_fit(ins))

            if data_aug:
                graph = random_rotate(graph)
                dataset.append(graph)
                subs+=num_subhalos

    print("Total number of halos", len(dataset), "Total number of subhalos",subs)

    node_features = tab.shape[1]-1  # Number of features. Substract halo index

    return dataset, node_features

#!/home/yuzhang/tensorflow-py3.6/bin/python3.6

# Author: Yu Zhang
# License: simplified BSD
# coding: utf-8

###define model for training
import os
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn import preprocessing


import torch
from torch.utils.data import Dataset, DataLoader
from torch_scatter import scatter_add
##from torch_geometric.utils import maybe_num_nodes


def spmm(index, value, m, matrix):
    """Matrix product of sparse matrix with dense matrix.

    Args:
        index (:class:`LongTensor`): The index tensor of sparse matrix.
        value (:class:`Tensor`): The value tensor of sparse matrix.
        m (int): The first dimension of sparse matrix.
        matrix (:class:`Tensor`): The dense matrix.

    :rtype: :class:`Tensor`
    """

    row, col = index
    matrix = matrix if matrix.dim() > 1 else matrix.unsqueeze(-1)

    out = matrix[col]
    out = out * value.unsqueeze(-1)
    out = scatter_add(out, row, dim=0, dim_size=m)

    return out


##use the old add_self_loop function from pytorch-geometric
def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes

def add_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    r"""Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted, all existent self-loops will be removed and
    replaced by weights denoted by :obj:`fill_value`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        fill_value (int, optional): If :obj:`edge_weight` is not :obj:`None`,
            will add self-loops with edge weights of :obj:`fill_value` to the
            graph. (default: :obj:`1`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    loop_index = torch.arange(0,
                              num_nodes,
                              dtype=torch.long,
                              device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = edge_weight.new_full((num_nodes, ), fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight


def extract_event_data(events_all_subjects_file,Trial_Num=None):
    ### loading event designs
    if not Trial_Num:
        Trial_Num = 284
    if not os.path.isfile(events_all_subjects_file):
        print("event file not exist:", events_all_subjects_file)
        return None
    
    print('Collecting trial info from file:', events_all_subjects_file)
    subjects_trial_labels = pd.read_csv(events_all_subjects_file,sep="\t",encoding="utf8")
    ###print(subjects_trial_labels.keys())

    try:
        label_matrix = subjects_trial_labels['label_data'].values
        print(subjects_trial_labels.shape)
        #xx = label_matrix[0].split(",")
        subjects_trial_label_matrix = []
        for subi in range(len(label_matrix)):
            xx = [x.replace("['", "").replace("']", "") for x in label_matrix[subi].split("', '")]
            subjects_trial_label_matrix.append(xx)
        subjects_trial_label_matrix = pd.DataFrame(data=(subjects_trial_label_matrix))
    except:
        subjects_trial_label_matrix = subjects_trial_labels.loc[:, 'trial1':'trial' + str(Trial_Num)]
    #subjects_trial_label_matrix = subjects_trial_labels.values.tolist()
    trialID = subjects_trial_labels['trialID']
    sub_name = subjects_trial_labels['subject'].tolist()
    coding_direct = subjects_trial_labels['coding']
    #print(np.array(subjects_trial_label_matrix).shape,len(sub_name),len(np.unique(sub_name)),len(coding_direct))
    
    return subjects_trial_label_matrix, trialID, sub_name, coding_direct

def load_fmri_data_from_lmdb(lmdb_filename,fmri_files=None,fmri_data_clean=None, write_frequency=10):
    ##lmdb_filename = pathout + modality + "_MMP_ROI_act_1200R_test_Dec2018_ALL.lmdb"
    ## read lmdb matrix
    import lmdb
    import os
    os.environ['TENSORPACK_SERIALIZE'] = 'msg'
    os.environ['TENSORPACK_ONCE_SERIALIZE'] = 'msg'
    from tensorpack.utils.serialize import loads
    help(loads)
    
    print('loading data from file: %s' % lmdb_filename)
    matrix_dict = []
    fmri_sub_name = []

    ##########################################33
    lmdb_env = lmdb.open(lmdb_filename, subdir=False)
    try:
        lmdb_txn = lmdb_env.begin()
        listed_fmri_files = loads(lmdb_txn.get(b'__keys__'))
        listed_fmri_files = [l.decode("utf-8") for l in listed_fmri_files]
        print('Stored fmri data from files:')
        print(len(listed_fmri_files))
    except:
        print('Search each key for every fmri file...')


    with lmdb_env.begin() as lmdb_txn:
        cursor = lmdb_txn.cursor()
        for key, value in cursor:
            # print(key)
            if key == b'__keys__':
                continue
            pathsub = Path(os.path.dirname(key.decode("utf-8")))
            ##subname_info = os.path.basename(key.decode("utf-8")).split('_')
            ##fmri_sub_name.append('_'.join((subname_info[0], subname_info[2], subname_info[3])))
            #############change due to directory switch to projects
            subname_info = str(Path(os.path.dirname(key.decode("utf-8"))).parts[-3])
            fmri_sub_name.append(Path(os.path.dirname(key.decode("utf-8"))).parts[-1].replace('tfMRI',subname_info))
            data = loads(lmdb_txn.get(key)).astype('float32', casting='same_kind')
            matrix_dict.append(np.array(data))
    lmdb_env.close()

    return matrix_dict, fmri_sub_name

##load fmri data from array
class HCP_taskfmri_matrix_datasets(Dataset):
    ##build a new class for own dataset
    import numpy as np
    def __init__(self, fmri_data_matrix, label_matrix, target_name,
                 isTrain='train', block_dura=1, transform=False):
        super(HCP_taskfmri_matrix_datasets, self).__init__()

        if not isinstance(fmri_data_matrix, np.ndarray):
            self.fmri_data_matrix = np.array(fmri_data_matrix)
        else:
            self.fmri_data_matrix = fmri_data_matrix
        
        self.Subject_Num = self.fmri_data_matrix.shape[0]
        self.Region_Num = self.fmri_data_matrix[0].shape[-1]

        if isinstance(label_matrix, pd.DataFrame):
            self.label_matrix = label_matrix
        elif isinstance(label_matrix, np.ndarray):
            self.label_matrix = pd.DataFrame(data=np.array(label_matrix))
        self.target_names = target_name

        self.block_dura = block_dura
        self.data_type = isTrain
        self.transform = transform

    def __len__(self):
        return self.Subject_Num

    def __getitem__(self, idx):
        #step1: get one subject data
        fmri_trial_data = self.fmri_data_matrix[idx]
        label_trial_data = self.label_matrix.iloc[idx]


        ##step3: match fmri and event acoording to time
        fmri_data, label_data = self.match_fmri_event_data(fmri_trial_data, label_trial_data,self.block_dura)
        #print(fmri_data.shape, label_data.shape)
        Nsamples = fmri_data.shape[0]
        if self.transform:
            fmri_data_2d = fmri_data.reshape(Nsamples,-1)
            scaler = preprocessing.StandardScaler()
            fmri_data = scaler.fit_transform(fmri_data_2d).reshape(Nsamples,self.block_dura,self.Region_Num)
            #print(fmri_data.shape, label_data.shape)

        tensor_x = torch.stack([torch.FloatTensor(fmri_data[ii].transpose()) for ii in range(len(label_data))])  # transform to torch tensors
        tensor_y = torch.stack([torch.LongTensor([label_data[ii]]) for ii in range(len(label_data))])

        return tensor_x, tensor_y

    def match_fmri_event_data(self,fmri_trial_data, label_trial_data,block_dura=1):
        ###matching between fmri data and event data
        condition_mask = pd.Series(label_trial_data).isin(self.target_names)
        tc_matrix_select = np.array(fmri_trial_data[condition_mask, :])
        label_data_select = np.array(label_trial_data[condition_mask])
        ##print(tc_matrix_select.shape,label_data_select.shape)

        le = preprocessing.LabelEncoder()
        le.fit(list(self.target_names))
        label_data_int = le.transform(label_data_select)

        ##cut the trials
        label_data_trial_block = np.array(np.split(label_data_select, np.where(np.diff(label_data_int))[0] + 1))
        fmri_data_block = np.array(np.array_split(tc_matrix_select, np.where(np.diff(label_data_int))[0] + 1, axis=0))

        trial_duras = [label_data_trial_block[ii].shape[0] for ii in range(len(label_data_trial_block))]
        ##cut each trial to blocks
        fmri_data_block_new = []
        label_data_trial_block_new = []
        for ti, dura in enumerate(trial_duras):
            if dura < block_dura:  ##one sample per each trial
                trial_num_used = min(dura, block_dura)
                xx = fmri_data_block[ti][:trial_num_used, :]
                xx2 = np.expand_dims(xx.take(range(0, block_dura), axis=0, mode='clip'), axis=0)
                yy = np.array([label_data_trial_block[ti][0]])
            else:  ##multiple samples from one trial
                trial_num_used = dura // block_dura * block_dura
                chunks = int(np.floor(trial_num_used // block_dura))
                xx2 = np.array(np.array_split(fmri_data_block[ti][:trial_num_used, :], chunks, axis=0))
                yy = np.array(np.array_split(label_data_trial_block[ti][:trial_num_used], chunks, axis=0))[:, 0]

            fmri_data_block_new.append(xx2)
            label_data_trial_block_new.append(yy)
        label_data_matrix = np.concatenate(label_data_trial_block_new, axis=0)
        fmri_data_matrix = np.array(np.vstack(fmri_data_block_new)).astype('float32', casting='same_kind')

        return fmri_data_matrix, le.transform(label_data_matrix)


#####load fmri data from nii files
class HCP_taskfmri_files(Dataset):
    ##build a new class for own dataset

    def __init__(self, fmri_files, event_files, modality, target_labels, mmp_atlas,
                 TR=0.72, block_dura=1, isTrain='train',randomseed=1234, transform=True):
        super(HCP_taskfmri_files, self).__init__()

        self.fmri_files = fmri_files
        self.event_files = event_files
        self.Subject_Num = len(fmri_files)

        self.modality = modality
        self.target_labels = target_labels
        self.target_names = list(target_labels.values())
        self.mmp_atlas = mmp_atlas

        ###basic information about task fmri
        rs = np.random.RandomState(randomseed)
        self.TR = TR
        self.block_dura = block_dura
        self.isTrain = (isTrain[:3].lower().strip() == 'tra')
        self.transform = transform

    def __len__(self):
        return self.Subject_Num

    def __getitem__(self, idx):
        ###return one subject fmri and event data at one time
        fmri_file = self.fmri_files[idx]
        ev_file = self.event_files[idx]
        
        ##step1: load fmri data
        #to avoid loading atlas many times, we preload the atlas before reading fmri ddata
        atlas_roi = nib.load(self.mmp_atlas).get_data()
        fmri_matrix = self.load_fmri_data(fmri_file, atlas_roi)
        if fmri_matrix is None:
            print("Skip data loading for subject: ", fmri_file)
            return [fmri_file, None]
        Trial_Num, self.Region_Num = fmri_matrix.shape
        ##fmri_matrix = preprocessing.scale(fmri_matrix,axis=-1)

        ##step2: load event data
        label_matrix = self.load_event_trials(ev_file, Trial_Num)
        #print(fmri_matrix.shape, label_matrix.shape)
        
        ##step3: match fmri and event acoording to time
        fmri_data, label_data = self.match_fmri_event_data(fmri_matrix, label_matrix,self.block_dura)
        #print(fmri_data.shape, label_data.shape)
        Nsamples = fmri_data.shape[0]
        if self.transform:
            fmri_data_2d = fmri_data.reshape(Nsamples,-1)
            scaler = preprocessing.StandardScaler()
            fmri_data = scaler.fit_transform(fmri_data_2d).reshape(Nsamples,self.block_dura,self.Region_Num)
            #print(fmri_data.shape, label_data.shape)

        tensor_x = torch.stack([torch.FloatTensor(fmri_data[ii].transpose()) for ii in range(len(label_data))])  # transform to torch tensors
        tensor_y = torch.stack([torch.LongTensor([label_data[ii]]) for ii in range(len(label_data))])
        # print(tensor_x.size(),tensor_y.size())

        return tensor_x, tensor_y
    
    def load_fmri_data(self, fmri_file, atlas_roi):
        ###mapping full brain fmri data onto the atlas by averaging
        Region_Num = len(np.unique(atlas_roi)) #removing label 0

        tc_matrix = nib.load(str(fmri_file)).get_data()
        if atlas_roi.shape[1] != tc_matrix.shape[1]:
            ##some vertex might not have labels in the atlas
            tc_matrix = tc_matrix[:, range(atlas_roi.shape[1])]

        ##print('Read data from: ', fmri_file)
        Trial_Num, Node_Num = tc_matrix.shape
        tc_matrix_df = pd.DataFrame(data=tc_matrix.ravel(), columns=['tc_signal'])
        tc_matrix_df['roi_label'] = np.repeat(atlas_roi, Trial_Num, axis=0).ravel()
        tc_matrix_df['trial_id'] = np.repeat(np.arange(Trial_Num).reshape((Trial_Num, 1)), Node_Num, axis=1).ravel()

        tc_roi = tc_matrix_df.groupby(['roi_label', 'trial_id'], as_index=False).mean()
        tc_roi_matrix = tc_roi['tc_signal'][tc_roi['roi_label'] != 0]
        tc_roi_matrix = tc_roi_matrix.values.reshape(Region_Num, Trial_Num).transpose()

        return tc_roi_matrix
    
    def load_event_trials(self, ev_file, Trial_Num=None):
        ##trial info per volume
        ###loading the event design
        trial_infos = pd.read_csv(event_file,sep="\t",encoding="utf8",header = None,names=['onset','duration','rep','task'])
        Onsets = np.array((trial_infos.onset/TR).astype(int)) #(trial_infos.onset/TR).astype(int)
        Duras = np.array((trial_infos.duration/TR).astype(int)) #(trial_infos.duration/TR).astype(int)
        Movetypes = list(trial_infos.task)
        move_mask = pd.Series(Movetypes).isin(self.target_labels.keys())
        Onsets = Onsets[move_mask]
        Duras = Duras[move_mask]
        Movetypes = [Movetypes[i] for i in range(len(move_mask)) if move_mask[i]]

        event_len = Onsets[-1] + Duras[-1]
        while event_len > Trial_Num:
            print('Remove last trials due to incomplete scanning...short of {} volumes out of {}'.format(event_len - Trial_Num, Trial_Num))
            Onsets = Onsets[:-1];
            Duras = Duras[:-1];
            del Movetypes[-1]
            event_len = Onsets[-1] + Duras[-1]
        ###build the labels
        labels = ["rest"] * Trial_Num;
        for start, dur, move in zip(Onsets, Duras, Movetypes):
            for ti in range(start, start + dur):
                try:
                    labels[ti] = self.target_labels[move]
                except:
                    print('Error loading for event file {}'.format(ev_file))
                    print('Run in to #{} volume when having total {} volumes'.format(ti, Trial_Num))
                    print(Onsets, Duras, Movetypes, move)

        labels = pd.Series(labels)
        return labels
    
    def match_fmri_event_data(self,fmri_trial_data, label_trial_data,block_dura=1):
        ###matching between fmri data and event data
        condition_mask = pd.Series(label_trial_data).isin(self.target_names)
        tc_matrix_select = np.array(fmri_trial_data[condition_mask, :])
        label_data_select = np.array(label_trial_data[condition_mask])
        ##print(tc_matrix_select.shape,label_data_select.shape)

        le = preprocessing.LabelEncoder()
        le.fit(list(self.target_names))
        label_data_int = le.transform(label_data_select)

        ##cut the trials
        label_data_trial_block = np.array(np.split(label_data_select, np.where(np.diff(label_data_int))[0] + 1))
        fmri_data_block = np.array(np.array_split(tc_matrix_select, np.where(np.diff(label_data_int))[0] + 1, axis=0))

        trial_duras = [label_data_trial_block[ii].shape[0] for ii in range(len(label_data_trial_block))]
        ##cut each trial to blocks
        fmri_data_block_new = []
        label_data_trial_block_new = []
        for ti, dura in enumerate(trial_duras):
            if dura < block_dura:  ##one sample per each trial
                trial_num_used = min(dura, block_dura)
                xx = fmri_data_block[ti][:trial_num_used, :]
                xx2 = np.expand_dims(xx.take(range(0, block_dura), axis=0, mode='clip'), axis=0)
                yy = np.array([label_data_trial_block[ti][0]])
            else:  ##multiple samples from one trial
                trial_num_used = dura // block_dura * block_dura
                chunks = int(np.floor(trial_num_used // block_dura))
                xx2 = np.array(np.array_split(fmri_data_block[ti][:trial_num_used, :], chunks, axis=0))
                yy = np.array(np.array_split(label_data_trial_block[ti][:trial_num_used], chunks, axis=0))[:, 0]

            fmri_data_block_new.append(xx2)
            label_data_trial_block_new.append(yy)
        label_data_matrix = np.concatenate(label_data_trial_block_new, axis=0)
        fmri_data_matrix = np.array(np.vstack(fmri_data_block_new)).astype('float32', casting='same_kind')

        return fmri_data_matrix, le.transform(label_data_matrix)

    
def fmri_samples_collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    images, targets = zip(*data)
    
    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.FloatTensor(torch.cat(images)) #.permute(0, 2, 1)
    targets = torch.LongTensor(torch.cat(targets).squeeze())
    #images = images.view(-1,np.prod(images.shape[1:]))

    #Nsamples = len(targets)
    #rand_slice = np.random.choice(Nsamples, 1, replace=False)
    #print(images.shape, targets.shape)
    
    return images, targets



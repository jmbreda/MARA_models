import numpy as np
import pandas as pd
import argparse
import h5py
from scipy.sparse import coo_matrix

def argparser():
    parser = argparse.ArgumentParser(description='Get regulatory regions')
    parser.add_argument('--pseudo_count', type=int, default=1, help='Pseudo count for log transformation')
    parser.add_argument('--window_kb', type=int, default=5, help='Window size for regulatory regions')
    parser.add_argument('--genome', type=str, default='mm10', help='Genome version')
    parser.add_argument('--promid_transcript', type=str, help='Dataset name')
    parser.add_argument('--promoterome', type=str, help='Promoterome table')
    parser.add_argument('--mrna_table', type=str, help='mRNA TPM table')
    parser.add_argument('--premrna_table', type=str, help='pre-mRNA TPM table')
    parser.add_argument('--regulatory_regions', type=str, help='Regulatory regions bed file')
    parser.add_argument('--Chip_sf', type=str, help='ChIP-seq signal matrix in npy format')
    parser.add_argument('--chip_experiments', type=str, help='ChIP-seq experiment table')
    parser.add_argument('--PWM_sm', type=str, help='PWM signal matrix in npy format')
    parser.add_argument('--jaspar_clusters_to_tf', type=str, help='Jaspar clusters to TF table')
    parser.add_argument('--output', type=str, help='hdf5 file')

    return parser

# remove version from transcript id
def remove_transcript_version(x):
    return x.split('.')[0]

# remove version from promoter id
def remove_promoter_version(x):
    return '_'.join( x.split('_')[2:] )

if __name__ == '__main__':
    
    args = argparser().parse_args()

    # get transcript to promid table
    promid_transcripts = pd.read_csv(args.promid_transcript, sep='\t', header=None)
    promid_transcripts.columns = ['prom','transcript']

    # fill in nan promoters
    for i in promid_transcripts.index:
        if pd.isnull(promid_transcripts.iloc[i, 0]):
            promid_transcripts.iloc[i, 0] = promid_transcripts.iloc[i-1, 0]

    # remove version from transcript and promoter ids and assert uniqueness
    promid_transcripts.loc[:,'transcript'] = promid_transcripts.loc[:,['transcript']].map(remove_transcript_version)
    promid_transcripts.loc[:,'prom'] =  promid_transcripts.loc[:,['prom']].map(remove_promoter_version)
    assert len(promid_transcripts.transcript.unique()) == len(promid_transcripts)

    # make dictionary
    transcript2promid = dict(zip(promid_transcripts.transcript.values, promid_transcripts.prom.values))

    # get the promoterome and id2idx mapping
    promoterome = pd.read_csv(args.promoterome, sep='\t')
    promoterome.loc[:,'center'] = (promoterome.start + promoterome.end)/2
    prom_id = []
    prom_idx = []
    for i in promoterome.index:
        prom_id.append(promoterome.loc[i,'id'])
        prom_idx.append(i)
        if '|' in promoterome.loc[i,'id']:
            for id in promoterome.loc[i,'id'].split('|'):
                prom_id.append(id)
                prom_idx.append(i)
    prom_id2idx = dict(zip(prom_id, prom_idx))

    # get gene expression data
    mrna = pd.read_csv(args.mrna_table, sep='\t', index_col=0)
    premrna = pd.read_csv(args.premrna_table, sep='\t', index_col=0)

    # clean transcript ids and get as index
    premrna.index = [id.split('::')[0].split('_')[1] for id in premrna.index]
    mrna.index = [id.split('|')[0] for id in mrna.index]
    premrna.index = premrna.index.map(remove_transcript_version)
    mrna.index = mrna.index.map(remove_transcript_version)
    assert np.all(premrna.index == mrna.index)

    # get the number of samples and transcripts
    E_ct = (mrna + premrna).T

    # get transcript to promoter idx mapping
    Map_tg = np.zeros((E_ct.shape[1], len(promoterome)))
    for i,transcript in enumerate(E_ct.columns):
        if transcript in transcript2promid:    
            promid = transcript2promid[transcript]
            if promid in prom_id2idx:
                j = prom_id2idx[promid]
                Map_tg[i,j] = 1

    # map expression matrix from transcriptome to promoterome
    E_cg = np.log2( E_ct.values @ Map_tg + args.pseudo_count )
    E_cg = pd.DataFrame(E_cg, columns=promoterome.id, index=E_ct.index)

    # get R_sg matrix mapping regulatory regions 's' to promoter 'g'
    
    Regulatory_regions_bed = pd.read_csv(args.regulatory_regions, sep='\t')
    Regulatory_regions_bed.loc[:,'center'] = (Regulatory_regions_bed.start + Regulatory_regions_bed.end)/2
    D_sg = np.zeros((Regulatory_regions_bed.shape[0],promoterome.shape[0]))*np.nan

    D_sg_ij = np.zeros([0,2]).astype(int)
    D_sg_v = np.zeros(0)
    for s in Regulatory_regions_bed.index:
        prom_id = Regulatory_regions_bed.loc[s,'name']
        g = prom_id2idx[prom_id]
        d = (Regulatory_regions_bed.loc[s,'center'] - promoterome.loc[g,'center'])
        D_sg_ij = np.vstack([D_sg_ij, [s,g]])
        D_sg_v = np.concatenate([D_sg_v, [d]])

    # get the N_sf and N_sm matrices
    Chip_sf = np.load(args.Chip_sf)/1000
    PWM_sm = np.load(args.PWM_sm)

    # to sparse tensor
    out = coo_matrix(Chip_sf)
    Chip_sf_ij = np.concatenate([out.row[:,None], out.col[:,None]],1).astype(int)
    Chip_sf_v = out.data

    out = coo_matrix(PWM_sm)
    PWM_sm_ij = np.concatenate([out.row[:,None], out.col[:,None]],1).astype(int)
    PWM_sm_v = out.data

    clusters_to_tf = pd.read_csv(args.jaspar_clusters_to_tf,sep='\t',header=None)

    chip_experiment = pd.read_csv(args.chip_experiments , sep='\t',index_col=0)
    chip_experiment = chip_experiment.loc[:,'antigen']
    TFs = chip_experiment.unique()

    c_index = np.array(E_cg.index)
    g_index = np.array(E_cg.columns)
    f_index = TFs
    m_index = clusters_to_tf.loc[:,1].values

    # save data
    with h5py.File(args.output, 'w') as f:
        f.create_group('data')
        f.create_dataset('data/E_cg', data=E_cg.values)
        f.create_group('data/D_sg')
        f.create_dataset('data/D_sg/index', data=D_sg_ij)
        f.create_dataset('data/D_sg/value', data=D_sg_v)
        f.create_dataset('data/D_sg/dim', data=D_sg.shape)
        f.create_group('data/Chip_sf')
        f.create_dataset('data/Chip_sf/index', data=Chip_sf_ij)
        f.create_dataset('data/Chip_sf/value', data=Chip_sf_v)
        f.create_dataset('data/Chip_sf/dim', data=Chip_sf.shape)
        f.create_group('data/PWM_sm')
        f.create_dataset('data/PWM_sm/index', data=PWM_sm_ij)
        f.create_dataset('data/PWM_sm/value', data=PWM_sm_v)
        f.create_dataset('data/PWM_sm/dim', data=PWM_sm.shape)
        f.create_group('index')
        f.create_dataset('index/c', data=c_index)
        f.create_dataset('index/g', data=g_index)
        f.create_dataset('index/f', data=f_index)
        f.create_dataset('index/m', data=m_index)
        f.attrs['pseudo_count'] = args.pseudo_count
        f.attrs['window_kb'] = args.window_kb
        f.attrs['genome'] = args.genome


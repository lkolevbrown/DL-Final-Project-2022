import logging

import numpy as np
import pandas as pd

from config_path import *

data_path = DATA_PATH
processed_path = join(PROSTATE_DATA_PATH, 'processed')

"""
patient profiles
"""
gene_final_no_silent_no_intron = 'P1000_final_analysis_set_cross__no_silent_no_introns_not_from_the_paper.csv'
cnv_filename = 'P1000_data_CNA_paper.csv'
gene_important_mutations_only = 'P1000_final_analysis_set_cross_important_only.csv'
gene_important_mutations_only_plus_hotspots = 'P1000_final_analysis_set_cross_important_only_plus_hotspots.csv'
gene_hotspots = 'P1000_final_analysis_set_cross_hotspots.csv'
gene_truncating_mutations_only = 'P1000_final_analysis_set_cross_truncating_only.csv'
gene_expression = 'P1000_adjusted_TPM.csv'
fusions_filename = 'p1000_onco_ets_fusions.csv'
cnv_burden_filename = 'P1000_data_CNA_burden.csv'
fusions_genes_filename = 'fusion_genes.csv'

"""
patient responses
"""
response_filename = 'response_paper.csv' #patient responses (i.e. labels, yes or no cancer)

"""
This method is used to load patient molecular profile data into pandas dataframes.
@param filename: the file to load patient data from; assumes the response filepath
@param selected_genes: the genes to selectively keep from patient data
@return x: the patient profiles with only the selected_genes and no labels with shape [num_patients, num_genes]
@return response: the patient labels (primary or metastatic prostate cancer) [num_patients, 1] (PANDAS?)
@return samples: the patient names (really barcode tag numbers) [num_patients]
@return genes: the set of genes that were present in both selected_genes and the patient dataset
"""
def load_data(filename, selected_genes=None):

    def get_response():
        logging.info('loading response from %s' % response_filename)
        labels = pd.read_csv(join(processed_path, response_filename))
        labels = labels.set_index('id')
        return labels


    filename = join(processed_path, filename) # put the file into the processed folder
    logging.info('loading data from %s,' % filename)

    data = pd.read_csv(filename, index_col=0) #pandas dataframe, first row is a column row
    logging.info(data.shape)
    labels = get_response() #pandas dataframe

    # remove all zeros columns (note the column may be added again later if another feature type belongs to the same gene has non-zero entries).
    # zero_cols = data.sum(axis=0) == 0
    # data = data.loc[:, ~zero_cols]

    # join the data with the labels
    all = data.join(labels, how='inner') #join the label to the patient data based on id (because labels is indexed by id), 'inner' means take the intersection.
    all = all[~all['response'].isnull()] #i think this only keeps the ones with a non-null response

    response = all['response'] #get all responses
    samples = all.index #get 

    #remove the response column and then get remaining column headers
    del all['response']
    x = all
    genes = all.columns

    if not(selected_genes is None): #if there are selected genes
        intersect = set.intersection(set(genes), selected_genes)
        if len(intersect) < len(selected_genes):
            logging.warning('some genes dont exist in the original data set')

        x = x.loc[:, intersect] #get the columns with only the desired genes
        genes = intersect
    logging.info('loaded data %d samples, %d variables, %d responses ' % (x.shape[0], x.shape[1], response.shape[0]))
    logging.info(len(genes))
    return x, response, samples, genes

"""
TMB: Tumor Mutational Burden
Effectively, find the TMB by finding the number of mutations a patient has, take some kind of log
@return x: the patient's TMB np array [num ]
@return response: the patient labels (primary or metastatic prostate cancer) [num_patients, 1] (numpy array)
@return samples: the patient names (really barcode tag numbers) [num_patients]
@return genes: the set of genes that were present in both selected_genes and the patient dataset
"""
def load_TMB(filename=gene_final_no_silent_no_intron):
    x, response, samples, genes = load_data(filename)
    x = np.sum(x, axis=1) #count num mutations
    x = np.array(x) #reshape?
    x = np.log(1. + x) #take log
    n = x.shape[0] #num samples
    response = response.values.reshape((n, 1)) #ensure shape?
    samples = np.array(samples) #turn into np
    cols = np.array(['TMB']) #create a new column 'TMB'
    return x, response, samples, cols

def load_data_type(data_type='gene', cnv_levels=5, cnv_filter_single_event=True, mut_binary=False, selected_genes=None):
    logging.info('loading {}'.format(data_type))
    if data_type == 'TMB':
        x, response, info, genes = load_TMB(gene_important_mutations_only)
    if data_type == 'mut_no_silent_no_intron':
        x, response, info, genes = load_data(gene_final_no_silent_no_intron, selected_genes)
        if mut_binary:
            logging.info('mut_binary = True')
            x[x > 1.] = 1.

    if data_type == 'mut_important':
        x, response, info, genes = load_data(gene_important_mutations_only, selected_genes)
        if mut_binary:
            logging.info('mut_binary = True')
            x[x > 1.] = 1.

    if data_type == 'mut_important_plus_hotspots':
        x, response, info, genes = load_data(gene_important_mutations_only_plus_hotspots, selected_genes)

    if data_type == 'mut_hotspots':
        x, response, info, genes = load_data(gene_hotspots, selected_genes)

    if data_type == 'truncating_mut':
        x, response, info, genes = load_data(gene_truncating_mutations_only, selected_genes)
        if mut_binary:
            logging.info('mut_binary = True')
            x[x > 1.] = 1.

    return x, response, info, genes


# complete_features: make sure all the data_types have the same set of features_processing (genes)
def combine(x_list, y_list, rows_list, cols_list, data_type_list, combine_type, use_coding_genes_only=False):
    cols_list_set = [set(list(c)) for c in cols_list]

    if combine_type == 'intersection':
        cols = set.intersection(*cols_list_set)
    else:
        cols = set.union(*cols_list_set)

    if use_coding_genes_only:
        f = join(data_path, 'genes/HUGO_genes/protein-coding_gene_with_coordinate_minimal.txt')
        coding_genes_df = pd.read_csv(f, sep='\t', header=None)
        coding_genes_df.columns = ['chr', 'start', 'end', 'name']
        coding_genes = set(coding_genes_df['name'].unique())
        cols = cols.intersection(coding_genes)

    # the unique (super) set of genes
    all_cols = list(cols)

    all_cols_df = pd.DataFrame(index=all_cols)

    df_list = []
    for x, y, r, c in zip(x_list, y_list, rows_list, cols_list):
        df = pd.DataFrame(x, columns=c, index=r)
        df = df.T.join(all_cols_df, how='right')
        df = df.T
        df = df.fillna(0)
        df_list.append(df)

    all_data = pd.concat(df_list, keys=data_type_list, join='inner', axis=1, )

    # put genes on the first level and then the data type
    all_data = all_data.swaplevel(i=0, j=1, axis=1)

    # order the columns based on genes
    order = all_data.columns.levels[0]
    all_data = all_data.reindex(columns=order, level=0)

    x = all_data.values

    reordering_df = pd.DataFrame(index=all_data.index)
    y = reordering_df.join(y, how='left')

    y = y.values
    cols = all_data.columns
    rows = all_data.index
    logging.info(
        'After combining, loaded data %d samples, %d variables, %d responses ' % (x.shape[0], x.shape[1], y.shape[0]))

    return x, y, rows, cols


def split_cnv(x_df):
    genes = x_df.columns.levels[0]
    x_df.rename(columns={'cnv': 'CNA_amplification'}, inplace=True)
    for g in genes:
        x_df[g, 'CNA_deletion'] = x_df[g, 'CNA_amplification'].replace({-1.0: 0.5, -2.0: 1.0})
        x_df[g, 'CNA_amplification'] = x_df[g, 'CNA_amplification'].replace({1.0: 0.5, 2.0: 1.0})
    x_df = x_df.reindex(columns=genes, level=0)
    return x_df


"""
This class is a wrapper and loader for the desired input/output patient data.
@param data_type : string | list: the type of patient data to be loaded (i.e. 'mut' for gene mutation data), can be a list or string and will load all data of those type(s)
^see load_data_type() for more options 
@param selected_genes : string | list: if list, assumed to be all the desired genes; if string, assumed to be the filepath to the csv of desired genes
@param shuffle : boolean : whether or not the shuffle the loaded input
@param selected_samples : string : a file containing only the samples (input pateint data) that are desired
"""
class ProstateDataPaper():

    def __init__(self, data_type='mut', account_for_data_type=None, cnv_levels=5,
                 cnv_filter_single_event=True, mut_binary=False,
                 selected_genes=None, combine_type='intersection',
                 use_coding_genes_only=False, drop_AR=False,
                 balanced_data=False, cnv_split=False,
                 shuffle=False, selected_samples=None):

        """
        THIS IS ALL PREPROCESSING
        """
        """
        GET SELECTED GENES
        """
        if selected_genes is not None:
            if type(selected_genes) == list: #if it is a list, assume it contains the list of desired genes
                # list of genes
                selected_genes = selected_genes
            else: #if it is a string, assume it contains the filepath to a csv of the desired genes
                # file that will be used to load list of genes
                selected_genes_file = join(data_path, 'genes')
                selected_genes_file = join(selected_genes_file, selected_genes)
                df = pd.read_csv(selected_genes_file, header=0)
                selected_genes = list(df['genes'])

        if type(data_type) == list:
            x_list = []
            y_list = []
            rows_list = []
            cols_list = []

            for t in data_type:
                x, y, rows, cols = load_data_type(t, cnv_levels, cnv_filter_single_event, mut_binary, selected_genes)
                x_list.append(x), y_list.append(y), rows_list.append(rows), cols_list.append(cols)
            x, y, rows, cols = combine(x_list, y_list, rows_list, cols_list, data_type, combine_type,
                                       use_coding_genes_only)
            x = pd.DataFrame(x, columns=cols)

        else:
            x, y, rows, cols = load_data_type(data_type, cnv_levels, cnv_filter_single_event, mut_binary,
                                              selected_genes)

        if drop_AR:

            data_types = x.columns.levels[1].unique()
            ind = True
            if 'cnv' in data_types:
                ind = x[('AR', 'cnv')] <= 0.
            elif 'cnv_amp' in data_types:
                ind = x[('AR', 'cnv_amp')] <= 0.

            if 'mut_important' in data_types:
                ind2 = (x[('AR', 'mut_important')] < 1.)
                ind = ind & ind2
            x = x.loc[ind,]
            y = y[ind]
            rows = rows[ind]

        if cnv_split:
            x = split_cnv(x)

        if type(x) == pd.DataFrame:
            x = x.values

        if balanced_data:
            pos_ind = np.where(y == 1.)[0]
            neg_ind = np.where(y == 0.)[0]

            n_pos = pos_ind.shape[0]
            n_neg = neg_ind.shape[0]
            n = min(n_pos, n_neg)

            pos_ind = np.random.choice(pos_ind, size=n, replace=False)
            neg_ind = np.random.choice(neg_ind, size=n, replace=False)

            ind = np.sort(np.concatenate([pos_ind, neg_ind]))

            y = y[ind]
            x = x[ind,]
            rows = rows[ind]

        if shuffle:
            n = x.shape[0]
            ind = np.arange(n)
            np.random.shuffle(ind)
            x = x[ind, :]
            y = y[ind, :]
            rows = rows[ind]

        if account_for_data_type is not None:
            x_genomics = pd.DataFrame(x, columns=cols, index=rows)
            y_genomics = pd.DataFrame(y, index=rows, columns=['response'])
            x_list = []
            y_list = []
            rows_list = []
            cols_list = []
            for t in account_for_data_type:
                x_, y_, rows_, cols_ = load_data_type(t, cnv_levels, cnv_filter_single_event, mut_binary,
                                                      selected_genes)
                x_df = pd.DataFrame(x_, columns=cols_, index=rows_)
                x_list.append(x_df), y_list.append(y_), rows_list.append(rows_), cols_list.append(cols_)

            x_account_for = pd.concat(x_list, keys=account_for_data_type, join='inner', axis=1)
            x_all = pd.concat([x_genomics, x_account_for], keys=['genomics', 'account_for'], join='inner', axis=1)

            common_samples = set(rows).intersection(x_all.index)
            x_all = x_all.loc[common_samples, :]
            y = y_genomics.loc[common_samples, :]

            y = y['response'].values
            x = x_all.values
            cols = x_all.columns
            rows = x_all.index

        if selected_samples is not None:
            selected_samples_file = join(processed_path, selected_samples)
            df = pd.read_csv(selected_samples_file, header=0)
            selected_samples_list = list(df['Tumor_Sample_Barcode'])

            x = pd.DataFrame(x, columns=cols, index=rows)
            y = pd.DataFrame(y, index=rows, columns=['response'])

            x = x.loc[selected_samples_list, :]
            y = y.loc[selected_samples_list, :]
            rows = x.index
            cols = x.columns
            y = y['response'].values
            x = x.values

        """
        Initialize instance variables
        x represents the patient gene data
        y represents the labels for the patients
        info represents the patient names
        columns represents the columns present inside the document (i.e. specific gene mutations, TMB, CNV)
        """
        self.x = x
        self.y = y
        self.info = rows
        self.columns = cols

    """
    Return instance variables
    @return x represents the patient gene data
    @return y represents the labels for the patients
    @return info represents the patient names
    @return columns represents the columns present inside the document (i.e. specific gene mutations, TMB, CNV)
    """
    def get_data(self):
        return self.x, self.y, self.info, self.columns

    """
    Get training and testing splits
    """
    def get_train_validate_test(self):
        info = self.info
        x = self.x
        y = self.y
        columns = self.columns

        #path is _database/prostate/splits
        splits_path = join(PROSTATE_DATA_PATH, 'splits')

        #assuming that there are csv files containing the id's of the samples desired for the training, validation and testing sets
        testing_set = pd.read_csv(join(splits_path, 'test_set.csv'))

        #use set intersection to get desired training, validation, and testing samples
        #use list for slicing
        info_train = list(set(info).difference(testing_set.id))
        info_test = list(set(info).intersection(testing_set.id))

        #get indices
        ind_train = info.isin(info_train)
        ind_test = info.isin(info_test)

        #split patient data
        x_train = x[ind_train]
        x_test = x[ind_test]

        #split patient labels
        y_train = y[ind_train]
        y_test = y[ind_test]

        #split patient id's
        info_train = info[ind_train]
        info_test = info[ind_test]

        return x_train, x_test, y_train, y_test, info_train.copy(), info_test.copy(), columns
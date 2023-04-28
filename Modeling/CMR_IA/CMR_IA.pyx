from __future__ import print_function
import mkl
mkl.set_num_threads(1)
import os
import sys
import math
import time
import json
import numpy as np
import scipy.io
from glob import glob
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.math cimport log, sqrt
cimport numpy as np
cimport cython

"""
Known differences from Lynn Lohnas's                                                                                                                                                                                                                                                                                                                                                                                                                                                               code:
1) Cycle counter starts at 0 in this code instead of 1 during leaky accumulator.
2) No empty feature vector is presented at the end of the recall period.
"""


# Credit to "senderle" for the cython random number generation functions used below. Original code can be found at:
# https://stackoverflow.com/questions/42767816/what-is-the-most-efficient-and-portable-way-to-generate-gaussian-random-numbers
@cython.cdivision(True)
cdef double random_uniform():
    cdef double r = rand()
    return r / RAND_MAX


@cython.cdivision(True)
cdef double random_gaussian():
    cdef double x1, x2, w
    w = 2.0
    while (w >= 1.0):
        x1 = 2.0 * random_uniform() - 1.0
        x2 = 2.0 * random_uniform() - 1.0
        w = x1 * x1 + x2 * x2

    w = ((-2.0 * log(w)) / w) ** 0.5
    return x1 * w


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void assign_random_gaussian_pair(double[:] out, int assign_ix):
    cdef double x1, x2, w
    w = 2.0
    while (w >= 1.0):
        x1 = 2.0 * random_uniform() - 1.0
        x2 = 2.0 * random_uniform() - 1.0
        w = x1 * x1 + x2 * x2

    w = sqrt((-2.0 * log(w)) / w)
    out[assign_ix] = x1 * w
    out[assign_ix + 1] = x2 * 2


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cython_randn(int n):
    cdef int i
    np_result = np.zeros(n, dtype='f8', order='C')
    cdef double[:] result = np_result
    for i in range(n // 2):  # Int division ensures trailing index if n is odd.
        assign_random_gaussian_pair(result, i * 2)
    if n % 2 == 1:
        result[n - 1] = random_gaussian()

    return result

class CMR2(object):

    def __init__(self, params, pres_mat, sem_mat,
                 source_mat=None, rec_mat=None, ffr_mat=None, cue_mat=None,
                 task='FR', mode='IFR', pair=False):
        """
        Initializes a CMR2 object and prepares it to simulate the session defined
        by pres_mat. [Modified by Beige]

        :param params: A dictionary of model parameters and settings to use for the
            simulation. Call the CMR_IA.make_params() function to get a
            dictionary template you can fill in with your desired values.
        :param pres_mat: A 2D array specifying the ID numbers of the words that will be
            presented to the model on each trial. Row i, column j should hold the ID number
            of the jth word to be presented on the ith trial. ID numbers are assumed to
            range from 1 to N, where N is the number of words in the semantic similarity
            matrix (sem_mat). 0s are treated as padding and are ignored, allowing you to
            zero-pad pres_mat if you wish to simulate trials with varying list length.
        :param sem_mat: A 2D array containing the pairwise semantic similarities between all
            words in the word pool. The ordering of words in the similarity matrix must
            match the word ID numbers, such that the scores for word k must be located along
            row k-1 and column k-1.
        :param source_mat: If None, source coding will not be used. Otherwise, source_mat
            should be a 3D array containing source features for each presented word. The
            matrix should have one row for each trial and one column for each serial
            position, with the third dimension having length equal to the number of source
            features you wish to simulate. Cell (i, j, k) should contain the value of the
            kth source feature of the jth item presented on list i. (DEFAULT=None)
        :param rec_mat: A 2D array specifying the ID numbers of the words that are recalled
            by the real subjects in a free recall phase on each trial. The rows are
            corresponding to pres_mat. Used in simulating PEERS recognition. (DEFAULT=None)
        :param ffr_mat: A 1D array specifying the ID numbers of the words that are recalled
            by the real subjects in a final free recall phase. Used in simulating
            PEERS recognition. (DEFAULT=None)
        :param cue_mat: A 1D array specifying the ID numbers of the words that are presented
            to the model in recognition. Used in simulating recognition. (DEFAULT=None)
        :param task: A string indicating the type of task to simulate. Set to 'FR' for free
            recall or 'CR' for cued recall or 'Recog' for recognition. (DEFAULT='FR')
        :param mode: A string indicating the type of task mode to simulate. For task = 'FR',
            set to 'IFR' for immediate free recall or 'DFR' for delayed recall; for task = 'CR',
            set to 'Simultaneous' for simultaneous encoding or 'Sequential' for sequential encoding;
            for task = 'Recog', set to 'Continuous' for continuous recognition or 'Final' for a
            recognition in final stage. (DEFAULT='IFR')
        :param pair: whether presenting word pair. (DEFAULT = False)
        """
        ##########
        #
        # Set up model parameters and presentation data
        #
        ##########

        # Convert input parameters
        self.recog_similarity = [] # [bj]
        self.params = params  # Dictionary of model parameters
        self.pres_nos = np.array(pres_mat, dtype=np.int16)  # Presented item ID numbers (trial x serial position)
        self.sem_mat = np.array(sem_mat, dtype=np.float32)  # Semantic similarity matrix (e.g. Word2vec, LSA, WAS)
        self.extra_distract = 0

        ### [Beige edited begin]
        # input cue mat
        if cue_mat is None:
            self.have_cue = False
        else:
            self.have_cue = True
            self.cues_nos = np.array(cue_mat, dtype=np.int16)
        # input recall mat
        if rec_mat is None:
            self.have_rec = False
        else:
            self.have_rec = True
            self.rec_nos = np.array(rec_mat, dtype=np.int16)
        # input ffr mat
        if ffr_mat is None:
            self.have_ffr = False
        else:
            self.have_ffr = True
            self.ffr_nos = np.array(ffr_mat, dtype=np.int16)
        # input source
        if source_mat is None:
            self.nsources = 0
        else:
            self.sources = np.atleast_3d(source_mat).astype(np.float32)
            self.nsources = self.sources.shape[2]
            if self.sources.shape[0:2] != self.pres_nos.shape[0:2]:
                raise ValueError('Source matrix must have the same number of rows and columns as the presented item matrix.')
        # input task
        if task not in ('FR', 'CR','Recog'): # Task must in FR or CR or Recog
            raise ValueError('Task must be "FR" or "CR" or "Recog", not %s.' % task)
        if (task is 'CR' or task is 'Recog') and cue_mat is None:
            raise ValueError('%s task must has a cue matrix.' % task)
        self.task = task
        # input mode
        if mode not in ('IFR', 'DFR','Simultaneous','Sequential','Continuous','Final', 'Hockley'):
            raise ValueError('Mode must be "IFR" or "DFR", not %s.' % mode)
        self.mode = mode
        # input learn_while_retrieving
        self.learn_while_retrieving = self.params['learn_while_retrieving'] if 'learn_while_retrieving' in self.params else False

        # Determine the number of lists and the maximum list length (how many words or word-pairs)
        self.nlists = self.pres_nos.shape[0]
        self.max_list_length = self.pres_nos.shape[1]

        # Determine and unpair when presented word pairs
        self.paired_pres = self.pres_nos.ndim == 3 # whether presented word-pairs
        if self.paired_pres: # unpair word-pairs
            self.pres_nos_unpair = np.reshape(self.pres_nos,(self.nlists,self.max_list_length*self.pres_nos.shape[2]))
        else:
            self.pres_nos_unpair = self.pres_nos

        # Determine when cue word pairs
        self.paired_cues = self.cues_nos.ndim == 2
        # if self.paired_cues: # unpair word-pairs
        #     self.cues_nos_unpair = self.cues_nos.flatten()
        # else:
        #     self.cues_nos_unpair = self.cues_nos

        # Create arrays of sorted and unique (nonzero) items
        self.pres_nonzero_mask = self.pres_nos > 0
        self.pres_nos_nonzero = self.pres_nos[self.pres_nonzero_mask] # reduce to 1D
        self.all_nos = self.pres_nos_nonzero
        if self.have_rec:
            self.rec_nonzero_mask = self.rec_nos > 0
            self.rec_nos_nonzero = self.rec_nos[self.rec_nonzero_mask]
            self.all_nos = np.concatenate((self.all_nos, self.rec_nos_nonzero), axis=None)
        if self.have_ffr:
            self.ffr_nonzero_mask = self.ffr_nos > 0
            self.ffr_nos_nonzero = self.ffr_nos[self.ffr_nonzero_mask]
            self.all_nos = np.concatenate((self.all_nos, self.ffr_nos_nonzero), axis=None)
            self.extra_distract += 1
        if self.have_cue:
            self.cues_nonzero_mask = self.cues_nos > 0
            self.cues_nos_nonzero = self.cues_nos[self.cues_nonzero_mask]  # reduce to 1D
            self.all_nos = np.concatenate((self.all_nos,self.cues_nos_nonzero),axis=None)
            if self.mode == 'Final': # for PEERS task
                self.extra_distract += 1
        self.all_nos_sorted = np.sort(self.all_nos)
        self.all_nos_unique = np.unique(self.all_nos_sorted) # 1D, order in feature vector

        # Convert presented item and cue item ID numbers to indexes within the feature vector
        self.pres_indexes = np.searchsorted(self.all_nos_unique, self.pres_nos)
        if self.have_rec:
            self.rec_indexes = np.searchsorted(self.all_nos_unique, self.rec_nos)
        if self.have_ffr:
            self.ffr_indexes = np.searchsorted(self.all_nos_unique, self.ffr_nos)
        if self.have_cue:
            cue_indexer = lambda x: np.searchsorted(self.all_nos_unique, x) if x > 0 else x
            cue_func = np.vectorize(cue_indexer)
            self.cues_indexes = cue_func(self.cues_nos)
            # self.cues_indexes = np.searchsorted(self.all_nos_unique, self.cues_nos)

        # Set up elevated-attention scaling vector for all itemno
        self.sem_mean = np.mean(self.sem_mat, axis=1)
        # self.att_vec = np.ones(len(self.sem_mean)) + self.params['n'] * np.exp(-1 * self.params['m'] * (self.sem_mean - self.params['t'])) + self.params['p']
        self.att_vec = self.params['m'] * self.sem_mean + self.params['n']
        self.att_vec[self.att_vec > 1/self.params['gamma_fc']] = 1/self.params['gamma_fc']
        self.att_vec[self.att_vec < 0] = 0
        self.c_vec = self.params['c1'] * self.sem_mean + self.params['c_thresh']

        # extract model-calculated word frequency
        self.b0 = 6.8657
        self.b1 = -12.7856
        self.cal_word_freq = np.exp(self.b0 + self.sem_mean * self.b1)
        ### [Beige edited end]

        # Cut down semantic matrix to contain only the items in the session
        self.sem_mat = self.sem_mat[self.all_nos_unique - 1, :][:, self.all_nos_unique - 1]
        # Make sure items' associations with themselves are set to 0
        np.fill_diagonal(self.sem_mat, 0)

        # Initial phase
        self.phase = None

        # Initial beta [Beige]
        self.beta = 0
        self.beta_source = 0

        ##########
        #
        # Set up context and feature vectors
        #
        ##########

        # Determine number of cells in each region of the feature/context vectors
        # self.nitems = self.pres_nos.size # [bj]
        self.nitems_unique = len(self.all_nos_unique) # [bj]
        self.ndistractors = self.nlists + self.extra_distract # One distractor prior to each list + ffr + recog
        if self.mode == 'DFR':
            self.ndistractors += self.nlists  # One extra distractor before each recall period if running DFR
        self.ntemporal = self.nitems_unique + self.ndistractors
        self.nelements = self.ntemporal + self.nsources

        # Create context and feature vectors
        self.f = np.zeros((self.nelements, 1), dtype=np.float32)
        self.c = np.zeros_like(self.f)
        self.c_old = np.zeros_like(self.f)
        self.c_in = np.zeros_like(self.f)

        ##########
        #
        # Set up weight matrices
        #
        ##########

        # Set up primacy scaling vector
        self.prim_vec = self.params['phi_s'] * np.exp(-1 * self.params['phi_d'] * np.arange(self.max_list_length)) + 1

        # Set up learning rate matrix for M_FC (dimensions are context x features)
        self.L_FC = np.empty((self.nelements, self.nelements), dtype=np.float32)
        if self.nsources == 0:
            self.L_FC.fill(self.params['gamma_fc']) # if no source, uniformly gamma_fc
        else:
            # Temporal Context x Item Features (items reinstating their previous temporal contexts)
            self.L_FC[:self.ntemporal, :self.ntemporal] = self.params['L_FC_tftc']
            # Temporal Context x Source Features (sources reinstating previous temporal contexts)
            self.L_FC[:self.ntemporal, self.ntemporal:] = self.params['L_FC_sftc']
            # Source Context x Item Features (items reinstating previous source contexts)
            self.L_FC[self.ntemporal:, :self.ntemporal] = self.params['L_FC_tfsc']
            # Source Context x Source Features (sources reinstating previous source contexts)
            self.L_FC[self.ntemporal:, self.ntemporal:] = self.params['L_FC_sfsc']

        # Set up learning rate matrix for M_CF (dimensions are features x context)
        self.L_CF = np.empty((self.nelements, self.nelements), dtype=np.float32)
        if self.nsources == 0:
            self.L_CF.fill(self.params['gamma_cf']) # if no source, uniformly gamma_cf
        else:
            # Item Features x Temporal Context (temporal context cueing retrieval of items)
            self.L_CF[:self.ntemporal, :self.ntemporal] = self.params['L_CF_tctf']
            # Item Features x Source Context (source context cueing retrieval of items)
            self.L_CF[:self.ntemporal, self.ntemporal:] = self.params['L_CF_sctf']
            # Source Features x Temporal Context (temporal context cueing retrieval of sources)
            self.L_CF[self.ntemporal:, :self.ntemporal] = self.params['L_CF_tcsf']
            # Source Features x Source Context (source context cueing retrieval of sources)
            self.L_CF[self.ntemporal:, self.ntemporal:] = self.params['L_CF_scsf']

        # Initialize weight matrices as identity matrices
        self.M_FC = np.identity(self.nelements, dtype=np.float32)
        self.M_CF = np.identity(self.nelements, dtype=np.float32)

        # Scale the semantic similarity matrix by s_fc (Healey et al., 2016) and s_cf (Lohnas et al., 2015)
        fc_sem_mat = self.params['s_fc'] * self.sem_mat
        cf_sem_mat = self.params['s_cf'] * self.sem_mat

        # Complete the pre-experimental associative matrices by layering on the
        # scaled semantic matrices
        # [bj] elements include distractors and items, sem_mat just apply to items
        self.M_FC[:self.nitems_unique, :self.nitems_unique] += fc_sem_mat
        self.M_CF[:self.nitems_unique, :self.nitems_unique] += cf_sem_mat

        # Scale pre-experimental associative matrices by 1 - gamma
        self.M_FC *= 1 - self.L_FC
        self.M_CF *= 1 - self.L_CF

        #####
        #
        # Initialize leaky accumulator and recall variables
        #
        #####

        self.ret_thresh = np.ones(self.nitems_unique, dtype=np.float32)  # Retrieval thresholds
        self.nitems_in_race = self.params['nitems_in_accumulator']  # Number of items in accumulator
        self.rec_items = []  # Recalled items from each trial
        self.rec_times = []  # Rectimes of recalled items from each trial

        # Calculate dt_tau and its square root based on dt
        self.params['dt_tau'] = self.params['dt'] / 1000.
        self.params['sq_dt_tau'] = np.sqrt(self.params['dt_tau'])

        ##########
        #
        # Initialize variables for tracking simulation progress
        #
        ##########

        self.trial_idx = 0  # Current trial number (0-indexed)
        self.serial_position = 0  # Current serial position (0-indexed)
        self.distractor_idx = self.nitems_unique  # Current distractor index
        self.first_source_idx = self.ntemporal  # Index of the first source feature

        # for test
        self.f_in_acc = []

    def run_fr_trial(self):
        """
        Simulates an entire standard trial, consisting of the following steps:
        1) A pre-trial context shift
        2) A sequence of item presentations
        3) A pre-recall distractor (only if the mode was set to 'DFR')
        4) A recall period
        [Unchanged from CMR2]
        """
        ##########
        #
        # Shift context before start of new list
        #
        ##########

        # On first trial, present orthogonal item that starts the system;
        # On subsequent trials, present an interlist distractor item
        # Assume source context changes at same rate as temporal between trials
        # initialize context vector to have non-zero elements, no updating matrix
        self.phase = 'pretrial'
        self.serial_position = 0
        self.beta = 1 if self.trial_idx == 0 else self.params['beta_rec_post'] # learn pre-trial distractor
        self.beta_source = 1 if self.trial_idx == 0 else self.params['beta_rec_post']
        # Treat initial source and intertrial source as an even mixture of all sources
        #source = np.zeros(self.nsources) if self.nsources > 0 else None
        source = self.sources[self.trial_idx, self.serial_position] if self.nsources > 0 else None
        self.present_item(self.distractor_idx, source, update_context=True, update_weights=False)
        self.distractor_idx += 1

        ##########
        #
        # Present items
        #
        ##########

        self.phase = 'encoding'
        for self.serial_position in range(self.pres_indexes.shape[1]):
            # Skip over any zero-padding in the presentation matrix in order to allow variable list length
            if not self.pres_nonzero_mask[self.trial_idx, self.serial_position].all:
                continue
            pres_idx = self.pres_indexes[self.trial_idx, self.serial_position] # [bj] if word-pair, give a pair
            source = self.sources[self.trial_idx, self.serial_position] if self.nsources > 0 else None
            self.beta = self.params['beta_enc']
            self.beta_source = self.params['beta_source'] if self.nsources > 0 else 0
            self.present_item(pres_idx, source, update_context=True, update_weights=True)

        ##########
        #
        # Pre-recall distractor (if delayed free recall)
        #
        ##########

        if self.mode == 'DFR':
            self.phase = 'distractor'
            self.beta = self.params['beta_distract']
            # Assume source context changes at the same rate as temporal during distractors
            self.beta_source = self.params['beta_distract']
            # By default, treat distractor source as an even mixture of all sources
            # [If your distractors and sources are related, you should modify this so that you can specify distractor source.]
            #source = np.zeros(self.nsources) if self.nsources > 0 else None
            source = self.sources[self.trial_idx, self.serial_position] if self.nsources > 0 else None
            self.present_item(self.distractor_idx, source, update_context=True, update_weights=False)
            self.distractor_idx += 1

        ##########
        #
        # Recall period
        #
        ##########

        self.phase = 'recall'
        self.beta = self.params['beta_rec']
        # Follow Polyn et al. (2009) assumption that beta_source is the same at encoding and retrieval
        self.beta_source = self.params['beta_source'] if self.nsources > 0 else 0
        self.rec_items.append([])
        self.rec_times.append([])

        if self.task == 'FR':
            if 'max_recalls' in self.params:  # Limit number of recalls per trial if user has specified a maximum
                self.simulate_recall(time_limit=self.params['rec_time_limit'], max_recalls=self.params['max_recalls'])
            else:
                self.simulate_recall(time_limit=self.params['rec_time_limit'])

        self.trial_idx += 1

    def run_cr_trial(self):
        """
        Simulates a standard trial of cued recall, consisting of the following steps:
        1) A pre-trial context shift
        2) A sequence of item (word pair) presentations
        3) A recall period
        [Newly added by Beige]
        """
        ##########
        #
        # Shift context before start of new list
        #
        ##########

        # On first trial, present orthogonal item that starts the system;
        # On subsequent trials, present an interlist distractor item
        # Assume source context changes at same rate as temporal between trials
        # initialize context vector to have non-zero elements, no updating matrix
        # print("Pretrial Start!")
        self.phase = 'pretrial'
        self.serial_position = 0
        self.beta = 1 if self.trial_idx == 0 else self.params['beta_rec_post'] # learn pre-trial distractor
        self.beta_source = 1 if self.trial_idx == 0 else self.params['beta_rec_post']
        # Treat initial source and intertrial source as an even mixture of all sources
        #source = np.zeros(self.nsources) if self.nsources > 0 else None
        source = self.sources[self.trial_idx, self.serial_position] if self.nsources > 0 else None
        self.present_item(self.distractor_idx, source, update_context=True, update_weights=False)
        self.distractor_idx += 1
        # print("pre f:\n", self.f)
        # print("pre c:\n", self.c)
        # print("pre M_FC:\n",self.M_FC)
        # print("pre M_CF:\n",self.M_CF)
        # print("Pretrial End!")

        ##########
        #
        # Present items
        #
        ##########

        # print("Encoding Start!")
        self.phase = 'encoding'
        for self.serial_position in range(self.pres_indexes.shape[1]):
            # Skip over any zero-padding in the presentation matrix in order to allow variable list length
            if not self.pres_nonzero_mask[self.trial_idx, self.serial_position].all:
                continue
            pres_idx = self.pres_indexes[self.trial_idx, self.serial_position] # [bj] if word-pair, give a pair
            source = self.sources[self.trial_idx, self.serial_position] if self.nsources > 0 else None
            self.beta = self.params['beta_enc']
            self.beta_source = self.params['beta_source'] if self.nsources > 0 else 0

            if self.mode == 'Simultaneous':
                self.present_item(pres_idx, source, update_context=True, update_weights=True)
            elif self.mode == 'Sequential':
                self.present_item(pres_idx[0], source, update_context=False, update_weights=True)
                # between encoding 2 items could be a potential context drift
                self.present_item(pres_idx[1], source, update_context=False, update_weights=True)
                # then update the context all at a time
                self.present_item(pres_idx, source, update_context=True, update_weights=False)
            # print("encoding",self.serial_position,"item index:\n", pres_idx)
            # print("encoding",self.serial_position,"f:\n", self.f)
            # print("encoding",self.serial_position,"c:\n", self.c)
            # print("encoding",self.serial_position,"M_FC:\n",self.M_FC)
            # print("encoding",self.serial_position,"M_CF:\n",self.M_CF)
        # print("Encoding End!")

        ##########
        #
        # Recall period
        #
        ##########

        # print("Recall Start!")
        self.phase = 'recall'
        self.beta = self.params['beta_rec']
        # Follow Polyn et al. (2009) assumption that beta_source is the same at encoding and retrieval
        self.beta_source = self.params['beta_source'] if self.nsources > 0 else 0
        self.rec_items.append([])
        self.rec_times.append([])
        for self.cue_idx in range(self.cues_nos.shape[1]):
            # print("Cue Index:\n", self.cues_indexes[self.trial_idx,self.cue_idx])
            self.simulate_cr(self.cue_idx,time_limit=self.params['rec_time_limit'])

        self.trial_idx += 1
        # print("Recall End!")

    def run_peers_recog_single_sess(self):
        """
        Simulates a session of PEERS1&3 recognition, consisting of the following steps:
        1) Pre-trial context initialization / between-trial distractor
        2) Item presentation as encoding
        3) Item presentation as actual free recall
        4) Loop step 1-3 for n times if you have n trials
        5) Item presentation as actural final free recall (If have one)
        6) Pre-recog distractor
        7) Recognition simulation
        [Newly added by Beige]
        """
        for self.trial_idx in range(self.nlists):
            ##########
            #
            # Shift context before start of new list
            #
            ##########

            # On first trial, present orthogonal item that starts the system;
            # On subsequent trials, present an interlist distractor item
            # Assume source context changes at same rate as temporal between trials
            # [bj] initialize context vector to have non-zero elements, no updating matrix
            # print("Pretrial Start!")
            self.phase = 'pretrial'
            self.serial_position = 0
            self.beta = 1 if self.trial_idx == 0 else self.params['beta_rec_post'] # jbg: learn pre-trial distractor
            self.beta_source = 1 if self.trial_idx == 0 else self.params['beta_rec_post']
            # Treat initial source and intertrial source as an even mixture of all sources
            #source = np.zeros(self.nsources) if self.nsources > 0 else None
            source = self.sources[self.trial_idx, self.serial_position] if self.nsources > 0 else None
            self.present_item(self.distractor_idx, source, update_context=True, update_weights=False)
            self.distractor_idx += 1
            # print("pre M_FC:\n",self.M_FC)
            # print("pre M_CF:\n",self.M_CF)
            # print("Pretrial End!")

            ##########
            #
            # Present items
            #
            ##########

            # print("Encoding Start!")
            self.phase = 'encoding'
            for self.serial_position in range(self.pres_indexes.shape[1]):
                # Skip over any zero-padding in the presentation matrix in order to allow variable list length
                if not self.pres_nonzero_mask[self.trial_idx, self.serial_position].all:
                    continue
                pres_idx = self.pres_indexes[self.trial_idx, self.serial_position] # [bj] if word-pair, give a pair
                source = self.sources[self.trial_idx, self.serial_position] if self.nsources > 0 else None
                self.beta = self.params['beta_enc']
                # f_s = self.att_vec[self.all_nos_unique[pres_idx] - 1]
                # b = np.log(1 / self.params['beta_enc'] - 1)
                # self.beta = 1 / (1 + np.exp(f_s + b)) if f_s > 0 else self.params['beta_enc']
                self.beta_source = self.params['beta_source'] if self.nsources > 0 else 0
                # print("encoding",self.serial_position,"item index:\n", pres_idx)  #
                # print("encoding",self.serial_position, "is", self.beta)  #
                self.present_item(pres_idx, source, update_context=True, update_weights=True)
                # print("encoding",self.serial_position,"M_FC:\n",self.M_FC)
                # print("encoding",self.serial_position,"phi:\n",self.prim_vec[self.serial_position])
                # print("encoding",self.serial_position,"M_CF:\n",self.M_CF)
            # print("Encoding End!")

            ##########
            #
            # Simulate the Actual Free Recall
            #
            ##########

            # print("Recall Start!")
            if self.have_rec:
                self.phase = 'recall'
                for self.rec_position in range(self.rec_indexes.shape[1]):
                    # Skip over any zero-padding in the presentation matrix in order to allow variable list length
                    if not self.rec_nonzero_mask[self.trial_idx, self.rec_position].all:
                        continue
                    rec_idx = self.rec_indexes[self.trial_idx, self.rec_position] # [bj] if word-pair, give a pair
                    self.beta = self.params['beta_rec']
                    self.beta_source = 0
                    self.present_item(rec_idx, source=None, update_context=True, update_weights=False)
            # print("Recall", "M_FC:\n", self.M_FC)
            # print("Recall", "M_CF:\n", self.M_CF)
            # print("Recall End!")

        ##########
        #
        # Simulate an actual final free recall
        #
        ##########

        if self.have_ffr:
            # shift the context before the final free recall
            # print("Pre FFR Start!")
            self.phase = 'shift'
            self.serial_position = 0
            self.beta = self.params['beta_rec_post']
            self.beta_source = self.params['beta_rec_post']
            source = None
            self.present_item(self.distractor_idx, source, update_context=True, update_weights=False)
            self.distractor_idx += 1
            # print("Pre FFR End!")

            # update the context as if we do the actual final free recall
            # print("FFR Start!")
            self.phase = 'FFR'
            for self.ffr_position in range(len(self.ffr_indexes)):
                # Skip over any zero-padding in the presentation matrix in order to allow variable list length
                if not self.ffr_nonzero_mask[self.ffr_position].all:
                    continue
                ffr_idx = self.ffr_indexes[self.ffr_position] # [bj] if word-pair, give a pair
                # print("FFR Index:\n", ffr_idx)
                self.beta = self.params['beta_rec']
                self.beta_source = 0
                self.present_item(ffr_idx, source=None, update_context=True, update_weights=False)
            # print("ffr M_FC:\n", self.M_FC)
            # print("ffr M_CF:\n", self.M_CF)
            # print("FFR End!")

        ###########
        #
        # Simulated Recognition Period
        #
        ##########

        # shift the context before the recognition
        # print("Pre Recog Start!")
        self.phase = 'shift'
        self.serial_position = 0
        self.beta = self.params['beta_rec_post']
        self.beta_source = self.params['beta_rec_post']
        source = None
        self.present_item(self.distractor_idx, source, update_context=True, update_weights=False)
        self.distractor_idx += 1
        # print("pre recog M_FC:\n", self.M_FC)
        # print("pre recog M_CF:\n", self.M_CF)
        # print("Pre Recog End!")

        # Finally, run the recognition!
        # print("Recog Start!")
        self.phase = 'recognition'
        self.beta = self.params['beta_rec']
        self.beta_source = self.params['beta_source'] if self.nsources > 0 else 0
        for self.cue_position in range(len(self.cues_indexes)):
            cue_idx = self.cues_indexes[self.cue_position]
            # print("Cue Index:\n", cue_idx)
            self.simulate_recog(cue_idx)
        # print("Recog End!")

    def run_conti_recog_single_sess(self):
        """
        Simulates a session of continuous recognition, consisting of the following steps:
        1) Pre-session context initialization / between-trial distractor
        2) Recognition simulation
        3) Item presentation as encoding
        4) Loop step 1-3
        9.12 update: beta_enc for new judgement and beta_rec for old judgement
        11.16? update: allow for paired presentation
        [Newly added by Beige]
        """
        for trial_idx in range(self.nlists):
            ##########
            #
            # Shift context before each trial
            #
            ##########

            # On first trial, present orthogonal item that starts the system;
            # On subsequent trials, present an interlist distractor item
            # Assume source context changes at same rate as temporal between trials
            # initialize context vector to have non-zero elements, no updating matrix
            # print("Pretrial Start!") #
            self.phase = 'pretrial'
            self.serial_position = 0
            self.beta = 1 if trial_idx == 0 else self.params['beta_rec_post'] # jbg: pre-session or between trial
            self.beta_source = 1 if trial_idx == 0 else self.params['beta_rec_post']
            source = None
            self.present_item(self.distractor_idx, source, update_context=True, update_weights=False)
            self.distractor_idx += 1
            # print("pre c_in:\n", self.c_in)  #
            # print("pre c:\n", self.c)  #
            # print("pre M_FC:\n",self.M_FC)  #
            # print("Pretrial End!") #

            ###########
            #
            # Simulated Recognition
            #
            ##########

            # print("Recog Start!") #
            self.phase = 'recognition'
            self.beta = 0
            self.beta_source = self.params['beta_source'] if self.nsources > 0 else 0
            cue_idx = self.cues_indexes[trial_idx]
            self.simulate_recog(cue_idx)
            # print("recog c_old:\n", self.c_old)
            # print("recog c_in:\n", self.c_in)  #
            # print("Recog End!")  #

            ##########
            #
            # Present items
            #
            ##########

            # print("Encoding Start!")  #
            self.phase = 'encoding'
            pres_idx = self.pres_indexes[trial_idx, self.serial_position]
            source = self.sources[trial_idx, self.serial_position] if self.nsources > 0 else None
            self.beta = self.params['beta_enc'] if self.rec_items[-1]==0 else self.params['beta_rec'] # if judge "new", beta_enc, if judge "old", beta_rec
            self.beta_source = self.params['beta_source'] if self.nsources > 0 else 0
            self.present_item(pres_idx, source, update_context=True, update_weights=True)
            # print("encoding c_in:\n", self.c_in)  #
            # print("encoding c:\n", self.c)  #
            # print("encoding M_FC:\n",self.M_FC) #
            # print("Encoding End!")  #

    def run_hockley_recog_single_sess(self):
        """
        Simulates a session of continuous recognition, consisting of the following steps:
        1) Pre-session context initialization / between-trial distractor
        3) Item presentation as encoding
        4) Loop step 1-3
        [Newly added by Beige]
        """

        phases = ['pretrial','encoding','recognition']
        for trial_idx in range(self.nlists):
            for phase in phases:
                self.phase = phase

                if self.phase == 'pretrial':
                    #####
                    # Shift context before each trial
                    #####
                    # On first trial, present orthogonal item that starts the system
                    # On subsequent trials, present an interlist distractor item
                    # Assume source context changes at same rate as temporal between trials
                    self.beta = 1 if trial_idx == 0 else self.params['beta_rec_post']
                    self.beta_source = 1 if trial_idx == 0 else self.params['beta_rec_post']
                    source = None
                    self.present_item(self.distractor_idx, source, update_context=True, update_weights=False)
                    self.distractor_idx += 1
                    self.serial_position = 0
                    # print("pre c_in:\n", self.c_in)  #
                    # print("pre c:\n", self.c)  #
                    # print("pre M_FC:\n",self.M_FC)  #

                if self.phase == 'encoding':
                    #####
                    # Present items
                    #####
                    # print(self.phase)
                    pres_idx = self.pres_indexes[trial_idx, self.serial_position]
                    # print(pres_idx)
                    source = None
                    self.beta = self.params['beta_enc']
                    self.beta_source = 0
                    self.present_item(pres_idx, source, update_context=True, update_weights=True)
                    # print("encoding c_in:\n", self.c_in)  #
                    # print("encoding c:\n", self.c)  #
                    # print("encoding M_FC:\n",self.M_FC) #

                if self.phase == 'recognition':
                    #####
                    # Simulate recognition
                    #####
                    # print(self.phase)
                    self.beta = self.params['beta_rec']
                    self.beta_source = 0
                    cue_idx = self.cues_indexes[trial_idx]
                    # print(cue_idx)
                    if len(cue_idx) == 2:
                        if cue_idx[1] == -1:
                            cue_idx = cue_idx[0].astype(int)
                    self.simulate_recog(cue_idx)
                    # print("recog c_old:\n", self.c_old)
                    # print("recog c_in:\n", self.c_in)  #

    def run_norm_cr_single_sess(self):
        """
        Simulates a standard trial of cued recall, consisting of the following steps:
        1) A pre-trial context shift
        2) A sequence of item (word pair) presentations
        3) A recall period
        [Newly added by Beige]
        """

        phases = ['pretrial', 'encoding', 'shift', 'recall']
        for trial_idx in range(self.nlists):
            for phase in phases:
                self.phase = phase

                if self.phase == 'pretrial' or self.phase == 'shift':
                    #####
                    # Shift context before each trial
                    #####
                    # On first trial, present orthogonal item that starts the system
                    # On subsequent trials, present an interlist distractor item
                    # Assume source context changes at same rate as temporal between trials
                    self.beta = 1 if self.phase == 'pretrial' else self.params['beta_rec_post']
                    self.beta_source = 1 if self.phase == 'pretrial' else self.params['beta_rec_post']
                    source = None
                    self.present_item(self.distractor_idx, source, update_context=True, update_weights=False)
                    self.distractor_idx += 1
                    self.serial_position = 0
                    # print("pre c_in:\n", self.c_in)  #
                    # print("pre c:\n", self.c)  #
                    # print("pre M_FC:\n",self.M_FC)  #

                if self.phase == 'encoding':
                    #####
                    # Present items
                    #####
                    for self.serial_position in range(self.pres_indexes.shape[1]):
                        # print(self.phase)
                        pres_idx = self.pres_indexes[trial_idx, self.serial_position]
                        # print(pres_idx)
                        source = None
                        self.beta = self.params['beta_enc']
                        self.beta_source = 0
                        self.present_item(pres_idx, source, update_context=True, update_weights=True)
                        # print("encoding c_in:\n", self.c_in)  #
                        # print("encoding c:\n", self.c)  #
                        # print("encoding M_FC:\n",self.M_FC) #

                if self.phase == 'recall':
                    #####
                    # Simulate cued recall
                    #####
                    # print(self.phase)
                    cue_cnt = 0
                    for cue_idx in self.cues_indexes:
                        self.beta = self.params['beta_rec']
                        self.beta_source = 0
                        # cue_idx = self.cues_indexes[trial_idx]
                        # print("before cue:\n",self.c)
                        # print(cue_idx)
                        self.simulate_cr(cue_idx)
                        cue_cnt += 1
                        if cue_cnt == len(self.cues_indexes)/2: # reset threshold when test2
                            self.ret_thresh = np.ones(self.nitems_unique, dtype=np.float32)
                        # print("recog c_old:\n", self.c_old) #
                        # print("recog c_in:\n", self.c_in)  #

    def present_item(self, item_idx, source=None, update_context=True, update_weights=True):
        """
        Presents a single item (or distractor) to the model by updating the
        feature vector. Also provides options to update context and the model's
        associative matrices after presentation. [Modified by Beige]

        :param item_idx: The index of the cell within the feature vector that
            should be activated by the presented item. If None, presents an
            empty feature vector.
        :param source: If None, no source features will be activated. If a 1D
            array, the source features in the feature vector will be set to
            match the numbers in the source array.
        :param update_context: If True, the context vector will update after
            the feature vector has been updated. If False, the feature vector
            will update but the context vector will not change.
        :param update_weights: If True, the model's weight matrices will update
            to strengthen the association between the presented item and the
            state of context at the time of presentation. If False, no learning
            will occur after item presentation.
        """
        ##########
        #
        # Activate item's features
        #
        ##########

        paired_pres = np.logical_not(np.isscalar(item_idx))

        # Activate the presented item itself
        self.f.fill(0)
        if item_idx is not None:
            self.f[item_idx] = 1

        # Activate the source feature(s) of the presented item
        if self.nsources > 0 and source is not None:
            self.f[self.first_source_idx:, 0] = np.atleast_1d(source)

        # copy c_old
        # [bj] put this out of update context
        self.c_old = self.c.copy()

        # Compute c_in
        self.c_in = np.dot(self.M_FC, self.f)

        # Normalize the temporal and source subregions of c_in separately
        norm_t = np.sqrt(np.sum(self.c_in[:self.ntemporal] ** 2))
        if norm_t != 0:
            self.c_in[:self.ntemporal] /= norm_t
        if self.nsources > 0:
            norm_s = np.sqrt(np.sum(self.c_in[self.ntemporal:] ** 2))
            if norm_s != 0:
                self.c_in[self.ntemporal:] /= norm_s

        ##########
        #
        # Update context
        #
        ##########

        if update_context:

            # Set beta separately for temporal and source subregions
            beta_vec = np.empty_like(self.c)
            beta_vec[:self.ntemporal] = self.beta
            beta_vec[self.ntemporal:] = self.beta_source

            # Calculate rho for the temporal and source subregions
            rho_vec = np.empty_like(self.c)
            c_dot_t = np.dot(self.c[:self.ntemporal].T, self.c_in[:self.ntemporal])
            rho_vec[:self.ntemporal] = math.sqrt(1 + self.beta ** 2 * (c_dot_t ** 2 - 1)) - self.beta * c_dot_t
            c_dot_s = np.dot(self.c[self.ntemporal:].T, self.c_in[self.ntemporal:])
            rho_vec[self.ntemporal:] = math.sqrt(1 + self.beta_source ** 2 * (c_dot_s ** 2 - 1)) - self.beta_source * c_dot_s

            # Update context
            self.c = (rho_vec * self.c_old) + (beta_vec * self.c_in)

        ##########
        #
        # Update weight matrices
        #
        ##########

        if update_weights:
            # self.M_FC += self.L_FC * np.dot(self.c_old, self.f.T)
            # [bj] to minimize computation load
            # if self.task == 'FR':
            #     if self.phase == 'encoding':  # Only apply primacy scaling during encoding
            #         self.M_CF += self.L_CF * self.prim_vec[self.serial_position] * np.dot(self.f, self.c_old.T)
            #     else:
            #         self.M_CF += self.L_CF * np.dot(self.f, self.c_old.T)
            if self.phase == 'encoding': # [bj] Only apply elevated-attention scaling during encoding
                self.M_FC[:self.nitems_unique,:self.nitems_unique] \
                    += self.L_FC[:self.nitems_unique,:self.nitems_unique] \
                       * np.dot(self.c_old[:self.nitems_unique], self.f[:self.nitems_unique].T) \
                       * np.mean(self.att_vec[self.all_nos_unique[item_idx]-1])
                self.M_CF[:self.nitems_unique,:self.nitems_unique] \
                    += self.L_CF[:self.nitems_unique,:self.nitems_unique] \
                       * np.dot(self.f[:self.nitems_unique], self.c_old[:self.nitems_unique].T) \
                       * np.mean(self.att_vec[self.all_nos_unique[item_idx]-1])
            else:
                self.M_FC[:self.nitems_unique,:self.nitems_unique] \
                    += self.L_FC[:self.nitems_unique,:self.nitems_unique] \
                       * np.dot(self.c_old[:self.nitems_unique], self.f[:self.nitems_unique].T)
                self.M_CF[:self.nitems_unique,:self.nitems_unique] \
                    += self.L_CF[:self.nitems_unique,:self.nitems_unique] \
                       * np.dot(self.f[:self.nitems_unique], self.c_old[:self.nitems_unique].T)
            if paired_pres: # [bj] pair association
                pair_ass = self.params['d_ass'] * np.dot(self.f, self.f.T)
                np.fill_diagonal(pair_ass, 0)
                self.M_FC += self.L_FC * pair_ass
                self.M_CF += self.L_CF * self.prim_vec[self.serial_position] * pair_ass

    def simulate_recall(self, time_limit=60000, max_recalls=np.inf):
        """
        Simulate a recall period, starting from the current state of context.
        [Unchanged from CMR2]

        :param time_limit: The simulated duration of the recall period (in ms).
            Determines how many cycles of the leaky accumulator will run before
            the recall period ends. (DEFAULT=60000)
        :param max_recalls: The maximum number of retrievals (not overt recalls)
            that the model is permitted to make. If this limit is reached, the
            recall period will end early. Use this setting to prevent the model
            from eating up runtime if its parameter set causes it to make
            hundreds of recalls per trial. (DEFAULT=np.inf)
        """
        cycles_elapsed = 0
        nrecalls = 0
        max_cycles = time_limit // self.params['dt']

        while cycles_elapsed < max_cycles and nrecalls < max_recalls:
            # Use context to cue items
            f_in = np.dot(self.M_CF, self.c)[:self.nitems_unique].flatten()

            # Identify set of items with the highest activation
            top_items = np.argsort(f_in)[self.nitems_unique-self.nitems_in_race:]
            top_activation = f_in[top_items]
            top_activation[top_activation < 0] = 0

            # Run accumulator until an item is retrieved
            winner_idx, ncycles = self.leaky_accumulator(top_activation, self.ret_thresh[top_items], int(max_cycles - cycles_elapsed))
            # Update elapsed time
            cycles_elapsed += ncycles
            nrecalls += 1

            # Perform the following steps only if an item was retrieved
            if winner_idx != -1:

                # Identify the feature index of the retrieved item
                item = top_items[winner_idx]

                # Decay retrieval thresholds, then set the retrieved item's threshold to maximum
                self.ret_thresh = 1 + self.params['alpha'] * (self.ret_thresh - 1)
                self.ret_thresh[item] = 1 + self.params['omega']

                # Present retrieved item to the model, with no source information
                if self.learn_while_retrieving:
                    self.present_item(item, source=None, update_context=True, update_weights=True)
                else:
                    self.present_item(item, source=None, update_context=True, update_weights=False)

                # Filter intrusions using temporal context comparison, and log item if overtly recalled
                c_similarity = np.dot(self.c_old[:self.ntemporal].T, self.c_in[:self.ntemporal])
                if c_similarity >= self.params['c_thresh']:
                    rec_itemno = self.all_nos_unique[item] # [bj]
                    self.rec_items[-1].append(rec_itemno)
                    self.rec_times[-1].append(cycles_elapsed * self.params['dt'])

    def simulate_recog(self, cue_idx):
        """
        Simulate a recognition. [Newly added by Beige]

        :param cue_idx: The index of the provided cue in the feature vector.
        """
        # Present cue and update the context
        paired_cue = np.logical_not(np.isscalar(cue_idx))
        if self.mode == "Final":
            self.present_item(cue_idx, source=None, update_context=True, update_weights=False)
        if self.mode == "Continuous":
            self.present_item(cue_idx, source=None, update_context=False, update_weights=False)
        if self.mode == "Hockley":
            if paired_cue: # pair cue
                # print("before item1 \n", self.c)
                self.present_item(cue_idx[0], source=None, update_context=True, update_weights=False)
                self.present_item(cue_idx[1], source=None, update_context=True, update_weights=False)
                # self.present_item(cue_idx, source=None, update_context=True, update_weights=False)
            else:
                self.present_item(cue_idx, source=None, update_context=True, update_weights=False)

        # Recognize or not using context similarity
        # c_similarity, rt = self.diffusion(self.c_old[:self.nitems_unique], self.c_in[:self.nitems_unique], max_time=self.params['rec_time_limit'])
        # print("c_old item1 \n", self.c_old)
        # print("c_inp item2 \n", self.c_in)
        c_similarity = np.dot(self.c_old[:self.nitems_unique].T, self.c_in[:self.nitems_unique]) # !! similarity should not include distractors
        rt = self.params['a'] * np.exp(-1 * self.params['b'] * np.abs(c_similarity - self.params['c_thresh'])) # !!
        self.recog_similarity.append(c_similarity.item())
        self.rec_times.append(rt.item())

        if paired_cue:
            thresh = self.params['c_thresh_ass']
        else:
            thresh = self.c_vec[self.all_nos_unique[cue_idx] - 1]
        # thresh = self.params['c_thresh'] if not paired_cue else self.params['c_thresh_ass']
        if c_similarity >= thresh:
            self.rec_items.append(1)  # YES
        else:
            self.rec_items.append(0)  # NO

    def simulate_cr(self, cue_idx, time_limit=5000):
        """
        Simulate a cued recall. [Newly added by Beige]

        :param cue_idx: The index of the provided cue in the feature vector.
        :param time_limit: The simulated duration of the recall period (in ms).
            Determines how many cycles of the leaky accumulator will run before
            the recall period ends. (DEFAULT=5000)
        """
        cycles_elapsed = 0
        max_cycles = time_limit // self.params['dt']

        # present cue and update the context
        self.present_item(cue_idx, source=None, update_context=True, update_weights=False)
        self.ret_thresh[cue_idx] = np.inf # can't recall the cue!
        # print("c_in\n:", self.c_in)
        # print("after cue\n:", self.c)

        # Use context to cue items
        f_in = np.dot(self.M_CF, self.c)[:self.nitems_unique].flatten()
        self.f_in = f_in # for test
        self.f_in_acc.append(f_in) # for test
        # print("recall", cue_idx, "f_in:\n", f_in)

        # Identify set of items with the highest activation
        # jbg: argsort returns the original index of the sorted order
        top_items = np.argsort(f_in)[self.nitems_unique - self.nitems_in_race:]
        top_activation = f_in[top_items]
        top_activation[top_activation < 0] = 0

        # print("recall", cue_idx,"threshold:\n",self.ret_thresh)
        # Run accumulator until an item is retrieved
        # jbg: winnder_idx is the index with in top_activation
        winner_idx, ncycles = self.leaky_accumulator(top_activation, self.ret_thresh[top_items],
                                                     int(max_cycles))
        cycles_elapsed += ncycles

        # Perform the following steps only if an item was retrieved
        if winner_idx != -1:

            # Identify the feature index of the retrieved item
            item = top_items[winner_idx]

            # Decay retrieval thresholds, then set the retrieved item's threshold to maximum
            self.ret_thresh = 1 + self.params['alpha'] * (self.ret_thresh - 1)
            self.ret_thresh[item] = 1 + self.params['omega']
            self.ret_thresh[cue_idx] = 1 # back to norm

            # Present retrieved item to the model, with no source information
            if self.learn_while_retrieving:
                self.present_item(item, source=None, update_context=True, update_weights=True)
            else:
                self.present_item(item, source=None, update_context=True, update_weights=False)

            # Filter intrusions using temporal context comparison, and log item if overtly recalled
            c_similarity = np.dot(self.c_old[:self.ntemporal].T, self.c_in[:self.ntemporal])
            if c_similarity >= self.params['c_thresh']:
                # print("recall", cue_idx, "recall item index:\n", item)
                rec_itemno = self.all_nos_unique[item] #[bj]
                self.rec_items.append(rec_itemno)
                self.rec_times.append(cycles_elapsed * self.params['dt'])
            else:
                self.rec_items.append(-2) # reject
                self.rec_times.append(-2)

        else:
            self.rec_items.append(-1) # fail
            self.rec_times.append(-1) 

    def diffusion(self, c1, c2,max_time=5000):
        """
        An experimental mechanism to calculate RT. Not used for now. [Newly added by Beige]
        """
        if len(c1) != len(c2):
            print('err')
        len_c = len(c1)

        dt = self.params['dt']
        dot_order = np.random.permutation(len_c)
        total_time = 0
        c_similarity = 0

        for i in dot_order:
            if total_time > max_time:
                total_time = max_time
                break

            c_similarity += c1[i] * c2[i]
            total_time += dt

            if c_similarity >= self.params['c_thresh']:
                break

        return c_similarity, total_time


    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    @cython.cdivision(True)  # Skip checks for division by zero
    def leaky_accumulator(self, float [:] in_act, float [:] x_thresholds, Py_ssize_t max_cycles):
        """
        Simulates the item retrieval process using a leaky accumulator. Loops
        until an item is retrieved or the recall period ends. [Unchanged from CMR2]

        :param in_act: A 1D array containing the incoming activation values
            for all items in the competition.
        :param x_thresholds: A 1D array containing the activation thresholds
            required to retrieve each item in the competition.
        :param max_cycles: The maximum number of cycles the accumulator can run
            before the recall period ends.

        :returns: The index of the retrieved item (or -1 if no item was
            retrieved) and the number of cycles that elapsed before retrieval.
        """
        # Set up indexes
        cdef Py_ssize_t i, j, cycle = 0
        cdef Py_ssize_t nitems_in_race = in_act.shape[0]

        # Set up time constants
        cdef float dt_tau = self.params['dt_tau']
        cdef float sq_dt_tau = self.params['sq_dt_tau']

        # Pre-scale decay rate (kappa) based on dt
        cdef float kappa = self.params['kappa']
        kappa *= dt_tau
        # Pre-scale inhibition (lambda) based on dt
        cdef float lamb = self.params['lamb']
        lamb *= dt_tau
        # Take sqrt(eta) and pre-scale it based on sqrt(dt_tau)
        # Note that we do this because (for cythonization purposes) we multiply the noise
        # vector by sqrt(eta), rather than directly setting the SD to eta
        cdef float eta = self.params['eta'] ** .5
        eta *= sq_dt_tau
        # Pre-scale incoming activation based on dt
        np_in_act_scaled = np.empty(nitems_in_race, dtype=np.float32)
        cdef float [:] in_act_scaled = np_in_act_scaled
        for i in range(nitems_in_race):
            in_act_scaled[i] = in_act[i] * dt_tau

        # Set up activation variables
        np_x = np.zeros(nitems_in_race, dtype=np.float32)
        cdef float [:] x = np_x
        cdef float act
        cdef float sum_x
        cdef float delta_x
        cdef double [:] noise_vec

        # Set up winner variables
        cdef int has_retrieved_item = 0
        cdef int nwinners = 0
        np_retrieved = np.zeros(nitems_in_race, dtype=np.int32)
        cdef int [:] retrieved = np_retrieved
        cdef int [:] winner_vec
        cdef int winner
        cdef (int, int) winner_and_cycle

        # Loop accumulator until retrieving an item or running out of time
        while cycle < max_cycles and not has_retrieved_item:

            # Compute sum of activations for lateral inhibition
            sum_x = 0
            i = 0
            while i < nitems_in_race:
                sum_x += x[i]
                i += 1

            # Update activation and check whether any items were retrieved
            noise_vec = cython_randn(nitems_in_race)
            i = 0
            while i < nitems_in_race:
                # Note that kappa, lambda, eta, and in_act have all been pre-scaled above based on dt
                x[i] += in_act_scaled[i] + (eta * noise_vec[i]) - (kappa * x[i]) - (lamb * (sum_x - x[i]))
                x[i] = max(x[i], 0)
                if x[i] >= x_thresholds[i]:
                    has_retrieved_item = 1
                    nwinners += 1
                    retrieved[i] = 1
                    winner = i
                i += 1

            cycle += 1

        # If no items were retrieved, set winner to -1
        if nwinners == 0:
            winner = -1
        # If multiple items crossed the retrieval threshold on the same cycle, choose one randomly
        elif nwinners > 1:
            winner_vec = np.zeros(nwinners, dtype=np.int32)
            i = 0
            j = 0
            while i < nitems_in_race:
                if retrieved[i] == 1:
                    winner_vec[j] = i
                    j += 1
                i += 1
            srand(time.time())
            rand_idx = rand() % nwinners  # see http://www.delorie.com/djgpp/doc/libc/libc_637.html
            winner = winner_vec[rand_idx]
        # If only one item crossed the retrieval threshold, we already set it as the winner above

        # Return winning item's index within in_act, as well as the number of cycles elapsed
        winner_and_cycle = (winner, cycle)
        return winner_and_cycle


##########
#
# Code to load data and run model
#
##########

def make_params(source_coding=False):
    """
    Returns a dictionary containing all parameters that need to be defined in
    order for CMR2 to run. Can be used as a template for the "params" input
    required by CMR2, run_cmr2_single_sess(), and run_cmr2_multi_sess().
    For notes on each parameter, see in-line comments. [Modified by Beige]

    :param source_coding: If True, parameter dictionary will contain the
        parameters required for the source coding version of the model. If
        False, the dictionary will only condain parameters required for the
        base version of the model.

    :returns: A dictionary containing all of the parameters you need to define
        to run CMR2.
    """
    param_dict = {
        # Beta parameters
        'beta_enc': None,  # Beta encoding
        'beta_rec': None,  # Beta recall
        'beta_rec_new': None, # [beige] Beta recall for new items
        'beta_rec_post': None,  # Beta post-recall
        'beta_distract': None,  # Beta for distractor task

        # Primacy and semantic scaling
        'phi_s': None,
        'phi_d': None,
        's_cf': None,  # Semantic scaling in context-to-feature associations
        's_fc': 0,  # Semantic scaling in feature-to-context associations (Defaults to 0)

        # Recall parameters
        'kappa': None,
        'eta': None,
        'omega': None,
        'alpha': None,
        'c_thresh': None,
        'c_thresh_ass': None,
        'd_ass': None,
        'lamb': None,

        # Timing & recall settings
        'rec_time_limit': 60000.,  # Duration of recall period (in ms) (Defaults to 60000)
        'dt': 10,  # Number of milliseconds to simulate in each loop of the accumulator (Defaults to 10)
        'nitems_in_accumulator': 50,  # Number of items in accumulator (Defaults to 50)
        'max_recalls': 50,  # Maximum recalls allowed per trial (Defaults to 50)
        'learn_while_retrieving': False,  # Whether associations should be learned during recall (Defaults to False)

        # [bj] Parameters for exponential RT in recognition
        'a': None,
        'b': None,

        # [bj] Elevated-attention parameters for WFE
        'm': None,
        'n': None,

        # [bj] Criterion-shift parameters for WFE
        'c1':None,
    }

    # If not using source coding, set up 2 associative scaling parameters (gamma)
    if not source_coding:
        param_dict['gamma_fc'] = None  # Gamma FC
        param_dict['gamma_cf'] = None  # Gamma CF

    # If using source coding, add an extra beta parameter and set up 8 associative scaling parameters
    else:
        param_dict['beta_source'] = None  # Beta source

        param_dict['L_FC_tftc'] = None  # Scale of items reinstating past temporal contexts (Recommend setting to gamma FC)
        param_dict['L_FC_sftc'] = 0  # Scale of sources reinstating past temporal contexts (Defaults to 0)
        param_dict['L_FC_tfsc'] = None  # Scale of items reinstating past source contexts (Recommend setting to gamma FC)
        param_dict['L_FC_sfsc'] = 0  # Scale of sources reinstating past source contexts (Defaults to 0)

        param_dict['L_CF_tctf'] = None  # Scale of temporal context cueing past items (Recommend setting to gamma CF)
        param_dict['L_CF_sctf'] = None  # Scale of source context cueing past items (Recommend setting to gamma CF or fitting as gamma source)
        param_dict['L_CF_tcsf'] = 0  # Scale of temporal context cueing past sources (Defaults to 0, since model does not recall sources)
        param_dict['L_CF_scsf'] = 0  # Scale of source context cueing past sources (Defaults to 0, since model does not recall sources)

    return param_dict

def make_default_params():
    """
    Returns a dictionary containing all parameters that need to be defined in
    order for CMR_IA to run, with default value. [Newly added by Beige]

    :returns: A dictionary containing all of the parameters you need to define
        to run CMRIA, with default value.
    """
    param_dict = make_params()
    param_dict.update(
        beta_enc = 0.5,
        beta_rec = 0.5,
        beta_rec_new = 0.5,
        beta_rec_post = 0.5,
        phi_s = 2,
        phi_d = 0.5,
        s_cf = 0,
        s_fc = 0,
        kappa = 0.5,
        eta = 0.5,
        omega = 8,
        alpha = 4,
        c_thresh = 0.5,
        c_thresh_ass = 0.5,
        d_ass = 1,
        lamb = 0.5,
        gamma_fc = 0.5,
        gamma_cf = 0.5,
        a = 2800,
        b = 20,
        m = 0,
        n = 1,
        c1 = 0,
    )

    return param_dict


def load_pres(path):
    """
    Loads matrix of presented items from a .txt file, a .json behavioral data,
    file, or a .mat behavioral data file. Uses numpy's loadtxt function, json's
    load function, or scipy's loadmat function, respectively. [Unchanged from CMR2]

    :param path: The path to a .txt, .json, or .mat file containing a matrix
        where item (i, j) is the jth word presented on trial i.

    :returns: A 2D array of presented items.
    """
    if os.path.splitext(path) == '.txt':
        data = np.loadtxt(path)
    elif os.path.splitext(path) == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
            data = data['pres_nos'] if 'pres_nos' in data else data['pres_itemnos']
    elif os.path.splitext(path) == '.mat':
        data = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)['data'].pres_itemnos
    else:
        raise ValueError('Can only load presented items from .txt, .json, and .mat formats.')
    return np.atleast_2d(data)


def split_data(pres_mat, identifiers, source_mat=None):
    """
    If data from multiple subjects or sessions are in one matrix, separate out
    the data into separate presentation and source matrices for each unique
    identifier. [Unchanged from CMR2]

    :param pres_mat: A 2D array of presented items from multiple consolidated
        subjects or sessions.
    :param identifiers: A 1D array with length equal to the number of rows in
        pres_mat, where entry i identifies the subject/session/etc. to which
        row i of the presentation matrix belongs.
    :param source_mat: (Optional) A trials x serial positions x nsources array of
        source information for each presented item in pres_mat.

    :returns: A list of presented item matrices (one matrix per unique
        identifier), an array of the unique identifiers, and a list of source
        information matrices (one matrix per subject, None if no source_mat provided).
    """
    # Make sure input matrices are numpy arrays
    pres_mat = np.array(pres_mat)
    if source_mat is not None:
        source_mat = np.atleast_3d(source_mat)

    # Get list of unique IDs
    unique_ids = np.unique(identifiers)

    # Split data up by each unique identifier
    data = []
    sources = None if source_mat is None else []
    for i in unique_ids:
        mask = identifiers == i
        data.append(pres_mat[mask, :])
        if source_mat is not None:
            sources.append(source_mat[mask, :, :])

    return data, unique_ids, sources


def run_cmr2_single_sess(params, pres_mat, sem_mat, source_mat=None, mode='IFR'):
    """
    Simulates a single session of free recall using the specified parameter set.
    [Unchanged from CMR2]

    :param params: A dictionary of model parameters and settings to use for the
        simulation. Call the CMR2_pack_cyth.make_params() function to get a
        dictionary template you can fill in with your desired values.
    :param pres_mat: A 2D array specifying the ID numbers of the words that will be
        presented to the model on each trial. Row i, column j should hold the ID number
        of the jth word to be presented on the ith trial. ID numbers are assumed to
        range from 1 to N, where N is the number of words in the semantic similarity
        matrix (sem_mat). 0s are treated as padding and are ignored, allowing you to
        zero-pad pres_mat if you wish to simulate trials with varying list length.
    :param sem_mat: A 2D array containing the pairwise semantic similarities between all
        words in the word pool. The ordering of words in the similarity matrix must
        match the word ID numbers, such that the scores for word k must be located along
        row k-1 and column k-1.
    :param source_mat: If None, source coding will not be used. Otherwise, source_mat
        should be a 3D array containing source features for each presented word. The
        matrix should have one row for each trial and one column for each serial
        position, with the third dimension having length equal to the number of source
        features you wish to simulate. Cell (i, j, k) should contain the value of the
        kth source feature of the jth item presented on list i. (DEFAULT=None)
    :param mode: A string indicating the type of free recall to simulate. Set to 'IFR'
        for immediate free recall or 'DFR' for delayed recall. (DEFUALT='IFR')

    :returns: Two 2D arrays. The first contains the ID numbers of the items the model
        recalled on each trial. The second contains the response times of each of
        those items relative to the start of the recall period.
    """
    ntrials = pres_mat.shape[0]

    # Simulate all trials of the session using CMR2
    cmr = CMR2(params, pres_mat, sem_mat, source_mat=source_mat, mode=mode)
    for i in range(ntrials):
        cmr.run_fr_trial()

    # Get the model's simulated recall data
    rec_items = cmr.rec_items
    rec_times = cmr.rec_times

    # Identify the max number of recalls made on any trial
    max_recalls = max([len(trial_data) for trial_data in rec_times])

    # Zero-pad response data into an ntrials x max_recalls matrix
    rec_mat = np.zeros((ntrials, max_recalls), dtype=int)
    time_mat = np.zeros((ntrials, max_recalls))
    for i, trial_data in enumerate(rec_items):
        trial_nrec = len(trial_data)
        if trial_nrec > 0:
            rec_mat[i, :trial_nrec] = rec_items[i]
            time_mat[i, :trial_nrec] = rec_times[i]

    return rec_mat, time_mat


def run_cmr2_multi_sess(params, pres_mat, identifiers, sem_mat, source_mat=None, mode='IFR'):
    """
    Simulates multiple sessions of free recall using a single set of parameters.
    [Unchanged from CMR2]

    :param params: A dictionary of model parameters and settings to use for the
        simulation. Call the CMR2_pack_cyth.make_params() function to get a
        dictionary template you can fill in with your desired values.
    :param pres_mat: A 2D array specifying the ID numbers of the words that will be
        presented to the model on each trial. Row i, column j should hold the ID number
        of the jth word to be presented on the ith trial. ID numbers are assumed to
        range from 1 to N, where N is the number of words in the semantic similarity
        matrix (sem_mat). 0s are treated as padding and are ignored, allowing you to
        zero-pad pres_mat if you wish to simulate trials with varying list length.
    :param identifiers: A 1D array of session numbers, subject IDs, or other values
        indicating how the rows/trials in pres_mat and source_mat should be divided up
        into sessions. For example, one could simulate two four-trial sessions by
        setting identifiers to np.array([0, 0, 0, 0, 1, 1, 1, 1]), specifying that the
        latter four trials come from a different session than the first four trials.
    :param sem_mat: A 2D array containing the pairwise semantic similarities between all
        words in the word pool. The ordering of words in the similarity matrix must
        match the word ID numbers, such that the scores for word k must be located along
        row k-1 and column k-1.
    :param source_mat: If None, source coding will not be used. Otherwise, source_mat
        should be a 3D array containing source features for each presented word. The
        matrix should have one row for each trial and one column for each serial
        position, with the third dimension having length equal to the number of source
        features you wish to simulate. Cell (i, j, k) should contain the value of the
        kth source feature of the jth item presented on list i. (DEFAULT=None)
    :param mode: A string indicating the type of free recall to simulate. Set to 'IFR'
        for immediate free recall or 'DFR' for delayed recall. (DEFUALT='IFR')

    :returns: Two 2D arrays. The first contains the ID numbers of the items the model
        recalled on each trial. The second contains the response times of each of
        those items relative to the start of the recall period.
    """
    now_test = time.time()

    # Split data based on identifiers provided
    pres, unique_ids, sources = split_data(pres_mat, identifiers, source_mat=source_mat)

    # Run CMR2 for each subject/session
    rec_items = []
    rec_times = []
    for i, sess_pres in enumerate(pres):
        sess_sources = None if sources is None else sources[i]
        out_tuple = run_cmr2_single_sess(params, sess_pres, sem_mat, source_mat=sess_sources, mode=mode)
        rec_items.append(out_tuple[0])
        rec_times.append(out_tuple[1])
    # Identify the maximum number of recalls made in any session
    max_recalls = max([sess_data.shape[1] for sess_data in rec_items])

    # Zero-pad response data into an total_trials x max_recalls matrix where rows align with those in the original data_mat
    total_trials = len(identifiers)
    rec_mat = np.zeros((total_trials, max_recalls), dtype=int)
    time_mat = np.zeros((total_trials, max_recalls))
    for i, uid in enumerate(unique_ids):
        sess_max_recalls = rec_items[i].shape[1]
        if sess_max_recalls > 0:
            rec_mat[identifiers == uid, :sess_max_recalls] = rec_items[i]
            time_mat[identifiers == uid, :sess_max_recalls] = rec_times[i]

    print("CMR Time: " + str(time.time() - now_test))

    return rec_mat, time_mat

def run_peers_recog_multi_sess(params, data_dict, sem_mat, source_mat=None, task='Recog', mode='Final'):
    """
    Simulates multiple sessions of peers recognition using a single set of parameters.
    [Newly added by Beige]

    :param params: A dictionary of model parameters and settings to use for the
        simulation. Call the CMR_IA.make_params() function to get a
        dictionary template you can fill in with your desired values.
    :param data_dict: a dictionary contains pres_mat, rec_mat, ffr_mat, cue_mat.
    :param sem_mat: A 2D array containing the pairwise semantic similarities between all
        words in the word pool. The ordering of words in the similarity matrix must
        match the word ID numbers, such that the scores for word k must be located along
        row k-1 and column k-1.
    :param source_mat: If None, source coding will not be used. Otherwise, source_mat
        should be a 3D array containing source features for each presented word. The
        matrix should have one row for each trial and one column for each serial
        position, with the third dimension having length equal to the number of source
        features you wish to simulate. Cell (i, j, k) should contain the value of the
        kth source feature of the jth item presented on list i. (DEFAULT=None)
    :param task: Recognition. (DEFAULT='Recog')
    :param mode: A string indicating the type of recognition to simulate. Set to 'Final'
        for PEERS recognition. (DEFUALT='IFR')

    :returns: Three dictionaries. The first contains the recognition response for each item.
        The second contains the reaction time for each response. The third contains the
        simulated context similarity for each item.
    """
    now_test = time.time()

    recog_dict = {}
    rt_dict = {}
    csim_dict = {}

    for sess in data_dict.keys():
        # extarct the session data
        pres_mat, rec_mat, ffr_mat, cue_mat = data_dict[sess]

        # run CMR for each session
        cmr = CMR2(params, pres_mat, sem_mat, source_mat=None,
                   rec_mat=rec_mat, ffr_mat=ffr_mat, cue_mat=cue_mat, task='Recog', mode=mode)
        cmr.run_peers_recog_single_sess()

        recog_dict[sess] = cmr.rec_items
        rt_dict[sess] = cmr.rec_times
        csim_dict[sess] = cmr.recog_similarity

    print("CMR Time: " + str(time.time() - now_test))

    return recog_dict, rt_dict, csim_dict

def run_norm_recog_multi_sess(params, df_study, df_test, sem_mat, source_mat=None, task='Recog', mode='Final'):
    """
    Simulates multiple sessions of normal recognition using a single set of parameters.
    [Newly added by Beige]

    :param params: A dictionary of model parameters and settings to use for the
        simulation. Call the CMR_IA.make_params() function to get a
        dictionary template you can fill in with your desired values.
    :param df_study: a dataframe containing study list.
    :param df_test: a dataframe containing test list.
    :param sem_mat: A 2D array containing the pairwise semantic similarities between all
        words in the word pool. The ordering of words in the similarity matrix must
        match the word ID numbers, such that the scores for word k must be located along
        row k-1 and column k-1.
    :param source_mat: If None, source coding will not be used. Otherwise, source_mat
        should be a 3D array containing source features for each presented word. The
        matrix should have one row for each trial and one column for each serial
        position, with the third dimension having length equal to the number of source
        features you wish to simulate. Cell (i, j, k) should contain the value of the
        kth source feature of the jth item presented on list i. (DEFAULT=None)
    :param task: Recognition. (DEFAULT='Recog')
    :param mode: A string indicating the type of recognition to simulate. Set to 'Final'
        for PEERS recognition. (DEFUALT='IFR')

    :returns: Three dictionaries. The first contains the recognition response for each item.
        The second contains the reaction time for each response. The third contains the
        simulated context similarity for each item.
    """
    now_test = time.time()

    sessions = np.unique(df_study.session)
    df_thin = df_test[['session','itemno']]
    df_thin = df_thin.assign(s_resp=np.nan, s_rt=np.nan, csim=np.nan)

    for sess in sessions:
        # extarct the session data
        pres_mat = df_study.loc[df_study.session==sess,'itemno'].to_numpy()
        pres_mat = np.reshape(pres_mat,(1, len(pres_mat)))
        cue_mat = df_thin.loc[df_thin.session==sess,'itemno'].to_numpy()

        # run CMR for each session
        cmr = CMR2(params, pres_mat, sem_mat, source_mat=None,
                   rec_mat=None, ffr_mat=None, cue_mat=cue_mat, task='Recog', mode=mode)
        cmr.run_peers_recog_single_sess()

        recs = cmr.rec_items
        rts = cmr.rec_times
        csims = cmr.recog_similarity
        result = np.column_stack((recs,rts,csims))

        df_thin.loc[df_thin.session==sess, ['s_resp','s_rt','csim']] = result

    print("CMR Time: " + str(time.time() - now_test))

    return df_thin

def run_conti_recog_multi_sess(params, df, sem_mat, source_mat=None, task='Recog', mode='Final'):
    """
    Simulates multiple sessions of continuous recognition using a single set of parameters.
    [Newly added by Beige]

    :param params: A dictionary of model parameters and settings to use for the
        simulation. Call the CMR_IA.make_params() function to get a
        dictionary template you can fill in with your desired values.
    :param df: A dataframe of experiment design. This dataframe should have these
        3 columns: "session", "position", "itemno". Each row corresponds to a trial.
        "session" specifies the session, "position" specifies the sequence of items
        within a session, and "itemno" specifies which item is presented. "itemno" should
        correspond to sem_mat.
    :param sem_mat: A 2D array containing the pairwise semantic similarities between all
        words in the word pool. The ordering of words in the similarity matrix must
        match the word ID numbers, such that the scores for word k must be located along
        row k-1 and column k-1.
    :param source_mat: If None, source coding will not be used. Otherwise, source_mat
        should be a 3D array containing source features for each presented word. The
        matrix should have one row for each trial and one column for each serial
        position, with the third dimension having length equal to the number of source
        features you wish to simulate. Cell (i, j, k) should contain the value of the
        kth source feature of the jth item presented on list i. (DEFAULT=None)
    :param task: Recognition. (DEFAULT='Recog')
    :param mode: A string indicating the type of free recall to simulate. Set to 'Continuous'
        for continuous recognition. (DEFUALT='Continuous')

    :returns: A dataframe with 6 columns: "session", "position", "itemno", "s_resp", "s_rt", "csim".
        The first three columns are identical to those in input df. The last three columns
        indicates the simulated response, reaction time and context similarity respectively.
    """
    now_test = time.time()

    sessions = np.unique(df.session)
    df_thin = df[['session','position','itemno']]
    df_thin = df_thin.assign(s_resp=np.nan, s_rt=np.nan, csim=np.nan)

    for sess in sessions:
        # extarct the session data
        cue_mat = df_thin.loc[df.session==sess,'itemno'].to_numpy()
        pres_mat = np.reshape(cue_mat,(len(cue_mat),1))

        # run CMR for each session
        cmr = CMR2(params, pres_mat, sem_mat, source_mat=None,
                   rec_mat=None, ffr_mat=None, cue_mat=cue_mat, task=task, mode=mode)
        cmr.run_conti_recog_single_sess()

        recs = cmr.rec_items
        rts = cmr.rec_times
        csims = cmr.recog_similarity
        result = np.column_stack((recs,rts,csims))

        df_thin.loc[df_thin.session==sess, ['s_resp','s_rt','csim']] = result

    print("CMR Time: " + str(time.time() - now_test))

    return df_thin

def run_hockley_recog_multi_sess(params, df, sem_mat, source_mat=None, task='Recog', mode='Hockley'):
    """
    Simulates multiple sessions of continuous recognition using a single set of parameters.
    [Newly added by Beige]

    :param params: A dictionary of model parameters and settings to use for the
        simulation. Call the CMR_IA.make_params() function to get a
        dictionary template you can fill in with your desired values.
    :param df: A dataframe of experiment design. This dataframe should have these
        3 columns: "session", "position", "itemno". Each row corresponds to a trial.
        "session" specifies the session, "position" specifies the sequence of items
        within a session, and "itemno" specifies which item is presented. "itemno" should
        correspond to sem_mat.
    :param sem_mat: A 2D array containing the pairwise semantic similarities between all
        words in the word pool. The ordering of words in the similarity matrix must
        match the word ID numbers, such that the scores for word k must be located along
        row k-1 and column k-1.
    :param source_mat: If None, source coding will not be used. Otherwise, source_mat
        should be a 3D array containing source features for each presented word. The
        matrix should have one row for each trial and one column for each serial
        position, with the third dimension having length equal to the number of source
        features you wish to simulate. Cell (i, j, k) should contain the value of the
        kth source feature of the jth item presented on list i. (DEFAULT=None)
    :param task: Recognition. (DEFAULT='Recog')
    :param mode: A string indicating the type of free recall to simulate. Set to 'Continuous'
        for continuous recognition. (DEFUALT='Continuous')

    :returns: A dataframe with 6 columns: "session", "position", "itemno", "s_resp", "s_rt", "csim".
        The first three columns are identical to those in input df. The last three columns
        indicates the simulated response, reaction time and context similarity respectively.
    """
    now_test = time.time()

    sessions = np.unique(df.session)
    df_thin = df[['session','position','study_itemno1','study_itemno2','test_itemno1','test_itemno2']]
    df_thin = df_thin.assign(s_resp=np.nan, s_rt=np.nan, csim=np.nan)

    for sess in sessions:
        # extarct the session data
        pres_mat = df_thin.loc[df_thin.session == sess, ['study_itemno1', 'study_itemno2']].to_numpy()
        pres_mat = np.reshape(pres_mat, (len(pres_mat), 1, 2))
        cue_mat = df_thin.loc[df_thin.session == sess, ['test_itemno1', 'test_itemno2']].to_numpy()

        # run CMR for each session
        cmr = CMR2(params, pres_mat, sem_mat, source_mat=None,
                   rec_mat=None, ffr_mat=None, cue_mat=cue_mat, task=task, mode=mode)
        cmr.run_hockley_recog_single_sess()

        recs = cmr.rec_items
        rts = cmr.rec_times
        csims = cmr.recog_similarity
        result = np.column_stack((recs,rts,csims))

        df_thin.loc[df_thin.session==sess, ['s_resp','s_rt','csim']] = result

    print("CMR Time: " + str(time.time() - now_test))

    return df_thin

def run_norm_cr_multi_sess(params, df_study, df_test, sem_mat, source_mat=None, task='CR', mode='Final'):
    """
    Simulates multiple sessions of continuous recognition using a single set of parameters.
    [Newly added by Beige]

    :param params: A dictionary of model parameters and settings to use for the
        simulation. Call the CMR_IA.make_params() function to get a
        dictionary template you can fill in with your desired values.
    :param df: A dataframe of experiment design. This dataframe should have these
        3 columns: "session", "position", "itemno". Each row corresponds to a trial.
        "session" specifies the session, "position" specifies the sequence of items
        within a session, and "itemno" specifies which item is presented. "itemno" should
        correspond to sem_mat.
    :param sem_mat: A 2D array containing the pairwise semantic similarities between all
        words in the word pool. The ordering of words in the similarity matrix must
        match the word ID numbers, such that the scores for word k must be located along
        row k-1 and column k-1.
    :param source_mat: If None, source coding will not be used. Otherwise, source_mat
        should be a 3D array containing source features for each presented word. The
        matrix should have one row for each trial and one column for each serial
        position, with the third dimension having length equal to the number of source
        features you wish to simulate. Cell (i, j, k) should contain the value of the
        kth source feature of the jth item presented on list i. (DEFAULT=None)
    :param task: Recognition. (DEFAULT='Recog')
    :param mode: A string indicating the type of free recall to simulate. Set to 'Continuous'
        for continuous recognition. (DEFUALT='Continuous')

    :returns: A dataframe with 6 columns: "session", "position", "itemno", "s_resp", "s_rt", "csim".
        The first three columns are identical to those in input df. The last three columns
        indicates the simulated response, reaction time and context similarity respectively.
    """
    now_test = time.time()

    sessions = np.unique(df_study.session)
    df_thin = df_test[['session','test_itemno']]
    df_thin = df_thin.assign(s_resp=np.nan, s_rt=np.nan)
    f_in = []

    for sess in sessions:
        # extarct the session data
        pres_mat = df_study.loc[df_study.session == sess, ['study_itemno1', 'study_itemno2']].to_numpy()
        pres_mat = np.reshape(pres_mat, (1, len(pres_mat), 2))
        cue_mat = df_thin.loc[df_thin.session == sess, 'test_itemno'].to_numpy()

        # run CMR for each session
        cmr = CMR2(params, pres_mat, sem_mat, source_mat=None,
                   rec_mat=None, ffr_mat=None, cue_mat=cue_mat, task=task, mode=mode)
        cmr.run_norm_cr_single_sess()

        recs = cmr.rec_items
        rts = cmr.rec_times
        result = np.column_stack((recs,rts))
        df_thin.loc[df_thin.session==sess, ['s_resp','s_rt']] = result
        # f_in.append(cmr.f_in) # only for testing
        f_in.append(cmr.f_in_acc)

    print("CMR Time: " + str(time.time() - now_test))

    return df_thin, f_in
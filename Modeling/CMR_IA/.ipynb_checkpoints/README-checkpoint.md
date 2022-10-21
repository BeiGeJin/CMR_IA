# FILES

Setup files and the core CMR-IA code can be found in the top-level directory. These include the following:

1) CMR_IA.pyx - Contains the Cython implementation of CMR-IA.
2) setup_cmr2.py - A script which can be used to build and install CMR-IA.
3) setup_env.sh - A script which can be used to create a new Python 3 environment able to run CMR-IA.

Code for fitting CMR-IA can be found in the fitting/ subdirectory:

4) fitting/pso_cmr2.py - Contains a Python function for running particle swarm optimization on CMR2. Includes implementations of many different particle swarm variants (see its docstring for reference).
5) fitting/optimization_utils.py - Contains a variety of utility functions used by the optimization algorithms, including the function that evaluates each parameter set by calling CMR2 and comparing its simulated behavior to the target behavior.
6) fitting/noise_maker_pso.py - Contains a function used by the PSO algorithm to generate random numbers used in particle initialization and motion.

Semantic similarity matrices and their associated wordpools are stored as text files in the wordpools/ subdirectory:

7) wordpools/PEERS_wordpool.txt - Contains a list of the 1638 words in the PEERS wordpool. Word ID numbers correspond to the ordering of words in this list, from 1 to 1638.
8) wordpools/PEERS_w2v.txt - Contains a matrix of semantic similarities between all pairs of words in the PEERS wordpool, determined as the cosine similarity between the words' representations in word2vec space. Word order matches that used in PEERS_wordpool.txt.

---

# SETUP

(1A) If you do not already have a Python 3 environment set up, use the provided shell script to set up an environment, build and install CMR-IA and its required packages, and set up a Jupyter kernel for the environment:

~~~
bash setup_env.sh
~~~

(1B) If you already have a Python 3 environment in Anaconda, simply activate it (replace ENV_NAME with the name of your Python 3 environment), make sure CMR2's dependencies are installed, and then run its installation script:

~~~
source activate ENV_NAME
conda install numpy scipy matplotlib mkl cython
python setup_cmr2.py install
~~~

(2) Regardless of which of the two methods you used to install CMR, you should now be able to import it into Python from anywhere using the following line:

~~~
import CMR_IA as cmr
~~~

Once you have imported it, you can use the functions from the package just like you would use any other package in Python. Note that the supplementary files containing optimization algorithms will not be installed as part of the package. If you wish to use these scripts, you will need to copy these files, modify them as needed to work with your research project, and then run the files directly (see below).


(3) Any time you change or update CMR_IA, you will need to rerun the following line in order to rebuild the code and update your installation:

~~~
python setup_cmr2.py install 
~~~

(4) If you wish to use the model fitting algorithms, you will also need to setup a couple scripts that will help you test multiple models in parallel. Thesefiles are located in the Modeling/CMR2/pgo_files/ directory.If you do not already have a bin folder in your home directory, create one:

~~~
mkdir ~/bin
~~~

Then copy the two scripts from pgo_files (pgo and runpyfile.sh) into ~/bin.Finally, edit the following line in your copy of runpyfile.sh, replacing ENV_NAME with the name of your Python 3 environment. If your Anaconda folderis named something other than anaconda3, you will also need to edit the filepath to match the name of your Anaconda folder:

~~~
PY_COMMAND="/home1/$USER/anaconda3/envs/base/bin/python"
~~~

You should now be able to call the pgo function in your terminal from anywhere. See the instructions below on how to use pgo in conjunction with the model fitting algorithms to optimize your model fits.

[Beige]: Another Way is to cd to Modeling/CMR2/fitting and type ./pgo to call pgo function directly.

---

# RUNNING CMR_IA

[Most Beige's modifications on previous CMR2 are illustrated in this section.]

Once you have imported CMR_IA, there are basically two approaches you can run simulations. The first is through the provided functions, which automatially initializes a CMR2 object and simulates a single session or multiple sessions using a given parameter set. This is the recommended approach. Another is to initialize a CMR2 object manually and then build your own simulation code around it.

CMR_IA is able to simulate three types of memory tasks: free recall, cued recall, and recognition.

## Free Recall

For free recall, CMR_IA performs the same simulation as CMR2 from Pazdera & Kahana (2022).

A couple helpful tips before we begin:
- For the "params" input to the functions below, you can use the function ``CMR_IA.make_params()`` to create a dictionary containing all of the parameters that must be included in the params input to the functions below. Just take the dictionary template it creates and fill in your desired settings.
- For the "pres_mat" input to the functions below, you can use the ``CMR_IA.load_pres()`` function to load the presented item matrix from a variety of data files, including the behavioral matrix files from ltp studies in .json and .mat format.
- For the "sem_mat" inputs to the functions below, if your simulation uses one of the wordpools provided in the wordpools subdirectory, you should be able to find a text file in there containing the semantic similarity matrix for that wordpool (e.g. PEERS_w2v.txt). Simply load it with np.loadtxt() and input it as your similarity matrix.

There are two provided functions for free recall.

~~~
run_cmr2_single_sess(params, pres_mat, sem_mat, source_mat=None, mode='IFR')
~~~

This function simulates a single session with CMR2 and returns two numpy arrays containing the ID numbers of recalled items and the response times of each of those recalls, respectively. See the function's docstring for details on its inputs, and note that you can choose whether to include source featuresin your model and whether to simulate immediate or delayed free recall.

~~~
run_cmr2_multi_sess(params, pres_mat, identifiers, sem_mat, source_mat=None, mode='IFR')
~~~

This function simulates multiple sessions with CMR2, using the same parameter set for all sessions. Like its single-session counterpart it returns two numpy arrays containing the ID numbers of recalled items and the response times of each of those recalls. See the function's docstring for details on its inputs, and again note that you can choose whether to include source features in your model and whether to simulate immediate or delayed free recall.

## Recognition

CMR_IA is able to simulate two recognition paradigms. The first is the continuous recognition paradigm where subjects memorize and response for each item presented continuously. The second is a final recognition paradigm used in PEERS, where subjects fisrt memorize a word list and then perform the recognition task.

There are two provided functions responsible for two paradigms respectively. These functions support both single session and multiple sessions. 

~~~
run_conti_recog_multi_sess(params, df, sem_mat, source_mat=None, task='Recog', mode='Continuous')
~~~

This function simulates multiple sessions of continuous recognition with CMR_IA with a given set of parameters. The model takes a dataframe specifying sessions, positions, and item ID numvers as input. It returns a dataframe containing each item's recognition response, reaction time, and context similarity. See the function's docstring for details on itsinputs.

~~~
run_peers_recog_multi_sess(params, data_dict, sem_mat, source_mat=None, task='Recog', mode='Final')
~~~

This function simulates multiple sessions of the recongition task in PEERS. Under developed.

## Cued Recall

Provided functions are under developed. Now CMR_IA only supports to simulate cued recall by initializing a CMR2 object manually (see the next chapter).

## Initialize a CMR2 Object

This chapter is for those who want to build its own simulation code rather than using provided functions.

The inputs when creating a CMR2 object are identical to those you would providewhen using run_cmr2_single_sess(). Indeed, that function simply creates a CMR2 object and calls methods of the class in order to simulate each trial and organize the results into recall and response time matrices. If desired, you can work with the CMR2 object directly rather than using one of the wrapper functions provided. You can then directly call the following methods of the CMR2 class:


- ``run_trial()``: Simulates a standard free recall trial, consisting of the following steps:
    1) A pre-trial context shift
    2) A sequence of item presentations
    3) A pre-recall distractor (only if the mode was set to 'DFR') 
    4) A recall period
- ``run_peers_recog_single_sess()``: Simulates a session of PEERS1&3 recognition, consisting of the following steps:
    1) Pre-trial context initialization / between-trial distractor
    2) Item presentation as encoding
    3) Item presentation as actual free recall
    4) Loop step 1-3 for n times if you have n trials
    5) Item presentation as actural final free recall (If have one)
    6) Pre-recognition distractor
    7) Recognition simulation
-  ``run_conti_recog_single_sess()``: Simulates a session of continuous recognition, consisting of the following steps:
     1) Pre-session context initialization / between-trial distractor
     2) Recognition simulation
     3) Item presentation as encoding
     4) Loop step 1-3
-   ``run_cr_trial()``:  Simulates a standard trial of cued recall, consisting of the following steps:
    1) A pre-trial context shift
    2) A sequence of item (word pair) presentations
    3) A recall period
- ``present_item(item_idx, source=None, update_context=True, update_weights=True)``: Presents a single item (or distractor) to the model. This includes options for setting the source features of the item, as well as for choosing whether the context vector and/or weight matrices should be updated after presenting the item. 
- ``simulate_recall(time_limit=60000, max_recalls=np.inf)``: Simulates a recall period with a duration of time_limit miliseconds. If the model makes max_recalls  retrievals, the recall period will terminate early (this can be used to avoid wasting time running models that make hundreds of recalls per trial). 

It is NOT RECOMMENDED to manually run present_item() and simulate_recall() unless you have read the code for run_trial() and understand the fields in the CMR2 object that need to be updated over the course of a trial. Manually running run_trial() and other functions is perfectly fine, however. 

---

# FITTING CMR_IA

In order to fit CMR2 to a set of data, you will need to use some type ofoptimization algorithm to search the paramaeter space and evaluate the goodnessof fit of different parameter sets. Although you can choose to use anyalgorithm you like, included is a particle swarm algorithm that has been used inprevious work. It can be found in the fitting/ subdirectory. In orderto make use of these functions, you will need to make copies of them andcustomize them for your purposes. You can find additional code inoptimization_utils.py to help you design your actual goodness-of-fit test,score your model's recall performance, and more.

Regardless of which algorithm you are using, you can run your optimization jobsin parallel by running the following commands. First, make sure your Python 3environment is active (source activate EV_NAME), then run:

~~~
pgo FILENAME N_JOBS
~~~

Where FILENAME is the optimization algorithm's file path and N_JOBS is the number of parallel jobs you wish to run. Remember you can view your jobs at any time with qstat and can manually kill jobs using qdel. Please cluster responsibly.

In the PSO script, parallel instances will automatically track one another's progress to make sure the next iteration starts once all jobs have finished  evaluating the current iteration. You therefore only need to run pgo once, rather than once per iteration.

IMPORTANT: The particle swarm leaves behind many files tracking intermediate steps of the algorithm. Once the algorithm has finished, remember to delete all tempfiles and keep only the files with the goodness of fit scores and parameter values from each iteration.

NOTE: Particle swarms are designed to test small numbers of parameter sets for hundreds/thousands of iterations. Parameter sets within an iteration can be tested in parallel. Each new iteration cannot start until all parameter sets from the current iteration have finished.

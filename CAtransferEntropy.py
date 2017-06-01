##############################################################################################################
##############################################################################################################
##############################################################################################################
#
#  Functions designed to accompany the notebook showing how to calculate local trasnfer entropy using the
#  tool that Lizer created. These functions were created by myself to optimize applying trasnfer entropy to
#  cellular automata
#
#  **Marc Brittain**
##############################################################################################################
##############################################################################################################
##############################################################################################################


import numpy as np
from jpype import *
import random


def createCA(rule, timesteps=600, n=100):
    """creates ECA and store in a matrix.


    parameters
    ----------

    rule : dictionary
        Dictionary containing the elementary cellular automata rule.
        keys and values must be of dtype string.

    timesteps : integer
        Number of timesteps to evolve CA. Default is 600

    n : integer
        width of the CA space. Default is 100.


    Returns
    -------

    ca : numpy matrix
        The cellular automata matrix with time evolving down.



    """


    ca_space = np.random.randint(0,2,size=n)
    ca = np.zeros((timesteps+1, n),dtype=int)

    ca_new = np.zeros(n,dtype=int)
    ca[0] = ca_space
    for t in range(timesteps):
        ln = np.roll(ca_space, 1, 0).astype(str)
        rn = np.roll(ca_space, -1, 0).astype(str)
        ca_str = ca_space.astype(str)
        for i in range(len(ca_space)):


            temp_rule = ln[i] + ca_str[i] + rn[i]

            update = rule[temp_rule]

            ca_new[i] = int(update)

        ca[t+1] = ca_new

        ca_space = ca_new

    return ca




#############################################################################################################
#############################################################################################################
#############################################################################################################



def teCA(ca, k_history, neighbor):
    """calculates the local transfer entropy for a given Elementary Cellular Automata

    Parameters
    ----------

    ca : numpy matrix
        Input cellular automata matrix

    k_history : Int
        history length for transfer entropy calculation

    neighbor : String
        specifying which neighbor to run the trasnfer entropy on. "L" = left, "R" = right



    Returns
    -------

    localTE : numpy matrix
        matrix containing the local trasnfer entropy values for the cellular automata




    """

    neighb = neighbor.upper()

    timesteps = []
    localTE = np.zeros(ca.shape, dtype=int)

    teCalcClass = JPackage("infodynamics.measures.discrete").TransferEntropyCalculatorDiscrete
    teCalc = teCalcClass(2,k_history)

    for column in range(ca.shape[1]):

        # using left neighbor for y timeseries

        x = ca[:,column]

        if neighb == "L":

            y = np.roll(ca,1,1)[:,column]

        if neighb == "R":

            y = np.roll(ca,-1,1)[:,column]


        y_JArray = JArray(JInt, 1)(y.tolist())
        x_JArray = JArray(JInt, 1)(x.tolist())

        teCalc.initialise()

        teCalc.addObservations(y_JArray, x_JArray)
        localTE[:,column] = np.array(teCalc.computeLocalFromPreviousObservations(y_JArray,x_JArray))


    return localTE

#############################################################################################################
#############################################################################################################
#############################################################################################################


def teCA_null(ca, k_history, numTrials):
    """performs a null test for to verify the transfer entropy for a given Elementary Cellular Automata

    Parameters
    ----------

    ca : numpy matrix
        Input cellular automata matrix

    k_history : Int
        history length for transfer entropy calculation

    numTrials : Int
        number of runs to perform the null test



    Returns
    -------

    localTE_null :  numpy matrix
        matrix containing the null test results of the local transfer entropy values for the cellular automata

    localTE_std : numpy matrix
        matrix containing the standard deviation results of the local transfer entropy for the cellular automata

    localTE_max : numpy matrix
        matrix containing the max transfer entropy values from the randomized time series for the cellular automata

    """


    localTE = np.zeros((ca.shape[0], ca.shape[1],numTrials), dtype=int)
    teCalcClass = JPackage("infodynamics.measures.discrete").TransferEntropyCalculatorDiscrete
    teCalc = teCalcClass(2,k_history)
    for i in range(numTrials):

        for column in range(ca.shape[1]):

            # using left neighbor for y timeseries

            x = ca[:,column].copy()
            y = ca[:,column].copy()

            np.random.shuffle(y)

            y_JArray = JArray(JInt, 1)(y.tolist())
            x_JArray = JArray(JInt, 1)(x.tolist())

            teCalc.initialise()

            teCalc.addObservations(y_JArray, x_JArray)
            localTE[:,column,i] = np.array(teCalc.computeLocalFromPreviousObservations(y_JArray,x_JArray))


    localTE_null = localTE.mean(axis=2,dtype=float)
    localTE_std = np.std(localTE,axis=2,dtype=float)
    localTE_max = np.max(localTE,axis=2)

    return localTE_null, localTE_std, localTE_max

#############################################################################################################
#############################################################################################################
#############################################################################################################


def teCA_Box(ca, k_history, neighbor):
    """calculates the local transfer entropy for a given Elementary Cellular Automata across all neighbors


    Parameters
    ----------

    ca : numpy matrix
        Input cellular automata matrix

    k_history : Int
        history length for transfer entropy calculation

    neighbor : String
        specifying which neighbor to run the trasnfer entropy on. "L" = left, "R" = right



    Returns
    -------

    localTE : numpy 3D matrix
        matrix containing the local trasnfer entropy values for the cellular automata for every neighbor




    """

    # convert neighbor to upper case to prevent any human errors.

    neighb = neighbor.upper()


    # creating the empty trasnfer entropy matrix. Shape is the same as Input
    # except now has a 3rd dimension that is the size of the space. Not time
    localTE = np.zeros((ca.shape[0],ca.shape[1],ca.shape[1]), dtype=int)


    # Initialising the trasnfer entropy calculator for the discrete calculation
    teCalcClass = JPackage("infodynamics.measures.discrete").TransferEntropyCalculatorDiscrete

    # setting the number of possible states, as well as, the k_history value here
    teCalc = teCalcClass(2,k_history)


    # nested for-loop to compute the local trasnfer entropy against every neighbor for a given cell
    for column in range(ca.shape[1]):

        for i in range(ca.shape[1]):

            # using left neighbor for y timeseries

            # by using the left neighbor, we are calculating the TE(Left_Neighbor ---> Current cell)

            x = ca[:,column]

            if neighb == "L":

                y = np.roll(ca,i,1)[:,column]

            if neighb == "R":

                y = np.roll(ca,-i,1)[:,column]


            # converting the numpy arrays to java arrays
            y_JArray = JArray(JInt, 1)(y.tolist())
            x_JArray = JArray(JInt, 1)(x.tolist())

            # initialise the calculator to accept new data
            teCalc.initialise()

            # Adding the java array data. Must be in format: Y, X for transfer
            # entropy calculation listed above

            teCalc.addObservations(y_JArray, x_JArray)
            

            # store the results
            localTE[:,column,i] = np.array(teCalc.computeLocalFromPreviousObservations(y_JArray,x_JArray))


    return localTE

#############################################################################################################
#############################################################################################################
#############################################################################################################



def greatestInfluence(teBox):
    """determines which town was the maxmimum forcer for each individual town and counts how many times the town showed up
    
        
    Parameters
    ----------

    teBox : numpy 3D matrix
        Input transfer entropy values for each every town forcing a single town



    Returns
    -------

    [indicies,towns] : list of numpy arrays
        indicies contains the town numbers, towns contains the count of which town had the greatest influence




    """
    
    
    towns = np.zeros(teBox.shape[0],dtype=int)
    indicies = np.arange(0,226,dtype=int)

    for i in range(teBox.shape[0]):
        index = np.argmax(teBox[i,:,:])
        index = np.unravel_index(index,teBox[i,:,:].shape)

        towns[indicies[i-index[1]]] += 1
        
    return [indicies,towns]



#############################################################################################################
#############################################################################################################
#############################################################################################################

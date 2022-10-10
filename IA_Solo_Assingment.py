# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
random.seed(1111)
np.random.seed(1111)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Loads a data file from a provided file location.
def load_data():    
    #loads the data from a given location as a dictionary
    loaded_data={}
    file_list=['PA1_train1','PA1_test1']
    keys = ['train', 'test']
    for i, file in enumerate(file_list):
        d=pd.read_csv(file+".csv")
        loaded_data[keys[i]]=d
    return loaded_data


# Implements dataset preprocessing, with boolean options to either normalize the data or not, 
# and to either drop the sqrt_living15 column or not.
#
# Note that you will call this function multiple times to generate dataset versions that are
# / aren't normalized, or versions that have / lack sqrt_living15.
def preprocess_data(df, normalize=True, sqft_living15=False):
    # Your code here:
    '''
    input variable data contains a tuple of (train, test and validation)
    '''
    for key in df.keys():

    # 1. Remove ID column and sqft_living15 if True
        if(key == 'train'):
            df[key].drop("id",axis=1,inplace=True)
            df[key].drop("sqft_living15","sqft_lot15","sqft_lot", axis=1,inplace=True)
            norm_zipcode = df[key].get_dummies(df[key]['zipcode'])
            df = pd.concat([df,norm_zipcode],axis = 1)
            df[key].drop("zipcode",axis=1,inplace=True)
        if sqft_living15:
            df[key].drop("sqft_living15","sqft_lot15","sqft_lot", axis=1,inplace=True)

    # 2. Split date into date, month, year
        df[key]['date']=pd.to_datetime(df[key]['date'])
        df[key]['year']=df[key]['date'].dt.year
        df[key]['month']=df[key]['date'].dt.month
        df[key]['day']=df[key]['date'].dt.day

    # (Do not add a dummy feature column now. Add it at the end of the normalization process if its done so that the variance calculation does not give an error)
        df[key]['dummy']=np.ones(len(df[key]),dtype=int)

    # 3. Replace yr_renovated column with age_since_renovated (important to remove the yr_renovated column at the end)
        for x in df[key]['yr_renovated']:
            if x==0:
                df[key]['age_since_renovated']=df[key]['year']-df[key]['yr_built']
            else:
                df[key]['age_since_renovated']=df[key]['year']-df[key]['yr_renovated']

    # 4. drop the date and yr_renovated columns
        df[key].drop(columns=['date', 'yr_renovated'], axis=1, inplace=True)

    # 5. The rest of the steps below have to be done if normalize is true
    # Points to note while normalizing:
        # 1. Ignore the waterfront column while normalizing and add it as it is into the preprocessed_data variable at the end
        # 2. Do not normalize price value
        # 3. Store the avg and mean in the train step for each column that has to be used 

    if normalize==True:
        for col in df['train'].drop(columns=["dummy","waterfront","price"],axis=1).columns:
            m=df['train'][col].mean() #calculates mean
            s=df['train'][col].std()  #calculates standard deviation
            for key in df.keys():
                df[key][col] = (df[key][col] - m) / s
                
    return df


# Implements the feature engineering required for part 4. Quite similar to preprocess_data.
# Expand the arguments of this function however you like to control which feature modification
# approaches are / aren't active.
def modify_features(data):
    # Your code here:
    modified_data = data
    return modified_data    


# Trains a linear model on the provided data and labels, using the supplied learning rate.
# weights should store the per-feature weights of the learned linear regression.
# losses should store the sequence of MSE losses for each epoch of training, which you will then plot.

def gd_train(data, labels, lr, stopConvVal=10e-6, numDivergentVals=10):
    # Your code here:
    # 1. Initialise random w vector of the size of number of columns (cols) in features dataframe
    rows, cols = data.shape
    weights = np.random.rand(cols)

    # 2. Initialize an empty losses list, divergent counter to track divergence incrementally, 
    #    completion_marker to store if the model completed, converged or diverged
    losses = []
    divergent_count = 0
    completion_marker = "Completed"

    # 3. Loop over number of epochs
    for epoch in range(4000):
        # 1. Randomly shuffle the data and the labels (ensure that the shuffle maintains the 1-to-1 correspondance)
        shuffle_index = list(range(rows))
        random.shuffle(shuffle_index)
        X = data[shuffle_index, :]
        Y = labels[shuffle_index]

        # 2. Initialize deltaLoss for the epoch to a 0 vector of length cols and also initialise MSEloss for the epoch as 0 (integer)

        # 3. Loop over the N training samples (using numpy functions instead of looping individually)
            # 1. Compute y' = wTx
        Y_hat = np.sum(weights * X, axis=1)
            # 2. Compute the direction for the sample x using (y' - y)*x
        grad_direction = np.dot(np.diag(Y_hat-Y), X)  
            # 3. Reinitialize deltaLoss by adding above computed loss vector to initial deltaLoss vector
        deltaLoss = np.sum(grad_direction, axis=0)
            # 4. Compute MSE loss for the sample as (y' - y)**2 
        epochLoss = np.sum(np.square(Y_hat - Y))

        # 4. Normalize deltaLoss as dL = (2/N) * deltaLoss
        dL = (2/rows) * deltaLoss

        # 5. append MSEloss/N to the losses list
        loss = epochLoss/rows
        losses.append(loss)

        # 6. Update weights w = w - lr*dL
        weights -= (lr * dL)

        # 7. check convergence
        if len(losses) > 1 and str(losses[-1]) != 'nan' and str(losses[-2]) != 'nan' and abs(losses[-2] - losses[-1]) <= stopConvVal:
            completion_marker = "EarlyConvergence"
            break

        # 8. check divergence
        if len(losses) > 1 and (loss > losses[-2] or str(loss) == 'nan'):
            divergent_count += 1
        else:
            divergent_count = 0

        if numDivergentVals == divergent_count:
            completion_marker = "Divergent"
            break
    

    return weights, losses, completion_marker

# Invoke the above functions to implement the required functionality for each part of the assignment.

# Part 0  : Data preprocessing.
# Your code here:
# 1. Load the two datasets using the load_data function
    

# Part 1 . Implement batch gradient descent and experiment with different learning rates.
# Your code here:

def grad_desc(features, y, title, lrs, stopEarlyConv, stopEarlyDiv):
    losses_matrix = []
    lr_plots = []
    weights_array = []
    completion_markers = []
    print("lr  :  status  :  Train MSE")
    for lr in lrs:
        weights, losses, completion_marker = gd_train(features, y, lr, stopEarlyConv, stopEarlyDiv)
        if completion_marker != "Divergent":
            losses_matrix.append(losses)
            lr_plots.append(lr)
            print(lr, " : ", completion_marker, " : ", str(losses[-1]))
        else:
            print(lr, " : ", completion_marker, " : ", str(losses[-1]))
        weights_array.append(weights)
        completion_markers.append(completion_marker)

    return weights_array, completion_markers
    
def validation(features, weights,output_df):

     Y_hat = np.sum(weights * features, axis=1)
     #sum_loss = np.sum(np.square(Y_hat - y))
     #loss = sum_loss/features.shape[0]
     output_df['price'] = Y_hat
     return output_df



       
# Part 1 batch gradient descent on normalized data 
# Your code here:
if True:
    print("------------------------------------------------------------------------------------------------------------------")
    print("------------------------------------------- Processing Normalized Data -------------------------------------------")
    print("------------------------------------------------------------------------------------------------------------------")
    data=load_data()
    normalized_data = preprocess_data(data, normalize=True)
    norm_feature_names = [col for col in normalized_data['train'].columns if col != 'price']
    norm_learning_rates = [1e-1]
    #print("Weights Feature Order: ", '   '.join(norm_feature_names))
    norm_train_features = data['train'][norm_feature_names].to_numpy()
    norm_y = data['train']['price'].to_numpy()
    #print("-------------------------------------------- Running Gradient Descent --------------------------------------------")
    norm_weights_array, norm_completion_markers = grad_desc(norm_train_features, norm_y, 'Normalised', norm_learning_rates, 10e-6, 10)
    print("----------------------------------------------- Running Validation -----------------------------------------------")
    norm_test_features = data['test'][norm_feature_names].to_numpy()
    #norm_test_y = data['test']['price'].to_numpy()
    #norm_weights_array = np.array([-0.2842115, 0.34837914, 0.87006086, 0.05846561, 0.02292471, 3.1335569,
     #                        0.4770081, 0.19157651, 1.11449918, 0.66323201, 0.10116509, -0.06705246,
      #                       -0.26514001, 0.83395034, -0.30246532, 0.13769728, -0.09955753, 0.16014805,
       #                      0.05612991, -0.04974887, 5.34055516, 0.73579381])
    output_df = data['test'][['id']]

    output_df = validation(norm_test_features, norm_weights_array,output_df)
    output_df.to_csv('out.csv',index=False)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx Processed Normalized Data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n\n\n")


    #/nfs/stak/users/sathea/IA_Solo_Assignment
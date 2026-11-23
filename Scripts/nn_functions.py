import torch
import math
#Give inputs and type of activation function and according to the type the chosen
#activation function will be implemented
def activation_function(u, act_type):
    #Convert to uppercase
    act_type = act_type.upper()

    if act_type == "RELU":
        return torch.clamp(u, min=0)
    elif act_type == "SIGMOID":
        return torch.sigmoid(u)
    elif act_type == "TANH":
        return torch.tanh(u)

#Same as activation function but with derivatives
def derivative_activation_function(u, act_type):
    act_type = act_type.upper()

    if act_type == "RELU":
        return (u > 0).to(u.dtype)
    elif act_type == "SIGMOID":
        a = torch.sigmoid(u)
        return a * (1 - a)
    elif act_type == "TANH":
        return 1 - torch.tanh(u) ** 2

#Give a batch (x) and array w as inputs and get the batch output before activation (u)
#And after activation (a)
#X -> M*(Image_pixels)
#W -> This layer's neurons x (weights connecting all the previous layers neurons with the i-th + 1)
#U -> M x This layers neurons
#A -> same with U
def layer(w, x, act_type):
    #In the beginning of each x add -1 to create the bias
    #Create a column tensor of -1 and concatenate it with the rest of x
    x_aug = torch.cat((x.new_full((x.size(0), 1), -1), x), dim=1)
    #Result = Matrix containing the non activated results of the layer(each line for different input)
    u = x_aug @ w.T
    #Matrix containing the activated results of the layer for (each line for different input)
    a = activation_function(u, act_type)
    return u, a

#This function will take as inputs all the weight tensors for each layer, a batch (x) and the type of
#Activation function we want to use and will produce the output
def forward_pass(W, x, act_type):
    #Create lists a and u that will contain activated and raw outputs of each layer
    #To use later on backpropagation
    #Initialize a with x's
    a_values = [x]
    u_values = []
    #initialize layer's input
    input_tensor = x
    #For each layer's weight matrix
    for w in W:
        #Calculate exit before and after activation
        u, a = layer(w, input_tensor, act_type)
        #Add the results to a_values and u_values
        u_values.append(u)
        a_values.append(a)
        #Update next layer's input
        input_tensor = a

    #Return entire a_values and u_values but also u_values[-1] which is the final exit of the network
    return a_values, u_values, u_values[-1]

#this function calculates the loss signals on the output layer
def loss_function(desired, y_out):
    # Apply softmax on the non activated output
    #Y_out is (batch_size) x (neurons of the layer) so dim = 1 because softmax through columns(per row)
    soft_yout = torch.softmax(y_out, dim=1)
    # Construct les of the output layer
    deltas = [desired - soft_yout]
    #Retrurn the deltas(local error signals ) of the output layer
    return deltas


#This function calculates backwards all the local error signals
def backpropagation(output_deltas, W, u_values, act_type):

    #Construct les of the output layer
    deltas = output_deltas
    #Compute les for all layers backwards
    #We ve already computed last layer so index will start from -2
    inverse_index = -2

    #Run the list of W Backwards [start:stop:step]
    for w in W[:0:-1]:
        #Take the next layer deltas
        next_deltas = deltas[-1]
        #Calculate fdot(u)
        derivative = derivative_activation_function(u_values[inverse_index], act_type)
        lse = (next_deltas @ w[:, 1:]) * derivative

        #Update deltas
        deltas.append(lse)
        #Update indexes
        inverse_index -= 1

    return deltas[::-1]

#Update all the weights
def update_weights(W, deltas, a_values, b, s, lamda = 0.0, beta = 0.9):
    #Calculate new weights for each weight array of the list
    w_new = []
    #i is enumerated while w,d,a are assigned their i-st list element
    for i, (w, d, a) in enumerate(zip(W, deltas, a_values[:-1])):
        #Add the -1 as first input at each layer (is for the threshold)
        a_aug = torch.cat((a.new_full((a.size(0), 1), -1), a), dim=1)
        #Calculate g
        #that way every element of g matrix (gij) is the sum of  deltas of i-th neurons of this layer
        #multiplied with input of the j-th neuron of the previous layer so dwij
        g = (d.T @ a_aug) / d.size(0)
        #Add the L2 on g (L2 dosen't apply on Bias weights)
        g[:, 1:] -=  2*lamda*w[:, 1:]
        #Create new matrix s for each wij
        s[i] = beta*s[i] + (1 - beta)* g**2
        #Create new g using RMSprop
        g = g / (s[i] + 1e-8).sqrt()
        #Finally update the weights
        updated = w + b * g

        w_new.append(updated)

    return w_new, s

# This function constructs the desired outcome.
# We have 10 classes, so there are 10 neurons in the output layer.
# If the label is 0, we want [1, 0, ..., 0].
# For each label, we produce the corresponding desired vector.
def construct_desired(labels):
    # Create a tensor with one row per batch's element label and 10 columns for the CIFAR-10 classes.
    # Initialize all elements to zero as a tensor on the GPU
    desired = torch.zeros(labels.size(0), 10, device=labels.device, dtype=torch.float32)

    # rows = [0,1, 2, .... M-1] at GPU to use it to the desired line below
    rows = torch.arange(labels.size(0), device=labels.device)

    # In each row, set the element at the label index to 1.
    desired[rows, labels] = 1

    return desired

#Each Weight tensor is (This layers Neurons) x (Last layer outputs+1 for bias)
# This function takes as inputs a list that contains the number of neurons on each layer
# and the input's size and initializes all weights
# Weights for biases will be initialized to zero while the rest of them following
# uniform distribution. So it returns a list of weight arrays
def initialize_network(neurons, input_size, device = None):
    # initialize list
    weights = []
    # Number of columns on first weight array will be input size + 1 for biases
    columns = input_size + 1
    for i in neurons:
        # Set uniform's distribution range (current layer neurons + previous layers neurons)
        b = math.sqrt(6 / (i + columns - 1))
        # Initialize the w array based on uniform distribution
        #Add device and dtype to store them directly on the GPU
        w = torch.empty(i, columns, device = device, dtype = torch.float32).uniform_(-b, b)
        # Set the bias weights to zero
        w[:, 0] = 0
        # Add the new array on the list
        weights.append(w)
        columns = i + 1

    return weights

#Function randomly sets  10% of training data as validation ones
def make_validation_and_train(train_data, train_label):
    #Take size of training data
    n = train_data.size(0)
    #Apply floor division because we need integers
    n_val = n // 10
    #Set indices randomly so [1,342,9,90,....]
    indices = torch.randperm(n, device=train_data.device)
    #Keep the first 10% as validation indexes
    val_idx = indices[:n_val]
    #And the rest as training ones
    train_idx = indices[n_val:]
    #Construct train and validation data
    val_data = train_data[val_idx]
    val_label = train_label[val_idx]

    train_data = train_data[train_idx]
    train_label = train_label[train_idx]

    return  train_data, train_label, val_data, val_label

#Epoch takes the train_data set splits it to batches and returns the new calculated weights
def epoch(train_data, train_label, batch_size, w, act_type, b, s, lamda, beta):

   new_weights = w
   # Loop for each batch
   #[start:stop:step]
   for start in range(0, train_data.size(0),  batch_size):
       #update stop index
       stop = start + batch_size
       #Gather a (outputs after activation), u(outputs before activation), yout
       a, u, y_out = forward_pass(new_weights, train_data[start:stop], act_type)

       #Construct desired vectors based on labels
       desired = construct_desired(train_label[start:stop])
       #Create output layer deltas
       output_deltas = loss_function(desired, y_out)
       #Calculate all deltas
       deltas = backpropagation(output_deltas, new_weights, u, act_type)
       #Update the new weights
       new_weights, s = update_weights(new_weights, deltas, a, b, s, lamda, beta)
   #Return weights  and s at the end of the season
   #So next season to be able to continue through there
   return new_weights, s

#Function takes test data and through forward pass checks if the nn's classification
#Matches the labels. Returns accuracy percentage...
def evaluate(w, test_data, act_type, test_labels, batch_size):
    correct = 0
    #Start, stop, step
    for start in range(0, test_data.size(0),batch_size):
        stop = start + batch_size
        #_ because we don't care about the rest of the return arguments
        _, _, y_out = forward_pass(w, test_data[start:stop], act_type)
        #Keep the index of the greatest value
        predictions = y_out.argmax(dim = 1)
        #If the index is the same as the corresponding label we add it
        #Checks if row = row and adds to the correct the sum of the matching rows
        correct = correct + (predictions == test_labels[start:stop]).sum()
        #.item converts tensor to float
    return 100*correct.item() / test_data.size(0)

#Train function uses part of train data as validation
def train(w, train_data, train_label, batch_size, b, act_type, val_data, val_label, lamda, beta):

    #Initalize best weights, best accuracy and no improvement reps
    best_weights = w
    weights = w
    best_accuracy = -1
    no_impr_epochs = 0
    total_epochs = 0
    b_change = 0
    #Initalize s as zero for RMSProp
    s = [torch.zeros_like(weight) for weight in w]

    #initalize validation history and bs_used
    validation_history = []
    bs_used = []

    # Run a maximum of 50 epochs
    for i in range(50):
        total_epochs += 1
        #Take weights and s to start the next epoch with them
        weights, s = epoch(train_data, train_label, batch_size, weights, act_type, b, s, lamda, beta)
        #Calculate accuracy
        accuracy = evaluate(weights, val_data, act_type, val_label, batch_size)
        #Update validation_history and learning rate used
        validation_history.append(accuracy)
        bs_used.append(b)

        #If epoch had improvements
        if accuracy > best_accuracy:
            #Set no imprv counter to zero
            no_impr_epochs = 0
            #Update best accuracy and best weights
            best_weights = weights
            best_accuracy = accuracy
        else:
            #Counter of epochs with no improvement goes up by one
            no_impr_epochs += 1

        #If we have three epochs without improvement
        if no_impr_epochs >= 3:
            #If we have cut in half b more than 4 times kill the training
            if b_change >= 4:
                break
            #Otherwise continue training with b reduced in half
            b /= 2
            b_change += 1
            no_impr_epochs = 0


    return best_weights, validation_history, bs_used

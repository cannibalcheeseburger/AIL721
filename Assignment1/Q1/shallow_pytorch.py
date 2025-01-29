# ================================ Imports ================================ #
import matplotlib.pyplot as plt
import torch
import numpy as np
import logging
plt.style.use('seaborn-v0_8')

# =============================== Variables ================================== #
torch.manual_seed(100) # Do not change the seed
np.random.seed(100) # Do not change the seed
torch.set_default_dtype(torch.float64)
logging.basicConfig(filename="avg-error-pytorch.log", filemode='w', format='%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
EPOCH = 10000

# ============================================================================ #


class shallow_network(torch.nn.Module):

    def __init__(self,input_size,num_hidden_units,output_size) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size,num_hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden_units,output_size),
            torch.nn.Sigmoid()
        )

    def forward(self,x):
        return self.model(x)




def plot_loss(train_loss):
    '''
    :param train_loss: list of training losses
    :return: saves a plot of the training losses vs epochs with file name loss-pytorch.png on current working directory
    '''
    # ========== Please do not edit this function

    plt.xlabel("Training Epoch")
    plt.ylabel("Training Loss")
    plt.plot(train_loss)
    plt.savefig("./loss-pytorch.png")


def main():

    # ========= Input Data
    X = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                X.append([i,j,k])
    X = torch.tensor(X).double()
    # ========= Ouput Labels
    Y  = outputs = torch.all(X == 1, axis=1).double()

    # =========== Write code to build neural net model
    input_size = 3
    num_hidden_units = 2
    output_size = 1
    model = shallow_network(input_size,num_hidden_units,output_size)
    lr = 0.1
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)

 
    # =========== Write code for training the model
    train_loss = [] # Use this list to store training loss per epoch
    for i in range(EPOCH): # Total training epoch is 10000
        
        outputs = model(X)

        loss = loss_function(outputs.squeeze(),Y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        print(f'Epoch: {i + 1} , Loss: {loss.item()}')

        
        train_loss.append(loss.item()) # Log the training loss for the epoch


    # =========== Plot Epoch Losses
    plot_loss(train_loss) # Do not change

    # =========== Predict
    X = torch.tensor(X).double()
    Y = torch.tensor(Y).double()
    error = []
    logger.info("===================")
    logger.info("   X       Y   Y' ")
    logger.info("===================")
    for i in range(Y.shape[0]):
        tstr = ""
        x = X[i]
        y_target = Y[i]
        y_pred = model.forward(x)
        loss = loss_function(y_pred.squeeze(),y_target)
        error.append(loss.item())
        x = x.data.numpy()
        y_target = int(y_target.item())
        y_pred = round(y_pred.item(), 1)
        tstr += str(x) + " "+ str(y_target)+" "+str(y_pred)
        logger.info(tstr)
    logger.info("Average Error: " + str(round(np.mean(error), 5)))



# =============================================================================== #

if __name__ == '__main__':
    main()

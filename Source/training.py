#----------------------------------------------------------------------
# Routines for training and testing the GNNs
# Author: Pablo Villanueva Domingo
# Last update: 10/11/21
#----------------------------------------------------------------------

from Source.constants import *
from scipy.spatial.transform import Rotation as Rot

# For perfoming symbolic regression, L1 regularization of messages is required
# Inputs and messages are also stored
# This functionality is not well tested

# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

# Training step
def train(loader, model, optimizer):
    model.train()

    loss_tot = 0
    for data in loader:  # Iterate in batches over the training dataset.

        # Rotate randomly for data augmentation
        rotmat = Rot.random().as_matrix()
        data.pos = torch.tensor([rotmat.dot(p) for p in data.pos], dtype=torch.float32)
        data.x[:,:3] = torch.tensor([rotmat.dot(p) for p in data.x[:,:3]], dtype=torch.float32)

        data.to(device)
        optimizer.zero_grad()  # Clear gradients.
        out = model(data)  # Perform a single forward pass.
        y_out, err_out = out[:,0], out[:,1]     # Take mean and standard deviation of the output

        # Compute loss as sum of two terms for likelihood-free inference
        loss_mse = torch.mean((y_out - data.y)**2 , axis=0)
        loss_lfi = torch.mean(((y_out - data.y)**2 - err_out**2)**2, axis=0)
        loss = torch.log(loss_mse) + torch.log(loss_lfi)

        # Message passing L1 regularization (for symbolic regression)
        if use_l1 and (model.namemodel=="PointNet" or model.namemodel=="EdgeNet"):
            mpL1 = l1_reg*torch.sum(torch.abs(model.layer1.messages))
            loss += mpL1

        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        loss_tot += loss.item()

    return loss_tot/len(loader)

# Testing/validation step
def test(loader, model, params, message_reg=sym_reg):
    model.eval()

    if model.namemodel=="PointNet":
        inputs = np.zeros((1,9))
    elif model.namemodel=="EdgeNet":
        inputs = np.zeros((1,12))
    #inputs = np.zeros((1,6))
    messgs = np.zeros((1,100))
    pools = np.zeros((1,100))
    outs = np.zeros((1))
    trues = np.zeros((1))
    yerrors = np.zeros((1))

    errs = []
    loss_tot = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        with torch.no_grad():

            data.to(device)
            out = model(data)  # Perform a single forward pass.
            y_out, err_out = out[:,0], out[:,1]
            err = (y_out.reshape(-1) - data.y)/data.y
            errs.append( np.abs(err.detach().cpu().numpy()).mean(axis=0) )

            # Compute loss as sum of two terms for likelihood-free inference
            loss_mse = torch.mean((y_out - data.y)**2 , axis=0)
            loss_lfi = torch.mean(((y_out - data.y)**2 - err_out**2)**2, axis=0)
            loss = torch.log(loss_mse) + torch.log(loss_lfi)

            # Message passing L1 regularization (for symbolic regression)
            if use_l1 and (model.namemodel=="PointNet" or model.namemodel=="EdgeNet"):
                mpL1 = l1_reg*torch.sum(torch.abs(model.layer1.messages))
                loss += mpL1
            loss_tot += loss.item()

            # Append true values and predictions
            outs = np.append(outs, y_out.detach().cpu().numpy(), 0)
            trues = np.append(trues, data.y.detach().cpu().numpy(), 0)
            yerrors = np.append(yerrors, err_out.detach().cpu().numpy(), 0)

            # If symbolic regression (pass otherwise)
            if message_reg and (model.namemodel=="PointNet" or model.namemodel=="EdgeNet"):

                ins = model.layer1.input
                mes = model.layer1.messages
                pool = model.pooled
                maxmes = mes
                maxpool = pool

                inputs = np.append(inputs, ins.detach().cpu().numpy(), 0)
                messgs = np.append(messgs, maxmes.detach().cpu().numpy(), 0)
                pools = np.append(pools, maxpool.detach().cpu().numpy(), 0)


    # Save true values and predictions
    np.save("Outputs/outputs_"+namemodel(params)+".npy",outs)
    np.save("Outputs/trues_"+namemodel(params)+".npy",trues)
    np.save("Outputs/errors_"+namemodel(params)+".npy",yerrors)

    # If symbolic regression (pass otherwise)
    if message_reg and (model.namemodel=="PointNet" or model.namemodel=="EdgeNet"):
        inputs, messgs, pools, outs, trues = np.delete(inputs,0,0), np.delete(messgs,0,0), np.delete(pools,0,0), np.delete(outs,0,0), np.delete(trues,0,0)
        np.save("Outputs/inputs_"+namemodel(params)+".npy",inputs)
        np.save("Outputs/messages_"+namemodel(params)+".npy",messgs)
        np.save("Outputs/poolings_"+namemodel(params)+".npy",pools)

    return loss_tot/len(loader), np.array(errs).mean(axis=0)

# Training procedure
def training_routine(model, train_loader, test_loader, params, verbose=True):

    use_model, learning_rate, weight_decay, n_layers, k_nn, n_epochs, training, simsuite, simset, n_sims = params

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses, valid_losses = [], []
    valid_loss_min, err_min = 1000., 1000.

    # Training loop
    for epoch in range(1, n_epochs+1):
        train_loss = train(train_loader, model, optimizer)
        test_loss, err = test(test_loader, model, params)
        train_losses.append(train_loss); valid_losses.append(test_loss)

        # Save model if it has improved
        if test_loss <= valid_loss_min:
            if verbose: print("Validation loss decreased ({:.2e} --> {:.2e}).  Saving model ...".format(valid_loss_min,test_loss))
            torch.save(model.state_dict(), "Models/"+namemodel(params))
            valid_loss_min = test_loss
            err_min = err

        if verbose: print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.2e}, Validation Loss: {test_loss:.2e}, Relative error: {err:.2e}')

    return train_losses, valid_losses

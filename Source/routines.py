import matplotlib.pyplot as plt
from Source.params import *
from pysr import pysr, best, best_tex, get_hof

# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

# Name of the model and hyperparameters
def namemodel(model):
    #return "model_"+model.__class__.__name__+"_lr_{:.2e}_weightdecay_{:.2e}_epochs_{:d}".format(learning_rate,weight_decay,epochs)
    return "model_"+model.namemodel+"_lr_{:.2e}_weightdecay_{:.2e}_epochs_{:d}".format(learning_rate,weight_decay,epochs)

# Training step
def train(loader, model, optimizer, criterion):
    model.train()

    loss_tot = 0
    for data in loader:  # Iterate in batches over the training dataset.

        data.to(device)
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x, data.batch)  # Perform a single forward pass.
        loss = criterion(out.reshape(-1), data.y)  # Compute the loss.
        # Message passing L1 regularization (for symbolic regression)
        if use_l1 and (model.namemodel=="PointNet" or model.namemodel=="EdgeNet"):
            mpL1 = l1_reg*torch.sum(torch.abs(model.layer1.messages))
            #mpL1 = l1_reg*torch.sum(torch.abs(model.h))
            loss += mpL1
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        loss_tot += loss

    return loss_tot/len(loader)

# Testing/validation step
def test(loader, model, criterion, message_reg=0):
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

    errs = []
    loss_tot = 0
    for data in loader:  # Iterate in batches over the training/test dataset.

        data.to(device)
        out = model(data.x, data.batch)  # Perform a single forward pass.
        err = (out.reshape(-1) - data.y)/data.y
        #errs.append( np.abs(err.detach().numpy()).mean(axis=0) )
        errs.append( np.abs(err.detach().cpu().numpy()).mean(axis=0) )
        loss = criterion(out.reshape(-1), data.y)
        # Message passing L1 regularization (for symbolic regression)
        if use_l1 and (model.namemodel=="PointNet" or model.namemodel=="EdgeNet"):
            mpL1 = l1_reg*torch.sum(torch.abs(model.layer1.messages))
            #mpL1 = l1_reg*torch.sum(torch.abs(model.h))
            loss += mpL1
        loss_tot += loss

        """ins = model.layer1.input
        mes = model.layer1.messages
        stds = mes.std(0)
        indstdmax = np.argmax(stds.detach().numpy())
        maxmes = mes[:,indstdmax]"""
        #print("in",model.layer1.input.shape)
        #print("mes",mes.shape, ins.shape)
        #print(type(mes), type(ins))
        #mes = mes.detach().numpy()
        #print(type(mes), type(ins))

        if message_reg and (model.namemodel=="PointNet" or model.namemodel=="EdgeNet"):

            #print(data.x.shape, model.h.shape)
            ins = model.layer1.input
            mes = model.layer1.messages
            #ins = data.x
            #mes = model.h
            pool = model.pooled
            #lin = model.lin
            #stds = mes.std(0)
            #indstdmax = np.argmax(stds.detach().numpy())
            #maxmes = mes[:,indstdmax]
            maxmes = mes
            #stds = pool.std(0)
            #indstdmax = np.argmax(stds.detach().numpy())
            #maxpool = pool[:,indstdmax]
            maxpool = pool

            #print(ins.shape, maxmes.shape, maxpool.shape, out.shape, mes.shape)
            inputs = np.append(inputs, ins.detach().cpu().numpy(), 0)
            messgs = np.append(messgs, maxmes.detach().cpu().numpy(), 0)
            pools = np.append(pools, maxpool.detach().cpu().numpy(), 0)

            #lins = np.append(lins, lin, 0)
            #inputs.append(ins), messgs.append(mes)

            """equations = pysr(ins, maxmes, niterations=10,
                binary_operators=["plus", "sub", "mult", "pow", "div"],
                unary_operators=[ "exp", "abs", "logm", "square", "cube", "sqrtm"])
            print(best(equations))"""

        outs = np.append(outs, out.reshape(-1).detach().cpu().numpy(), 0)
        trues = np.append(trues, data.y.detach().cpu().numpy(), 0)

    #inputs, messgs = np.array(inputs), np.array(messgs)
    #flat_list = [item for sublist in t for item in sublist]
    #for input in inputs:

    if message_reg and (model.namemodel=="PointNet" or model.namemodel=="EdgeNet"):
        inputs, messgs, pools, outs, trues = np.delete(inputs,0,0), np.delete(messgs,0,0), np.delete(pools,0,0), np.delete(outs,0,0), np.delete(trues,0,0)
        #print(inputs.shape, messgs.shape, outs.shape, trues.shape)
        np.save("Models/inputs_"+namemodel(model)+".npy",inputs)
        np.save("Models/messages_"+namemodel(model)+".npy",messgs)
        np.save("Models/poolings_"+namemodel(model)+".npy",pools)

    np.save("Models/outputs_"+namemodel(model)+".npy",outs)
    np.save("Models/trues_"+namemodel(model)+".npy",trues)

    #inputs, messgs = inputs.reshape((-1, 100)), messgs.reshape((-1,9))
    #print(inputs.shape, messgs.shape)
    return loss_tot/len(loader), np.array(errs).mean(axis=0)

# Training procedure
def training_routine(model, train_loader, test_loader, learning_rate, weight_decay, verbose=True):

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    train_losses, valid_losses = [], []
    valid_loss_min, err_min = 1000., 1000.
    for epoch in range(1, epochs+1):
        train_loss = train(train_loader, model, optimizer, criterion)
        test_loss, err = test(test_loader, model, criterion)
        train_losses.append(train_loss); valid_losses.append(test_loss)

        # Save model if it has improved
        if test_loss <= valid_loss_min:
            if verbose: print("Validation loss decreased ({:.2e} --> {:.2e}).  Saving model ...".format(valid_loss_min,test_loss))
            torch.save(model.state_dict(), "Models/"+namemodel(model))
            valid_loss_min = test_loss
        if err < err_min:
            err_min = err

        if verbose: print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.2e}, Validation Loss: {test_loss:.2e}, Relative error: {err:.2e}')

    return train_losses, valid_losses

def plot_losses(train_losses, valid_losses, test_loss, err_min, model):

    plt.plot(range(epochs), np.array(train_losses), "r-",label="Training")
    plt.plot(range(epochs), np.array(valid_losses), "b:",label="Validation")
    plt.legend()
    plt.yscale("log")
    #plt.title(f"Minimum relative error: {err_min:.2e}")
    plt.title(f"Test loss: {test_loss:.2e}, Minimum relative error: {err_min:.2e}")
    plt.savefig("Plots/loss_"+namemodel(model)+".pdf")

# Visualization routine
def visualize_points(data, edge_index=None, index=None):

    pos = data.x[:,:2]
    c_o_m = data.y
    fig = plt.figure(figsize=(4, 4))
    if edge_index is not None:
        for (src, dst) in edge_index.t().tolist():
            src = pos[src].tolist()
            dst = pos[dst].tolist()
            plt.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=1, color='black')
    if index is None:
        plt.scatter(pos[:, 0], pos[:, 1], s=50, zorder=1000)
    else:
       mask = torch.zeros(pos.size(0), dtype=torch.bool)
       mask[index] = True
       plt.scatter(pos[~mask, 0], pos[~mask, 1], s=50, color='lightgray', zorder=1000)
       plt.scatter(pos[mask, 0], pos[mask, 1], s=50, zorder=1000)

    plt.axis('off')
    #plt.savefig("loss.pdf")
    plt.show()

# Creates a complete graph. For a given set of nodes, creates a set of edges connecting all the nodes
def build_complete_graph(num_nodes):

    # Initialize edge index matrix
    E = torch.zeros((2, num_nodes * (num_nodes - 1)), dtype=torch.long)

    # Populate 1st row
    for node in range(num_nodes):
        for neighbor in range(num_nodes - 1):
            E[0, node * (num_nodes - 1) + neighbor] = node

    # Populate 2nd row
    neighbors = []
    for node in range(num_nodes):
        neighbors.append(list(np.arange(node)) + list(np.arange(node+1, num_nodes)))
    E[1, :] = torch.Tensor([item for sublist in neighbors for item in sublist])

    return E

# Creates an empty graph, without edges
def build_empty_graph():

    edge_index = torch.tensor([[], []], dtype=torch.int64)
    return edge_index

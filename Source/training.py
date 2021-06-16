from Source.constants import *
#from pysr import pysr, best, best_tex, get_hof
from torch_geometric.transforms import Compose, RandomRotate

random_rotate = Compose([
    RandomRotate(degrees=180, axis=0),
    RandomRotate(degrees=180, axis=1),
    RandomRotate(degrees=180, axis=2),
])


# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

# Training step
def train(loader, model, optimizer, criterion):
    model.train()

    loss_tot = 0
    for data in loader:  # Iterate in batches over the training dataset.

        random_rotate(data)

        data.to(device)
        optimizer.zero_grad()  # Clear gradients.
        out = model(data)  # Perform a single forward pass.
        y_out, err_out = out[:,0], out[:,1]

        #loss_mse = criterion(y_out, data.y)  # Compute the loss.
        loss_mse = torch.mean((y_out - data.y)**2 , axis=0)
        loss_lfi = torch.mean(((y_out - data.y)**2 - err_out**2)**2, axis=0)
        loss = torch.log(loss_mse) + torch.log(loss_lfi)

        # Message passing L1 regularization (for symbolic regression)
        if use_l1 and (model.namemodel=="PointNet" or model.namemodel=="EdgeNet"):
            mpL1 = l1_reg*torch.sum(torch.abs(model.layer1.messages))
            #mpL1 = l1_reg*torch.sum(torch.abs(model.h))
            loss += mpL1
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        loss_tot += loss.item()

    return loss_tot/len(loader)

# Testing/validation step
def test(loader, model, criterion, params, message_reg=0):
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
            #errs.append( np.abs(err.detach().numpy()).mean(axis=0) )
            errs.append( np.abs(err.detach().cpu().numpy()).mean(axis=0) )

            #loss_mse = criterion(y_out, data.y)  # Compute the loss.
            loss_mse = torch.mean((y_out - data.y)**2 , axis=0)
            loss_lfi = torch.mean(((y_out - data.y)**2 - err_out**2)**2, axis=0)
            loss = torch.log(loss_mse) + torch.log(loss_lfi)

            # Message passing L1 regularization (for symbolic regression)
            if use_l1 and (model.namemodel=="PointNet" or model.namemodel=="EdgeNet"):
                mpL1 = l1_reg*torch.sum(torch.abs(model.layer1.messages))
                #mpL1 = l1_reg*torch.sum(torch.abs(model.h))
                loss += mpL1
            loss_tot += loss.item()

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

            outs = np.append(outs, y_out.detach().cpu().numpy(), 0)
            trues = np.append(trues, data.y.detach().cpu().numpy(), 0)
            yerrors = np.append(yerrors, err_out.detach().cpu().numpy(), 0)


    #inputs, messgs = np.array(inputs), np.array(messgs)
    #flat_list = [item for sublist in t for item in sublist]
    #for input in inputs:

    if message_reg and (model.namemodel=="PointNet" or model.namemodel=="EdgeNet"):
        inputs, messgs, pools, outs, trues = np.delete(inputs,0,0), np.delete(messgs,0,0), np.delete(pools,0,0), np.delete(outs,0,0), np.delete(trues,0,0)
        #print(inputs.shape, messgs.shape, outs.shape, trues.shape)
        np.save("Outputs/inputs_"+namemodel(params)+".npy",inputs)
        np.save("Outputs/messages_"+namemodel(params)+".npy",messgs)
        np.save("Outputs/poolings_"+namemodel(params)+".npy",pools)

    np.save("Outputs/outputs_"+namemodel(params)+".npy",outs)
    np.save("Outputs/trues_"+namemodel(params)+".npy",trues)
    np.save("Outputs/errors_"+namemodel(params)+".npy",yerrors)

    #inputs, messgs = inputs.reshape((-1, 100)), messgs.reshape((-1,9))
    #print(inputs.shape, messgs.shape)
    return loss_tot/len(loader), np.array(errs).mean(axis=0)

# Training procedure
def training_routine(model, train_loader, test_loader, params, verbose=True):

    use_model, learning_rate, weight_decay, n_layers, k_nn, n_epochs, training, simtype, simset, n_sims = params

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    train_losses, valid_losses = [], []
    valid_loss_min, err_min = 1000., 1000.
    for epoch in range(1, n_epochs+1):
        train_loss = train(train_loader, model, optimizer, criterion)
        test_loss, err = test(test_loader, model, criterion, params)
        train_losses.append(train_loss); valid_losses.append(test_loss)

        # Save model if it has improved
        if test_loss <= valid_loss_min:
            if verbose: print("Validation loss decreased ({:.2e} --> {:.2e}).  Saving model ...".format(valid_loss_min,test_loss))
            torch.save(model.state_dict(), "Models/"+namemodel(params))
            valid_loss_min = test_loss
        if err < err_min:
            err_min = err

        if verbose: print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.2e}, Validation Loss: {test_loss:.2e}, Relative error: {err:.2e}')

    return train_losses, valid_losses


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

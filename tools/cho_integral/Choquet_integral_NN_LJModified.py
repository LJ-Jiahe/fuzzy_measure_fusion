import torch
import numpy as np


# Convert decimal to binary string
def sources_and_subsets_nodes(N):
    str1 = "{0:{fill}"+str(N)+"b}"
    a = []
    for i in range(1,2**N):
        a.append(str1.format(i, fill='0'))

    sourcesInNode = []
    sourcesNotInNode = []
    subset = []
    sourceList = list(range(N))
    # find subset nodes of a node
    def node_subset(node, sourcesInNodes):
        return [node - 2**(i) for i in sourcesInNodes]
    
    # convert binary encoded string to integer list
    def string_to_integer_array(s, ch):
        N = len(s) 
        return [(N - i - 1) for i, ltr in enumerate(s) if ltr == ch]
    
    for j in range(len(a)):
        # index from right to left
        idxLR = string_to_integer_array(a[j],'1')
        sourcesInNode.append(idxLR)  
        sourcesNotInNode.append(list(set(sourceList) - set(idxLR)))
        subset.append(node_subset(j,idxLR))

    return sourcesInNode, subset


def subset_to_indices(indices):
    return [i for i in indices]


def train_chinn(chinn, lr, criterion, optimizer, num_epoch, train_d, train_label):
    """
    Train nn for fuzzy measure
    """
    for epoch in range(num_epoch):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = chinn(train_d)
        # Compute the loss
        loss = criterion(torch.squeeze(y_pred, dim=1), train_label)
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class Choquet_Integral_NN(torch.nn.Module):
    
    def __init__(self, N_in, N_out):
        super(Choquet_Integral_NN,self).__init__()
        self.N_in = N_in
        self.N_out = N_out
        self.nVars = 2**self.N_in - 2
        
        # The FM is initialized with mean
        dummy = (1./self.N_in) * torch.ones((self.nVars, self.N_out), requires_grad=True)
#        self.vars = torch.nn.Parameter( torch.Tensor(self.nVars,N_out))
        self.vars = torch.nn.Parameter(dummy)
        
        # following function uses numpy vs pytorch
        self.sourcesInNode, self.subset = sources_and_subsets_nodes(self.N_in)
        
        self.sourcesInNode = [torch.tensor(x) for x in self.sourcesInNode]
        self.subset = [torch.tensor(x) for x in self.subset]
        
    def forward(self,inputs):    
        self.FM = self.chi_nn_vars(self.vars)
        sortInputs, sortInd = torch.sort(inputs,1, True)
        M, N = inputs.size()
        sortInputs = torch.cat((sortInputs, torch.zeros(M,1)), 1)
        sortInputs = sortInputs[:,:-1] -  sortInputs[:,1:]
        
        out = torch.cumsum(torch.pow(2,sortInd),1) - torch.ones(1, dtype=torch.int64)
        
        data = torch.zeros((M,self.nVars+1))
        
        for i in range(M):
            data[i,out[i,:]] = sortInputs[i,:] 
        
        ChI = torch.matmul(data,self.FM)
        return ChI
    
    # Converts NN-vars to FM vars
    def chi_nn_vars(self, chi_vars):
#        nVars,_ = chi_vars.size()
        chi_vars = torch.abs(chi_vars)
        #        nInputs = inputs.get_shape().as_list()[1]
        
        FM = chi_vars[None, 0,:]
        for i in range(1,self.nVars):
            indices = subset_to_indices(self.subset[i])
            if (len(indices) == 1):
                FM = torch.cat((FM,chi_vars[None,i,:]),0)
            else:
                #         ss=tf.gather_nd(variables, [[1],[2]])
                maxVal,_ = torch.max(FM[indices,:],0)
                temp = torch.add(maxVal,chi_vars[i,:])
                FM = torch.cat((FM,temp[None,:]),0)
              
        FM = torch.cat([FM, torch.ones((1,self.N_out))],0)
        FM = torch.min(FM, torch.ones(1))  
        
        return FM
    



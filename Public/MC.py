import torch
import opt_einsum as oe
import itertools
import scipy.io
import numpy as np
import base
import skimage.metrics as skm
from scipy import io
import time
class MLP(torch.nn.Module):
    '''
    Our MLP neural networks.
    '''
    def __init__(self, input_shape, rank):
        super(MLP, self).__init__()
        self.FC = torch.nn.Sequential(
            torch.nn.Linear(input_shape, rank+50, bias = 1),
            torch.nn.ELU(),
            # torch.nn.Linear(rank+20, rank+10, bias = 1),
            # torch.nn.ELU(),
            # torch.nn.Linear(128, 128),
            # torch.nn.ReLU(),
            torch.nn.Linear(rank+50, rank, bias = 1),
        )
        
    
    def forward(self, x):
        out = self.FC(x)
        return out

def init_weights(layer):
    if type(layer) == torch.nn.Conv2d:
        torch.nn.init.normal_(layer.weight, mean=0, std=0.5)
    elif type(layer) == torch.nn.Linear:
        torch.nn.init.uniform_(layer.weight, a = -0.5, b = 0.5)
        torch.nn.init.constant_(layer.bias, 0.1)


class TN_NN_Decomposition:
    '''
    Decomposition via Neural Networks.
    '''
    def __init__(self, tn_shape, tn_name='CP'):
        self.tensorn_name = tn_name # The name of Decomposition name.
        self.tensor_shape = tn_shape # The shape of Decomposed tensor.
        self.tensor_num = len(tn_shape) # The number of tensor cores.
        self.mode_nn = [] #Neural Networks.

    def fit(self, TensorData, Omega = None, rank = None, epochs = 1000, lr = 0.01, batch_size = 512, mode='batch'):
        '''
        :param TensorData: Data to be decomposed.
        :param rank: The output dimention of last layer.
        :param epochs: Epochs.
        :param lr: Learning rate.
        :param mode:  Optimize Full tensor or Some indexs.
        :param batch_size: Batch_size
        '''
        
        if self.tensorn_name == 'CP':
            param = []
            for i in range(self.tensor_num):
                self.mode_nn.append(MLP(self.tensor_shape[i], rank).apply(init_weights))  # Neural Networks
                param.append(self.mode_nn[i].parameters())
            param = itertools.chain(*param)  # Parameters to be optimized.
            
            if mode == 'full':
                optimzer = torch.optim.Adam(param, lr=lr, betas=(0.9, 0.999))  # batch 不适合
                Observed_index = np.array(np.where(Omega == 1))
                index_list = torch.tensor(base.generateIndex_list(Omega))
                index = base.batch_full_onehot(self.tensor_shape, index_list)
                full_size = len(index[0])
                Data = TensorData.float()
                
                for epoch in range(epochs):
                    output = []
                    for i in range(self.tensor_num):
                        output.append(self.mode_nn[i].forward(index[i]))  # Get output Matrix

                    einsum_list_before = ''
                    init_code = 0
                    for i in range(len(output)):
                        einsum_list_before += oe.get_symbol(init_code) + oe.get_symbol(init_code+1) + ','
                    einsum_list_res = oe.get_symbol(init_code)
                    einsum_str = einsum_list_before[:-1] + '->' + einsum_list_res  # The tensor operations. ba,ca->bc

                    Tensor_result = oe.contract(einsum_str, *output)  # The output prediction.
                    lossfn = torch.nn.MSELoss(reduce=True, size_average=True)  # RMSE loss.
                    Data = TensorData.float()
                    loss = lossfn(Tensor_result, Data[Observed_index])

                    optimzer.zero_grad()  # Optimization
                    loss.backward()
                    optimzer.step()
                        
                    Tensor_result = self.get_prediction()
                    loss = lossfn(Tensor_result[Observed_index], Data[Observed_index])
                    print(f"Epoch {epoch+1}, MSE is {loss}")

            elif mode == 'batch':
                optimzer = torch.optim.SGD(param, lr = lr)
                Observed_index = np.array(np.where(Omega == 1))
                index_list = torch.tensor(base.generateIndex_list(Omega))
                index = base.batch_full_onehot(self.tensor_shape, index_list)
                full_size = len(index[0])
                Data = TensorData.float()
                
                for epoch in range(epochs):
                    for batch_num in range(int(full_size/batch_size)+1):
                        if batch_num == int(full_size/batch_size):
                            if full_size % batch_size == 0:
                                break
                            else:
                                start = batch_num*batch_size
                                end = None
                        else:
                            start = batch_num * batch_size
                            end = (batch_num + 1) * batch_size
                            
                        output = []
                        for i in range(self.tensor_num):
                            output.append(self.mode_nn[i].forward(index[i][start:end]))  # Get output Matrix
    
                        einsum_list_before = ''
                        init_code = 0
                        for i in range(len(output)):
                            einsum_list_before += oe.get_symbol(init_code) + oe.get_symbol(init_code+1) + ','
                        einsum_list_res = oe.get_symbol(init_code)
                        einsum_str = einsum_list_before[:-1] + '->' + einsum_list_res  # The tensor operations. ba,ca->bc
    
                        Tensor_result = oe.contract(einsum_str, *output)  # The output prediction.
                        lossfn = torch.nn.MSELoss(reduce=True, size_average=True)  # RMSE loss.
                        Data = TensorData.float()
                        loss = lossfn(Tensor_result, Data[Observed_index[:,start:end]])
    
                        optimzer.zero_grad()  # Optimization
                        loss.backward()
                        optimzer.step()
                        
                    Tensor_result = self.get_prediction()
                    loss = lossfn(Tensor_result[Observed_index], Data[Observed_index])
                    print(f"Epoch {epoch+1}, MSE is {loss}")

    def get_prediction(self):
        '''
        Get the result output.
        :return:
        '''
        index = []
        for i in range(len(self.tensor_shape)):
            index.append(base.onehot(self.tensor_shape[i],
                                     torch.tensor(range(self.tensor_shape[i]))))  # index list
        output = []
        for i in range(self.tensor_num):
            output.append(self.mode_nn[i].forward(index[i]))  # Get output Matrix

        einsum_list_res = ''
        einsum_list_before = ''
        init_code = 0
        for i in range(len(output)):
            einsum_list_before += oe.get_symbol(init_code + i + 1) + oe.get_symbol(init_code) + ','
            einsum_list_res += oe.get_symbol(init_code + i + 1)
        einsum_str = einsum_list_before[:-1] + '->' + einsum_list_res  # The tensor operations. ba,ca->bc

        Tensor_result = oe.contract(einsum_str, *output)  # The output prediction

        return Tensor_result


if __name__ == "__main__":
    
    M_Omega = torch.tensor(base.loaddata('M_Omega.mat')['M_Omega'])
    Omega = base.loaddata('Omega.mat')['Omega']
    M = torch.tensor(base.loaddata('M.mat')['M']).float()
    
    model = TN_NN_Decomposition(M_Omega.shape)
    
    start = time.time()
    uv = model.fit(M_Omega, Omega = Omega, rank = 10, epochs = 1000, lr = 0.01, batch_size = 128, mode='full')
    end = time.time()
    # Add RMSE Computation
    
    Tensor_result = model.get_prediction()                 # The output prediction.
    lossfn = torch.nn.MSELoss(reduce=True, size_average=True)       # MSE loss.
    loss = lossfn(Tensor_result, M)
    print('Executing time:', end - start)
    print(f"Final MSE is {loss}")
    psnr = skm.peak_signal_noise_ratio(M.detach().numpy(),Tensor_result.detach().numpy())
    print(f"Final MSE is {loss}, PSNR is {psnr}")
    
    #np.save("Windows_Fixed005.npy",Tensor_result.detach().numpy())
    # io.savemat('Pillarfixed.mat', {'Pillar': Tensor_result.detach().numpy()})
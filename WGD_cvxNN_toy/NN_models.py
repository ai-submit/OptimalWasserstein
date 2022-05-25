import numpy as np
import torch

class Feedforward_basic(torch.nn.Module):
    def __init__(self, input_size, hidden_size,activation='ReLU'):
        super(Feedforward_basic, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        if activation=='ReLU':
            self.activation = torch.nn.ReLU()
            self.activation_diff = lambda X: (X>=0).double()
            self.activation_hess = lambda X: X*0
        elif activation=='Sigmoid':
            self.activation = torch.nn.Sigmoid()
            self.activation_diff = lambda X: self.activation(X)*(1-self.activation(X))
            self.activation_hess = lambda X: self.activation(X)*(1-self.activation(X))*(1-2*self.activation(X))
        elif activation=='cos':
            self.activation = lambda X: torch.cos(X)
            self.activation_diff = lambda X: -torch.sin(X)
            self.activation_hess = lambda X: -torch.cos(X)
        elif activation=='ReLU_sq':
            self.activation = lambda X: torch.nn.ReLU()(X)**2
            self.activation_diff = lambda X: torch.nn.ReLU()(X)*2
            self.activation_hess = lambda X: (X>=0).double()*2
        elif activation=='zero':
            self.activation = lambda X: 0*X
            self.activation_diff = lambda X: 0*X
            self.activation_hess = lambda X: 0*X
        else:
            raise Exception('Not implemented')
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = self.activation(hidden)
        output = self.fc2(hidden)
        return output
    
    def grad(self, x):
        hidden = self.fc1(x)
        hidden = self.activation_diff(hidden)
        W = list(self.fc1.parameters())[0]
        a = list(self.fc2.parameters())[0]
        output = (a*hidden)@W
        return output
        
    def laplace(self,x):
        hidden = self.fc1(x)
        hidden = self.activation_hess(hidden)
        W = list(self.fc1.parameters())[0]
        a = list(self.fc2.parameters())[0]
        W_sum = torch.sum(W**2,1)
        output = torch.sum(a*hidden*W_sum,1)
        return output

class Feedforward_poly(Feedforward_basic):
    def __init__(self, input_size, hidden_size,activation='ReLU',poly_degree=2):
        super(Feedforward_poly, self).__init__(input_size, hidden_size,activation)
        self.fc1_poly=torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2_poly=torch.nn.Linear(self.hidden_size, 1)
        self.poly_degree = poly_degree

    def forward(self, x):
        poly_degree = self.poly_degree
        hidden = self.fc1(x)
        hidden = self.activation(hidden)
        output = self.fc2(hidden)
        hidden = (self.fc1_poly(x))**poly_degree
        output = output+self.fc2_poly(hidden)
        return output
    
    def grad(self, x):
        poly_degree = self.poly_degree
        hidden = self.fc1(x)
        hidden = self.activation_diff(hidden)
        W = list(self.fc1.parameters())[0]
        a = list(self.fc2.parameters())[0]
        output = (a*hidden)@W 
        hidden = poly_degree*(self.fc1_poly(x))**(poly_degree-1)
        W = list(self.fc1_poly.parameters())[0]
        a = list(self.fc2_poly.parameters())[0]
        output2 = (a*hidden)@W
        output = output+output2
        return output
        
    def laplace(self,x):
        poly_degree = self.poly_degree
        hidden = self.fc1(x)
        hidden = self.activation_hess(hidden)
        W = list(self.fc1.parameters())[0]
        a = list(self.fc2.parameters())[0]
        W_sum = torch.sum(W**2,1)
        output = torch.sum(a*hidden*W_sum,1)
        hidden = poly_degree*(poly_degree-1)*(self.fc1_poly(x))**(poly_degree-2)
        W = list(self.fc1_poly.parameters())[0]
        a = list(self.fc2_poly.parameters())[0]
        W_sum = torch.sum(W**2,1)
        output = output+torch.sum(a*hidden*W_sum,1)
        return output



class Feedforward_vector(torch.nn.Module):
    def __init__(self, input_size, hidden_size,activation='ReLU'):
        super(Feedforward_vector, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        if activation=='ReLU':
            self.activation = torch.nn.ReLU()
            self.activation_diff = lambda X: (X>=0).double()
        elif activation=='Sigmoid':
            self.activation = torch.nn.Sigmoid()
            self.activation_diff = lambda X: self.activation(X)*(1-self.activation(X))
        elif activation=='cos':
            self.activation = lambda X: torch.cos(X)
            self.activation_diff = lambda X: -torch.sin(X)
        elif activation=='ReLU_sq':
            self.activation = lambda X: torch.nn.ReLU()(X)**2
            self.activation_diff = lambda X: torch.nn.ReLU()(X)*2
        elif activation=='zero':
            self.activation = lambda X: 0*X
            self.activation_diff = lambda X: 0*X
        else:
            raise Exception('Not implemented')
        self.fc2 = torch.nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = self.activation(hidden)
        output = self.fc2(hidden)
        if self.add_poly:
            for i in range(self.poly_num):
                hidden = (self.fc1_poly[i](x))**(i+1)
                output = output+self.fc2_poly[i](hidden)
        return output
        
    def divergence(self,x):
        hidden = self.fc1(x)
        hidden = self.activation_diff(hidden)
        W = list(self.fc1.parameters())[0]
        U = list(self.fc2.parameters())[0]
        WU = torch.sum(W*U.T,1)
        output = torch.sum(hidden*WU,1)
        if self.add_poly:
            for i in range(self.poly_num):
                hidden = (i+1)*(self.fc1_poly[i](x))**i
                W = list(self.fc1_poly[i].parameters())[0]
                U = list(self.fc2_poly[i].parameters())[0]
                WU = torch.sum(W*U.T,1)
                output = output+torch.sum(hidden*WU,1)
        return output

class Feedforward_vector_basic(torch.nn.Module):
    def __init__(self, input_size, hidden_size,activation='ReLU',use_bias=False):
        super(Feedforward_vector_basic, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.input_size, bias=use_bias)
        if activation=='ReLU':
            self.activation = torch.nn.ReLU()
            self.activation_diff = lambda X: (X>=0).double()
        elif activation=='Sigmoid':
            self.activation = torch.nn.Sigmoid()
            self.activation_diff = lambda X: self.activation(X)*(1-self.activation(X))
        elif activation=='cos':
            self.activation = lambda X: torch.cos(X)
            self.activation_diff = lambda X: -torch.sin(X)
        elif activation=='ReLU_sq':
            self.activation = lambda X: torch.nn.ReLU()(X)**2
            self.activation_diff = lambda X: torch.nn.ReLU()(X)*2
        elif activation=='zero':
            self.activation = lambda X: 0*X
            self.activation_diff = lambda X: 0*X
        else:
            raise Exception('Not implemented')

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = self.activation(hidden)
        output = self.fc2(hidden)
        return output
        
    def divergence(self,x):
        hidden = self.fc1(x)
        hidden = self.activation_diff(hidden)
        W = list(self.fc1.parameters())[0]
        U = list(self.fc2.parameters())[0]
        WU = torch.sum(W*U.T,1)
        output = torch.sum(hidden*WU,1)
        return output

class Feedforward_vector_poly(Feedforward_vector_basic):
    def __init__(self, input_size, hidden_size,activation='ReLU',use_bias=False,poly_degree=2):
        super(Feedforward_vector_poly, self).__init__(input_size, hidden_size,activation,use_bias)
        self.fc1_poly=torch.nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.fc2_poly=torch.nn.Linear(self.hidden_size, self.input_size, bias=use_bias)
        self.poly_degree = poly_degree

    def forward(self, x):
        poly_degree = self.poly_degree
        hidden = self.fc1(x)
        hidden = self.activation(hidden)
        output = self.fc2(hidden)
        hidden = (self.fc1_poly(x))**poly_degree
        output = output+self.fc2_poly(hidden)
        return output
        
    def divergence(self,x):
        poly_degree = self.poly_degree
        hidden = self.fc1(x)
        hidden = self.activation_diff(hidden)
        W = list(self.fc1.parameters())[0]
        U = list(self.fc2.parameters())[0]
        WU = torch.sum(W*U.T,1)
        output = torch.sum(hidden*WU,1)
        hidden = poly_degree*(self.fc1_poly(x))**(poly_degree-1)
        W = list(self.fc1_poly.parameters())[0]
        U = list(self.fc2_poly.parameters())[0]
        WU = torch.sum(W*U.T,1)
        output = output+torch.sum(hidden*WU,1)
        return output
import numpy as np


class RBFNN:
    
    def __init__(self, kernels,centers, beta=1,lr=0.1,epochs=80) -> None:
        
        self.kernels = kernels
        self.centers = centers
        self.beta = beta
        self.lr = lr
        self.epochs = epochs
        
        self.W = np.random.randn(kernels,1)
        self.b = np.random.randn(1,1)
        
        # to save the errors evolution
        # in case we want to check them later
        self.errors = []
        
        # to save the gradients 
        # calculated by the network
        # for verification reasons
        self.gradients = []
    
    
    def rbf_activation(self,x,center):
        return np.exp(-self.beta*np.linalg.norm(x - center)**2)
        
    
    def linear_activation(self,A):
        return self.W.T.dot(A) + self.b
    
    def least_square_error(self,pred,y):
        return (y - pred).flatten()**2
    
    def _forward_propagation(self,x):
        
        a1 = np.array([
            [self.rbf_activation(x,center)] 
            for center in self.centers
        ])
        
        a2 = self.linear_activation(a1)
        
        return a2, a1
    
    def _backpropagation(self, y, pred,a1):
        # Back propagation
        dW = -(y - pred).flatten()*a1
        db = -(y - pred).flatten()
        
        # Updating the weights
        self.W = self.W -self.lr*dW
        self.b = self.b -self.lr*db
        return dW, db
        
    
    def fit(self,X,Y):
        
        for _ in range(self.epochs):
            
            for x,y in list(zip(X,Y)):
                # Forward propagation
                pred, a1 = self._forward_propagation(x)
                
                error = self.least_square_error(pred[0],y[0,np.newaxis])
                self.errors.append(error)
                
                # Back propagation
                dW, db = self._backpropagation(y,pred,a1)
                self.gradients.append((dW,db))
    
    def predict(self,x):
        a2,a1 = self._forward_propagation(x)
        return 1 if np.squeeze(a2) >= 0.5 else 0
        
    
    
def main():
        X = np.array([
            [0,0],
            [0,1],
            [1,0],
            [1,1]
        ])
        Y = np.array([
            [0],
            [1],
            [1],
            [0]
        ])
        
        
        rbf = RBFNN(kernels=2,
                    centers=np.array([
                        [0,1],
                        [1,0]
                        
                    ]),
                    beta=1,
                    lr= 0.1,
                    epochs=80
                    )
    
        rbf.fit(X,Y)            
        
        print(f"RBFN weights : {rbf.W}")
        print(f"RBFN bias : {rbf.b}")
        print()
        print("-- XOR Gate --")
        print(f"| 1 xor 1 : {rbf.predict(X[3])} |")
        print(f"| 0 xor 0 : {rbf.predict(X[0])} |")
        print(f"| 1 xor 0 : {rbf.predict(X[2])} |")
        print(f"| 0 xor 1 : {rbf.predict(X[1])} |")
        print("_______________")
    
if __name__ == "__main__":    
    main()


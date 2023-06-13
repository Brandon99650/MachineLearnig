import sys
import time
import numpy as np

class HyperPlane():

    def __init__(self, weight:np.ndarray) -> None:
        """
        weight : 
            A numpy ndarray with dimension (N+1, 1) 
            for the features with N dimensions.

            Note that the weight may *(-1) 
            to make weight[1] (i.e. coefficient of x1) > 0
        """
        self.weight = weight.astype(np.float64)
        if self.weight[1][0] < 0:
            self.weight = -self.weight
        self.dim = weight.shape[0] - 1

    def _expand_ones(self, x:np.ndarray)->np.ndarray:
        """
        expanding one column vector on the first index of
        the column of x as the constant term.
        """
        return np.hstack(
            [np.ones((x.shape[0], 1)), x]
        ).astype(np.float64)
        
    def f(self, x:np.ndarray)->np.ndarray:
        """
        x :
            A numpy ndarray for a set of $X$
            in row form
        
        will return x_@self.weight such that 
        x_ is x with expanded extra "ones" columns
        if the dimensions of x is N, 
        otherwise return x@self.weight directly.

        f(x) = w0*1 +  w1*x1 + w2*x2 + ... + wn*xn
        """
        if x.shape[1] == self.weight.shape[0]:
            return x@self.weight
        return self._expand_ones(x)@self.weight

    def makepoints(self, Xn_1:np.ndarray, concatation=False)->np.ndarray:
        """
        Will using the equation:
        x_n = - [(weight[0])*1 +(weight[1])*x_1 + ... + \
            weight[n-1]*x_(n-1) ] /weight[n]
        to make the points.

        Please notice the error of the floating point 
        operation.
        """
        Xn_1_exp = (Xn_1)
        if Xn_1.shape[1] == self.dim-1:
            """
            hasn't expand yet 
            """
            Xn_1_exp = self._expand_ones(Xn_1)
        Xn = -(Xn_1_exp @ (self.weight[:-1]))/self.weight[-1][0]
        if not concatation:
            return Xn
        else:
            if Xn_1.shape[1] == self.dim-1:
                return np.hstack([Xn_1,Xn])
            else:
                return np.hstack([Xn_1[:, 1:], Xn])

class LinearPerceptron(HyperPlane):

    def __init__(self, dim:int, w_init:np.ndarray=None) -> None:
        if w_init is not None:
            if w_init.shape[0] == dim+1:
                super().__init__(weight=w_init)
            else:
                print(
                    f"dimensions of initial weight \
                        {w_init.shape[0]} \
                        and wanted dimesions \
                        {dim}+1 are not eqaul."
                    )
                return
        else:
            super().__init__(weight=np.random.randn(dim+1,1))

    def __enhance_weight(self, target:np.ndarray, sign:int, debug=False)->np.ndarray:
        enhance = (sign*target).reshape(-1,1)
        _weight = self.weight+enhance
        if debug:
            print(f"enhance: {enhance}")
            print(f"after weight: {_weight}")
        if _weight[1][0] < 0:
            _weight = -_weight
        return _weight

    def train(
        self, x_:np.ndarray, ans:np.ndarray, method="navie",
        max_iter = 100000, log=sys.stdout, debug=False
    ):

        if x_.shape[0] != ans.shape[0]:
            print("ans is not as many as dataset")
            return
        if x_.shape[1] != self.dim:
            print(f"dimesion error,\
                this Perceptron is {self.dim}, \
                but x is {x_.shape[1]}"
            )
            return

        x = self._expand_ones(x_)
        iteration = 0 
        #tolerance = 0
        #max_tolerance = x.shape[0]
        print(f"{method}")
        least_mistakes = (self.predict(x) - ans).nonzero()[0]
        current_least = least_mistakes.size
        mistakerecords = [current_least]
        mistakeprovement = [current_least]
        w_index = 0
        s=time.time()

        while(iteration < max_iter):
    
            if least_mistakes.size == 0:
                print("\nOptimal for training data",file=log)
                break
            elif least_mistakes.size == x.shape[0]:
                print("\nOptimal for training data", file=log)
                self.weight = -self.weight
                break
        
            iteration += 1
            
            #amistake = np.random.choice(least_mistakes)
            amistake = least_mistakes[w_index]
            new_weight = self.__enhance_weight(
                ans[amistake][0],x[amistake] 
            )
            new_mistakes = (np.sign(x@new_weight) - ans).nonzero()[0]
            mistakerecords.append(new_mistakes.size)

            print(f"iteration:{iteration}, least:{least_mistakes.size}, new:{new_mistakes.size}|", end="\r")
            
            if method == "navie":
                """
                directly update the weight
                """
                least_mistakes = new_mistakes
                self.weight = new_weight
                if least_mistakes.shape[0] > 0:
                    w_index = np.random.randint(
                        low=0, high=least_mistakes.shape[0]
                    )
                
                print(end="\r")
                continue
            
            
            if new_mistakes.size < current_least:
                """
                if new weight can make fewer 
                mistake, update to that weight
                """
                least_mistakes = new_mistakes
                self.weight = new_weight
                w_index = 0

            else:
                #Not update the weight
                w_index += 1
                if w_index > least_mistakes.shape[0]-1:
                    """
                    if all the new wiehgt enhanced by
                    x that are iteration's mistakes term
                    can't reduce the mistake number,
                    the loop is done.
                    """
                    break
            
            current_least = least_mistakes.size
            mistakeprovement.append(current_least)
         
        e = time.time()
        training_metrics = self.metrics(x = x,ans=ans)
        print()
        return {
            "iterations":iteration, 
            "times":e-s,
            "metrics":training_metrics,
            "#mistake":mistakerecords,
            "mistake provement":mistakeprovement
        }
    
    
    def predict(self, x:np.ndarray)->np.ndarray:
        return np.sign(self.f(x))

    def metrics(self, x:np.ndarray, ans:np.ndarray=None):
        pred = self.predict(x)
        wrong = (pred - ans).nonzero()[0]
        correct = x.shape[0]-wrong.size
        acc = correct/x.shape[0]
        return {
            "accuracy":acc, 
            "#wrong":wrong.size,
            "wrongidx":wrong.tolist(), 
            "prediction":pred.tolist(),
        }


"""
    def __navie_train(
        self, x:np.ndarray, ans:np.ndarray,
        max_iter = 100000, log=sys.stdout,debug=False
    ):
    
        iteration = 0
        records = []
        while (iteration < max_iter):

            mistakes = (self.predict(x) - ans).nonzero()[0]
            records.append(mistakes.size)
            if mistakes.size == 0:
                print("Optimal for training data", file=log)
                break
            elif mistakes.size == x.shape[0]:

                #Find the opposition norm vector,
                #just need to reverse it and then 
                #can get the optimal model.
                 
                print("Optimal for training data", file=log)
                self.weight = -self.weight
                break

            iteration += 1
            amistake = np.random.choice(mistakes)
            self.weight = self.__enhance_weight(
                ans[amistake][0],x[amistake] , debug=debug
            )
            print(f"{iteration}",end="\r", file=log)
        print(f"iterations : {iteration}", file=log)

        return iteration, records
"""
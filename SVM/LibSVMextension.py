import numpy as np
from libsvm.svmutil import *
import itertools
import os
from tqdm import tqdm
import copy
from sklearn.datasets import dump_svmlight_file
import sys
import pandas as pd
import json 

STDOUT = sys.stdout

class LIBSVM_Dataset():
    
    def __init__(self, generator=None, data={}) -> None:
        """
        ## parameter:
        - generator:
        A function, have to return 
            a dict ```{'data':X,'label':Y}```
                - X, Y are np.ndarray
        
            if it is None, it means no need to generate and just
            using the exist file directly
        
        """
        self.__generator = generator
        self.__data:dict = data
        
    def gen(self,needreturn=False, **kwargs)->dict:
        
        """
        Please note that if self.__generator is None, 
        calling this member function is illegal.
        ## parameters:
            - needreturn :
                wether to return the libsvm data, defualt is False.
            
            - kwargs: A dict that 
                contains the parameters ```self.generator``` needs.
                
        Once the data is generated, since it is np.ndarray,
        it will direclty be saved to ```savepath``` in order to 
        get libsvm data format by using svm_read_problem from ```libsvm.svmutil```
        """
        
        if self.__generator is None:
            print("Error ! no generator is given")
            return None
        
        self.__data = self.__generator(**kwargs)
        self.__data['label'] = self.__data['label'].reshape(-1,)
  
        if needreturn:
            return self.get_data()

    def get_data(self, fromfile:str=None)->dict:
        
        """
        ## parameter:
            - fromfile:
                ```Warning!``` if it is set, then all the previous
                data will be replace by it.
        """
        
        if fromfile is None:
            if self.__data is None:
                print("Error !")
                print("Since no data has been generated, it needs files dir to load data")
                print("will return None")
                return None
        else:
            y, x = svm_read_problem(data_file_name=fromfile,return_scipy=True)
            self.__data = {'data':x.toarray(),'label':y}
        
        return copy.deepcopy(self.__data)
    
    def scaling(self,scaler, label=False):
        """
        ## parameter:
        - scaler:
            the sklearn scaler ```instance```.
        """
        scaling_d = self.get_data()
        
        scaling_d['data'] =scaler.fit_transform(scaling_d['data'])
        if label:
            scaling_d['label'] = scaler.transform(
                scaling_d['label'].reshape(-1,1)
            ).reshape(-1,)
        
        return LIBSVM_Dataset(data=scaling_d)
        
    @staticmethod
    def savedata(data, savepath)->str:
        """
        will return the file path
        """
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        
        svmdata = os.path.join(savepath, "data.svm")
        print(f"writting data to {svmdata} ..")
        
        with open(svmdata, "wb+") as f:
            dump_svmlight_file(
                X = data['data'], y = data['label'], f = f
            )
        
        return svmdata
   
def cross_validation(parameters:dict, data:LIBSVM_Dataset, saveroot=os.path.join("result"), k=5):
    
    print(saveroot)
    
    def grid_search(p:list):
        """
        Doing the cartesian product as Grid-search parameters
        """
        return np.array(
            list(list(x) for x in itertools.product(*p))
        )
    
    def make_arg(paranames, para, cv=k):
        p = ""
        if cv:
            p = f"-v {cv} -q "
        show = ""
        for _, pname in enumerate(paranames):
            p += f"-{pname} {para[_]:.16f}"
            show += f"{pname}:{para[_]}"
            if _ != len(para_order):
                p+=" "
                show += " "
        return p, show
        
    para_grid = grid_search(p=list(v for k,v in parameters.items()))
    para_order = list(parameters.keys())
    
    d = data.get_data()
    X=d['data']
    Y=d['label']
    prob = svm_problem(y = Y, x = X)
    accs = np.zeros((para_grid.shape[0],))
    log = open(os.path.join(saveroot,"cvlog.txt"), "w+")
    
    cvabar = tqdm(para_grid)
    for i, para in enumerate(cvabar):
        sys.stdout = log
        p, show = make_arg(paranames=para_order, para=para)
        print(p)
        parameter = svm_parameter(p)
        acci = svm_train(prob, parameter)/100
        print()
        accs[i] = acci
        sys.stdout=STDOUT
        cvabar.set_postfix_str(f"{show},'acc':{acci}")
    log.close()
    
    best = np.argmax(accs)
    print(f"CV best acc: {accs[best]}")
    cvrecords = pd.DataFrame(
        {'C':para_grid[:, 0],'gamma':para_grid[:, 1],'CV_acc':accs}
    )
    cvrecords.to_csv(
        os.path.join(saveroot,f"{k}-fold-cv_metrics.csv"),
        index=False
    )
    p, _ = make_arg(
        paranames=para_order, para = para_grid[best], cv=0
    )
   
    parameter = svm_parameter(p+" -q")
    model = svm_train(prob, parameter)
    train_result = svm_predict(y=Y, x=X, m=model)
    
    trainacc = train_result[1][0]
    print(f"training acc: {trainacc/100}")

    svm_save_model(os.path.join(saveroot,"model_svm"), model)
    with open(os.path.join(saveroot, "paras.txt"), "w+") as paraf:
        paraf.write(p)
    with open(os.path.join(saveroot, "acc.json"), "w+") as jf:
        json.dump(
            {'cv_best':accs[best], 'training':trainacc/100},
            jf, indent=4, ensure_ascii=False
        )
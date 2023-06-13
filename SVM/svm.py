import numpy as np
import os
from tqdm import tqdm
from LibSVMextension import cross_validation, LIBSVM_Dataset
from sklearn.preprocessing import MinMaxScaler,StandardScaler


def makedir(d):
    if not os.path.exists(d):
        os.mkdir(d)
    return d

def generate_data(region=(-100, 100), N=500, gaussian_noise_para=(0.0,1.0)):
    
    """
    y = 2x+eps~Normal(guassian_noise_para)
    """
    print("data information:")
    print(f"N: {N}, xregion:{region[0]}~{region[1]}", end= " ")
    print(f"N({gaussian_noise_para[0]}, {gaussian_noise_para[1]})")
    
    x = np.linspace(region[0], region[1], N, dtype=np.float64).reshape(-1,1)
    eps = np.random.normal(
        loc=gaussian_noise_para[0], 
        scale=gaussian_noise_para[1],
        size=x.shape
    )
    label = (eps >= 0).astype(np.int16)
    biased = 2*x+ eps
    return {'data':np.hstack((x,biased)),'label':label}

def main(saveroot, datafile=None):
    dataset = None
    if datafile is None:
        print("Generate new data to train")
        dataset = LIBSVM_Dataset(generator=generate_data)
        dataset.gen(needreturn=False)
    
        _ = LIBSVM_Dataset.savedata(
            data=dataset.get_data(),
            savepath=makedir(os.path.join(saveroot,"data"))
        )
    else:
        print(f"Use the pre-generated data from : {datafile}")
        dataset = LIBSVM_Dataset()
        dataset.get_data(fromfile=datafile)
    
    

    pc = np.arange(-5, 17, 2)
    pg = np.arange(-15, 5, 2)
    parameters={
        'c':np.power(np.ones(pc.shape[0])*2 ,pc),
        'g':np.power(np.ones(pg.shape[0])*2 ,pg)
    }
    cross_validation(
        parameters=parameters,
        data = dataset.scaling(scaler=StandardScaler(), label=False),
        saveroot=makedir(os.path.join(saveroot,"std"))
    )
    
    cross_validation(
        parameters=parameters,
        data = dataset, 
        saveroot= makedir(os.path.join(saveroot,"origin"))
    )

    cross_validation(
        parameters=parameters,
        data =dataset.scaling(scaler=MinMaxScaler((0,1)), label=False), 
        saveroot=makedir(os.path.join(saveroot,"scaling_01"))
    )
    
    
    
def Do_The_Report_data():
    saveroot = makedir(os.path.join("result"))
    datafile=os.path.join(saveroot, "data","data.svm")
    main(saveroot=saveroot, datafile=datafile)

def Try_all_program():
    saveroot = makedir(os.path.join("result1"))
    main(saveroot=saveroot)
    
if __name__ == "__main__":
    #Do_The_Report_data()
    
    Try_all_program()
    
import os 
import json 
import numpy as np
import matplotlib.pyplot as plt
from Perceptron import LinearPerceptron, HyperPlane


def savearray(a, savedir, savename)->None:
    np.save(os.path.join(savedir, savename), a)
    np.savetxt(
        os.path.join(savedir, f"{savename}.csv"), a, 
        delimiter=","
    )

def plot_change(a:list, names:list, title, saveplt=None, notebook=False):
    plt.figure(dpi=800)
    for idx, line in enumerate(a):
        e = list(i for i in range(len(line)))
        plt.plot(e, line, label = names[idx])
    plt.xlabel("epoch")
    plt.title(title)
    plt.legend()
    if saveplt is not None:
        plt.savefig(saveplt)
    if notebook:
        plt.show()
    else:
        plt.close()

def linear_2D_dataset(
    N:int, hp:HyperPlane, val_range:tuple=(0,1),
    alongwith_biased=True, flush=True, bias_amplitude = 1.0
)->tuple:
        X0 = np.linspace(
            start=val_range[0], stop=val_range[1], 
            num=N, dtype=np.float64
        ).reshape(-1, 1)
        X1 = hp.makepoints(X0)
        unbiased_dataset = np.hstack([X0, X1])
        if not alongwith_biased:
            return (unbiased_dataset, None)
    
        bias =  bias_amplitude * np.random.uniform(
            low = 0.1, high = 1.1,size=X0.shape
        )
        random_indices = np.random.permutation(
            np.array(list(i for i in range(X0.shape[0])))
        )
    
        leftpart = random_indices[:N//2]
        rightpart = random_indices[N//2:]
        Xleft = X0[leftpart] - bias[:N//2]
        Xright = X0[rightpart] + bias[N//2:]
        X1left = X1[leftpart]
        X1right = X1[rightpart]

        neg = np.hstack([Xleft, X1left])
        pos = np.hstack([Xright, X1right])
        X = np.vstack([neg, pos])
        Y = np.vstack([-np.ones((neg.shape[0],1)), np.ones((pos.shape[0],1))])
        
        if flush:
            flush_index = np.random.permutation(X.shape[0])
            X = X[flush_index]
            Y = Y[flush_index]

        return (
            unbiased_dataset, 
            {
                "negative":neg, 
                "positive":pos
            }, 
            {
                "X":X,
                "Y":Y
            }
        )

def makepath(root, subdir=[])->dict:
    if root is None:
        return None
    if not os.path.exists(root):
        os.mkdir(root)
    ret = {}
    for subd in subdir:
        d = os.path.join(root, subd)
        if not os.path.exists(d):
            os.mkdir(d)
        ret[f"{subd}"]=d
    return ret

def plot_2D_hp(data:np.ndarray, lines:dict,saveplt=None, notebook=False):

    plt.figure(dpi = 800)
    for k, v in lines.items():
        plt.plot(
            v['point'][:, 0], v['point'][:, 1],
            linewidth=0.5, label=k, c=v['color']
        )

    plt.scatter(
        data['negative'][:, 0],data['negative'][:, 1],
        c = 'blue', marker='^', s=1, label="negative",
        alpha=0.3
    )

    plt.scatter(
        data['positive'][:, 0],data['positive'][:, 1],
        c = 'red', marker=',', s=1, label="postive",
        alpha=0.3 
    )

    plt.legend()
    
    if saveplt is not None:
        plt.savefig(saveplt)
    
    if notebook :
        plt.show()
    else:
        plt.close()

def PLA2D(
    groundtruth:np.ndarray, N=30, 
    sample_region=(0,1), save_path:os.PathLike=None,
    training_method="navie"
)->tuple:
    """
    hypothesis_line:np.ndarray, gth_line:np.ndarray,
        optimal_line:np.ndarray=None,
    """
    pathes = makepath(save_path, subdir=["model", "data"])
   
    gth_line = LinearPerceptron(dim = 2, w_init=groundtruth)
    gth_line_samples, biased_data, gened_dataset = \
    linear_2D_dataset(
        N, hp=gth_line, val_range=sample_region
    )
    X = gened_dataset['X']
    Y = gened_dataset['Y']

    hypo_line =  LinearPerceptron(dim=2)

    linesamples ={}
    
    if save_path is not None:
        """
        Saving the gth, init_model and dataset

        both gth and hypo model having a .npy version 
        and .txt version
        """
        print("Saving init ... ")
        savearray(np.hstack([X,Y]), pathes['data'],"data")
        savearray(gth_line.weight, pathes['model'], "gth_weight")
        savearray(hypo_line.weight, pathes['model'], "init_weight")
        
        linesamples['gth'] = {
            'point':gth_line_samples,'color':'green'
        }
        linesamples['hypo']={
            'point':linear_2D_dataset(
                N, hp=hypo_line, val_range=sample_region,
                alongwith_biased=False
            )[0], 
            'color':'darkorange'
        }
        
        plot_2D_hp(
            data=biased_data, lines=linesamples,
            saveplt=os.path.join(pathes['model'],"init_status.jpg")
        )
        print("OK ..")
    
    training_detail = hypo_line.train(
        x_=X, ans=Y, max_iter=np.inf,
        method=training_method
    )

    if save_path is not None:
        print("Saving result .. ")
        savearray(hypo_line.weight, pathes['model'], "optimal_weight")
        
        with open(os.path.join(pathes['model'],"detail.json"), "w+") as jf:
            json.dump(training_detail, jf,ensure_ascii=False,indent=4)
        
        linesamples['optimal'] = {
            'point':linear_2D_dataset(
                    N, hp=hypo_line, val_range=sample_region, 
                    alongwith_biased=False
                )[0],
            'color':'purple'
        }

        plot_2D_hp(
            data=biased_data, lines=linesamples,
            saveplt=os.path.join(pathes['model'],"optimal_status.jpg")
        )
        plot_change(
            a = [training_detail['#mistake']],
            names=["mistake"] ,title="mistake",
            saveplt=os.path.join(pathes['model'],"num_mistakes.jpg")
        )
        print("OK")
    
    print(
        f"accuracy : {training_detail['metrics']['accuracy']*100} %"
    )
    return gened_dataset, hypo_line

def Comparasion_PLA_Pocket(
    groundtruth:np.ndarray, N = 2000, sample_region=(-20.0, 20.0),
    save_path:os.PathLike=None
):
    pathes = makepath(save_path, subdir=["model", "data"])
   
    gth_line = LinearPerceptron(dim = 2, w_init=groundtruth)
    gth_line_samples, biased_data, gened_dataset \
    = linear_2D_dataset(
        N, hp=gth_line, val_range=sample_region,
        bias_amplitude=5.0
    )
    X = gened_dataset['X']
    Y = gened_dataset['Y']

    hypo_line = LinearPerceptron(dim = 2)
    hypo_line2 = LinearPerceptron(dim = 2, w_init=hypo_line.weight)
    hypo_line2 = LinearPerceptron(dim = 2, w_init=hypo_line.weight)
    linesamples = {}

    if save_path is not None:
        """
        Saving the gth, init_model and dataset

        both gth and hypo model having a .npy version 
        and .txt version
        """
        print("Saving init ... ")
        savearray(np.hstack([X,Y]), pathes['data'],"data")
        savearray(gth_line.weight, pathes['model'], "gth_weight")
        savearray(hypo_line.weight, pathes['model'], "init_weight")
        
        linesamples['gth'] = {
            'point':gth_line_samples,'color':'green'
        }
        linesamples['hypo']={
            'point':linear_2D_dataset(
                N, hp=hypo_line, val_range=sample_region,
                alongwith_biased=False
            )[0], 
            'color':'darkorange'
        }
        
        plot_2D_hp(
            data=biased_data, lines=linesamples,
            saveplt=os.path.join(pathes['model'],"init_status.jpg")
        )
        print("OK ..")
    navie_training_detail= hypo_line2.train(
        x_=X, ans=Y, max_iter=np.inf, 
        method="navie"
    )
    pocket_training_detail = hypo_line.train(
        x_=X, ans=Y, max_iter=np.inf,
        method="pocket"
    )

    if save_path is not None:
        print("Saving result .. ")
        savearray(hypo_line.weight, pathes['model'], "pocket_optimal_weight")
        savearray(hypo_line2.weight, pathes['model'], "navie_optimal_weight")
        
        with open(os.path.join(pathes['model'],"pocket_detail.json"), "w+") as jf:
            json.dump(pocket_training_detail, jf,ensure_ascii=False,indent=4)
        
        with open(os.path.join(pathes['model'],"navie_detail.json"), "w+") as jf:
            json.dump(navie_training_detail, jf,ensure_ascii=False,indent=4)
        

        linesamples['pocket'] = {
            'point':linear_2D_dataset(
                    N, hp=hypo_line, val_range=sample_region, 
                    alongwith_biased=False
                )[0],
            'color':'purple'
        }
        linesamples['navie'] = {
            'point':linear_2D_dataset(
                    N, hp=hypo_line2, val_range=sample_region, 
                    alongwith_biased=False
                )[0],
            'color':'cyan'
        }

        plot_2D_hp(
            data=biased_data, lines=linesamples,
            saveplt=os.path.join(pathes['model'],"optimal_status.jpg")
        )
        
        plot_change(
            a = [
            pocket_training_detail['#mistake'],
            pocket_training_detail['mistake provement']
            ],names=["mistake","provement"] ,title="pocket_mistake",
            saveplt=os.path.join(pathes['model'],"num_mistakes.jpg")
        )

        print("OK")
    
    print(f"accuracy : {pocket_training_detail['metrics']['accuracy']*100} %")
    return gened_dataset, hypo_line


def PLA30point(resultdir):
    print("30 points 3 set")
    if not os.path.exists(resultdir):
        os.mkdir(resultdir)
    
    data1 ,plav1 = PLA2D(
        groundtruth=np.array([[1.5],[-0.3],[-1]]),
        save_path=os.path.join(resultdir,"v1")
    )
    
    data2, plav2 = PLA2D(
        groundtruth=np.array([[0.5],[-0.62],[0.14]]),
        save_path=os.path.join(resultdir,"v2")
    )

    data3, plav3 = PLA2D(
        groundtruth=np.array([[-0.8],[0.52],[-0.04]]),
        save_path=os.path.join(resultdir,"v3")
    )

def Comparation1000point(resultdir):
    if not os.path.exists(resultdir):
        os.mkdir(resultdir)
    gendata, pla = Comparasion_PLA_Pocket(
        groundtruth=np.array([[0.5],[-0.62],[0.14]]),
        save_path=resultdir
    )

def mislabeling(dataset, init_weight, gth, savedir):
    X = dataset[:, :-1]
    Y = dataset[:, -1].reshape(-1,1)
    pathes = makepath(root=savedir,subdir=["model", "biaseddata"])
    
    postive_index = np.where(Y == 1)[0]
    negative_index = np.where(Y == -1)[0]

    mis_pos_idx = np.random.choice(postive_index, size=50,replace=False)
    mis_neg_idx = np.random.choice(negative_index, size=50, replace=False)
    misY = np.copy(Y)
    misY[mis_pos_idx] = -1
    misY[mis_neg_idx] = 1
    flush = np.random.permutation(X.shape[0])
    biased_perc = LinearPerceptron(dim=2, w_init=init_weight)
    detail = biased_perc.train(
        x_ = X[flush], ans=misY[flush], method="pocket",
        max_iter=np.inf
    )
    print((Y-misY).nonzero()[0].size)
    true_metrices = biased_perc.metrics(x=X, ans=Y)
    detail['True'] = true_metrices

    if savedir is not None:
        linesamples = {
            'gth':{
                'point':linear_2D_dataset(
                    X.shape[0], hp=HyperPlane(weight=gth), 
                    val_range=(-20.0, 20.0), 
                    alongwith_biased=False
                )[0],
                'color':'green'
            },
            'hypo':{
                'point':linear_2D_dataset(
                    X.shape[0], hp=HyperPlane(init_weight), 
                    val_range=(-20.0, 20.0), 
                    alongwith_biased=False
                )[0],
                'color':'darkorange'
            },

            'biased pocket':{
                'point':linear_2D_dataset(
                    X.shape[0], hp=biased_perc, 
                    val_range=(-20.0, 20.0), 
                    alongwith_biased=False
                )[0],
                'color':'purple'
            }
        }
        savearray(np.hstack([X, misY]), pathes['biaseddata'], "mislabel")
        savearray(biased_perc.weight, pathes['model'],"biased_pocket")
        with open(os.path.join(pathes['model'], "pocket_detail.json"), "w+") as jf:
            json.dump(detail, jf, indent=4, ensure_ascii=False)

        plot_2D_hp(
            data={
                "positive":dataset[np.where(misY==1)[0]], 
                "negative":dataset[np.where(misY==-1)[0]]
            }, 
            lines=linesamples,
            saveplt=os.path.join(pathes['model'],"optimal_status.jpg")
        )

        plot_change(
            a = [detail['#mistake'],detail['mistake provement']
            ],names=["mistake","provement"] ,title="pocket_mistake",
            saveplt=os.path.join(pathes['model'],"num_mistakes.jpg")
        )

    


if __name__ == "__main__":
    
    #PLA30point(resultdir = os.path.join("PLA30_1"))
    
    #Comparation1000point(resultdir = os.path.join("Pocket1000"))


    dataset = np.load(os.path.join("Pocket1000", "data","data.npy"))
    
    gth = np.load(os.path.join("Pocket1000", "model","gth_weight.npy"))
    w = np.load(os.path.join("Pocket1000", "model","init_weight.npy"))
    mislabeling(
        dataset=dataset, gth=gth,
        init_weight=w, 
        savedir=os.path.join("Pocket1000","mislabel")
    )

    

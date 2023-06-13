# Assignment 4 SVM by using libsvm

资工三 408410086 常博爱 

## 如何執行:

下命令: ```python ./svm.py``` 即可

### 所需要的第三方套件:

- sklearn
- numpy 
- sicpy
- pandas 
- libsvm 
  - ```pip install -U libsvm-official```
      - from https://www.csie.ntu.edu.tw/~cjlin/libsvm/

## 訓練

附在 ```./result/``` 裡面

### 資料
產生方式依照要求為 $y = 2x+\epsilon, x \in [-100,100], \epsilon\sim N(0,1)$ 。500個 x； y則是依據 $\epsilon_i$ 的正負 : 正或0的為 label 1，負的就為 label 1。

5-fold cross validation : 使用 
```libsvm.svmutil.train_svm``` 增加 options ```-v 5``` 來使用他提供的 cross validation 的功能。

Grid search 的參數:
- $c: 2^{-5}, 2^{-3}, ... , 2^{17}$
- $\gamma : 2^{-15}, 2^{-13}, ... ,2^{5}$

### Scaling
- 沒有 scaling :
  - Best $\{c,\gamma\} =\{2^{15}, 2^{-15}\} = \{32768, 3.0517578125\times10^{-5}\}$ 
  - Execution time: CV: 27s
  - |method|accuracy|
    |-|-|
    |5-fold cv|97.6%|
    |whole data|98.8%|

- Normalize to $[0,1]$:
  - $\{c,\gamma\} =\{2^{15}, 2\} = \{32768, 2\}$  
  - Execution time: CV: 13s
  - |method|accuracy|
    |-|-|
    |5-fold cv|98.2%|
    |whole data|99%|

- Standardize to $(\mu, \sigma)=(0,1)$
  - Best $\{c,\gamma\} =\{2^{11}, 2^{-1}\} = \{2048, 0.5\}$ 
  - Execution time: 19s
  - |method|accuracy|
    |-|-|
    |5-fold cv|97.6%|
    |whole data|98%|


## 討論: 
依據我的實驗結過，只要有適當的參數，好像其實以上兩種 scaling 方法跟原始資料所能達到的 accuracy 差不多。不過 Normalize 後的 accuracy 有稍微好一些。 

可以看到 $\gamma$ 再經過兩種 scaling 後，有出現比較有意義的值，不會像原始資料直接丟進去後跑出來幾乎為 0。可能是因為 $\gamma$ 作用在 $exp$ 上，所以資料值太大的話會迫使 $\gamma$ 縮小來去 fit y。可以看出使用 normalize / standardize 後，數值在一個較為有限制的範圍，會 trian 出叫有意義的參數，我想這也可能是 Normalize 後的 accuracy 有稍微好一些的原因之一 。

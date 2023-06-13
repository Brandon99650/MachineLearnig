# Perceptron Learning Algorithm for 2 dimensions data
資工四408410086 常博爱

第三方套件需求:
- numpy 
- matplotlib


執行:
- 下命令 ```python pla2d.py``` 即可

其結果會寫在工作資料夾底下

我有把報告中所用的該次的實驗數據放在資料夾這個裡面。

## Q1 生成資料集的方法:
給定直線 

$$ \begin{bmatrix} 1 & x_1 & x_2 \end{bmatrix} \times \begin{bmatrix} 
w_0 \\  
w_1 \\  
w_2
\end{bmatrix} = 0 $$

後，將 $(0, 1)$ 分成所要資料個數的N個分點 ( ```使用np.linespace()``` )，視為   $X_1$ ，並將 $X_2$ 用上述方程式算出來，成為 unbiased 的資料集。

之後，在使用 ```np.random.uniform(low = 0.1, high = 1.1, (N,1))``` 生成一系列偽隨機數當作 bias。其中，```low``` 設為0.1是為了防止產生 0 。

把 $X_1$ 一半的部分減去一半的bias，那些 $(x_i-\text{bias}_i, y_i)$ 在直線左邊，可以用來當作 negative samples ；剩下的一半加到另一半的 $X_1$， $(x_j+\text{bias}_j, y_j)$  在直線的右半邊，可以用來當作 positive samples

## Q2:
### 實驗結果

訓練 3 次以不同直線生成的資料集。

weight 的數據四捨五入到小數點第3位；

視覺化圖表中:
- gth (綠色) : 用來生成資料集的直線
- hypo (橘色) : 一開始的隨機指定weight所形成的直線
- optimal (紫色) : PLA演算法從hypo開始，找到的最佳直線

結果如下: 
#### 第一次: 
  - 使用直線: 
  
$$ \begin{bmatrix} 1 & x_1 & x_2 \end{bmatrix} \times \begin{bmatrix}
  1.5 \\
  -0.3 \\
  -1
\end{bmatrix} = 0 $$

$$x_2 = 0.3x_1 + 1.5$$

  - iterations: 5
  - 程式執行結果:
  
$$\begin{bmatrix} 1 & x_1 &x_2\end{bmatrix}\times
\begin{bmatrix}
  -1.190 \\
  3.330 \\
  -0.565
\end{bmatrix} = 0$$

$$x_2 = -2.106x_1 + 5.894$$

視覺化:<img src="PLA30\v1\model\optimal_status.jpg">

#### 第二次: 
  - 使用直線: 
  
$$\begin{bmatrix}1 &x_1 &x_2\end{bmatrix}\times
    \begin{bmatrix}
      0.5 \\
      -6.2 \\
      0.14\end{bmatrix} = 0$$

$$x_2 = 4.429x_1 + (-3.571)$$

  - iterations: 30
  
  - 程式執行結果:
  
  $$ \begin{bmatrix}1 &x_1 & x_2\end{bmatrix}\times
     \begin{bmatrix}
      -6.579 \\
      8.522 \\
      -1.083
      \end{bmatrix} = 0 $$

$$x_2 = 7.872x_1 + (-6.077)$$


視覺化:<img src="PLA30\v2\model\optimal_status.jpg">

#### 第三次: 
  - 使用直線: 
  
$$\begin{bmatrix}1 &x_1 &x_2\end{bmatrix}\times
    \begin{bmatrix}
      -0.8\\
      0.52\\
      -0.04
    \end{bmatrix} = 0$$

$$x_2 = 13x_1 + (-20)$$

  - iterations: 276
  
  - 程式執行結果:
  
$$\begin{bmatrix}1 &x_1 &x_2\end{bmatrix}\times
     \begin{bmatrix}
      -63.892\\
      44.052\\
      -3.161
      \end{bmatrix} = 0$$

$$x_2 = 13.935x_1 + (-20.211)$$


視覺化:<img src="PLA30\v3\model\optimal_status.jpg">


平均迭代次數: $(5+30+276)/3 \approx 104$

## Q1 & Q2 討論:

除了上述這3次，我也有使用相同的那3個groundtrutht生成資料集，只是一開始的hypothesis line 不同跑其他更多的實驗次數。

而平均所需要的迭代次數是100 ~ 1000 左右，不過起伏很大，有50幾以內，也有到8、900甚至破千次，也有幾次是跑不完的。

跑不完得情況有可能是我演算法寫錯了；或是我目前能想到的有可能有幾個樣本離groundtruth line很近，導致有可能因浮點數運算誤差而持續震盪而不收斂。

而看到 實驗一 的結果，找到的最佳化直線雖然說能夠分出訓練資料集，但斜率甚至跟groundtruth 相反 (一個+ 一個 - )。我想會有這樣的狀況，是給的樣本的值域太小 ($x_0$才 0 ~ 1而已)，才會發生就算連斜率都不對，但卻可以分出樣本的情況。

透過這個例子，我更加了解樣本蒐集的廣泛性之重要程度，如果蒐集到的樣本有侷限性，或許能在training data 上表現的好，但拿到真實世界可能會錯的一塌糊塗。

## Q3
資料集的生成大致上和前面相同，只是由於各有1000個點(共2000個sample)，所以 $x_1$ 的範圍調整為 $(-20.0, 20.0)$；而在加/減 bias 之前，會先把從 unifrom 得到的數值都 $\times 5$ 在加/減到 $x_1$，讓樣本間隔較寬。

另外，在使用pocket algorithm更新權重的時候，我將每次都隨機選改成從此次錯誤第一個的 $x_{\text{wrong}_0}$ 開始，如果選到的前一個不能改進，就試試它的下一個 $x_{\text{wrong}_1}$，直到這個 weight 都不能被此次犯錯的$x_i$ 改進為止就結束，所以理論上執行速度會比原本的版本快一點。

### 實驗結果
- 使用直線: 

$$ \begin{bmatrix}1 &x_1 &x_2\end{bmatrix}\times
    \begin{bmatrix}
      -0.5\\
      0.62\\
      -0.14
      \end{bmatrix} = 0 $$

$$x_2 = 4.429x_1 + (-3.571)$$

- 訓練相關數據:
    
    |method|iterations|times|training accuracy|
    |-|-|-|-|
    |navie PLA|747|0.633 s|100 %|
    |pocket|854|0.274 s|86 %|

程式執行結果:

__navie PLA__ :

$$\begin{bmatrix}1 &x_1 &x_2\end{bmatrix}\times
      \begin{bmatrix}
        -333.685 \\
        1078.664 \\
        -243.472
      \end{bmatrix} = 0$$
    
  $$x_2 = 4.430x_1 + (-1.371)$$

__pocket__ :
  
  $$\begin{bmatrix}1 &x_1 &x_2\end{bmatrix}\times
      \begin{bmatrix}
        -1.685 \\
        4.812 \\
        -0.877
      \end{bmatrix} = 0$$

  $$x_2 = 5.484x_1 + (-1.921)$$


視覺化:<img src="Pocket1000\model\optimal_status.jpg">
(亮藍色的為使用原始PLA之結果)

## Q4
使用與 Q3 相同的初始權重與資料，並將該資料50個正樣本標為負；50個負樣本標為正，使用pocket algorithm 進行訓練。

### 實驗結果:
- 程式執行結果
  
$$\begin{bmatrix}1 &x_1 &x_2\end{bmatrix}\times
  \begin{bmatrix} 
    0.315\\
    2.442\\
    -0.488
  \end{bmatrix} = 0$$

$$x_2 = 5.004x_1 + 0.645$$

- Accuracy
  |training with mislabel dataset|Actual|
  |-|-|
  |86.9%|91.1%|

視覺化:<img src="Pocket1000\mislabel\model\optimal_status.jpg">


## Q3 & Q4 討論:

以下是Q3 執行時，不同的iteraiton 其 mistakes 的數量:
<img src="Pocket1000\mislabel\model\num_mistakes.jpg">
藍色為每一次嘗試修正weight後所產的錯誤數量；橘色為截至目前這個iteration時，最少犯錯的數量。

令我訝異的是即使資料線性可分割，使用pocket algorithm 還是無法達到trainig accuracy 100%。

除了上述的結果，我還有跑其他次，跑出來的accuracy 約在50% ~ 80% 左右，其實不是很理想。也或許是我寫的 pocket algorithm 演算法有誤。

我想這應該是上課講到的如果資料是linear speratable，則PLA「每一次的改進都是更好的」(p.27)，所以在這個資料集就是linear speratable 的情況，Pocket algorithm 只看是否能減少錯誤來決定是否要更新其實比一般的PLA還不好。

不過，真實世界不太可能所有資料都是linear speratable，所以pocket algorithm 還是有其意義在的。

而在 Q4 的實驗，還蠻訝異真實的accuracy還比training accuracy 還要高，目前我還想不到可能原因。
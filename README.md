Piacherski Maksim, 556241

# ML labs

In this repository you can find a realization for different ML problems which is a practice part of the Machine Learning class.

## Lab1: Image classification
2 approaches of the image classification problem' solving were chosen to take a hands on this work: _Linear Regression_ was implemnted with the help of Sklearn lib, and _Neural Network_ with the help of Tensorflow&Keras resource.

Classifier type can be chosen by passing an arg to `classify("nn") | classify("")` function. Linear regression will be chosen by default. 

Images are displayed with the help of PIL library. 

The big amount of the execution time is taking by the `compute_hashes and remove_hashes` functions, they offer us an ability to check the dataset for the dupes by calculating hash number for the content of each file, get rid of the empty files, and remove them from the dataset.

Training set size is about ~200000 examples, all data are flattened and converted to the grayscale range of shades.

File `finalized_model.sav` has saved `LinearRegression` model.
### Task:
1. Download data and show a few examples with the help of the python.
2. Check if data is balanced.
3. Divide data into 3 datasets: train, test, validation
4. Get rid of duplicates
5. Build a classifier

### Learning curves:
<img width="409" alt="Screenshot 2022-11-03 at 17 47 30" src="https://user-images.githubusercontent.com/43992068/199783049-0aba818f-f75a-4257-be47-0045039c96d5.png">


## Lab2: Image classification (DNN)
In this stage a deep neural network was designed and implemented to solve image classification task with increased accuracy. Designed netowrk consists of 3 inner layers alongside with some utilitary steps. A different layer configurations were used to achieve the highest possible classification precision. Inner layers utilize **ReLU** (rectified linear unit) activation function.

### Learning curves:
<img width="627" alt="Screenshot 2022-11-03 at 16 40 24" src="https://user-images.githubusercontent.com/43992068/199783007-9b851e65-75a7-43ec-b1b0-515f5a2ccb72.png">

**Result:** average validation precision score with training dataset of 200.000 items is approx 10% higher than in task #1.

On the next stage a dropout layer was introduced to mitigate risks of network overfitting. Network was trained and validated with different combination of 1 to 3 dropout layers using various parameters.

**Result:** neural network with introduced dropout layer did not show significant increase in classification precision – the result was improved by 1-1.5% of validation accuracy. This is possible because training dataset is not too big to introduce overfitting of the neural network.

On the last stage of the task a variable learning rate was introduced. This change did not show any significant impact on the overall classification precision. During this stage we could notice, that accuracy value wasn't increasing so fast, the range was about 0.0015 - 0.003 for an epoch.

**Result:** the highest achieved classification precision with training dataset of 200.000 items is 91% and traingng accuracy is near is 95%.

### Learning curves:
![Screenshot 2022-11-09 at 23 18 47](https://user-images.githubusercontent.com/43992068/201393660-262548fc-7cc4-4bd1-9095-7e795bf84102.png)

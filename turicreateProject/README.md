## Training Output

~~~python
+------------------+--------------+------------------+
| Images Processed | Elapsed Time | Percent Complete |
+------------------+--------------+------------------+
| 63               | 5.82s        | 100%             |
+------------------+--------------+------------------+
Logistic regression:
--------------------------------------------------------
Number of examples          : 63
Number of classes           : 4
Number of feature columns   : 1
Number of unpacked features : 2048
Number of coefficients      : 6147
Starting L-BFGS
--------------------------------------------------------
+-----------+----------+-----------+--------------+-------------------+
| Iteration | Passes   | Step size | Elapsed Time | Training Accuracy |
+-----------+----------+-----------+--------------+-------------------+
| 0         | 5        | 0.039852  | 0.214507     | 0.301587          |
| 1         | 9        | 0.836884  | 0.384588     | 0.507937          |
| 2         | 10       | 1.000000  | 0.473797     | 0.920635          |
| 3         | 11       | 1.000000  | 0.546340     | 0.968254          |
| 4         | 12       | 1.000000  | 0.615277     | 1.000000          |
| 9         | 18       | 1.000000  | 0.973880     | 1.000000          |
+-----------+----------+-----------+--------------+-------------------+
Analyzing and extracting image features.
+------------------+--------------+------------------+
| Images Processed | Elapsed Time | Percent Complete |
+------------------+--------------+------------------+
| 21               | 1.96s        | 100%             |
+------------------+--------------+------------------+

Test Accuracy:
0.7619047619047619

~~~


 
## Data Pipeline
The images I have come from reddit and stockx. I have data for jordan `one, three, four, eleven` (these are models of jordans). I took a screenshot of each shoe at different angles. I have about an average of 150 for each jordan. Some are higher quality while some are blurry, some are being worn, some are just shoes, some have 2 shoes in one photo, different angles (ie. back, front, left, right), etc. I also have a dataset with all the same images but with a filter applied to them in `filteredDataset` which I then created a model using those images to see if this would get me better results. All classes have 150 images, some 160. Here are some examples of the images I have.
 
![alt example](./dataset/jordan_one/readMeImg.png)
![alt example](./dataset/jordan_one/readMeImgEx.png)
 
I knew I needed to increase my amount of data so I created a file called 'addFilterToImages.py'. In this file I run all all the images in the folder `dataset` through a function that applies a filters to them. Each photo would get 5 filters applied to them (flipped vertically, flipped horizontally, black and white, saturated, blurred). Once I did this I ended up with about 900 images for each class. Here are some examples:
 
![alt example](./expandedDataset/jordan_eleven/1jordanElevenRed.png)
![alt example](./expandedDataset/jordan_eleven/2jordanElevenRed.png)
![alt example](./expandedDataset/jordan_eleven/3jordanElevenRed.png)
![alt example](./expandedDataset/jordan_eleven/4jordanElevenRed.png)
![alt example](./expandedDataset/jordan_eleven/5jordanElevenRed.png)
![alt example](./expandedDataset/jordan_eleven/6jordanElevenRed.png)
 
 
I then tried to run my model again and see if test accuracy would change but there was little to know change. My model was performing best when my classes were slightly disbalanced. This could mean my model was predicting a certain class more often because of my dataset and there for getting high test accuracy. I can try and prove this by creating a model with a very disbalanced dataset and see how that turns out.
 
 
I created a model using turi create with images that I also found and took myself. A pipeline that would be good to have would be to have a brand identification then once we identify the brand we can send it to another model that specifies a shoe number. When testing out my models with images I took of my own shoes the most common mistake that would happen is labeling any shoe as a `jordan_eleven`
 
 
## Evaluating
I tested my model with images I took of my own shoes. Overall it predicted the shoe well. One thing I did realize is that it's more likely to predict a shoe over another. The predictions can be seen in `modelEval.ipynb`
 



~~~python 
Analyzing and extracting image features.
+------------------+--------------+------------------+
| Images Processed | Elapsed Time | Percent Complete |
+------------------+--------------+------------------+
| 64               | 8.62s        | 2%               |
| 128              | 16.58s       | 4.25%            |
| 192              | 25.14s       | 6.25%            |
| 256              | 33.85s       | 8.5%             |
| 320              | 45.22s       | 10.5%            |
| 448              | 1m 6s        | 14.75%           |
| 512              | 1m 17s       | 17%              |
| 576              | 1m 28s       | 19%              |
| 640              | 1m 38s       | 21.25%           |
| 704              | 1m 50s       | 23.25%           |
| 768              | 2m 0s        | 25.5%            |
| 832              | 2m 10s       | 27.5%            |
| 896              | 2m 20s       | 29.75%           |
| 960              | 2m 30s       | 31.75%           |
| 1024             | 2m 40s       | 34%              |
| 1088             | 2m 50s       | 36%              |
| 1152             | 3m 0s        | 38.25%           |
| 1216             | 3m 11s       | 40.25%           |
| 1280             | 3m 17s       | 42.5%            |
| 1344             | 3m 24s       | 44.5%            |
| 1408             | 3m 31s       | 46.75%           |
| 1472             | 3m 38s       | 48.75%           |
| 1536             | 3m 45s       | 51%              |
| 1600             | 3m 51s       | 53%              |
| 1664             | 3m 58s       | 55.25%           |
| 1728             | 4m 5s        | 57.25%           |
| 1792             | 4m 12s       | 59.5%            |
| 1856             | 4m 18s       | 61.5%            |
| 1920             | 4m 25s       | 63.75%           |
| 1984             | 4m 32s       | 65.75%           |
| 2048             | 4m 39s       | 68%              |
| 2112             | 4m 45s       | 70%              |
| 2176             | 4m 52s       | 72.25%           |
| 2240             | 4m 59s       | 74.25%           |
| 2304             | 5m 6s        | 76.5%            |
| 2368             | 5m 12s       | 78.5%            |
| 2432             | 5m 19s       | 80.75%           |
| 2496             | 5m 26s       | 82.75%           |
| 2560             | 5m 33s       | 85%              |
| 2624             | 5m 39s       | 87%              |
| 2688             | 5m 46s       | 89.25%           |
| 2752             | 5m 53s       | 91.25%           |
| 2816             | 5m 59s       | 93.5%            |
| 2880             | 6m 6s        | 95.5%            |
| 2944             | 6m 11s       | 97.75%           |
| 2985             | 6m 17s       | 100%             |
+------------------+--------------+------------------+
PROGRESS: Creating a validation set from 5 percent of training data. This may take a while.
          You can set ``validation_set=None`` to disable validation tracking.

Logistic regression:
--------------------------------------------------------
Number of examples          : 2835
Number of classes           : 4
Number of feature columns   : 1
Number of unpacked features : 2048
Number of coefficients      : 6147
Starting L-BFGS
--------------------------------------------------------
+-----------+----------+-----------+--------------+-------------------+---------------------+
| Iteration | Passes   | Step size | Elapsed Time | Training Accuracy | Validation Accuracy |
+-----------+----------+-----------+--------------+-------------------+---------------------+
| 0         | 5        | 0.105322  | 0.910528     | 0.456790          | 0.440000            |
| 1         | 9        | 2.211769  | 1.802593     | 0.557319          | 0.480000            |
| 2         | 14       | 1.770433  | 2.860240     | 0.600705          | 0.540000            |
| 3         | 15       | 1.770433  | 3.257373     | 0.681129          | 0.606667            |
| 4         | 21       | 0.536856  | 4.520679     | 0.738977          | 0.686667            |
| 9         | 28       | 1.000000  | 6.896338     | 0.832099          | 0.746667            |
+-----------+----------+-----------+--------------+-------------------+---------------------+
Analyzing and extracting image features.
+------------------+--------------+------------------+
| Images Processed | Elapsed Time | Percent Complete |
+------------------+--------------+------------------+
| 64               | 6.55s        | 7.5%             |
| 128              | 13.39s       | 15.25%           |
| 192              | 20.14s       | 23%              |
| 256              | 26.98s       | 30.75%           |
| 320              | 33.49s       | 38.25%           |
| 448              | 47.16s       | 53.75%           |
| 512              | 53.61s       | 61.5%            |
| 576              | 1m 0s        | 69%              |
| 640              | 1m 7s        | 76.75%           |
| 704              | 1m 13s       | 84.5%            |
| 768              | 1m 20s       | 92.25%           |
| 771              | 1m 20s       | 100%             |
+------------------+--------------+------------------+
0.7600518806744487
~~~
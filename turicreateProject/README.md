 ## Evaluating
I tested my model with images I took of my own shoes. Overall it predicted the shoe well. One thing I did realize is that it's more likely to predict a shoe over another. The predictions can be seen in `modelEval.ipynb`



~~~python 

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
Test Accuracy:
0.7600518806744487
~~~
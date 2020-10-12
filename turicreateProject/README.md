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



## Where the Dataset Came From
The images come from reddit and stockx. I have data for jordan `one,three,four,eleven` (these are models of jordans). I took a screenshot of each shoe some at different angles. I have about an average of 130 for each jordan. Some are high quality some are blury, some are being worn, some are just shoes, some have 2 shoes in one photo, different angles (ie. back, front, left, right). I also have a dataset with all the same images but with a filter applied to them in `filteredDataset`. I also created a model using those images. All classes have 150 images some 160. Here are some examples of the images I have.

![alt example](./dataset/jordan_one/readMeImg.png)
![alt example](./dataset/jordan_one/readMeImgEx.png)


## Data Pipeline 
I create a model using turicreate with iamges that I also found and took myself. A pipeline that would be good to have would be to have a brand identification then once we identify the brand we can send it to another model that specifies in shoe number. When testing out my models with images I took of my own shoes the most common mistake that would happen is labeling any shoe as a `jordan_eleven`


## How to Test Models
I can have a function that runs all the models on the training data and use different techniques to measure them. I can also have a function that tests with images I took myself and see where it fails.


## Notes:
>- Are the classes balanced
>- Do I have enough data
>- Add Images under where the datset came from
>- Link to Blog posts

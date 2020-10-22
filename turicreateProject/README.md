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
The images come from reddit and stockx. I have data for jordan `one,three,four,eleven` (these are models of jordans). I took a screenshot of each shoe some at different angles. I have about an average of 130 for each jordan. Some are high quality some are blury, some are being worn, some are just shoes, some have 2 shoes in one photo, different angles (ie. back, front, left, right). I also have a dataset with all the same images but with a filter applied to them in `filteredDataset` which I then created a model using those images to see if this would get me better results. All classes have 150 images some 160. Here are some examples of the images I have. 

![alt example](./dataset/jordan_one/readMeImg.png)
![alt example](./dataset/jordan_one/readMeImgEx.png)

I know I needed to get more data so to increase my data I created a file callled 'addFilterToImages.py'. In this file I run all all the images in the folder `dataset` through a function that applies a filter to them. Each photo would get 5 filters applied to them (fliped vertically, fillped horizontally, black and white, saturated, blurred). Once I did this I ended up with about 900 images for each class. Here are some examples:

![alt example](./dataset/jordan_one/1Screen Shot 2020-09-13 at 3.57.04 PM.png)
![alt example](./dataset/jordan_one/21Screen Shot 2020-09-13 at 3.57.04 PM.png)
![alt example](./dataset/jordan_one/31Screen Shot 2020-09-13 at 3.57.04 PM.png)
![alt example](./dataset/jordan_one/41Screen Shot 2020-09-13 at 3.57.04 PM.png)
![alt example](./dataset/jordan_one/51Screen Shot 2020-09-13 at 3.57.04 PM.png)
![alt example](./dataset/jordan_one/61Screen Shot 2020-09-13 at 3.57.04 PM.png)


I then tried to run my model again and see if test accuracy would change but there was little to know change. My model was performing best when my classes where slightly disbalanced. This could mean my model was predicting a certain class more often because of my dataset and there for getting high test accuracy. I can try and prove this by creating a model with a very disbalanced dataset and see how that turns out.


## Data Pipeline 
I create a model using turicreate with iamges that I also found and took myself. A pipeline that would be good to have would be to have a brand identification then once we identify the brand we can send it to another model that specifies in shoe number. When testing out my models with images I took of my own shoes the most common mistake that would happen is labeling any shoe as a `jordan_eleven`


## Testing with my own images
I tested my model with images I took of my own shoes. Overall it predicted the shoe well. One thing I did realize is that its more likely to predict a shoe over another. The predictions can be seen in `modelEval.ipynb`

## Notes:
>- Are the classes balanced
>- Do I have enough data
>- Add Images under where the datset came from
>- Link to Blog posts

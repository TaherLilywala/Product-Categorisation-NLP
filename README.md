# MIDAS-Task-3

### Overview
IIIT-D MIDAS Internship Test Task 3 by Taher Lilywala.

The aim is to predict a Product's Category given its description. Two NLP models, an LSTM and a Transformer have been used to this end. The transformer managed to achieve an accuracy of 65%

### The Data
Taking a look at a few entries, the data is a csv with 15 columns, two of which, *product_category_tree* and *description* are relevant to us.

A snippet of some data along these columns:
| uniq_id       | product_category_tree    | description     |
| :------------- | :----------: | -----------: |
| f449ec65dcbc041b6ae5e6a32717d01b   | ["Footwear >> Women's Footwear >> Ballerinas >> AW Bellies"] | Key Features of AW Bellies Sandals Wedges Heel Casuals,AW Bellies Price: Rs. 499 Material: Synthetic Lifestyle: Casual Heel Type: Wedge Warranty Type: Manufacturer Product Warranty against manufacturing defects: 30 days Care instructions: Allow your pair of shoes to air and de-odorize at regular basis; use shoe bags to prevent any stains or mildew; dust any dry dirt from the surface using a clean cloth; do not use polish or shiner,Specifications of AW Bellies General Ideal For Women Occasion Casual Shoe Details Color Red Outer Material Patent Leather Heel Height 1 inch Number of Contents in Sales Package Pack of 1 In the Box One Pair Of Shoes |
|  c2d766ca982eca8304150849735ffef9 | ["Clothing >> Women's Clothing >> Lingerie, Sleep & Swimwear >> Shorts >> Alisha Shorts >> Alisha Solid Women's Cycling Shorts"]   | Key Features of Alisha Solid Women's Cycling Shorts Cotton Lycra Navy, Red, Navy,Specifications of Alisha Solid Women's Cycling Shorts Shorts Details Number of Contents in Sales Package Pack of 3 Fabric Cotton Lycra Type Cycling Shorts General Details Pattern Solid Ideal For Women's Fabric Care Gentle Machine Wash in Lukewarm Water, Do Not Bleach Additional Details Style Code ALTHT_3P_21 In the Box 3 shorts    |
| 0973b37acd0c664e3de26e97e5571454   | ["Clothing >> Women's Clothing >> Lingerie, Sleep & Swimwear >> Shorts >> Alisha Shorts >> Alisha Solid Women's Cycling Shorts"] |Key Features of Alisha Solid Women's Cycling Shorts Cotton Lycra Black, Red,Specifications of Alisha Solid Women's Cycling Shorts Shorts Details Number of Contents in Sales Package Pack of 2 Fabric Cotton Lycra Type Cycling Shorts General Details Pattern Solid Ideal For Women's Fabric Care Gentle Machine Wash in Lukewarm Water, Do Not Bleach Additional Details Style Code ALTGHT_11 In the Box 2 shorts |

### Working With The Data - Cleanup
#### Input Data
Now, I did some basic NLP pre-processing to tidy up the description data.

These include:
- Converting to lower case
- Removing tags, URLs, punctuation/symbols, etc
- Removing non-alphabetical characters and ones of length less than 1
- Removing digits, stray whitespaces
- Lemmanizing the words to work with base forms
- Removing instances of null description

A Word Map of the Prevalent Words in the Description Text:

![image](https://user-images.githubusercontent.com/73401457/114269503-131bf980-9a25-11eb-99ad-2385533010b2.png)

Next, I tokenized the product descriptions in a numpy array.  
Here is the first entry, tokenized:
```
['key', 'feature', 'alisha', 'solid', 'woman', 'cycling', 'short', 'cotton', 'lycra', 'navy', 'red', 'navyspecifications', 'alisha', 'solid', 'woman', 'cycling', 'short', 'short', 'detail', 'number', 'content', 'sale', 'package', 'pack', 'fabric', 'cotton', 'lycra', 'type', 'cycling', 'short', 'general', 'detail', 'pattern', 'solid', 'ideal', 'woman', 'fabric', 'care', 'gentle', 'machine', 'wash', 'lukewarm', 'water', 'bleach', 'additional', 'detail', 'style', 'code', 'box', 'short']
```
Now, I proceeded to create a bag of words embedding and padded them to the maximum length of any instance. I didn't use any standard embedding, as there were plenty of novel words such as brand names. These could act as useful predictors.  
Here is the embedding of the earlier tokenized example:
```
[    0     0     0     0     0     0     0     0     0     0     0     0
    37    16  4195    33     4  1091   144    32   554   812   106 16215
  4195    33     4  1091   144   144    18    36    76    50    40    30
    23    32   554    19  1091   144    24    18    78    33    38     4
    23   102   521   225   111   773   137   274    90    18    53   110
    27   144]
```

#### Output Data
For the categories, I just extracted the first word found, situated between the delimiters of (") and (>). Since only the primary category is needed for prediction, only this was used.  
One such entry:
```
category[100]
> 'Watches'
```
Now, the indices of both the input and categories are shuffled, to make sure the data is randomized.  
Checking the shapes at this point:
```
print(review_pad.shape)
> (19998, 62)

print(category.shape)
> (19998,)
```
This means two entries were dropped, likely due to being empty.

### Encoding Categories, Dropping Rare Categories
Since we are dealing with Categorical Data, I used a Label Encoder to encode the Category array. A total of 266 Categories were found.
Mapping these out:

![image](https://user-images.githubusercontent.com/73401457/114271393-b7ef0480-9a2e-11eb-857f-8e6385151377.png)

Having a look at the frequency distribution, we can see that many of the classes have extremely low frequencies:
```
[[   0    1]
 [   1    1]
 [   2    2]
 [   3    1]
 [   4    1]
 [   5    1]
 [   6    1]
 [   7    1]
 [   8    2]
 [   9    1]
 [  10    1]
 [  11    1]
 [  12    1]
 [  13    1]
 [  14    1]
 [  15    1]
 [  16    1]
 [  17    1]
 [  18    1]
 [  19    1]
 [  20    1]
 [  21 1012]
 [  22    1]
 [  23    1]
 [  24    1]
 [  25    1]
 [  26  481]
 [  27  265]
 [  28    1]
 [  29  710]]
```
As such, I dropped all categories with a frequency less than 1000, or 5% of the dataset.
After this, 5 categories remain, which are the most prominent ones.

### OverSampling
To begin working, I created a train-test split, giving about 2000 samples to testing and the rest for training.
Even after dropping the categories, a lot of desparity remained. As such I oversampled the first 3000 indices using the 'all' sampling strategy, which brings all minority classes upto par, creating a uniform distribution. To these, I appended the remainder of the training samples at the end of the oversampled distribution.  
Here is the graph before and after doing this:

![image](https://user-images.githubusercontent.com/73401457/114272026-f0dca880-9a31-11eb-8692-887f830feba1.png)

## Models

### LSTM Model
Following is a summary of the LSTM model.  
The approach was straightforward. One embedding layer followed by two bidirectional LSTMs, which end in a fully connected layer that ends in a softmax layer to predict the category. The paramters were reached by trying around various configurations.
```
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_8 (Embedding)      (None, 62, 100)           2133100   
_________________________________________________________________
bidirectional_4 (Bidirection (None, 62, 2000)          8808000   
_________________________________________________________________
bidirectional_5 (Bidirection (None, 1000)              10004000  
_________________________________________________________________
dense_17 (Dense)             (None, 500)               500500    
_________________________________________________________________
dense_18 (Dense)             (None, 5)                 2505      
=================================================================
Total params: 21,448,105
Trainable params: 21,448,105
Non-trainable params: 0
_________________________________________________________________
```
Graphs:


### Transformer Model
Following is a summary of the Transformer.  
The input shape was constrained by the max length, in other words the size of the input description array. The transformer model used has two attention heads and 32 neurons in its feed forward layer. It was largely taken from the Keras Documentation, linked [here](https://keras.io/examples/nlp/text_classification_with_transformer/)
```
Layer (type)                 Output Shape              Param #   
=================================================================
input_14 (InputLayer)        [(None, 62)]              0         
_________________________________________________________________
token_and_position_embedding (None, 62, 100)           5006200   
_________________________________________________________________
transformer_block_13 (Transf (None, 62, 100)           87632     
_________________________________________________________________
global_average_pooling1d_13  (None, 100)               0         
_________________________________________________________________
dropout_55 (Dropout)         (None, 100)               0         
_________________________________________________________________
dense_66 (Dense)             (None, 500)               50500     
_________________________________________________________________
dropout_56 (Dropout)         (None, 500)               0         
_________________________________________________________________
dense_67 (Dense)             (None, 500)               250500    
_________________________________________________________________
dense_68 (Dense)             (None, 5)                 2505      
=================================================================
Total params: 5,397,337
Trainable params: 5,397,337
Non-trainable params: 0
_________________________________________________________________
```
Metrics:

![image](https://user-images.githubusercontent.com/73401457/114274020-37360580-9a3a-11eb-998c-a51e1c783bb7.png)
![image](https://user-images.githubusercontent.com/73401457/114274034-3bfab980-9a3a-11eb-8270-0a04a53d03d9.png)

Through experimentation, I found that 35 epochs was the ideal, beyond which it started overfitting. Finally, an accuracy of around 65% was achieved.

### Analysis
Dropping categories and oversampling the remaining ones were the major driving force behind optimising the model. Beyond what was used here, deeper networks didn't show much more improvement. Future lines of improvement could be using Time Series for the LSTM layers.

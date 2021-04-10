# MIDAS-Task-3

### Overview
IIIT-D MIDAS Internship Test Task 3 by Taher Lilywala.

The aim is to predict a Product's Category given its description. Two NLP models, an LSTM and a Transformer have been used to this end.

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



### Model

##### transformer
##### LSTM

### Analysis

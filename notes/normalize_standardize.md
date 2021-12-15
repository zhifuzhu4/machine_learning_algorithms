### Why Scaling? 
[Wikipedia: Feature scaling](https://en.wikipedia.org/wiki/Feature_scaling#cite_note-1)

Since the range of values of raw data varies widely, 
in some machine learning algorithms, objective functions will not work properly 
without normalization. For example, many classifiers calculate the distance 
between two points by the Euclidean distance. If one of the features has 
a broad range of values, the distance will be governed by this particular feature. 
Therefore, the range of all features should be normalized so that each feature 
contributes approximately proportionately to the final distance.

Another reason why feature scaling is applied is that gradient descent 
converges much faster with feature scaling than without it.

It's also important to apply feature scaling if regularization is used 
as part of the loss function (so that coefficients are penalized appropriately).

<br/>

Feature scaling is one of the most important data preprocessing step 
in machine learning. Algorithms that compute the distance between the features 
are biased towards numerically larger values if the data is not scaled.

Tree-based algorithms are fairly insensitive to the scale of the features. 
Also, feature scaling helps machine learning, and deep learning algorithms 
train and converge faster.

Therefore, we need to bring all features to the
same level of magnitudes. This can be acheived by scaling.

### Normalization

**Normalization (Min-Max Scaling)** rescales the values into a range of [0, 1].      
![min max scaling](https://github.com/zhifuzhu4/ml_algorithms/blob/main/images/min_max_scaling.png)

This scales the range to [0, 1] or sometimes [-1, 1]. 
Geometrically speaking, transformation squishes the n-dimensional data 
into an n-dimensional unit hypercube. Normalization is useful when there 
are no outliers as it cannot cope up with them. 
Usually, we would scale age and not incomes because only a few people have high incomes 
but the age is close to uniform.

Selecting the target range depends on the nature of the data.

To rescale a range between an arbitrary set of values [a, b], the formula becomes:
![rescale to a range](https://github.com/zhifuzhu4/ml_algorithms/blob/main/images/range_rescaling.png)

### Mean normalization

![mean normalization](https://github.com/zhifuzhu4/ml_algorithms/blob/main/images/mean_normalization.png)

### Standardization

**Standardization (Z-Score Normalization)** rescales the data to have a 
mean of 0 and standard deviation of 1 (unit variance). <br>
![standardization](https://github.com/zhifuzhu4/ml_algorithms/blob/main/images/standardization.png)

Standardization can be helpful in cases where the data follows a Gaussian distribution. 
However, this does not have to be necessarily true. Geometrically speaking, 
it translates the data to the mean vector of original data to the origin and squishes 
or expands the points if std is 1 respectively. We can see that we are just changing mean 
and standard deviation to a standard normal distribution which is still normal thus 
the shape of the distribution is not affected.

Standardization does not get affected by outliers because there is no 
predefined range of transformed features.

Standardisation and Mean Normalization can be used for algorithms that 
assumes zero centric data like Principal Component Analysis(PCA).

<br/>

[Normalization vs Standardization](https://www.geeksforgeeks.org/normalization-vs-standardization/)

![normalize vs standardize](https://github.com/zhifuzhu4/ml_algorithms/blob/main/images/max_min_normalization.png)


### When to scale?
[Why, How and When to Scale your Features](https://medium.com/greyatom/why-how-and-when-to-scale-your-features-4b30ab09db5e)

Rule of thumb I follow here is any algorithm that computes distance 
or assumes normality, scale your features!!!

Some examples of algorithms where feature scaling matters are:
- k-nearest neighbors with an Euclidean distance measure is sensitive to 
magnitudes and hence should be scaled for all features to weigh in equally.
- Scaling is critical, while performing Principal Component Analysis(PCA). 
PCA tries to get the features with maximum variance and the variance is high 
for high magnitude features. This skews the PCA towards high magnitude features.
- We can speed up gradient descent by scaling. This is because θ will
descend quickly on small ranges and slowly on large ranges, 
and so will oscillate inefficiently down to the optimum when the variables 
are very uneven.
- Tree based models are not distance based models and can handle varying ranges 
of features. Hence, Scaling is not required while modelling trees.
- Algorithms like Linear Discriminant Analysis(LDA), Naive Bayes are by 
design equipped to handle this and gives weights to the features accordingly. 
Performing a features scaling in these algorithms may not have much effect.

  
# Recommendation Models: Collaborative Filtering, Neural CTR, and DLRM

## 1. Collaborative Filtering

Collaborative Filtering is a recommendation method that uses the past behavior of users to make recommendations.

### Two Approaches in Collaborative Filtering

There are two main approaches used in this task:

**1. k-Nearest Neighbors (kNN)**

kNN recommends items by finding users or items that are most similar to the current user or item.

For example, if two users have rated many of the same movies and their ratings are similar, they are considered neighbors. We can use the ratings or preferences of these nearest neighbors to make recommendations.

The basic idea is:

**Find similar users/items -> Look at their preferences -> Make a recommendation**

Similarity can be calculated using measures such as cosine similarity.

**2. Matrix Factorization**

In Matrix Factorization, the user-item rating matrix is divided into two smaller matrices containing latent factors (hidden features).

For example:

**User-Item Matrix = User Factors × Item Factors**

Each user gets a learned vector, and each item gets a learned vector. The dot product between these vectors can be used to predict how much a user may like an item.

$$
\hat{r}_{ui}=p_u^Tq_i
$$

where:

* $p_u$ = vector representing user \(u\)
* $q_i$ = vector representing item \(i\)
* $\hat{r}_{ui}$ = predicted rating

The model learns these vectors from the existing user-item ratings.

### Main Difference

**kNN:** Makes recommendations by looking at similar users/items.

**Matrix Factorization:** Learns hidden representations (latent factors) for users and items and uses them to predict preferences.

---

## 2. Neural CTR

CTR stands for Click-Through Rate.

The goal of a CTR model is to predict the probability that a user will click on an item, advertisement, or recommendation.

For example, given information such as:

* User
* Age
* Device
* Advertisement
* Location
* Time
* Other categorical features

the model predicts:


$P_{click}$=1


For example:


$P_{click}$=0.82

means the model predicts an 82% probability of a click.

### How the model works

Categorical features are first converted into IDs and then into **embedding vectors**.

For example:


UserID=15 -> $e_{user}$

AdID=42 -> $e_{ad}$


Numerical features can be used directly or passed through neural network layers.

These representations are then given to a neural network, usually an MLP.

The final layer produces a value that is converted into a probability using a sigmoid:


$$\hat y=\sigma(z)$$


where

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

The model is trained using the actual click information.

For example:

```text
Predicted click probability = 0.8
Actual click = 1
```

The model should learn that this prediction was good.

If:

```text
Predicted click probability = 0.8
Actual click = 0
```

the model receives a larger error and updates its parameters.

### Main idea

Features -> embedding -> MLP -> prediction

Neural CTR models are more flexible than basic collaborative filtering because they can use many different types of features.

---

# 3. DLRM

DLRM (Deep Learning Recommendation Model) is used for recommendation and click prediction. It works with both numerical and categorical features.

The categorical features are converted into IDs and then into embeddings. The numerical features are passed through a small neural network. DLRM then looks at the interactions between these features, mainly using dot products.

Finally, these values are passed through another neural network to predict the output, such as whether a user will click on an item.

**Basic idea :**

Numerical features -> Neural Network
Categorical features -> Embeddings ->
Feature interactions -> Neural Network -> Prediction

The main idea of DLRM is to learn how different features interact with each other to make better recommendations.



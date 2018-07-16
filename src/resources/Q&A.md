## Thread

* _Critical Section_: a code segment that accessed shared variables and has to be executed as an atomic action.
* _Mutual Exclusion_ a.k.a mutex: is a mechanism while a thread is accessing a shared variable, no other thread can access it by `mutex.lock()`
* _Semaphores_: similar to mutex, while having a counter to control how many resources may access resource simultaneously.
* **Semaphores and Mutual Exclusion are two mechanism to realize critical section**

## General

* `StringBuffer` thread safe, while `StringBuilder` is not
* `HashTable` thread safe not allow `null`, while `HashMap` is not and allow `null`
* How hash works: keys -> hash(key) -> find buckets -> entries (with linked listed)
* When implement equal it must implement hashcode, as equal is used to equality of hashcode of two objects.
* `LinkList` vs. `ArrayList`: LinkedList is faster in add and remove, but slower in get.
In brief, LinkedList should be preferred if: 
    * there are no large number of random access of element
    * there are a large number of add/remove operations
* `Hashmap`: array of reference, entry table 16 brackets, link-listed, load factor: 0.75, if greater, rehashing, double size.
* `Hashmap` contains an array of node and node can represent a class having following objects: `int hash`, `K key`, `V value`, `Node next`

## Design Patterns
* **Factory Pattern**: define an interface for creating an object, but let subclasses decide which class to instantiate.
e.g. `class NYPizzaStore extends PizzaStore`
* **Strategy Pattern**: defines a family of algorithms, encapsulates each one, and makes them interchangeable, strategy lets the algorithm vary independent from client and that use it.
e.g. 
    ```
    FlyWithWings extends FlyBehavior {...}
    FlyNoWay extends FlyBehavior {...}
    class Duck{FlyBehavior flyBehavor ...}
    ```
* **Observer Pattern**: defines a one-to-many dependency between objects so that when one object change state, all of its dependents are notified and updated automatically.
* **Decorator Pattern**: attaches additional responsibility to an object dynamically, provide a flexible alternative to subclassing for extending functionality.


## Machine Learning
* Bias-Variance Tradeoff: **Bias** is error due to overly simplistic assumptions, can be caused by underfitting data or small train set.
**Variance** is error due to too much complexity, highly sensitive, overfit data, carry too much noise from training data for model to be useful for test data.
* **Supervised learning** vs. **Unsupervised learning**: labeled data vs. not labeled data.
* KNN a.k.a **K-Nearest Neighbors** vs. **k-means clustering**: supervised learning vs. unsupervised learning. For KNN, once labeled, find the data point which is neared the center of cluster. While k-means use simliar way but gradually figure out which belongs to which group by computing the mean of the distance between different points.
* Explain ROC a.k.a **Receiver Operating Characteristic** curve: graphical representation of the contract between TP and FP at various thresholds. Used as an indicator for the trade-off between sensitivity of the model(TP) vs the fall-out, the probability it will trigger a false alarm.(FP)
* **Precision** vs. **Recall**: precision is the fraction of relevant instances among the retrieved instance, while recall is the relevant instances among the total amount of relevant instance. 
e.g. 8 dogs recognized in a picture of 12 dogs and some cats. Of the 8 identified as dogs, 5 actually are dogs(TP), while the rest of it are cats(FP).
precision = 5/8; recall = 5/12;
    * Precision(positive predictive value): how useful the result are. How many selected items are relevant.
    * Recall(sensitivity): how complete the results are. How many relevant items are selected.  
    ![alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Precisionrecall.svg/350px-Precisionrecall.svg.png "Logo Title Text 1")
* What is **Bayes’ Theorem**? How is it useful in a machine learning context? Bayes’ Theorem gives you the posterior probability of an event given what is known as prior knowledge.  
e.g. Say you had a 60% chance of actually having the flu after a flu test, but out of people who had the flu, the test will be false 50% of the time, and the overall population only has a 5% chance of having the flu. Would you actually have a 60% chance of having the flu after having a positive test?  
Bayes’ Theorem says no. It says that you have a (.6 * 0.05) (True Positive Rate of a Condition Sample) / (.6\*0.05)(True Positive Rate of a Condition Sample) + (.5*0.95) (False Positive Rate of a Population)  = 0.0594 or 5.94% chance of getting a flu.
* Why is “Naive” Bayes naive?
Despite its practical applications, especially in text mining, Naive Bayes is considered “Naive” because it makes an assumption that is virtually impossible to see in real-life data: the conditional probability is calculated as the pure product of the individual probabilities of components. This implies the absolute independence of features — a condition probably never met in real life.  
As a Quora commenter put it whimsically, a Naive Bayes classifier that figured out that you liked pickles and ice cream would probably naively recommend you a pickle ice cream.
e.g.  
P(y/X) = P(X/y) * P(y) = P(x1,x2, ... x10/ y) * P(y)  
So for, it is still not Naive. However, it is hard to calculate P(x1,x2, ... x10/ Y), so we assume the features to be independent, this is what we call the Naive assumption, hence, we end up with the following formula instead  
P(y/X) = P(x1/y) * P(x2/y) * ... P(x10/y) * P(y)
* Explain the difference between L1 and L2 regularization:  
L2 regularization tends to spread error among all the terms, while L1 is more binary/sparse, with many variables either being assigned a 1 or 0 in weighting.  
 The key difference between these techniques is that Lasso(L1) shrinks the less important feature’s coefficient to zero thus, removing some feature altogether. So, this works well for feature selection in case we have a huge number of features.
* What’s the difference between Type I and Type II error?
    
  Don’t think that this is a trick question! Many machine learning interview questions will be an attempt to lob basic questions at you just to make sure you’re on top of your game and you’ve prepared all of your bases.
  
  Type I error is a false positive, while Type II error is a false negative. Briefly stated, Type I error means claiming something has happened when it hasn’t, while Type II error means that you claim nothing is happening when in fact something is.
  
  A clever way to think about this is to think of Type I error as telling a man he is pregnant, while Type II error means you tell a pregnant woman she isn’t carrying a baby.  
  **False positive: false to positive, supposed to be negative**  
  **False negative: false to negative, supposed to be positive**
* What is deep learning, and how does it contrast with other machine learning algorithms?
    
  Deep learning is a subset of machine learning that is concerned with neural networks: how to use backpropagation and certain principles from neuroscience to more accurately model large sets of unlabelled or semi-structured data.
* What’s the F1 score? How would you use it?
  
  The F1 score is a measure of a model’s performance. It is a weighted average of the precision and recall of a model, with results tending to 1 being the best, and those tending to 0 being the worst. You would use it in classification tests where true negatives don’t matter much.

* How would you handle an imbalanced dataset?
    An imbalanced dataset is when you have, for example, a classification test and 90% of the data is in one class. That leads to problems: an accuracy of 90% can be skewed if you have no predictive power on the other category of data! Here are a few tactics to get over the hump:
    
    1- Collect more data to even the imbalances in the dataset.
    
    2- Resample the dataset to correct for imbalances.
    
    3- Try a different algorithm altogether on your dataset.
* When should you use classification over regression?
    
  Classification produces discrete values and dataset to strict categories, while regression gives you continuous results that allow you to better distinguish differences between individual points. You would use classification over regression if you wanted your results to reflect the belongingness of data points in your dataset to certain explicit categories
* How do you ensure you’re not overfitting with a model?
    
  This is a simple restatement of a fundamental problem in machine learning: the possibility of overfitting training data and carrying the noise of that data through to the test set, thereby providing inaccurate generalizations.
  
  There are three main methods to avoid overfitting:
  
  1- Keep the model simpler: reduce variance by taking into account fewer variables and parameters, thereby removing some of the noise in the training data.
  
  2- Use cross-validation techniques such as k-folds cross-validation.
  
  3- Use regularization techniques such as LASSO that penalize certain model parameters if they’re likely to cause overfitting.
  
## Machine Learning Section II.
1. How would you define Machine Learning? :: Machine Learning is great for complex problems for which we have no algorithmic solution, to replace long lists of hand-tuned rules, to build systems that adapt to fluctuating environments, and finally to help humans learn (e.g., data mining).
2. Can you name four types of problems where it shines? :: Machine Learning is great for complex problems for which we have no algorithmic solution, to replace long lists of hand-tuned rules, to build systems that adapt to fluctuating environments, and finally to help humans learn (e.g., data mining).
3. What is a labeled training set? :: A labeled training set is a training set that contains the desired solution (a.k.a. a label) for each instance.
4. What are the two most common supervised tasks? :: The two most common supervised tasks are regression and classification.
5. Can you name four common unsupervised tasks? :: Common unsupervised tasks include clustering, visualization, dimensionality reduction, and association rule learning.
6. What type of Machine Learning algorithm would you use to allow a robot to walk in various unknown terrains? :: Reinforcement Learning is likely to perform best if we want a robot to learn to walk in various unknown terrains since this is typically the type of problem that Reinforcement Learning tackles. It might be possible to express the problem as a supervised or semisupervised learning problem, but it would be less natural.
7. What type of algorithm would you use to segment your customers into multiple groups? :: If you don’t know how to define the groups, then you can use a clustering algorithm (unsupervised learning) to segment your customers into clusters of similar customers. However, if you know what groups you would like to have, then you can feed many examples of each group to a classification algorithm (supervised learning), and it will classify all your customers into these groups.
8. Would you frame the problem of spam detection as a supervised learning problem or an unsupervised learning problem? :: Spam detection is a typical supervised learning problem: the algorithm is fed many emails along with their label (spam or not spam).
9. What is an online learning system? :: An online learning system can learn incrementally, as opposed to a batch learning system. This makes it capable of adapting rapidly to both changing data and autonomous systems, and of training on very large quantities of data.
10. What is out-of-core learning? :: Out-of-core algorithms can handle vast quantities of data that cannot fit in a computer’s main memory. An out-of-core learning algorithm chops the data into mini-batches and uses online learning techniques to learn from these mini-batches.
11. What type of learning algorithm relies on a similarity measure to make predictions? :: An instance-based learning system learns the training data by heart; then, when given a new instance, it uses a similarity measure to find the most similar learned instances and uses them to make predictions.
12. What is the difference between a model parameter and a learning algorithm’s hyperparameter? :: A model has one or more model parameters that determine what it will predict given a new instance (e.g., the slope of a linear model). A learning algorithm tries to find optimal values for these parameters such that the model generalizes well to new instances. A hyperparameter is a parameter of the learning algorithm itself, not of the model (e.g., the amount of regularization to apply).
13. What do model-based learning algorithms search for? What is the most common strategy they use to succeed? How do they make predictions? :: Model-based learning algorithms search for an optimal value for the model parameters such that the model will generalize well to new instances. We usually train such systems by minimizing a cost function that measures how bad the system is at making predictions on the training data, plus a penalty for model complexity if the model is regularized. To make predictions, we feed the new instance’s features into the model’s prediction function, using the parameter values found by the learning algorithm.
14. Can you name four of the main challenges in Machine Learning? :: Some of the main challenges in Machine Learning are the lack of data, poor data quality, nonrepresentative data, uninformative features, excessively simple models that underfit the training data, and excessively complex models that overfit the data.
15. If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name three possible solutions? :: If a model performs great on the training data but generalizes poorly to new instances, the model is likely overfitting the training data (or we got extremely lucky on the training data). Possible solutions to overfitting are getting more data, simplifying the model (selecting a simpler algorithm, reducing the number of parameters or features used, or regularizing the model), or reducing the noise in the training data.
16. What is a test set and why would you want to use it? :: A test set is used to estimate the generalization error that a model will make on new instances, before the model is launched in production.
17. What is the purpose of a validation set? :: A validation set is used to compare models. It makes it possible to select the best model and tune the hyperparameters.
18. What can go wrong if you tune hyperparameters using the test set? :: If you tune hyperparameters using the test set, you risk overfitting the test set, and the generalization error you measure will be optimistic (you may launch a model that performs worse than you expect).
19. What is cross-validation and why would you prefer it to a validation set? :: Cross-validation is a technique that makes it possible to compare models (for model selection and hyperparameter tuning) without the need for a separate validation set. This saves precious training data.

## Training Models
1.  What Linear Regression training algorithm can you use if you have a training set with millions of features?
* If you have a training set with millions of features you can use Stochastic Gradient Descent or Mini-batch Gradient Descent, and perhaps Batch Gradient Descent if the training set fits in memory. But you cannot use the Normal Equation because the computational complexity grows quickly (more than quadratically) with the number of features.
```
Normal Equation vs. Gradient Descent

Disadvantages of gradient descent:
-   you need to choose the learning rate, so you may need to run the algorithm at least a few times to figure that out.
-   it needs many more iterations, so, that could make it slower
Compared to the normal equation:
-   you don't need to choose any learning rate
-   you don't need to iterate
Disadvantages of the normal equation:
-   Normal Equation is computationally expensive when you have a very large number of features ( n features ), because you will ultimately need to take the inverse of a n x n matrix in order to solve for the parameters data.
Compared to gradient descent:
-   it will be reasonably efficient and will do something acceptable when you have a very large number ( millions ) of features.
So if n is large then use gradient descent.
If n is relatively small ( on the order of a hundred ~ ten thousand ), then the normal equation
```    
2.  Suppose the features in your training set have very different scales. What algorithms might suffer from this, and how? What can you do about it? 
* If the features in your training set have very different scales, the cost function will have the shape of an elongated bowl, so the Gradient Descent algorithms will take a long time to converge. To solve this you should scale the data before training the model. Note that the Normal Equation will work just fine without scaling. Moreover, regularized models may converge to a suboptimal solution if the features are not scaled: indeed, since regularization penalizes large weights, features with smaller values will tend to be ignored compared to features with larger values.
3.  Can Gradient Descent get stuck in a local minimum when training a Logistic Regression model?
* Gradient Descent cannot get stuck in a local minimum when training a Logistic Regression model because the cost function is convex
4.  Do all Gradient Descent algorithms lead to the same model provided you let them run long enough?
 * If the optimization problem is convex (such as Linear Regression or Logistic Regression), and assuming the learning rate is not too high, then all Gradient Descent algorithms will approach the global optimum and end up producing fairly similar models. However, unless you gradually reduce the learning rate, Stochastic GD and Mini-batch GD will never truly converge; instead, they will keep jumping back and forth around the global optimum. This means that even if you let them run for a very long time, these Gradient Descent algorithms will produce slightly different models.
5.  Suppose you use Batch Gradient Descent and you plot the validation error at every epoch. If you notice that the validation error consistently goes up, what is likely going on? How can you fix this?
* If the validation error consistently goes up after every epoch, then one possibility is that the learning rate is too high and the algorithm is diverging. If the training error also goes up, then this is clearly the problem and you should reduce the learning rate. However, if the training error is not going up, then your model is overfitting the training set and you should stop training.
6.  Is it a good idea to stop Mini-batch Gradient Descent immediately when the validation error goes up? 
    * Due to their random nature, neither Stochastic Gradient Descent nor Mini-batch Gradient Descent is guaranteed to make progress at every single training iteration. So if you immediately stop training when the validation error goes up, you may stop much too early, before the optimum is reached. A better option is to save the model at regular intervals, and when it has not improved for a long time (meaning it will probably never beat the record), you can revert to the best saved model.
7.  Which Gradient Descent algorithm (among those we discussed) will reach the vicinity of the optimal solution the fastest? Which will actually converge? How can you make the others converge as well?
    * Stochastic Gradient Descent has the fastest training iteration since it considers only one training instance at a time, so it is generally the first to reach the vicinity of the global optimum (or Mini-batch GD with a very small mini-batch size). However, only Batch Gradient Descent will actually converge, given enough training time. As mentioned, Stochastic GD and Mini-batch GD will bounce around the optimum, unless you gradually reduce the learning rate.
8.  Suppose you are using Polynomial Regression. You plot the learning curves and you notice that there is a large gap between the training error and the validation error. What is happening? What are three ways to solve this?
    * If the validation error is much higher than the training error, this is likely because your model is overfitting the training set. One way to try to fix this is to reduce the polynomial degree: a model with fewer degrees of freedom is less likely to overfit. Another thing you can try is to regularize the model—for example, by adding an ℓ2 penalty (Ridge) or an ℓ1 penalty (Lasso) to the cost function. This will also reduce the degrees of freedom of the model. Lastly, you can try to increase the size of the training set.
9.  Suppose you are using Ridge Regression and you notice that the training error and the validation error are almost equal and fairly high. Would you say that the model suffers from high bias or high variance? Should you increase the regularization hyperparameter  α  or reduce it?
 * If both the training error and the validation error are almost equal and fairly high, the model is likely underfitting the training set, which means it has a high bias. You should try reducing the regularization hyperparameter α.
10.  Why would you want to use:
    
    -   Ridge Regression instead of plain Linear Regression (i.e., without any regularization)?
        A model with some regularization typically performs better than a model without any regularization, so you should generally prefer Ridge Regression over plain Linear Regression.
    -   Lasso instead of Ridge Regression?
        Lasso Regression uses an ℓ1 penalty, which tends to push the weights down to exactly zero. This leads to sparse models, where all weights are zero except for the most important weights. This is a way to perform feature selection automatically, which is good if you suspect that only a few features actually matter. When you are not sure, you should prefer Ridge Regression.
    -   Elastic Net instead of Lasso?
        Elastic Net is generally preferred over Lasso since Lasso may behave erratically in some cases (when several features are strongly correlated or when there are more features than training instances). However, it does add an extra hyperparameter to tune. If you just want Lasso without the erratic behavior, you can just use Elastic Net with an `l1_ratio` close to 1.
11.  Suppose you want to classify pictures as outdoor/indoor and daytime/nighttime. Should you implement two Logistic Regression classifiers or one Softmax Regression classifier?
   *  If you want to classify pictures as outdoor/indoor and daytime/nighttime, since these are not exclusive classes (i.e., all four combinations are possible) you should train two Logistic Regression classifiers.
12.  Implement Batch Gradient Descent with early stopping for Softmax Regression  (without using Scikit-Learn).
## Support Vector Machine
1.  What is the fundamental idea behind Support Vector Machines?
    * The fundamental idea behind Support Vector Machines is to fit the widest possible “street” between the classes. In other words, the goal is to have the largest possible margin between the decision boundary that separates the two classes and the training instances. When performing soft margin classification, the SVM searches for a compromise between perfectly separating the two classes and having the widest possible street (i.e., a few instances may end up on the street). Another key idea is to use kernels when training on nonlinear datasets.
2.  What is a support vector?
    
3.  Why is it important to scale the inputs when using SVMs?
    
4.  Can an SVM classifier output a confidence score when it classifies an instance? What about a probability?
    
5.  Should you use the primal or the dual form of the SVM problem to train a model on a training set with millions of instances and hundreds of features?
    
6.  Say you trained an SVM classifier with an RBF kernel. It seems to underfit the training set: should you increase or decrease  _γ_(`gamma`)? What about  `C`?
    
7.  How should you set the QP parameters (**H**,  **f**,  **A**, and  **b**) to solve the soft margin linear SVM classifier problem using an off-the-shelf QP solver?
    
8.  Train a  `LinearSVC`  on a linearly separable dataset. Then train an  `SVC`  and a  `SGDClassifier`  on the same dataset.  See if you can get them to produce roughly the same model.
    
9.  Train an SVM classifier on the MNIST dataset. Since SVM classifiers are binary classifiers, you will need to use  one-versus-all to classify all 10 digits. You may want to tune the hyperparameters using small validation sets to speed up the process. What accuracy can you reach?
    
10.  Train an SVM regressor on the California housing  dataset.
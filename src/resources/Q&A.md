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

## Kafka 

#### Message Model
1. Messaging traditionally has two models: **queuing** and **publish-subscribe**. 

2. In a queue, a pool of consumers may read from a server and each record goes to one of them;
 
3. In publish-subscribe the record is broadcast to all consumers. 

4. The strength of queuing is that it allows you to divide up the processing of data over multiple consumer instances, which lets you scale your processing. Unfortunately, queues aren't multi-subscriber—once one process reads the data it's gone. 

5. Publish-subscribe allows you broadcast data to multiple processes, but has no way of scaling processing since every message goes to every subscriber.

6. The consumer group concept in Kafka generalizes these two concepts. As with a queue the consumer group allows you to divide up processing over a collection of processes (the members of the consumer group). As with publish-subscribe, Kafka allows you to broadcast messages to multiple consumer groups.

#### Ordering
1. A traditional queue retains records in-order on the server, and if multiple consumers consume from the queue then the server hands out records in the order they are stored. 

2. However, although the server hands out records in order, the records are delivered asynchronously to consumers, so they may arrive out of order on different consumers. This effectively means the ordering of the records is lost in the presence of parallel consumption. 

3. Messaging systems often work around this by having a notion of "exclusive consumer" that allows only one process to consume from a queue, but of course this means that there is no parallelism in processing.

4. Kafka only provides a total order over messages within a partition, not between different partitions in a topic.

5. In Kafka, assigning the partitions in the topic to the consumers in the consumer group so that each partition is consumed by exactly one consumer in the group. By doing this we ensure that the consumer is the only reader of that partition and consumes the data in order.

#### As Storage System

1. Data written to Kafka is written to disk and replicated for fault-tolerance. 

2. Kafka allows producers to wait on acknowledgement so that a write isn't considered complete until it is fully replicated and guaranteed to persist even if the server written to fails.

3. The disk structures Kafka uses scale well—Kafka will perform the same whether you have 50 KB or 50 TB of persistent data on the server.
   
4. As a result of taking storage seriously and allowing the clients to control their read position, you can think of Kafka as a kind of special purpose distributed filesystem dedicated to high-performance, low-latency commit log storage, replication, and propagation.

#### As Streaming System

![](https://docs.confluent.io/current/_images/streams-architecture-overview.jpg)

1. A **topology** is a graph of stream processors (nodes) that are connected by streams (edges).

2. **Source Processor**: A source processor is a special type of stream processor that does not have any upstream processors. It produces an input stream to its topology from one or multiple Kafka topics by consuming records from these topics and forward them to its down-stream processors.

3. **Sink Processor**: A sink processor is a special type of stream processor that does not have down-stream processors. It sends any received records from its up-stream processors to a specified Kafka topic.

4. Kafka Streams provides two APIs to define stream processors:
    - The declarative, functional DSL is the recommended API for most users – and notably for starters – because most data processing use cases can be expressed in just a few lines of DSL code. Here, you typically use built-in operations such as map and filter.
    - The imperative, lower-level Processor API provides you with even more flexibility than the DSL but at the expense of requiring more manual coding work. Here, you can define and connect custom processors as well as directly interact with state stores.
    
5. Any stream processing technology must therefore provide **first-class support for streams and tables**.

6. The **stream-table duality** describes the close relationship between streams and tables.
   
   - **Stream as Table**: A stream can be considered a changelog of a table, where each data record in the stream captures a state change of the table. A stream is thus a table in disguise, and it can be easily turned into a “real” table by replaying the changelog from beginning to end to reconstruct the table. Similarly, aggregating data records in a stream will return a table. For example, we could compute the total number of pageviews by user from an input stream of pageview events, and the result would be a table, with the table key being the user and the value being the corresponding pageview count.
   - **Table as Stream**: A table can be considered a snapshot, at a point in time, of the latest value for each key in a stream (a stream’s data records are key-value pairs). A table is thus a stream in disguise, and it can be easily turned into a “real” stream by iterating over each key-value entry in the table.
   
7. A **KStream** is an abstraction of a **record stream**, where each data record represents a self-contained datum in the unbounded data set. Using the table analogy, data records in a record stream are always interpreted as an **“INSERT”**

8. A **KTable** is an abstraction of a changelog stream, where each data record represents an update. More precisely, the value in a data record is interpreted as an **“UPSERT”**

9. KTable also provides an ability to look up current values of data records by keys.

10. Like a KTable, a **GlobalKTable** is an abstraction of a **changelog stream**, where each data record represents an update.
    
    A GlobalKTable differs from a KTable in the data that they are being populated with, i.e. which data from the underlying Kafka topic is being read into the respective table. Slightly simplified, imagine you have an input topic with 5 partitions. In your application, you want to read this topic into a table. Also, you want to run your application across 5 application instances for maximum parallelism.
    
    If you read the input topic into a KTable, then the “local” KTable instance of each application instance will be populated with data from only 1 partition of the topic’s 5 partitions.
    If you read the input topic into a GlobalKTable, then the local GlobalKTable instance of each application instance will be populated with data from all partitions of the topic.
    
11. Kafka Streams supports the following notions of **time**:
    - Event-time: The point in time when an event or data record occurred, i.e. was originally created “by the source”. 
    - Processing-time: The point in time when the event or data record happens to be processed by the stream processing application
    - Ingestion-time: The point in time when an event or data record is stored in a topic partition by a Kafka broker. 
    
## HBase

1. **HBase** and **Hive** both are completely different Hadoop based technologies
    - Hive is a data warehouse infrastructure on top of Hadoop, Hive helps SQL savvy people to run MapReduce jobs, Hive is an ideal choice for analytical querying of data collected over the period of time.
    - HBase is a NoSQL key-value store that runs on top of Hadoop, HBase supports 4 primary operations-put, get, scan and delete. HBase is ideal for real-time querying of big data.
    
2. **Row key**: Every row in an HBase table has a unique identifier known as RowKey. It is used for grouping cells logically and it ensures that all cells that have the same RowKeys are co-located on the same server. RowKey is internally regarded as a byte array.

3. RDBMS is a schema-based database whereas HBase is schema-less data model.
   
   RDBMS does not have support for in-built partitioning whereas in HBase there is automated partitioning.
   
   RDBMS stores normalized data whereas HBase stores de-normalized data.
   
4. Record Level Operational Commands in HBase are - put, get, increment, scan and delete.
   
   Table Level Operational Commands in HBase are - describe, list, drop, disable and scan.
   
5. There are two important catalog tables in HBase, are ROOT and META. ROOT table tracks where the META table is and META table stores all the regions in the system.

6. On issuing a delete command in HBase through the HBase client, data is not actually deleted from the cells but rather the cells are made invisible by setting a tombstone marker. The deleted cells are removed at regular intervals during compaction.

7. Explain about HLog and WAL in HBase. All edits in the HStore are stored in the HLog. Every region server has one HLog. HLog contains entries for edits of all regions performed by a particular Region Server.WAL abbreviates to Write Ahead Log (WAL) in which all the HLog edits are written immediately.WAL edits remain in the memory till the flush period in case of deferred log flush.

8. The HFile is the underlying storage format for HBase. HFiles belong to a column family and a column family can have multiple HFiles. But a single HFile can’t have data for multiple column families

## Spark

1. Apache Spark is an open-source cluster computing framework for real-time processing.

2. Multi-languages: scala, java, R and Python

3. Support multiple format, data sources, pluggable data source connector 

4. lazy evaluation, for transformation, spark add them to DAG of computation and only driver request data, DAG gets actually executed.

5. RDD stands for Resilient Distribution Datasets. An RDD is a fault-tolerant collection of operational elements that run in parallel. The partitioned data in RDD is immutable and distributed in nature. There are primarily two types of RDD:
   
   - Parallelized Collections: Here, the existing RDDs running parallel with one another.
   - Hadoop Datasets: They perform functions on each file record in HDFS or other storage systems.
   
6. Partition is a smaller and logical division of data similar to ‘split’ in MapReduce. It is a logical chunk of a large distributed data set. 

7. Partitioning is the process to derive logical units of data to speed up the processing process. Spark manages data using partitions that help parallelize distributed data processing with minimal network traffic for sending data between executors. 

8. By default, Spark tries to read data into an RDD from the nodes that are close to it. Since Spark usually accesses distributed partitioned data, to optimize transformation operations it creates partitions to hold the data chunks. Everything in Spark is a partitioned RDD.

9. RDDs support two types of operations: transformations and actions. 
   
   - Transformations: Transformations create new RDD from existing RDD like map, reduceByKey and filter we just saw. Transformations are executed on demand. That means they are computed lazily.
   
   - Actions: Actions return final results of RDD computations. Actions triggers execution using lineage graph to load the data into original RDD, carry out all intermediate transformations and return final results to Driver program or write it out to file system.

10. Spark Streaming is used for processing real-time streaming data. Thus it is a useful addition to the core Spark API. It enables high-throughput and fault-tolerant stream processing of live data streams. The fundamental stream unit is DStream which is basically a series of RDDs (Resilient Distributed Datasets) to process the real-time data.

11. Spark does not support data replication in the memory and thus, if any data is lost, it is rebuild using RDD lineage. RDD lineage is a process that reconstructs lost data partitions. The best is that RDD always remembers how to build from other datasets.

12. Spark SQL is capable of:
    - Loading data from a variety of structured sources.
    - Querying data using SQL statements, both inside a Spark program and from external tools that connect to Spark SQL through standard database connectors (JDBC/ODBC). For instance, using business intelligence tools like Tableau. 
    - Providing rich integration between SQL and regular Python/Java/Scala code, including the ability to join RDDs and SQL tables, expose custom functions in SQL, and more. 
    
## Hive

1. **Partitioning** and **Bucketing** of tables is done to improve the query performance. Partitioning helps execute queries faster, only if the partitioning scheme has some common range filtering i.e. either by timestamp ranges, by location, etc. Bucketing does not work by default.
Partitioning helps eliminate data when used in WHERE clause. Bucketing helps organize data inside the partition into multiple files so that same set of data will always be written in the same bucket. Bucketing helps in joining various columns.
In partitioning technique, a partition is created for every unique value of the column and there could be a situation where several tiny partitions may have to be created. However, with bucketing, one can limit it to a specific number and the data can then be decomposed in those buckets.
Basically, a bucket is a file in Hive whereas partition is a directory.

2. **External** table: 
       
    - External table stores files on the HDFS server but tables are not linked to the source file completely.
    
    - If you delete an external table the file still remains on the HDFS server.
        
    - External table files are accessible to anyone who has access to HDFS file structure and therefore security needs to be managed at the HDFS file/folder level.
    
    - Meta data is maintained on master node, and deleting an external table from HIVE only deletes the metadata not the data/file.
    
3. **Internal** table:
    - Stored in a directory based on settings in hive.metastore.warehouse.dir, by default internal tables are stored in the following directory “/user/hive/warehouse” you can change it by updating the location in the config file .
    - Deleting the table deletes the metadata and data from master-node and HDFS respectively.
    - Internal table file security is controlled solely via HIVE. Security needs to be managed within HIVE, probably at the schema level (depends on organization).
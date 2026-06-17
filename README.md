## Attention Explaination

If we weighted sum all the previous contexts in the T-th token, we can then predict the next token only given the T-th token efficiently. So our most important task to find out the weights(masked) for the averaging operation. Ask yourself, which weights should be higher for average upto the T-th token, where the T-th token gets the averaged value ? Probably the ones the T-th token find more useful for prediction. So it spits out a query vector for all the tokens upto itelf to release their key vector for it to calculate a attention score, to be then softmaxed and and calculate the weights.

To calculate the weighted avg itself, each vector has a different set of vectors called values, onto which the avg is calculated

Two type of broad distinction can be made in the attention mechanism, between self attention and cross-attention and between encoder and decoder attention module.

To understand the difference between them, take a step back and think about how general the attention mechanism itself is: you give it s 'set' of vectors, some of the vectors are directed to some of the other vectors or itself. This forms a directed graph, which represents a single batch of attention. The set of all the vectors a each vector is pointed to by participates in calculating the attentyion score for this vector, i.e. when a vectors spits out a query, the keys and values are given by only the set of vectors it is being pointed to by. It does not need you to be autoregressive(i.e. causal) or fully connected or anything like that.

With this view in mind, the difference between the mechanisms should fairly simple. In self attention each vector in the set gives a query and a subset of vectors in the same set gives their key and values for each query. In cross attention, you can divide the vectors into two subsets, one of which gives out query but the keys and the values is only given by the other subset of vectors and the other set of vectors is pointed to by none, so they dont give query vectors. So the edges go from only one set of vectors to the other and neither the other way nor to any vector in itself.

The encoder and dedocer is most widely used but specific type of attention. The encoder is a complete directed graph, where each vector points to each of the vectors, including itself, so each one gives a query and all the vectors attend to it with keys and values. On the other hand decoder block is used for causal auto regressive tasks where the vectors are ordered and each vector is pointed by the all the vectors ordered upto itself.

Because or the exponential (peaky) nature of softmax, we used scaled dot product attention, other wise the softmax converges to one-hot vectors to the max value.
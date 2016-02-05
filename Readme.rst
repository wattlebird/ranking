======
Rankit
======
Game Start!

***************
What is rankit?
***************
Rankit is created for the purpose of a more "scientific" ranking of rankable objects.

We rank objects by giving objects a score, we call that score rating. Traditionally, people would generate ratings by calculating average score, median or some other statistical meaningful numbers. However, eventhough this method is widely accepted, it can have bias in some extreme cases. Average score would easily be manipulated if the number of scores are unrestricted. One cope to this cheat is weighting, but this can only leverage the problem but not solving it.

Here in rankit, we solve the problem by borrowing wisdoms from linear algebra. Every rankable objects are ranked based on their performance when they compare with each other. We compare each pair of rankable objects, and we generate a matrix. Then we get the ranking from that matrix, using different ranking methods. Most of the ranking methods are based on a same idea: ranking is a state that should be extracted from observations which are encoded in matrix. As you can see, rankit is a native solution in sport ranking field.

To further compete with ranking cheating, rankit also included ranking merging methods and provided measures to measure distance between different ranking results.

All the algorithms implemented in rankit have been described in `Who's \#1? The Science of Rating and Ranking <http://www.amazon.com/Whos-1-Science-Rating-Ranking/dp/0691154228/ref=sr_1_1?s=books&ie=UTF8&qid=1454383363&sr=1-1&keywords=who%27s+1+the+science+of+rating+and+ranking>`_. In fact, rankit is a sheer implementation of this book.

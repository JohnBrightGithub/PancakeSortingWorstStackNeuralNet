# PancakeSortingWorstStackNeuralNet 
#Introduction 
This repo is for a neural network used to find the "worst" stacks for the pancake sorting problem described here: https://en.wikipedia.org/wiki/Pancake_sorting. By worst we mean the stack with the greatest distance from the sorted stack. 

For a brief background, a pancake stack is a permutation of the numbers 1-n. The stack can be sorted by prefix flips. Ie: [1,3,2] can be flipped to become [3,1,2] or [2,3,1] only. [1,3,2] is the worst pancake stack for n=3. This is better illustrated by the picture found on the wikipedia page:

![](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Pancake_graph_g3.svg/1024px-Pancake_graph_g3.svg.png)

In the pancake graph the nodes represent stacks and the edges represent flips. As can be seen in the image [1,3,2] requires the most flips to get to the sorted stack [1,2,3].

An adjacency is counted when two consecutive elements of the permutation have a difference of 1 OR the last element is n. For instance [1,2,3] has 3 adjacencies, [2,1,3] has 2 adjacencies and [1,3,2] has 0. 

An efficient flip is one that increases the number of adjacencies and a waste is a flip that does not increase the number of adjacencies. 

Since a flip can at most add 1 adjacency, the distance of any stack is at least n-#of adjacencies


#Prior work
There has been much work in finding bounds for the distance of the worst stack for a given value of n.  Bill Gates and Christos Papadimitriou gave the first lower bound of 17/16n. This was done by creating “chunks” of length 7. The chunks are created starting with the stack [1,7,5,3,6,4,2] and for the next chunk we add 7 to each entry. This makes the stack with 2 chunks: [1,7,5,3,6,4,2,8,14,12,10,13,11,9] Then it is proven that for every 16 elements that one waste flip must be performed. The current best known bounds are 15/14 n and 18/11 n. 

The paper proving the 15/14n bound by Mohammad H. Heydari and I. Hal Sudborough uses the same idea and the same stack, but proves that every 14 elements requires a waste flip. In the paper they mention that 4 chunks [1,7,5,3,6,4,2] has distance 31. They conjecture that the stack made of these chunks has a lower bound of 8/7n-1.

The similar stack [1,7,5,3,6,2,4] is not mentioned in this paper, but at 4 chunks it has distance 32. Later in this readme we’ll post other stacks of length 28 with distance 32, as well as stacks with length 26 and distance 30.

#Neural Network 
Data fed into the neural network is as follows:
We take a stack with known distance of length n, then create a new stack by
appending {n+1} on to the end.
Flip the whole stack
Complete another random flip between 2 and n elements.

For example   
[18, 16, 1, 3, 7, 5, 2, 6, 4, 8, 12, 14, 10, 13, 11, 9, 15, 17]  
[18, 16, 1, 3, 7, 5, 2, 6, 4, 8, 12, 14, 10, 13, 11, 9, 15, 17, 19]  
[19, 17, 15, 9, 11, 13, 10, 14, 12, 8, 4, 6, 2, 5, 7, 3, 1, 16, 18]  
[10, 13, 11, 9, 15, 17, 19, 14, 12, 8, 4, 6, 2, 5, 7, 3, 1, 16, 18]  
  
Because appending {n+1} onto a stack doesn’t affect the distance of a stack, and two flips were used, the distance of the new stack is within 2 of the original. 

We put a label with every pair of stacks generated this way, 0-4. 0 if the new stack’s distance is 2 less and 4 if the new stack’s distance is 2 more.

The two stacks are also turned into 0-1 matrices with 35 columns and 35 rows so the previous two stacks would become:

[[0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
  [0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]  
 
Since the original stack has distance 20 and the new stack has distance 21, we give this pair the label 3.

#How to Run 
The first script to run is the training and testing script, type:

```
python -m TestPancakeWorstStack
```
with optional parameters:

"-mc", "--MaxColumns" - max columns for neural network data (default = 35)  
"-tf", "--TrainFirst" - train a new model before testing (default = False)  
"-it", "--Iterations" - number of iterations for the test (default = 1000)  
"-n", "--TestN"       - n value for test (default = 19)  
"-d", "--TestDist"    - dist value for test (default = 22)  
(currently training parameters have to be changed manually in the WorstStackNN/makeTrainingData.py file)
The terminal will print out the percentage accuracy of the neural net compared to the accuracy of choosing randomly.

The next script to run is the one that gets results.  
First it takes each stack from the chosen n and dist values, then it generates every stack that can be made by:  
    appending {n+1} on to the end.  
    Flip the whole stack  
    Complete another flip between 2 and (n-1) elements.  
(for n=27 this will create 26 new stacks)  
Then the neural network model is used to determine which stacks are good candidates (ie: they have distance 2 more than the original stack)  
To run:
```
python -m getResults
```
with optional paramters:

"-p", "--nPlus" - number 0-4 representing the minimum threshold (-2 to +2 original distance) for which stacks to calculate distance for (default = 4)  
"-xM", "--xMoreThanN" - minimum threshold (-2 to +2 original distance) for which stacks to write to Results folder (default = -2)  
"-n", "--n" - n value of stacks to test (default = 21)  
"-dst", "--origDist" - dist value of stacks to test (default = 23)  
"-parts", "--Parts" - for large files, reads file in #parts parts (default = 1)  
"-cores", "--numCores" - number of cores to use during (default = maxCores)  
"-pd", "--PrintDiff" - set to True to print arrays that are removed when removing duplicates (default = False)   

The results of this are put into the Results folder.  

Finally we have a brute force script (checkAll) for stacks that are made out of k chunks of size 7 or what we call mixed stacks which are made of (k-1) chunks of size 7 and 1 chunk of size 5.  
Here is an example of a stack made of 4 chunks of size 7 (here the chunk is 1 7 5 3 6 2 4)  
1 7 5 3 6 2 4 8 14 12 10 13 9 11 15 21 19 17 20 16 18 22 28 26 24 27 23 25  

And here is a mixed stack with 3 chunks of size 7 and 1 chunk of size 5:  
1 7 5 3 6 2 4 8 14 12 10 13 9 11 15 21 19 17 20 16 18 22 25 23 26 24  

To run the bruteForce with 4 chunks of size 7:

```
python -m checkAll -k 4 -mc True
```
The results will be put into BruteForce\\testchunks.txt

To run the bruteForce with mixed stacks with 3 chunks of size 7 and 1 chunk of size 5:

```
python -m checkAll -k 3 -mc False
```
The results will be put into BruteForce\\testchunksMixed.txt

Both commands check the distance using all permutations of size 7 with 0 adjacencies (or the two permutations of size 5 with no adjacencies)

#Results 
The following 18 stacks of length 28 and dist 32 have been found using the getEstimates on n=27, dist=30

1 3 7 5 2 6 4 8 14 11 13 10 12 9 15 23 21 18 20 17 19 22 16 24 26 28 25 27  
1 3 7 5 2 6 4 8 13 11 14 10 12 9 15 23 21 18 20 17 19 22 16 24 26 28 25 27  
1 3 7 5 2 6 4 8 14 12 10 13 11 9 15 23 21 18 20 17 19 22 16 24 26 28 25 27  
1 3 7 5 2 6 4 8 14 12 10 13 9 11 15 23 21 18 20 17 19 22 16 24 26 28 25 27  
1 3 7 5 2 6 4 8 12 10 14 9 13 11 15 23 21 18 20 17 19 22 16 24 26 28 25 27  
1 5 2 7 4 6 3 8 12 14 10 13 11 9 15 23 21 18 20 17 19 22 16 24 26 28 25 27  
1 5 2 7 4 6 3 8 13 11 14 10 12 9 15 23 21 18 20 17 19 22 16 24 26 28 25 27  
1 5 2 7 4 6 3 8 12 10 14 9 13 11 15 23 21 18 20 17 19 22 16 24 26 28 25 27  
1 3 7 5 2 6 4 8 12 14 10 13 11 9 15 23 21 18 20 17 19 22 16 24 26 28 25 27  
1 3 7 5 2 6 4 8 14 11 13 9 12 10 15 23 21 18 20 17 19 22 16 24 26 28 25 27  
1 7 4 6 2 5 3 8 14 11 13 9 12 10 15 21 18 20 16 19 17 22 28 25 27 23 26 24  
1 5 7 3 6 4 2 8 12 14 10 13 11 9 15 19 21 17 20 18 16 22 26 28 24 27 25 23  
1 5 2 7 4 6 3 8 12 9 14 11 13 10 15 19 16 21 18 20 17 22 26 23 28 25 27 24  
1 3 7 5 2 6 4 8 10 14 12 9 13 11 15 17 21 19 16 20 18 22 24 28 26 23 27 25  
1 6 4 7 3 5 2 8 13 11 14 10 12 9 15 20 18 21 17 19 16 22 27 25 28 24 26 23  
1 5 3 7 4 2 6 8 12 10 14 11 9 13 15 19 17 21 18 16 20 22 26 24 28 25 23 27  
1 6 3 7 5 2 4 8 13 10 14 12 9 11 15 20 17 21 19 16 18 22 27 24 28 26 23 25  
1 5 7 4 2 6 3 8 12 14 11 9 13 10 15 19 21 18 16 20 17 22 26 28 25 23 27 24   
1 6 3 5 2 7 4 8 13 10 12 9 14 11 15 20 17 19 16 21 18 22 27 24 26 23 28 25  
1 7 5 3 6 2 4 8 14 12 10 13 9 11 15 21 19 17 20 16 18 22 28 26 24 27 23 25  
  
10 of these stacks are made by repeating chunks of size 7 for 4 times (similar to what is done in the Gates and Papadimitriou paper) and 8 are given by the getEstimates script.

We have found 4 stacks of length 26 and distance 30 by repeating 3 chunks of size 7, then adding on a chunk of size 5 with 0 adjacencies:

1 6 4 7 3 5 2 8 13 11 14 10 12 9 15 20 18 21 17 19 16 22 24 26 23 25  
1 6 4 7 3 5 2 8 13 11 14 10 12 9 15 20 18 21 17 19 16 22 25 23 26 24  
1 7 5 3 6 2 4 8 14 12 10 13 9 11 15 21 19 17 20 16 18 22 24 26 23 25  
1 7 5 3 6 2 4 8 14 12 10 13 9 11 15 21 19 17 20 16 18 22 25 23 26 24  

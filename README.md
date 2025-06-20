# SiMPle-Fast-Extension
Two implementations of the SiMPle-Fast method for retrieving matrix profiles on multi-dimensional data. 
These implementations can retrieve the top *k* closest matching windows (instead of the single closest window) and include cosine distance as an additional metric.

## Implementation 1 (simple_fast.py):

Given a data matrix and query matrix, find the *k* nearest windows in the query matrix for all sliding windows in the data matrix.
The *k* nearest query windows are outputted in ascending order.

## Implementation 2 (multi_fast.py):

Given any number of data and query features (which represent the same underlying time-series data), find the *k* nearest windows in the query matrix for all sliding windows in the data matrix by combining the distance values of all features. 

- Each data-query feature pair is given a weight
- The distance values of the individual data-query feature pairs can be combined with the following methods:
	- Arithmetic Mean
	- Geometric Mean
	- Harmonic Mean
	- Root Mean Square
	- Softmax

## References

These implementations expand on the following work:

1. 
	@ARTICLE{8392419,
 	author = {Silva, Diego F. and Yeh, Chin-Chia M. and Zhu, Yan and Batista, Gustavo E. A. P. A. and Keogh, Eamonn},
  	journal={IEEE Transactions on Multimedia}, 
  	title={Fast Similarity Matrix Profile for Music Analysis and Exploration}, 
 	year={2019},
  	volume={21},
  	number={1},
  	pages={29-38},
  	doi={10.1109/TMM.2018.2849563}}

   
2. https://github.com/acmiyaguchi/birdclef-2021/tree/main/simple

# SiMPle-Fast-Extension
Two implementations of the SiMPle-Fast method for retrieving similarity matrix profiles on multi-dimensional data. These implementations include cosine distance as an additional metric. Furthermore, instead of returning the best single match per window, these implementations retrieve an arbitrary top 'x' matches per window in ascending order.

Implementation 1 (simple_fast.py):

  Given query and reference time-series data, find the ‘x’ lowest distance windows in the reference data, for each sliding window in the ‘query’ data.

Implementation 2 (multi_fast.py):

  Given some number of query feature matrices, reference feature matrices, and weights associated with each feature matrix pair, find the ‘x’ lowest distance windows in the reference data by combining the information   of the feature matrix pairs, for each sliding window in the ‘query’.

  Note that, the all feature matrices associated with the query (or the reference) must have the same number of rows (aka, the same time-axis size). For a query / reference feature matrix pair, these matrices           must have the same number of columns (aka. the same feature dimensions). 

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
2. 
	https://github.com/acmiyaguchi/birdclef-2021/tree/main/simple

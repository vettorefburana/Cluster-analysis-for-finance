# Cluster analysis for building diversified portfolios 

Objectives: 
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The returns of approximately 200 randomly selected stocks are divided into clusters based on return correlation, using the following partitional and hierarchical clustering algorithms:

* K-means 
* Bounded K-means
* Hierarchical risk parity

The goodness of each clustering algorithm is evaluated on the basis of the infra- and intra-cluster correlation.

The assets in each cluster are aggregated into equally weighted portfolios. A cluster portfolio is then formed by calculating the maximum sharpe ratio weights for each cluster.

In addition to the static optimization, the optimal portfolio weights are calculated on daily rolling windows of annual width, so that the portfolios are rebalaced daily.

For each clustering algorithm, the results of the static and rolling allocations are compared in order to calculate the excess return due to rebalancing the portfolio.

The out of sample performance of each clustering portfolio is compared with the tangency portfolio and with an equally weighted portfolio. 

Results: 
-----------------------------------------------------------------------------------------------------------------------------------------------------------------
* The Bounded K-Means portfolio yields the highest excess-return; 
* The tangency portfolio has the highest Sharpe ratio

References: 
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

* De Prado, M. L. (2016). Building diversified portfolios that outperform out of sample. The Journal of Portfolio Management, 42(4), 59-69.
* Ganganath, N., Cheng, C. T., & Chi, K. T. (2014). Data clustering with cluster size constraints using a modified k-means algorithm. In 2014 International Conference on Cyber-Enabled Distributed Computing and Knowledge Discovery (pp. 158-161). IEEE.
* Markowitz H. (1959). Portfolio Selection: Efficient Diversification of Investment. (J. Wiley, New York).
* Tola, V., Lillo, F., Gallegati, M., & Mantegna, R. N. (2008). Cluster analysis for portfolio optimization. Journal of Economic Dynamics and Control, 32(1), 235-258.
* https://github.com/hellojinwoo/CA_GMVP

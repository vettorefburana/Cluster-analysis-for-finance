# Cluster analysis for building diversified portfolios 

The returns of approximately 200 randomly selected stocks are divided into clusters based on return correlation, using the following partitional and hierarchical clustering algorithms:

* K-means 
* Bounded K-means
* Hierarchical risk parity

The goodness of each clustering algorithm is evaluated on the basis of the infra- and intra-cluster correlation.

The assets in each cluster are aggregated into equally weighted portfolios. A cluster portfolio is then formed by calculating the maximum sharpe ratio weights for each cluster.

In addition to the static optimization, the optimal portfolio weights are calculated on daily rolling windows of annual width, so that the portfolios are rebalaced daily.

For each clustering algorithm, the results of the static and rolling allocations are compared in order to calculate the excess return due to rebalancing the portfolio.

The out of sample performance of each clustering portfolio is compared with the tangency portfolio and with an equally weighted portfolio. 

###############################################################################

from pandas import DataFrame



class StatisticsHelper():

	@staticmethod
	def getFeatureStats(series):
		stats = series.describe()
		stats["nan_count"] = len(series)-series.count()
		stats["nb_unique_values"] = series.nunique()
		return DataFrame(stats)

	@staticmethod
	def drawFeatureDistribution(series, col):
		min_bins = min(20, series.nunique())
		return series.plot.hist(bins=min_bins, title=col, grid=True,
			                    rwidth=0.99)
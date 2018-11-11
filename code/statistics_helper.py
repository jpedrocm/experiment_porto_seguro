###############################################################################

from pandas import DataFrame



class StatisticsHelper():

	@staticmethod
	def getFeatureStats(series):
		stats = series.describe()
		stats["nan_count"] = len(series)-series.count()
		stats["nb_unique_values"] = series.nunique()
		stats["col_nan_rate"] = stats["nan_count"]/stats["count"]
		stats["std_fraction_of_mean"] = stats["std"]/stats["mean"]
		stats["dtype"] = series.dtype
		return DataFrame(stats)

	@staticmethod
	def drawFeatureDistribution(series, col):
		min_bins = min(20, series.nunique())
		return series.plot.hist(bins=min_bins, title=col, grid=True,
								ylim=(0, 600000))
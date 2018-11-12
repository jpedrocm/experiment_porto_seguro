###############################################################################

from pandas import DataFrame



class StatisticsHelper():

	@staticmethod
	def get_feature_stats(series):
		stats = series.describe()
		stats["mode"] = series.mode().iloc[0]
		stats["median"] = series.median(skipna=True, numeric_only=False)
		stats["std_fraction_of_mean"] = stats["std"]/stats["mean"]
		stats["nan_count"] = len(series)-series.count()
		stats["col_nan_rate"] = stats["nan_count"]/stats["count"]
		stats["nb_unique_values"] = series.nunique()
		stats["dtype"] = series.dtype
		return DataFrame(stats)

	@staticmethod
	def draw_feature_distribution(series, col):
		min_bins = min(20, series.nunique())
		return series.plot.hist(bins=min_bins, title=col, grid=True,
								ylim=(0, 900000))
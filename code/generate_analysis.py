###############################################################################

from io_helper import IOHelper
from statistics_helper import StatisticsHelper



if __name__ == "__main__":

	train_data = IOHelper.read_dataset("train")
	for col in train_data.columns:
		series= train_data[col]
		stats = StatisticsHelper.getFeatureStats(series)
		StatisticsHelper.drawFeatureDistribution(series, col)
		IOHelper.store_analysis(stats, col)
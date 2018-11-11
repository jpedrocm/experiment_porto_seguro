###############################################################################

from io_helper import IOHelper
from statistics_helper import StatisticsHelper



if __name__ == "__main__":

	train_data = IOHelper.read_dataset("train")
	for col in train_data.columns:
		series= train_data[col]
		stats = StatisticsHelper.get_feature_stats(series)
		StatisticsHelper.draw_feature_distribution(series, col)
		IOHelper.store_analysis(stats, col)
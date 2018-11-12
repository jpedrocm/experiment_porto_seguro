###############################################################################

from io_helper import IOHelper
from statistics_helper import StatisticsHelper
from data_helper import DataHelper
from config_helper import ConfigHelper



if __name__ == "__main__":

	dataset_name = ConfigHelper.analysis_dataset

	train_data = IOHelper.read_dataset(dataset_name)
	for col in train_data.columns:
		series= train_data[col]

		stats = StatisticsHelper.get_feature_stats(series)
		StatisticsHelper.draw_feature_distribution(series, col)
		IOHelper.store_analysis(stats, col, dataset_name)

	DataHelper.fill_missing_data(train_data, is_train=True)
	for col in train_data.columns:
		series= train_data[col]

		col = col + "_filled"
		stats = StatisticsHelper.get_feature_stats(series)
		StatisticsHelper.draw_feature_distribution(series, col)
		IOHelper.store_analysis(stats, col, dataset_name)
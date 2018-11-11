###############################################################################

from io_helper import IOHelper
from data_helper import DataHelper
from  config_helper import ConfigHelper



if __name__ == "__main__":

	train_data = IOHelper.read_dataset("train")
	train_X, train_y = DataHelper.extract_feature_labels(train_data)

	predef = ConfigHelper.use_predefined_cols

	DataHelper.add_nan_indication_cols(train_X)
	DataHelper.remove_high_correlation_cols(train_X, predef)
	DataHelper.remove_high_nan_rate_cols(train_X, predef)
	DataHelper.remove_small_variance_cols(train_X, predef)

	train_y = DataHelper.remove_high_nan_rate_rows(train_X, train_y)
	DataHelper.fill_missing_data(train_X, is_train=True)
	DataHelper.split_categorical_cols(train_X, inplace=True, is_train=True)
	DataHelper.scale_continuous_cols(train_X, inplace=True, is_train=True)
	DataHelper.select_best_features(train_X, inplace=True, is_train=True)

	test_X = IOHelper.read_dataset("test")

	DataHelper.add_nan_indication_cols(test_X)
	DataHelper.remove_high_correlation_cols(test_X, predef)
	DataHelper.remove_high_nan_rate_cols(test_X, predef)
	DataHelper.remove_small_variance_cols(test_X, predef)

	DataHelper.fill_missing_data(test_X, is_train=False)
	DataHelper.split_categorical_cols(test_X, inplace=True, is_train=False)
	DataHelper.scale_continuous_cols(test_X, inplace=True, is_train=False)
	DataHelper.select_best_features(test_X, inplace=True, is_train=False)

	for name, model in ConfigHelper.get_models():

		print "Training"
		model.fit(train_X, train_y)

		print "Assessing"
		predicted = model.predict(test_X)


		print "Saving submission"
		IOHelper.store_submission(metrics, name)
###############################################################################

from io_helper import IOHelper
from data_helper import DataHelper
from  config_helper import ConfigHelper



if __name__ == "__main__":

	print "Reading dataset"
	data = IOHelper.read_dataset("train")
	feats, labels = DataHelper.extract_feature_labels(data)

	print "Making data adjustments"
	predef = ConfigHelper.use_predefined_cols

	DataHelper.remove_small_variance_cols(feats, predef, inplace=True)
	DataHelper.remove_high_correlation_cols(feats, predef, inplace=True)
	DataHelper.remove_high_nan_rate_cols(feats, predef, inplace=True)

	for train_idxs, val_idxs in ConfigHelper.k_fold_cv(labels):
		train_X = DataHelper.select_rows(feats, train_idxs, copy=True)
		train_y = DataHelper.select_rows(labels, train_idxs, copy=False)
		val_X = DataHelper.select_rows(feats, val_idxs, copy=True)
		val_y = DataHelper.select_rows(labels, val_idxs, copy=False)

		print "Applying feature selection"
		DataHelper.remove_high_nan_rate_rows(train_X, inplace=True)
		DataHelper.fill_missing_data(train_X, inplace=True, is_train=True)
		DataHelper.split_categorical_cols(train_X, inplace=True, is_train=True)
		DataHelper.normalize_data(train_X, inplace=True, is_train=True)
		DataHelper.select_best_features(train_X, inplace=True, is_train=True)

		DataHelper.fill_missing_data(val_X, inplace=True, is_train=False)
		DataHelper.split_categorical_cols(val_X, inplace=True, is_train=False)
		DataHelper.normalize_data(train_X, inplace=True, is_train=False)
		DataHelper.select_best_features(val_X, inplace=True, is_train=False)

		for model in ConfigHelper.get_models():

			print "Training model"
			model.fit(train_X, train_y)

			print "Assessing model"
			predicted = model.predict(val_X)
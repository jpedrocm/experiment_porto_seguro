###############################################################################

from copy import deepcopy



class DataHelper():
	small_variance_cols = []
	high_correlation_cols = []
	high_nan_rate_cols = []
	label_col = "target"

	@staticmethod
	def _remove_cols(dataframe, inplace, cols):
		return dataframe.drop(columns=cols, inplace=inplace)

	@staticmethod
	def extract_feature_labels(dataframe):
		labels = dataframe["target"]
		feats = dataframe.drop(columns=DataHelper.label_col)
		return feats, labels

	@staticmethod
	def remove_small_variance_cols(dataframe, inplace):
		return DataHelper._remove_cols(dataframe, inplace, 
			                           DataHelper.small_variance_cols)

	@staticmethod
	def remove_high_correlation_cols(dataframe, inplace):
		return DataHelper._remove_cols(dataframe, inplace, 
			                           DataHelper.high_correlation_cols)

	@staticmethod
	def remove_high_nan_rate_cols(dataframe, inplace):
		return DataHelper._remove_cols(dataframe, inplace, 
			                           DataHelper.high_nan_rate_cols)

	@staticmethod
	def remove_high_nan_rate_rows(dataframe, inplace):
		pass

	@staticmethod
	def fill_missing_data(dataframe, inplace, is_train):
		pass

	@staticmethod
	def split_categorical_cols(dataframe, inplace, is_train):
		pass

	@staticmethod
	def select_best_features(dataframe, inplace, is_train):
		pass

	@staticmethod
	def select_rows(dataframe, idxs):
		return deepcopy(dataframe.iloc[idxs])
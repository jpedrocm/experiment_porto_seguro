###############################################################################

from copy import deepcopy

from pandas import get_dummies



class DataHelper():
	small_var_cols = []
	high_corr_cols = []
	high_nan_rate_cols = []
	categorical_cols = []
	best_cols = []

	label_col = "target"
	min_full_rate_row = 0.8
	max_nan_rate_col = 0.9
	small_var_rate = 0.1

	@staticmethod
	def _remove_cols(df, inplace, cols):
		return df.drop(columns=cols, inplace=inplace)

	@staticmethod
	def _select_small_variance_cols(df):
		new_df = df.loc[:, df.std() < DataHelper.small_var_rate*df.mean()]
		return new_df.columns

	@staticmethod
	def _select_high_nan_rate_cols(df):
		new_df = df.loc[:, df.count() < len(df)*DataHelper.max_nan_rate_col]
		return new_df.columns

	@staticmethod
	def _select_high_correlation_cols(df):
		#TODO
		pass

	@staticmethod
	def extract_feature_labels(dataframe):
		labels = dataframe["target"]
		feats = dataframe.drop(columns=DataHelper.label_col)
		return feats, labels

	@staticmethod
	def remove_small_variance_cols(dataframe, predef_cols, inplace):
		if predef_cols==False:
			DataHelper.small_var_cols = DataHelper._select_small_variance_cols(dataframe)

		return DataHelper._remove_cols(dataframe, inplace, 
									   DataHelper.small_var_cols)

	@staticmethod
	def remove_high_nan_rate_cols(dataframe, predef_cols, inplace):
		if predef_cols==False:
			DataHelper.high_nan_rate_cols = DataHelper._select_high_nan_rate_cols(dataframe)

		return DataHelper._remove_cols(dataframe, inplace, 
									   DataHelper.high_nan_rate_cols)

	@staticmethod
	def remove_high_correlation_cols(dataframe, predef_cols, inplace):
		#if predef_cols==False:
		#	DataHelper.high_corr_cols = DataHelper._select_high_correlation_cols(dataframe)

		#return DataHelper._remove_cols(dataframe, inplace, 
		#							   DataHelper.high_corr_cols)
		pass

	@staticmethod
	def select_rows(dataframe, idxs, copy):
		sel = dataframe.iloc[idxs]
		return deepcopy(sel) if copy == True else sel

	@staticmethod
	def remove_high_nan_rate_rows(dataframe, inplace):
		all_nb_cols = len(dataframe.columns)
		min_nb_full_cols = int(DataHelper.min_full_rate_row*all_nb_cols)
		return dataframe.dropna(thresh=min_nb_full_cols, inplace=inplace)

	@staticmethod
	def fill_missing_data(dataframe, inplace, is_train):
		pass

	@staticmethod
	def split_categorical_cols(dataframe, inplace, is_train):
		pass

	@staticmethod
	def normalize_data(dataframe, inplace, is_train):
		pass

	@staticmethod
	def select_best_features(dataframe, inplace, is_train):
		pass
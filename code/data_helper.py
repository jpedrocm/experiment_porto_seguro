###############################################################################

from copy import deepcopy

from pandas import get_dummies



class DataHelper():
	small_var_cols = []
	high_corr_cols = []
	high_nan_rate_cols = []
	categorical_cols = []
	continuous_cols = []
	mean_cols = []
	mode_cols = []
	median_cols = []
	best_cols = []

	fill_vals = {}

	label_col = "target"
	min_full_rate_row = 0.8
	max_nan_rate_col = 0.9
	small_var_rate = 0.05

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
	def _fill_with_mean(df, is_train):
		for col in DataHelper.mean_cols:
			if is_train==True:
				value = df[col].mean()
				DataHelper.fill_vals[col] = value
			df[col].fillna(value=value, inplace=True)

	@staticmethod
	def _fill_with_mode(df, is_train):
		for col in DataHelper.mode_cols:
			if is_train==True:
				value = df[col].mode()
				DataHelper.fill_vals[col] = value
			df[col].fillna(value=value, inplace=True)

	@staticmethod
	def _fill_with_median(df, is_train):
		for col in DataHelper.median_cols:
			if is_train==True:
				value = df[col].median(skipna=True)
				DataHelper.fill_vals[col] = value
			df[col].fillna(value=value, inplace=True)

	@staticmethod
	def extract_feature_labels(dataframe):
		print "Extracting features and labels"

		labels = dataframe["target"]
		feats = dataframe.drop(columns=DataHelper.label_col)

		print "Columns: " + str(len(feats.columns))
		print "Rows: " + str(len(labels))
		return feats, labels

	@staticmethod
	def remove_small_variance_cols(dataframe, predef_cols):
		print "Removing small variance columns"

		if predef_cols==False:
			DataHelper.small_var_cols = DataHelper._select_small_variance_cols(dataframe)
		DataHelper._remove_cols(dataframe, True, DataHelper.small_var_cols)

		print "Columns: " + str(len(dataframe.columns))

	@staticmethod
	def remove_high_nan_rate_cols(dataframe, predef_cols):
		print "Removing highly NaN rated columns"

		if predef_cols==False:
			DataHelper.high_nan_rate_cols = DataHelper._select_high_nan_rate_cols(dataframe)
		DataHelper._remove_cols(dataframe, True, DataHelper.high_nan_rate_cols)

		print "Columns: " + str(len(dataframe.columns))

	@staticmethod
	def remove_high_correlation_cols(dataframe, predef_cols):
		print "Removing highly correlated columns"

		#if predef_cols==False:
		#	DataHelper.high_corr_cols = DataHelper._select_high_correlation_cols(dataframe)

		#DataHelper._remove_cols(dataframe, True, DataHelper.high_corr_cols)

		print "Columns: " + str(len(dataframe.columns))

	@staticmethod
	def add_nan_indication_cols(dataframe):
		print "Adding NaN indication columns"

		cols = dataframe.columns
		for col in cols:
			if dataframe[col].count() < len(dataframe[col]):
				dataframe["nan_"+col] = dataframe[col].notna()
				dataframe["nan_"+col]=dataframe["nan_"+col].astype(int)

		print "Columns: " + str(len(dataframe.columns))

	@staticmethod
	def select_rows(dataframe, idxs, copy):
		print "Selecting rows of data"

		sel = dataframe.iloc[idxs]
		sel = deepcopy(sel) if copy == True else sel

		print "Rows: " + str(len(sel))
		return sel

	@staticmethod
	def remove_high_nan_rate_rows(dataframe_x, dataframe_y):
		print "Removing rows with high NaN rate"

		all_nb_cols = len(dataframe_x.columns)
		min_nb_full_cols = int(DataHelper.min_full_rate_row*all_nb_cols)
		dataframe_x.dropna(thresh=min_nb_full_cols, inplace=True)
		new_labels = dataframe_y.loc[dataframe_x.index]

		print "Rows: " + str(len(dataframe_x))
		return new_labels

	@staticmethod
	def fill_missing_data(dataframe, is_train):
		print "Filling missing data with mean, mode and median"
		dataframe.fillna(0, inplace=True)

		#DataHelper._fill_with_mean(dataframe, is_train)
		#DataHelper._fill_with_mode(dataframe, is_train)
		#DataHelper._fill_with_median(dataframe, is_train)

	@staticmethod
	def split_categorical_cols(dataframe, inplace, is_train):
		print "Splitting categorical columns to binary"
		print "Columns: " + str(len(dataframe.columns))

	@staticmethod
	def scale_continuous_cols(dataframe, inplace, is_train):
		print "Scaling continuous columns"

	@staticmethod
	def select_best_features(dataframe, inplace, is_train):
		print "Selecting best features"

		if is_train==True:
			#select best cols
			#assigned best cols to static variable
			pass
		else:
			pass
			#drop all columns not in best_cols
			#add extra columns of best_cols
			#return dataframe.drop(columns=DataHelper.best_cols,
			#					  inplace=inplace)
			#reindex columns in dataframe

		print "Columns: " + str(len(dataframe.columns))
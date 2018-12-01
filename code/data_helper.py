###############################################################################

from copy import deepcopy

from pandas import get_dummies
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif



class DataHelper():
	small_var_cols = []
	high_corr_cols = []
	high_nan_rate_cols = []
	categorical_cols = ["ps_reg_02", "ps_reg_01", "ps_ind_15", "ps_ind_14",
						"ps_ind_05_cat", "ps_ind_03", "ps_ind_02_cat", 
						"ps_ind_01"]
	continuous_cols = ["ps_reg_03"]
	mean_cols = []
	mode_cols = ["ps_ind_05_cat", "ps_ind_04_cat", "ps_ind_02_cat", 
				 "ps_reg_03"]
	median_cols = []
	best_cols = []

	fill_vals = {}

	label_col = "target"
	min_full_rate_row = 0.8
	max_nan_rate_col = 0.9
	small_var_rate = 0.05

	continuous_scaler = StandardScaler()

	@staticmethod
	def _remove_cols(df, inplace, cols):
		return df.drop(columns=cols, inplace=inplace)

	@staticmethod
	def _select_small_variance_cols(df):
		new_df = df.loc[:, df.std() < DataHelper.small_var_rate*df.mean()]
		return new_df.columns.tolist()

	@staticmethod
	def _select_high_nan_rate_cols(df):
		new_df = df.loc[:, df.count() < len(df)*DataHelper.max_nan_rate_col]
		return new_df.columns.tolist()

	@staticmethod
	def _fill_with_mean(df, is_train):
		for col in DataHelper.mean_cols:
			if col in df.columns:
				if is_train==True:
					value = df[col].mean()
					DataHelper.fill_vals[col] = value
				df[col].fillna(value=DataHelper.fill_vals[col], inplace=True)

	@staticmethod
	def _fill_with_mode(df, is_train):
		for col in DataHelper.mode_cols:
			if col in df.columns:
				if is_train==True:
					value = df[col].mode().iloc[0]
					DataHelper.fill_vals[col] = value
				df[col].fillna(value=DataHelper.fill_vals[col], inplace=True)

	@staticmethod
	def _fill_with_median(df, is_train):
		for col in DataHelper.median_cols:
			if col in df.columns:
				if is_train==True:
					value = df[col].median(skipna=True)
					DataHelper.fill_vals[col] = value
				df[col].fillna(value=DataHelper.fill_vals[col], inplace=True)

	@staticmethod
	def _get_different_cols(cols_big, cols_short):
		short_set = set(cols_short)
		return [col for col in cols_big if col not in short_set]	

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

		#DataHelper._fill_with_mean(dataframe, is_train)
		DataHelper._fill_with_mode(dataframe, is_train)
		#DataHelper._fill_with_median(dataframe, is_train)
		dataframe.fillna(0, inplace=True)

	@staticmethod
	def split_categorical_cols(dataframe, is_train):
		print "Splitting categorical columns to binary"

		new_df = get_dummies(dataframe, columns=DataHelper.categorical_cols,
					   		dummy_na=False, drop_first=True)

		print "Columns: " + str(len(new_df.columns))

		return new_df

	@staticmethod
	def reset_scaler():
		DataHelper.continuous_scaler = StandardScaler()

	@staticmethod
	def scale_continuous_cols(dataframe, is_train):
		print "Scaling continuous columns"

		copy_cols = [c for c in DataHelper.continuous_cols if c in dataframe.columns]
		DataHelper.continuous_cols = copy_cols

		if len(DataHelper.continuous_cols) > 0:
			if is_train==True:
				dataframe[DataHelper.continuous_cols] = DataHelper.continuous_scaler.fit_transform(
						dataframe[DataHelper.continuous_cols].values)
			else:
				dataframe[DataHelper.continuous_cols] = DataHelper.continuous_scaler.transform(
						dataframe[DataHelper.continuous_cols].values)

	@staticmethod
	def select_best_features(main_dataframe, train_dataframe, labels,
							 k, is_train):
		print "Selecting best features"

		if is_train==True:
			selector = SelectKBest(f_classif, k=int(k))
			selector.fit(main_dataframe.values, labels.values)
			selected_mask = selector.get_support()
			selected_idxs = [i for i in range(len(selected_mask)) \
							 if selected_mask[i]==True]
			selected_cols = list(main_dataframe.columns[selected_idxs])
			DataHelper.best_cols = selected_cols
			return main_dataframe[selected_cols]
		else:
			test_cols = main_dataframe.columns.tolist()
			extra_test_cols = DataHelper._get_different_cols(
													test_cols, 
													DataHelper.best_cols)
			DataHelper._remove_cols(main_dataframe, True, extra_test_cols)

			test_cols = main_dataframe.columns.tolist()
			unnecessary_train_cols = DataHelper._get_different_cols(
													DataHelper.best_cols,
													test_cols)
			DataHelper._remove_cols(train_dataframe, True, unnecessary_train_cols)

			if main_dataframe.columns.tolist()!=train_dataframe.columns.tolist():
				raise IndexError("Erro de ordem")

		print "Columns: " + str(len(main_dataframe.columns))
###############################################################################

# Do not change this for reproducibility
import random as rnd
from numpy import random as rnp
rnd.seed(2789)
rnp.seed(3056)
########################################

import time

from io_helper import IOHelper
from data_helper import DataHelper
from config_helper import ConfigHelper
from metrics_helper import MetricsHelper



if __name__ == "__main__":

	data = IOHelper.read_dataset("train")
	feats, labels = DataHelper.extract_feature_labels(data)

	predef = ConfigHelper.use_predefined_cols

	DataHelper.add_nan_indication_cols(feats)
	DataHelper.remove_high_nan_rate_cols(feats, predef)
	DataHelper.remove_small_variance_cols(feats, predef)

	for e in xrange(ConfigHelper.nb_executions):
		print "Execution: " + str(e+1)

		MetricsHelper.reset_metrics()

		for f, (train_idxs, val_idxs) in enumerate(ConfigHelper.k_fold_cv(labels)):
			start_time = time.time()
			print "Fold: " + str(f+1)

			DataHelper.reset_scaler()

			train_X = DataHelper.select_rows(feats, train_idxs, copy=True)
			train_y = DataHelper.select_rows(labels, train_idxs, copy=False)
			val_X = DataHelper.select_rows(feats, val_idxs, copy=True)
			val_y = DataHelper.select_rows(labels, val_idxs, copy=False)

			train_y = DataHelper.remove_high_nan_rate_rows(train_X, train_y)

			DataHelper.fill_missing_data(train_X, is_train=True)
			train_X = DataHelper.split_categorical_cols(train_X, is_train=True)
			DataHelper.scale_continuous_cols(train_X, is_train=True)
			train_X = DataHelper.select_best_features(train_X, None, train_y,
												ConfigHelper.max_nb_features, 
												is_train=True)

			DataHelper.fill_missing_data(val_X, is_train=False)
			val_X = DataHelper.split_categorical_cols(val_X, is_train=False)
			DataHelper.scale_continuous_cols(val_X, is_train=False)
			DataHelper.select_best_features(val_X, train_X, None,
											ConfigHelper.max_nb_features,
											is_train=False)

			MetricsHelper.store_gold(val_y)

			for name, model in ConfigHelper.get_training_models():
				print "Model: " + name

				print "Training"
				model.fit(train_X, train_y)

				print "Predicting"
				probs = model.predict_proba(val_X)
				MetricsHelper.store_probs(probs, name)
			print("--- %s seconds ---" % (time.time() - start_time))

		MetricsHelper.calculate_metrics()
	MetricsHelper.summarize_metrics()
	IOHelper.store_results(MetricsHelper.metrics, ConfigHelper.metrics_file)
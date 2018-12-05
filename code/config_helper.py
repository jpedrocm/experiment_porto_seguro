###############################################################################

from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC as SVM
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GB


class ConfigHelper():
	# Do not change
	k_folds = 2
	nb_executions = 5
	max_nb_features = 50
	use_predefined_cols = False
	analysis_dataset = "train"
	###########################

	metrics_file = "cv_metrics" #Change this when trying new experiments


	@staticmethod
	def k_fold_cv(labels):
		skf = StratifiedKFold(n_splits=ConfigHelper.k_folds, shuffle=True)
		return skf.split(X=range(len(labels)), y=labels)

	@staticmethod
	def get_submission_models():
		return [("GB_Final", GB(n_estimators=250, learning_rate=0.1, subsample=1.0,
				   max_depth=3, min_samples_split=20)),
			   ]

	@staticmethod
	def get_training_models():
		return [
				("MLP_RELU", MLP(hidden_layer_sizes=(100, ), alpha=0.0001,
					activation="relu", learning_rate_init=0.001,
					tol=0.0001, max_iter=200)),
				("GB_50", GB(n_estimators=250, learning_rate=0.1, subsample=1.0,
				   max_depth=3, min_samples_split=20)),
				 ("RF_FINAL", RF(n_estimators=250, max_depth=None, min_samples_split=2,
				   bootstrap=True, n_jobs=-1)),
				]
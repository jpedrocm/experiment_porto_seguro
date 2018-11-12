###############################################################################

from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC as SVM
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GB


class ConfigHelper():
	k_folds = 2
	nb_executions = 5
	use_predefined_cols = False
	metrics_file = "rf_min_samples"

	@staticmethod
	def k_fold_cv(labels):
		skf = StratifiedKFold(n_splits=ConfigHelper.k_folds, shuffle=True)
		return skf.split(X=range(len(labels)), y=labels)

	@staticmethod
	def get_submission_models():
		return [("RF_10", RF(n_estimators=320, max_depth=9, min_samples_split=10,
				   bootstrap=False, n_jobs=-1)),
			   ]

	@staticmethod
	def get_training_models():
		return [#("SVM_RBF", SVM(probability=True, tol=0.001, C=1, 
				#			   kernel="rbf", max_iter=-1, gamma="auto")),
				#("SVM_POLY", SVM(probability=True, tol=0.001, C=1, 
				#			   kernel="poly", max_iter=-1, gamma="auto")),
				#("SVM_SIGMOID", SVM(probability=True, tol=0.001, C=1, 
				#			   kernel="sigmoid", max_iter=-1, gamma="auto")),
				#("MLP_RELU", MLP(hidden_layer_sizes=(100, ), alpha=0.0001,
				#	activation="relu", learning_rate_init=0.001,
				#	tol=0.0001, max_iter=200)),
				#("MPL_TANH", MLP(hidden_layer_sizes=(100, ), alpha=0.0001,
				#	activation="tanh", learning_rate_init=0.001,
				#	tol=0.0001, max_iter=200)),
				#("MLP_LOG", MLP(hidden_layer_sizes=(100, ), alpha=0.0001,
				#	activation="logistic", learning_rate_init=0.001,
				#	tol=0.0001, max_iter=200)),
				("RF_6", RF(n_estimators=320, max_depth=9, min_samples_split=6,
				   bootstrap=False, n_jobs=-1)),
				("RF_4", RF(n_estimators=320, max_depth=9, min_samples_split=4,
				   bootstrap=False, n_jobs=-1)),
				("RF_8", RF(n_estimators=320, max_depth=9, min_samples_split=8,
				   bootstrap=False, n_jobs=-1)),
				("RF_10", RF(n_estimators=320, max_depth=9, min_samples_split=10,
				   bootstrap=False, n_jobs=-1)),
				#("GB_10", GB(n_estimators=10, learning_rate=0.1, subsample=1.0,
				#   max_depth=3, min_samples_split=2, tol=0.0001)),
				#("GB_20", GB(n_estimators=20, learning_rate=0.1, subsample=1.0,	
				#   max_depth=3, min_samples_split=2, tol=0.0001)),
				#("GB_40", GB(n_estimators=40, learning_rate=0.1, subsample=1.0,
				#   max_depth=3, min_samples_split=2, tol=0.0001))
				]
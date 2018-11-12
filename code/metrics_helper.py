###############################################################################

from sklearn.metrics import f1_score
import numpy as np

from pandas import DataFrame



class MetricsHelper():
	probs = {}
	gold = []
	metrics = DataFrame(columns=["model", "gini", "f1_macro"])

	@staticmethod
	def _gini(gold, pred, cmpcol=0, sortcol=1):  
		assert(len(gold) == len(pred))

		all_preds = np.asarray(np.c_[gold, pred, np.arange(len(gold))],
							   dtype=np.float)
		all_preds = all_preds[np.lexsort((all_preds[:,2], -1*all_preds[:,1]))]
		total_losses = all_preds[:,0].sum()
		gini_sum = all_preds[:,0].cumsum().sum() / total_losses
		gini_sum -= (len(gold)+1) / float(2)

		return gini_sum/len(gold)

	@staticmethod
	def _calculate_metrics(model, gold, prob):
		pos_prob = [row[1] for row in prob]
		pred = np.argmax(prob, axis=1)

		gini_score = MetricsHelper._gini(gold, pos_prob)/ \
						MetricsHelper._gini(gold, gold)
		f1_macro = f1_score(gold, pred, average="macro")

		return [model, gini_score, f1_macro]

	@staticmethod
	def reset_metrics():
		MetricsHelper.probs = {}
		MetricsHelper.gold = []

	@staticmethod
	def store_gold(gold):
		MetricsHelper.gold.extend(gold.values)

	@staticmethod
	def store_probs(prob, model):
		prob = prob.tolist()

		if model not in MetricsHelper.probs:
			MetricsHelper.probs[model] = prob
		else:
			MetricsHelper.probs[model].extend(prob)

	@staticmethod
	def calculate_metrics():
		print "Calculating metrics"

		for model, probs in MetricsHelper.probs.iteritems():
			idx = len(MetricsHelper.metrics)
			m = MetricsHelper._calculate_metrics(model, MetricsHelper.gold, 
												 probs)
			MetricsHelper.metrics.loc[idx] = m

	@staticmethod
	def summarize_metrics():
		print "Getting summary of metrics"

		grouped = MetricsHelper.metrics.groupby(by=["model"], as_index=True)
		MetricsHelper.metrics = grouped.agg([np.mean, np.std, np.amin,
											 np.amax])
		print MetricsHelper.metrics

	@staticmethod
	def get_submission(idxs, prob):
		pos_prob = [row[1] for row in prob]

		submission = DataFrame(columns=["target"], index=idxs)
		submission["target"] = pos_prob

		return submission
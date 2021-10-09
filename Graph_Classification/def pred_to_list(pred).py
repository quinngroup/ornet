def pred_to_list(pred):
	pred_list = []
	for i in range(pred.numpy().size):
		pred_list.append(pred.numpy()[i][0][0])

	return pred_list


def target_to_list(target):
	target_list = []
	for i in range(target.numpy().size):
		target_list.append(target.numpy()[i][0])
	return target_list


def my_round(prd):
	round_pred = []
	for i in prd:
		if i>=0.5:
			round_pred.append(1)
		else:
			round_pred.append(0)
	return(round_pred)



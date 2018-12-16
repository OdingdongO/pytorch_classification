import time
import copy
import pandas as pd
from collections import defaultdict
import numpy as np
class2label={"DESERT":0,"MOUNTAIN":1,"OCEAN":2,"FARMLAND":3,"LAKE":4,"CITY":5,"UNKNOW":6}
label2class=["DESERT","MOUNTAIN","OCEAN","FARMLAND","LAKE","CITY","UNKNOW"]

result_ratios = defaultdict(lambda: 0)
mode="val"
result_ratios['resnet50'] = 0.5
# result_ratios['resnet101'] = 0.5
result_ratios['densent121'] = 0.5
# result_ratios['densent169'] = 0.25

assert sum(result_ratios.values()) == 1
def str2np(str):
	return np.array([float(x) for x in str.split(";")])
def np2str(arr):
	return ";".join(["%.16f" % x for x in arr])
for index, model in enumerate(result_ratios.keys()):
	print('ratio: %.3f, model: %s' % (result_ratios[model], model))
	result = pd.read_csv('tiangong/csv/{}_{}_prob.csv'.format(model,mode),names=["filename","label","probability"])
	# result = result.sort_values(by='filename').reset_index(drop=True)
	result['probability'] = result['probability'].apply(lambda x: str2np(x))
	print(result.head())

	if index == 0:
		ensembled_result = copy.deepcopy(result)
		ensembled_result['probability'] =0

	ensembled_result['probability'] = ensembled_result['probability'] + result['probability']*result_ratios[model]
	print(ensembled_result.head())
def parase_prob(x):
	if np.max(x)<0.33:
		label=label2class[6]
	else:
		label = label2class[int(np.argmax(x))]
	return label
def get_class_pro(x,class_name):
	# sum =np.sum(x)
	# prob=x[class2label[class_name]]*1.0/sum
	prob=x[class2label[class_name]]
	prob =np.round(prob,8)
	# return str("({},{})".format(class_name,str("%.8f" % prob)))
	return "("+class_name+","+str(prob)+")"

ensembled_result['label'] = ensembled_result['probability'].apply(lambda x: parase_prob(x))
for class_name in label2class[:-1]:
	ensembled_result[class_name] = ensembled_result['probability'].map(lambda x: get_class_pro(x,class_name))
# ensembled_result['probability'] = ensembled_result['probability'].map(lambda x: np2str(x))
# ensembled_result[["filename","label","OCEAN","MOUNTAIN","LAKE","FARMLAND","DESERT","CITY"]].to_csv('%s_ensembled_result.csv' % time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time())), header=None,index=False)
# ensembled_result[["filename","label","OCEAN","MOUNTAIN","LAKE","FARMLAND","DESERT","CITY"]].to_csv('ensembled_result2.csv', header=None,index=False)
with open('ensembled_result.csv',"w") as f:
	for filename,label,OCEAN,MOUNTAIN,LAKE,FARMLAND,DESERT,CITY in zip(ensembled_result["filename"],
																	   ensembled_result["label"],
																	   ensembled_result["OCEAN"],
																	   ensembled_result["MOUNTAIN"],
																	   ensembled_result["LAKE"],
																	   ensembled_result["FARMLAND"],
																	   ensembled_result["DESERT"],
																	   ensembled_result["CITY"]):
		output_str=",".join([x for x in [filename,label,OCEAN,MOUNTAIN,LAKE,FARMLAND,DESERT,CITY]])
		f.write(output_str)
		f.write("\n")

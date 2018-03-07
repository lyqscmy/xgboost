import xgboost as xgb
import numpy as np

with open("ads-model.bin",'rb') as f:
    model = f.read()
    bst = xgb.Booster(model_file=bytearray(model))

dtest = xgb.DMatrix('ads-test.txt')
predicts = bst.predict(dtest, pred_leaf=True)

with open("labels.npy",'rb') as f:
    ans = np.load(f)


result = True
with open('ads-test.txt') as f:
    count = 0
    for line in f:
        data = line.split(' ')
        label = float(data[0])
        xs = [tuple(i.split(':')) for i in data[1:]]
        indices = [int(index) for index,_ in  xs]
        values = [float(value) for _, value in  xs]
        labels = bst.predictLeafInst(indices, values)
        result &= (np.all(predicts[count]==labels))
        count+=1

print(result)


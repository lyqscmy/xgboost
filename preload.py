import xgboost as xgb

with open("demo/data/model.bin",'rb') as f:
    model = f.read()
    bst = xgb.Booster(model_file=bytearray(model))

dtest = xgb.DMatrix('demo/data/test.txt')
label = bst.predict(dtest)
print(label)

with open('demo/data/test.txt') as f:
    data = f.read().split(' ')
    label = float(data[0])
    xs = [tuple(i.split(':')) for i in data[1:]]
    indices = [int(index) for index,_ in  xs]
    values = [float(value) for _, value in  xs]

label = bst.predictInst(indices, values)
print(label)

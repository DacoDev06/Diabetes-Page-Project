import pickle
from fastapi.responses import JSONResponse

clf=pickle.load(open('model.pkl','rb'))

def predict(list):
    result = clf.predict([list])
    if result == 0:
        return 0
    else:
        return 1
print(predict([40, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]))
print(predict([40, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1]))

    
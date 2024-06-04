import pickle

clf=pickle.load(open('./model.pkl','rb'))

def predict(list):
    newpat2 = [[20,0,1,1,1,1,1,1,0,0,1,1,0,1,1,0]]
    newpat1 = [[20,1,0,0,0,0,0,0,1,1,0,0,1,0,0,1]]
    print(list)
    result = clf.predict([list])
    if result == 0:
        return "Alta probabilidad de poseer diabetes"
    else:
        return "Baja probabilidad de poseer diabetes"

    
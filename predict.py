import pickle


clf=pickle.load(open('./model.pkl','rb'))

newpat2 = [[20,0,1,1,1,1,1,1,0,0,1,1,0,1,1,0]]
newpat1 = [[20,1,0,0,0,0,0,0,1,1,0,0,1,0,0,1]]

print(type(newpat1))
result = clf.predict(newpat1)
print(result)
if result == 0:
    print("Alta probabilidad de poseer diabetes")
    
else:
    print("Baja probabilidad de poseer diabetes")
    
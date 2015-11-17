import cPickle as pickle

from SuperScale import SuperScale

SuperScale.initData()
ds = SuperScale('train')
monitoringDs = SuperScale('valid')

# cree un fichier en pickle et serialisation des donnees
with open('dataset_app.pkl','wb') as f:
    pickle.dump(ds,f)

with open('dataset_val.pkl', 'wb') as f:
    pickle.dump(monitoringDs,f)

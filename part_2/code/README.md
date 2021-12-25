In order to reproducee the results and clusters:

1. Create a python virtual environment ```python -m venv venv```
2. Activate the venv ```source venv/bin/activate```
3. Install requirements ```pip install -r requirements.txt```
4. Change the path for the labeled / unlabeled dataset in the following files:
- train_vectorizer.py  
- train_clustering.py  
- company2cluster.py  
5. Run ```python train_vectorizer.py```
6. Run ```python train_clustering.py```
7. Run ```python company2cluster.py```
# Détection de transactions illicites sur Bitcoin

Ce projet étudie la détection de transactions illicites à partir de graphes de transactions Bitcoin. Il s'appuie sur le jeu Elliptic et l'article de Weber et al. (2019), *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics*.

## Structure du dépôt

- `src/` : chargement et préparation des données, modèles et métriques du protocole.
- `notebooks/01_exploration_elliptic.ipynb` : exploration et vérifications des données.
- `notebooks/02_baselines_ml.ipynb` : baselines historiques.
- `notebooks/03_gcn.ipynb` : expérimentation GCN préliminaire, conservée pour l'historique ; ne pas utiliser ses chiffres.
- `notebooks/04_protocole_commun.ipynb` : comparaison des modèles avec le protocole commun.
- `notebooks/05_evaluation_finale.ipynb` : évaluation après réentraînement final.
- `results/` : résultats générés à l'exécution, ignorés par Git.
- `data/raw/` : données Elliptic brutes, ignorées par Git.

## Installation (Windows PowerShell)

Utilisez Python 3.11 ou 3.12. Python 3.14 n'est pas recommandé pour cet environnement.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Sans GPU NVIDIA, installez d'abord la version CPU de PyTorch, puis les dépendances du projet :

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Si pip répond `from versions: none`, vérifiez qu'une configuration `PIP_NO_INDEX` ne désactive pas les index :

```powershell
echo $env:PIP_NO_INDEX
```

Enregistrez le kernel Jupyter :

```powershell
python -m ipykernel install --user --name fraud-blockchain
```

## Données

Téléchargez manuellement le jeu Kaggle [Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set), ou utilisez l'API Kaggle. Pour l'API, fournissez la clé via la variable d'environnement `KAGGLE_API_TOKEN` ; ne placez jamais de clé dans le dépôt. Placez directement les trois CSV dans `data/raw/` :

- `elliptic_txs_features.csv`
- `elliptic_txs_edgelist.csv`
- `elliptic_txs_classes.csv`

Si l'archive contient le sous-dossier `elliptic_bitcoin_dataset`, déplacez les CSV hors de ce sous-dossier. Les dimensions de contrôle du jeu sont : features `(203769, 167)`, edges `(234355, 2)` ; classes : `1` = 4 545 illicites, `2` = 42 019 licites et `unknown` = 157 205. Le jeu comprend 49 pas de temps. Depuis la racine du dépôt, cette commande vérifie les dimensions et les effectifs des classes :

```powershell
python -c "import pandas as pd; p='data/raw/'; f=pd.read_csv(p+'elliptic_txs_features.csv',header=None); e=pd.read_csv(p+'elliptic_txs_edgelist.csv'); c=pd.read_csv(p+'elliptic_txs_classes.csv'); print('features',f.shape,'edges',e.shape,'classes',c.iloc[:,-1].value_counts().to_dict(),'pas',f.iloc[:,1].nunique())"
```

## Protocole commun et évaluation finale

Dans le notebook 04, le découpage temporel est train sur les pas 1–24, validation sur 25–34 et test sur 35–49. Les transactions `unknown` sont exclues des métriques. Le scaler est ajusté sur l'entraînement ; le seuil est choisi sur la validation. Les graines utilisées sont 42 à 46. Les métriques suivies sont F1, précision, rappel, PR-AUC et F1 par pas de temps.

Le notebook 05 choisit le seuil et le nombre d'époques avec le protocole train/validation, puis réentraîne sur les pas 1–34. Le seuil et les époques choisis sont figés pour le test final sur les pas 35–49. Les transactions `unknown` restent exclues des métriques ; le scaler est ajusté sur l'entraînement correspondant.

## Exécuter les notebooks

Réglez les constantes en tête du notebook (`SEEDS`, `GCN_EPOCHS`, `MLP_EPOCHS`, `MODELES`) si nécessaire et sauvegardez le fichier avant de lancer l'exécution. Depuis `notebooks/`, par exemple :

```powershell
python -m jupyter nbconvert --to notebook --execute 05_evaluation_finale.ipynb --output-dir $env:TEMP\nb_runs --output 05_run.ipynb --ExecutePreprocessor.timeout=-1
```

`GCN_EPOCHS = 20` correspond à un mode de test et ne convient pas aux résultats ; utilisez 600 pour les exécutions de résultats. Les CSV sous `results/` sont écrasés à chaque exécution : copiez-les ailleurs avant de relancer si vous devez conserver un résultat. N'exécutez pas les notebooks depuis le sandbox de Codex, où le kernel Jupyter est bloqué ; utilisez un terminal normal.

## Durées observées sur CPU

Sur une machine sans GPU, les durées mesurées étaient d'environ 9 minutes pour les modèles classiques sur 5 seeds, et 92 minutes pour GCN + Skip-GCN sur 1 seed avec un maximum de 600 époques. Ce sont des mesures faites sur une seule machine et elles peuvent varier selon le matériel et la charge.

## Résultats

Aucun chiffre de performance n'est repris ici. Après exécution, consultez les fichiers produits dans `results/` ; les résultats actuels sont également présentés dans le document de synthèse du projet.

## Avancement et suite

Le protocole commun, les baselines et l'évaluation finale sont présents dans le dépôt. Les pistes à traiter sont : EvolveGCN, Random Forest avec embeddings GCN, analyse comportementale, dérive temporelle, SHAP, application Streamlit et décision sur l'utilisation d'Ethereum/Web3.py.

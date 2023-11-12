# Projet-DNN

# Dependences

Comme le model donnée n'est compatible que TensorFlow 1,
et que TensorFlow est passé à la version depuis 2019, le 
modèle est donc déprécié.

Heuresement, Félix à eu le courage de recoder le modèle 
en `pytorch` (merci à lui).

Ce programme a été testé avec python 3.
Donc il est nécessaire d'avoir les packages python :
    - `pytorch`
    - `torchvision`
    - `tqdm`
    
Enfin il est nécessaire d'avoir z3 d'installé.
Et le package python `z3-solver`


# Lancer le programme

Pour lancer le programme il suffit de faire :
`python3 main.py`

Ou si le programme a les droits d'éxécution :
`./main.py`

# Paramètre
Il est possible dans l'entête du programme de choisir 
l'intervalle de confiance que l'on souhaite pour epsilon.
Et de déterminer si l'on veut ou non telecharher le dataset (MNIST) pour
entrainer l'ia présente. (Il est recommander de le télecherger lors de la
première exécution du programme).

# Problème
Il semble que z3 soit vraiment lent, le cas avec epsilon < 0,
dans la formule du inf, est très long.

Mais normalement le programme effectue bien la tâche demandé



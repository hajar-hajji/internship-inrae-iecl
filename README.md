# Analyse des effets mémoire de la sécheresse sur l'écosystème forestier


<img width="850" alt="logos1" src="https://github.com/hajar-hajji/test_git/assets/120224819/d92599d9-7f5d-4b49-8880-c3d21a33e208">

-------------

La finalité de ce projet est d'étudier l'influence des sécheresses répétées sur les flux de carbone. Pour ce faire, nous adopterons une approche boîte noire (black-box approach) en tirant parti du potentiel des réseaux de neurones. Nous aurons également recours à des méthodes d'explicabilité de l'IA afin de retracer et comprendre les mécanismes sous-jacents au fonctionnement de ces algorithmes. 

--------

## Structure
Le répertoire est organisé de la manière suivante :

- `data/` : Ce dossier contient les datasets utilisés dans le projet.
- `code/` : Les carnets Jupyter ainsi les codes utilisés (les notebooks sont à importer de préférence sur google colab pour utiliser le GPU)
- `models/` : Les modèles que j'ai entraînés.

## Expérimentations
### Dataset
Dans le cadre de ce stage, nous avons restreint notre analyse aux données provenant du site FR-Bil. Nous avons choisi de ne considérer que les variables pertinentes pour notre étude. En effet, nous avons limité les entrées à un petit nombre de prédicteurs qui sont connus pour être des facteurs environnementaux (à savoir le SWC, VPD, Rg et Ta) afin d'éviter le surapprentissage (over-fitting), qui se produit lorsque le modèle apprend de la variabilité aléatoire dans l'ensemble de données d'entraînement plutôt que des relations réelles existantes entre les prédicteurs et la variable de réponse (GPP dans ce cas), ce qui peut entraver la capacité de généralisation du modèle. Une description concise de ces variables est fournie dans le tableau ci-dessous.

#### Variables

| Variable    | Description                                             | Unité                  |
|-------------|---------------------------------------------------------|------------------------|
| `GPP`       | Quantité de CO2 assimilée par la forêt via la photosynthèse | µmol m^-2 s^-1 |
| `SWC`       | Teneur en eau du sol                                    | %                      |
| `VPD`       | Déficit de pression de vapeur                           | kPa                    |
| `Rg`        | Rayonnement solaire entrant                              | Wm^-2                  |


### Première approche
Avant d'aborder la partie sur la causalité, il apparait pertinent de commencer par entraîner des modèles de prédiction basés sur les RNNs (réseaux de neurones récurrents) et leurs architectures, notamment les LSTMs (Long Short-Term Memory) et les GRUs (Gated Recurrent Units). Ensuite, nous pourrons essayer d'optimiser les résultats en mettant en place des mécanismes d'attention. Enfin, nous allons remplacer les RNNs par un modèle alternatif, à savoir les TCNs (Temporal Convolutional Networks), qui ont démontré de bons résultats selon l'état de l'art. Le remplacement des RNNs nous a semblé judicieux pour maintes raisons, en effet, contrairement aux RNNs (et naturellement les LSTMs et les GRUs), qui utilisent le rétropropagation à travers le temps et peuvent pâtir de la diminution rapide des gradients ou leur explosion, les TCNs se démarquent par des gradients plus stables.

En ce qui est du prétraitement des données, nous avons décomposé la série chronologique représenté par le GPP de la façon suivante : 

$$X_t = 
\overline{X}^{365d}_t + 
\overline{X_t - \bar{X}^{365d}_t} + 
\tilde{X_t}$$

avec : 
$$\overline{X}^{T} = \frac{1}{N} \sum_{k=1}^{N} X_{t+kT}$$

où la moyenne annuelle \overline{X}^{365d}_t est calculée avec N=2 et le cycle jour et nuit avec N=7, le dernier terme représente les anomalies.

En raison de la pénurie de données d'entrée principales (à savoir le SWC et le VPD), nous nous contentons dans un premier temps d'une seule entrée, qui est le temps représenté par des signaux compréhensibles par le modèle (pour mettre en évidence la périodicité quotidienne et annuelle) ainsi que le GPP que nous souhaitons modéliser.  Le jeu de données modifié, comprenant les colonnes ajoutées (le temps et la décomposition du GPP) est sauvegardé sous le nom de `"preprocessed_dataICOS"` dans le dossier `/data` afin d'éviter de reprendre le calcul à chaque utilisation du dataset. En analysant le dataset,  il paraît que la plupart des valeurs manquantes se trouvent en 2014 et en 2022. En revanche, entre 2015 et 2021, il existe quelques valeurs manquantes de SWC_3 et sont principalement localisées en 2016, plus précisément lors de la première semaine de Septembre (voir `"preprocessing_dataICOS"` dans les `/code`). D'où la possibilité de travailler dans cette période avec le GPP, le temps ainsi que le VPD et le SWC. Nous le ferons avec les TCNs puisqu'on va travailler directement sur le dataset initial (pas besoin de décomposer le GPP manuellement...).

**to be continued**

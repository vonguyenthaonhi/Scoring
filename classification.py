import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold, GridSearchCV

import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, log_loss



class SelectionVariable:
    """
    Classe pour procéder à la sélection de variables en prenant en compte le VIF et la corrélation.
    """
    def __init__(self, seuil_vif=5):
        self.seuil_vif = seuil_vif  #seuil de VIF au-dessus duquel les variables seront supprimées
        self.donnees_vif = None
        self.donnees_correlation = None
        self.donnees_combinees = None


    def calcul_vif(self, X):
        """
        Calcul du VIF (Variance Inflation Factor) pour chaque variable dans X.
        """
        
        donnees_vif = pd.DataFrame()
        donnees_vif["feature"] = X.columns
        donnees_vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        self.donnees_vif = donnees_vif
        return donnees_vif
    
    
    def calcul_correlation(self, X, y):
        """
        Calcul de la corrélation entre chaque variable de X et la variable cible y.
        """
        
        donnees_correlation = X.apply(lambda x: x.corr(y))
        donnees_correlation = donnees_correlation.reset_index()
        donnees_correlation.columns = ["feature", "correlation"]
        self.donnees_correlation = donnees_correlation
        return donnees_correlation
    

    def remove_highvif_lowcorrelation(self, X, y):
        """
        Suppression des variables avec un VIF élevé (VIF>5) et une corrélation faible avec la cible y 
        jusqu'à ce que toutes les variables restantes aient un VIF inférieur au seuil.
        """

        self.calcul_vif(X)
        self.calcul_correlation(X, y)

        # On fusionne les résultats du VIF et des corrélations
        donnees_combinees = pd.merge(self.donnees_vif, self.donnees_correlation, on="feature")
        self.donnees_combinees = donnees_combinees

    
        while donnees_combinees['VIF'].max() > self.seuil_vif:
            donnees_combinees = donnees_combinees.sort_values(by=['VIF', 'correlation'], ascending=[False, True])
            var_a_eliminer = donnees_combinees.iloc[0]['feature']  #la variable à supprimer
            print(f"On élimine '{var_a_eliminer}' avec un VIF de: {donnees_combinees.iloc[0]['VIF']} et une corrélation de: {donnees_combinees.iloc[0]['correlation']}")

            X= X.drop(columns=[var_a_eliminer]) 
            self.calcul_vif(X)
            self.calcul_correlation(X, y)
            donnees_combinees = pd.merge(self.donnees_vif, self.donnees_correlation, on="feature") 

        self.donnees_combinees = donnees_combinees
        return X, donnees_combinees



class LogisticRegressionModèle:
    """
    Entraînement, prédiction et évaluation du modèle de régression logistique avec grille de paramètres et CV.
    """

    def __init__(self, verbose=True, n_iterations=10): 
        self.grille_param = [
            {'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100]},
            {'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag'], 'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10, 100]},
            {'solver': ['saga'], 'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100]},
            {'solver': ['saga'], 'penalty': ['elasticnet'], 'C': [0.01, 0.1, 1, 10, 100], 'l1_ratio': [0.1, 0.5, 0.7]}
        ]
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.modele = LogisticRegression(max_iter=10000, fit_intercept=True, class_weight='balanced')
        self.meilleur_modele = None
        self.verbose = verbose
        self.n_iterations = n_iterations
        self.metriques_aleatoires = {  # Dictionnaire pour stocker les métriques des itérations aléatoires
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': [],
            'ROC AUC': [],
            'Log Loss': []
        }

    def entrainer(self, X_train, y_train):
        """
        Entraînement du modèle de régression logistique avec GridSearchCV.
        """
        grid_search = GridSearchCV(self.modele, self.grille_param, cv=self.cv, scoring='f1', verbose=0)
        grid_search.fit(X_train, y_train)
        self.meilleur_modele = grid_search.best_estimator_
        if self.verbose:
            print("Meilleur modèle trouvé avec les paramètres:", grid_search.best_params_)
            print("Modèle optimal:", self.meilleur_modele)

    def predire(self, X_test):
        """
        Prédictions avec le meilleur modèle.
        """
        y_pred = self.meilleur_modele.predict(X_test)
        y_proba = self.meilleur_modele.predict_proba(X_test)[:, 1]
        return y_pred, y_proba

    def evaluer_resultats(self, y_test, y_pred, y_proba):
        """
        Affichage des métriques d'évaluation et de la matrice de confusion.
        """
        resultats = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_proba),
            'Log Loss': log_loss(y_test, y_proba)
        }
        if self.verbose:
            print(pd.DataFrame(resultats.items(), columns=['Metric', 'Score']).to_string(index=False))
            print("\nMatrice de confusion:\n", confusion_matrix(y_test, y_pred))

    def liste_metriques(self, y_test, y_pred, y_proba):
        """
        Retour d'une liste des métriques d'évaluation - utile pour visualisation après.
        """
        return [
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred, zero_division=0),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred),
            roc_auc_score(y_test, y_proba),
            log_loss(y_test, y_proba)
        ]
    
    def evaluation_aleatoire(self, X_train, y_train, X_test, y_test):
        """
        Évaluation du modèle avec labels permutés sur plusieurs itérations.
        """
        for i in range(self.n_iterations):
            y_train_permute = np.random.permutation(y_train)
            self.entrainer(X_train, y_train_permute)
            y_pred, y_proba = self.predire(X_test)
            # Stockage des métriques
            self.metriques_aleatoires['Accuracy'].append(accuracy_score(y_test, y_pred))
            self.metriques_aleatoires['Precision'].append(precision_score(y_test, y_pred, zero_division=0))
            self.metriques_aleatoires['Recall'].append(recall_score(y_test, y_pred))
            self.metriques_aleatoires['F1-Score'].append(f1_score(y_test, y_pred))
            self.metriques_aleatoires['ROC AUC'].append(roc_auc_score(y_test, y_proba))
            self.metriques_aleatoires['Log Loss'].append(log_loss(y_test, y_proba))
            if self.verbose:
                print(f"Iteration {i+1}/{self.n_iterations} terminée.")
        
        return {metric: np.mean(values) for metric, values in self.metriques_aleatoires.items()}

    def comparer_modeles(self, original_scores):
        """
        Affiche un graphique de comparaison entre le modèle original et les scores moyens aléatoires.
        """
        random_scores = [np.mean(values) for values in self.metriques_aleatoires.values()]
        labels = list(self.metriques_aleatoires.keys())
        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots()
        ax.bar(x - width/2, original_scores, width, label='Modèle original')
        ax.bar(x + width/2, random_scores, width, label='Modèle aléatoire')

        ax.set_ylim(0, 1)
        ax.set_xlabel('Métriques')
        ax.set_ylabel('Score')
        ax.set_title('Comparaison des performances entre modèle original et modèle aléatoire')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        # Ajout des scores sur chaque barre
        for i, v in enumerate(original_scores):
            ax.text(i - width/2, v + 0.02, f"{v:.2f}", ha='center')
        for i, v in enumerate(random_scores):
            ax.text(i + width/2, v + 0.02, f"{v:.2f}", ha='center')

        plt.tight_layout()
        plt.show()

    def comparer_performance_train_test(self, X_train, y_train, X_test, y_test):
            """
            Affichage une graphique qui compare les performances du modèle entre les ensembles de données train et de test
            pour vérifier l'overfitting.
            """
            # Calcul des scores pour les ensembles de données d'entraînement et de test
            y_pred_train, y_proba_train = self.predire(X_train)
            train_scores = self.liste_metriques(y_train, y_pred_train, y_proba_train)

            y_pred_test, y_proba_test = self.predire(X_test)
            test_scores = self.liste_metriques(y_test, y_pred_test, y_proba_test)

            # Étiquettes des métriques et configuration des barres
            label_metriques = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC', 'Log Loss']
            x = np.arange(len(label_metriques))
            width = 0.35

            fig, ax = plt.subplots()
            bar1 = ax.bar(x - width/2, test_scores, width, label='Performance sur X_test', color='cornflowerblue')
            bar2 = ax.bar(x + width/2, train_scores, width, label='Performance sur X_train', color='lightcoral')

            ax.set_ylim(0, 1)
            ax.set_xlabel('Métriques')
            ax.set_ylabel('Score')
            ax.set_title('Comparaison des performances entre X_test et X_train')
            ax.set_xticks(x)
            ax.set_xticklabels(label_metriques)

            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

            # Annotation des valeurs sur chaque barre
            for bars in [bar1, bar2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.2f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 5), 
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=8) 

            plt.tight_layout()
            plt.show()


class EntrainerModèle:
    """
    Entrainement des autres modèles.
    """
    def __init__(self):
        
        self.modeles = {
            'SVM': SVC(probability=True), 'GradientBoosting': GradientBoostingClassifier(), 
            'KNN': KNeighborsClassifier(), 'DecisionTree' : DecisionTreeClassifier(random_state=42),
            'RandomForest' : RandomForestClassifier(random_state=42), 'NaiveBayes': GaussianNB()
        }
        
        self.grille_param = {
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }, 
            'GradientBoosting' : {
                'n_estimators': [100, 200, 300],        
                'learning_rate': [0.01, 0.1, 0.2],     
                'max_depth': [3, 4, 5],                
                'subsample': [0.7, 0.8, 1.0],          
                'min_samples_split': [2, 5, 10],       
                'min_samples_leaf': [1, 2, 4],         
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9],        
                'weights': ['uniform', 'distance'],  
                'metric': ['euclidean', 'manhattan', 'hamming'], 
                'p': [1, 2]  
            },
            'DecisionTree' : {
                'max_depth': [3, 5, 10],
                'min_samples_split': [2, 10, 20]
            },
            'RandomForest' : {
                'n_estimators': [50, 100, 200],  
                'max_depth': [5, 10, 20],  
                'min_samples_split': [2, 10, 20]  
            },
            'NaiveBayes' : {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]  
            }
        }

        #les meilleurs modèles sauvegardés après optimisation + la sauvegarde des métriques pour la visualisation
        self.meilleurs_modeles = {}
        self.stockage_metriques = []

    def entrainement(self, nom_modele, X_train, y_train):
        """
        Entrainement du modele avec GridSearch et CV.
        """
        print(f"Entrainement {nom_modele}...")
        
        model = self.modeles.get(nom_modele)
        param_grid = self.grille_param.get(nom_modele)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='f1', verbose=1)
        grid_search.fit(X_train, y_train)
        
        #on sauvegarde le meilleur modèle trouvé
        self.meilleurs_modeles[nom_modele] = grid_search.best_estimator_
        print(f"Les meilleurs paramètres pour {nom_modele}: {grid_search.best_params_}")
        
   
    def prediction(self, model_name, X_test):
        """
        Prédiction avec le meilleur modèle selectionné.
        """
        model = self.meilleurs_modeles.get(model_name)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  #probabilité pour la classe 1
        return y_pred, y_proba
        
    
    def evaluation(self, y_test, y_pred, y_proba):
        """
        Évaluation du modèle avec plusieurs métriques.
        """
        metriques = {
            'Accuracy': accuracy_score,
            'Precision': precision_score,
            'Recall': recall_score,
            'F1-Score': f1_score, 
            'ROC AUC': roc_auc_score,  
            'Log Loss': log_loss        
        }

        resultats = {}
        for nom_metrique, fonction_metrique in metriques.items():
            if nom_metrique in ['ROC AUC', 'Log Loss']:  #on a besoin de probabilités pour ces 2 métriques
                resultats[nom_metrique] = fonction_metrique(y_test, y_proba)
            else:
                resultats[nom_metrique] = fonction_metrique(y_test, y_pred)

        return resultats
    
     
    def evaluation_stockage(self, nom_modele, y_test, y_pred, y_proba):
        """
        Evaluation et stockage des resultats.
        """

        resultats = self.evaluation(y_test, y_pred, y_proba)
        resultats['Model'] = nom_modele
        self.stockage_metriques.append(resultats)

        df_resultats = pd.DataFrame(resultats.items(), columns=['Metric', 'Score'])
        print(f"Résultats pour le modèle {nom_modele} :\n{df_resultats.to_string(index=False)}")


        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalisation par ligne
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_normalized, annot=cm, fmt="d", 
                    cmap=sns.color_palette(["#ffcccc", "#ff9999", "#ff6666", "#cc0000"], as_cmap=True), 
                    cbar=False, linecolor='gray', linewidths=0.5,
                    xticklabels=['Classe 0', 'Classe 1'],
                    yticklabels=['Classe 0', 'Classe 1'])
        plt.xlabel("Prédiction")
        plt.ylabel("Vraie classe")
        plt.title(f"Matrice de confusion pour {nom_modele}")
        plt.show()
        
        
    def metriques_dataframe(self):
        """
        Stockage des metriques sous forme de df - utile pour les graphiques.
        """
    
        return pd.DataFrame(self.stockage_metriques)

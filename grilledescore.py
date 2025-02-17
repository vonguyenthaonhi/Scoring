import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

class ModeleGrilleDeScore:
    def __init__(self, train, cible, score_base, facteur=20):
        """
        Initialise le modèle de grille de score avec les paramètres fournis.
        """
        self.train = train
        self.cible = cible
        self.score_base = score_base
        self.facteur = facteur
        self.coefficients = None
        self.points = None
        self.resultat_modele = None

    def ajuster_regression_logistique(self):
        """
        Ajuste une régression logistique sur les données d'entraînement pour estimer les coefficients.
        """
        X = self.train.drop(columns=[self.cible])  
        X = sm.add_constant(X)  
        y = self.train[self.cible] 
        logit_model = sm.Logit(y, X)
        self.resultat_modele = logit_model.fit()
        self.coefficients = self.resultat_modele.params  

    def calculer_points(self):
        """
        Calcule les points de score pour chaque variable en fonction des coefficients estimés
        et les stocke dans un dictionnaire.
        """
        self.points = {}
        for variable, coefficient in self.coefficients.items():
            if variable == 'const':
                continue
            self.points[variable] = coefficient / np.log(2) * self.facteur  

    def calculer_score_individuel(self, ligne):
        """
        Calcule le score pour une ligne individuelle en additionnant le score de base
        avec les points associés à chaque variable.
        """
        score = self.score_base
        for variable in self.points:
            score += self.points[variable] * ligne[variable]
        return score

    def generer_scores(self):
        """
        Calcule les scores pour toutes les lignes du DataFrame d'entraînement
        et les ajoute dans une nouvelle colonne 'Score'.
        """
        self.train['Score'] = self.train.apply(self.calculer_score_individuel, axis=1)
        return self.train['Score']

    def creer_grille_de_score(self):
        """
        Crée une grille de score en compilant les informations des points pour chaque variable
        et leurs valeurs uniques. La contribution de chaque variable au modèle est également calculée.
        """
        result_list = []
        abs_coefficients = sum(abs(self.coefficients[variable]) for variable in self.points.keys())
        
        for variable in self.points:
            unique_values = self.train[variable].unique()
            for value in unique_values:
                subset = self.train[self.train[variable] == value]
                if subset.empty:
                    continue
                
                coefficient = self.coefficients[variable]
                point_value = self.points[variable]
                
                # Contribution de la variable au modèle
                variable_contribution_percentage = (abs(coefficient) / abs_coefficients) * 100
                
                # Fréquence de la valeur dans la variable
                frequency = len(subset) / len(self.train) * 100
                
                # Contribution de chaque valeur
                value_contribution = variable_contribution_percentage * (frequency / 100)
                
                result_list.append({
                    'Variable': variable,
                    'Valeur': value,
                    'Coefficient': coefficient,
                    'Points': point_value,
                    'Fréquence (%)': frequency,
                    'Contribution (%)': value_contribution
                })
        
        grille_de_score = pd.DataFrame(result_list)
        return grille_de_score

    def mettre_a_jour_taux_defaut(self, grille_de_score):
        """
        Met à jour la grille de score avec les taux de défaut pour chaque valeur de chaque variable.
        """
        taux_defaut = {}

        for variable in self.train.columns.drop(self.cible):
            taux_par_niveau = self.train.groupby(variable)[self.cible].mean()
            taux_defaut[variable] = taux_par_niveau

        for variable, taux in taux_defaut.items():
            if variable in grille_de_score['Variable'].values:
                for valeur, taux_value in taux.items():
                    grille_de_score.loc[
                        (grille_de_score['Variable'] == variable) & 
                        (grille_de_score['Valeur'] == valeur), 
                        'Taux de défaut'] = taux_value

        return grille_de_score

    def ajuster(self):
        """
        Exécute toutes les étapes pour ajuster le modèle et générer la grille de score.
        """
        self.ajuster_regression_logistique()
        self.calculer_points()
        scores = self.generer_scores()
        grille_de_score = self.creer_grille_de_score()
        grille_de_score = self.mettre_a_jour_taux_defaut(grille_de_score)
        return grille_de_score, scores, self.resultat_modele
    
    def calculer_roc_auc_gini(self):
        """
        Calcule le score AUC et le coefficient de Gini pour le modèle.
        """
        fpr, tpr, thresholds = roc_curve(self.train[self.cible], self.train['Score'])
        roc_auc = auc(fpr, tpr)
        gini_coefficient = 2 * roc_auc - 1
        print(f'AUC: {roc_auc}')
        print(f'Gini Coefficient: {gini_coefficient}')
        return fpr, tpr, roc_auc
    
    @staticmethod
    def plot_roc_curve(fpr_train, tpr_train, roc_auc_train, fpr_test, tpr_test, roc_auc_test):
        """
        Trace la courbe ROC pour les ensembles d'entraînement et de test.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(fpr_train, tpr_train, color='blue', label='Courbe ROC d\'entraînement (AUC = {:.2f})'.format(roc_auc_train))
        plt.plot(fpr_test, tpr_test, color='green', label='Courbe ROC de test (AUC = {:.2f})'.format(roc_auc_test))
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')  
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux de Faux Positifs')
        plt.ylabel('Taux de Vrais Positifs')
        plt.title('Courbe Caractéristique de Fonctionnement (ROC) de l\'entraînement et du test')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

    @staticmethod
    def plot_densite_conditionnelle(df):
        """
        Trace la densité conditionnelle des scores pour les individus en fonction de la variable cible.
        """
        plt.figure(figsize=(10, 6))
        sns.kdeplot(df[df['BAD'] == 1]['Score'], color='blue', label='BAD = 1', fill=True, alpha=0.5)
        sns.kdeplot(df[df['BAD'] == 0]['Score'], color='cyan', label='BAD = 0', fill=True, alpha=0.5)
        plt.title("Distribution des scores des individus conditionnellement au défaut", fontsize=16)
        plt.xlabel('Score', fontsize=14)
        plt.ylabel('Densité', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()

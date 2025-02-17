
import pandas as pd
from scipy import stats
import numpy as np

from optbinning import OptimalBinning


class DiscretisationCategorie:

    """
    Discrétisation des variables catégorielles.
    """

    def discretisation_de_variable_categorielle_train(self, train, variable, target):
        """
        Discrétise une variable catégorielle du train.
        """
      
        taux_default = train.groupby(variable)[target].mean()
        taux_default = taux_default.sort_values(ascending=True)

        # Diviser les catégories en 3 groupes à parts égales
        labels = ['classe_1', 'classe_2', 'classe_3']
        discretisation = pd.qcut(taux_default, q=3, labels=labels)

        train[variable + '_discret'] = train[variable].map(discretisation)

        #affichage
        regroupement = pd.DataFrame({
            'Categorie': taux_default.index,
            'Taux_default': taux_default.values,
            'Classe': discretisation.values
        })

        print(regroupement)

        return train, discretisation

    def appliquer_discretisation_test(self, test, variable, discretisation):
        """
        Application de la disrétisation sur les données test.
        """
        
        test[variable + '_discret'] = test[variable].map(discretisation)

        return test


    def tschuprow_coefficient(self,contingency_table):
        """
        Formules pour le calcule de coeficient Tschuprow.
        """

        chi2, _, _, _ = stats.chi2_contingency(contingency_table)
        n = np.sum(contingency_table)
        r, c = contingency_table.shape
        tschuprow = np.sqrt(chi2 / (n * np.sqrt((r - 1) * (c - 1))))
        return tschuprow
    
    def tshuprow_coef_calcul (self, train, variables, target) -> None:
        """
        Calcule de coeficient Tschuprow.
        """
        for variable in variables :
            contingency_table = pd.crosstab(train[variable], train[target])
            coeff_tschuprow = self.tschuprow_coefficient(contingency_table.values)

            print(f"Coefficient de Tschuprow pour {variable} : {coeff_tschuprow:.4f}")


     
class DiscretisationNumerique:
    """
    Discrétisation des variables numériques.
    """
    
    def __init__(self):
        self.binning_models = {}

    def extraction_des_bounds(self, bin_str):
        """
        Nettoyage des bornes pour les tranformer en numérique après.
        """
         
        bin_str_clean = bin_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "") # suppresion des paranthèses
        lower_bound, upper_bound = bin_str_clean.split(", ") # on reçoit les bornes des classes
        lower_bound = float('-inf') if lower_bound == '-inf' else float(lower_bound) # transformation en variable
        upper_bound = float('inf') if upper_bound == 'inf' else float(upper_bound)
        return lower_bound, upper_bound


    def manual_encoding_des_interval(self, bin_str, sorted_intervals):
        """
         Création de l'encodage manuel, transformation en une valeur numérique ordonnée en fonction de la position dans l'ordre trié.       
        """
        lower_bound, upper_bound = self.extraction_des_bounds(bin_str)
          
        if lower_bound == float('-inf'):
            return 1  #si la borne inférieure est = à -inf, alors elle est = à 1
        else:
            interval_tuple = (lower_bound, upper_bound)

            return sorted_intervals.index(interval_tuple) + 1  # les autres bornes sont encodées de manière ordinale dans l'ordre croissante


    def application_de_manual_encoding_numerical(self, train, test, variables_numeriques):
        """
         Application de l'encodage créé pour chaque variable.    
        """
        
        for var in variables_numeriques:
            bin_order = train[f'{var}_binned'].unique()

            numeric_intervals = [self.extraction_des_bounds(bin_str) for bin_str in bin_order]
            sorted_intervals = sorted(numeric_intervals, key=lambda x: (x[0], x[1])) # trier de intervals

            # Encodage
            train[f'{var}_binned_encoded'] = train[f'{var}_binned'].map(lambda x: self.manual_encoding_des_interval(x, sorted_intervals))
            test[f'{var}_binned_encoded'] = test[f'{var}_binned'].map(lambda x: self.manual_encoding_des_interval(x, sorted_intervals))
        
        
        return train, test
    
    
    def appliquer_optimal_binning(self, train, test, variables, target, max_n_bins=5):

        """
         Discrétisation des variables en utilisant OptimalBinning.   
        """
        for variable in variables:
            print(f"\nBinning for variable: {variable}")
            
            optb = OptimalBinning(name=variable, dtype="numerical", max_n_bins=max_n_bins)
            optb.fit(train[variable], train[target])
            
            self.binning_models[variable] = optb
            
            train[f'{variable}_binned'] = optb.transform(train[variable], metric="bins")
            test[f'{variable}_binned'] = optb.transform(test[variable], metric="bins")

            binning_table = optb.binning_table.build()
            print(binning_table[['Bin', 'Count (%)', 'WoE', 'IV']])

        # Encodage des variables
        train, test = self.application_de_manual_encoding_numerical(train, test, variables)

        return train, test
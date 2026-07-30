import pickle
import numpy as np
import warnings
from typing import List, Dict, Any, Optional

class UniversalModelExporter:
    """
    Bibliothèque utilitaire pour exporter différents modèles ML (custom, Sklearn, PyTorch)
    dans un format dictionnaire standardisé que l'orchestrateur peut lire
    sans avoir besoin du code/classes d'entraînement d'origine.
    """

    @staticmethod
    def export_custom_linear(service_instance, 
                             output_path: str, 
                             description: str = "Custom Linear Model"):
        """
        Exporte votre DonorPredictionService/PersistentAdamAgent personnalisé.
        """
        agent = service_instance.agent
        
        # 1. Extraire les statistiques de mise à l'échelle (convertir les stats en ligne en moyenne/écart-type)
        means = dict(agent.standardizer.means)
        stds = {}
        if hasattr(agent, 'standardizer'):
            for feat, m2 in agent.standardizer.m2s.items():
                count = agent.standardizer.counts[feat]
                stds[feat] = np.sqrt(m2 / (count - 1)) if count > 1 else 1.0

        # 2. Extraire les encodeurs catégoriels
        encoders_clean = {k: list(v.classes_) for k, v in agent.encoders.items()}

        # 3. Créer l'enveloppe standardisée
        universal_model = {
            "architecture": "linear_custom", 
            "description": description,
            "params": {
                "weights": dict(agent.weights),
                "intercept": float(agent.intercept),
                "means": means,
                "stds": stds,
                "encoders": encoders_clean
            },
            "metadata": {
                "features": list(agent.active_features),
                "version": "1.0",
                "framework": "custom_numpy"
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(universal_model, f)
        print(f"📦 [Custom] Saved to {output_path}")

    @staticmethod
    def export_sklearn(model, 
                       feature_names: List[str], 
                       output_path: str, 
                       description: str = "Scikit-Learn Model"):
        """
        Exporte un modèle Scikit-Learn (RandomForest, SVM, etc).
        CRITIQUE : fournir feature_names pour que l'orchestrateur connaisse l'ordre des entrées.
        """
        universal_model = {
            "architecture": "sklearn",
            "description": description,
            "model_binary": model, # Objet sklearn réel
            "metadata": {
                "features": feature_names,
                "framework": "sklearn",
                "type": type(model).__name__
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(universal_model, f)
        print(f"📦 [Sklearn] Saved to {output_path}")

    @staticmethod
    def export_pytorch(model, 
                       feature_names: List[str], 
                       output_path: str, 
                       input_shape: tuple = None,
                       description: str = "PyTorch Model"):
        """
        Exporte un modèle PyTorch.
        """
        import torch
        
        # Déplacer sur CPU pour l'export afin d'éviter les erreurs CUDA sur les machines d'inférence
        model.cpu()
        model.eval()
        
        universal_model = {
            "architecture": "pytorch",
            "description": description,
            "model_binary": model, # Sauvegarde du modèle complet (nécessite généralement la définition de la classe)
            # En général, il vaut mieux utiliser state_dict, mais pour un usage simple l'objet complet convient
            # à condition que la classe soit disponible. Pour une vraie portabilité, utiliser TorchScript ci-dessous.
            "metadata": {
                "features": feature_names,
                "framework": "pytorch",
                "input_shape": input_shape
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(universal_model, f)
        print(f"📦 [PyTorch] Saved to {output_path}")

    @staticmethod
    def export_xgboost(model, 
                       feature_names: List[str], 
                       output_path: str, 
                       description: str = "XGBoost Model"):
        """
        Exporte un modèle XGBoost.
        """
        universal_model = {
            "architecture": "xgboost",
            "description": description,
            "model_binary": model,
            "metadata": {
                "features": feature_names,
                "framework": "xgboost"
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(universal_model, f)
        print(f"📦 [XGBoost] Saved to {output_path}")

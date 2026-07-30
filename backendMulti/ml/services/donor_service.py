import pandas as pd
import numpy as np
import pickle
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from sklearn.preprocessing import LabelEncoder
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# ==========================================
# NORMALISATION EN LIGNE (Algorithme de Welford)
# ==========================================
class OnlineStandardizer:
    """Calcul en flux de la moyenne et de la variance pour la mise à l'échelle des caractéristiques"""
    def __init__(self):
        self.means = defaultdict(float)
        self.m2s = defaultdict(float)  # Pour le calcul de la variance
        self.counts = defaultdict(int)
        self.minmax = {}
        self.initialized = set()
    
    def partial_fit(self, x: Dict[str, float]):
        """Met à jour les statistiques en cours avec une seule observation"""
        for feat, val in x.items():
            if not isinstance(val, (int, float)):
                continue
                
            # Met à jour la moyenne courante et la somme des différences au carré (Welford)
            self.counts[feat] += 1
            n = self.counts[feat]
            delta = val - self.means[feat]
            self.means[feat] += delta / n
            delta2 = val - self.means[feat]
            self.m2s[feat] += delta * delta2
            
            # Suivre le min/max pour limiter les valeurs aberrantes
            if feat not in self.minmax:
                self.minmax[feat] = [val, val]
            else:
                self.minmax[feat][0] = min(self.minmax[feat][0], val)
                self.minmax[feat][1] = max(self.minmax[feat][1], val)
    
    def transform(self, x: Dict[str, float]) -> Dict[str, float]:
        """Standardise les valeurs en utilisant les statistiques courantes"""
        result = {}
        for feat, val in x.items():
            if feat not in self.means or self.counts[feat] < 12:
                # Pas encore assez de données, retourne normalisé uniquement par la magnitude
                result[feat] = val / (abs(val) + 1.0)
                continue
            
            mean = self.means[feat]
            var = self.m2s[feat] / (self.counts[feat] - 1) if self.counts[feat] > 1 else 1.0
            std = np.sqrt(var) if var > 0 else 1.0
            
            # Limite les valeurs extrêmes (mise à l'échelle robuste)
            z_score = (val - mean) / std if std > 0 else 0.0
            z_score = np.clip(z_score, -5, 5)  # Limiter à 5 sigmas
            
            result[feat] = z_score
        return result
    
    def get_stats(self):
        return {
            'means': dict(self.means),
            'stds': {k: np.sqrt(v/max(1, self.counts[k]-1)) for k, v in self.m2s.items()},
            'counts': dict(self.counts)
        }

# ==========================================
# AGENT OPTIMISÉ AVEC FILE D'ATTENTE DE CARACTÉRISTIQUES
# ==========================================
class PersistentAdamAgent:
    def __init__(self, learning_rate=0.01, l1_penalty=0.0001, l2_penalty=0.01,
                 feature_prune_threshold=0.0,  # 0 = désactiver l'élagage au départ  
                intercept_penalty=0.1,  # NOUVEAU : régularisation L2 forte sur l'intercept
                 interaction_prune_threshold=0.10,   
                 feature_grace_period=50000,   # Période de grâce très longue
                 interaction_grace_period=8000,       
                 interaction_search_interval=2000, 
                 interaction_threshold=0.15,
                 interaction_shield=True, beta1=0.9, beta2=0.999, eps=1e-8,
                 feature_blacklist=None,
                 grad_clip=1.0, model_id="agent_v1", standardize=True):
        
        # Suivre les statistiques du donneur idéal - STOCKE MAINTENANT LES VALEURS BRUTES
        self.ideal_stats = {
            'continuous': {},      # caractéristique -> {'median': float, 'mean': float, 'q25': float, 'q75': float}
            'categorical': {},     # caractéristique -> {'mode': str, 'categories': Counter}
            'counts': defaultdict(int)
        }
        self.has_ideal_stats = False

        self.lr = learning_rate
        self.l1 = l1_penalty
        self.l2 = l2_penalty
        # Si le seuil est à 0, l'élagage est désactivé
        self.feat_threshold = feature_prune_threshold if feature_prune_threshold > 0 else float('inf')
        self.int_threshold = interaction_prune_threshold
        self.feat_grace = feature_grace_period
        self.int_grace = interaction_grace_period
        self.interaction_search_interval = interaction_search_interval
        self.interaction_threshold = interaction_threshold
        self.interaction_shield = interaction_shield
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.grad_clip = grad_clip
        self.model_id = model_id
        self.standardize = standardize
        self.intercept_penalty = intercept_penalty  # Éviter la dérive
        self.feature_blacklist = set(feature_blacklist or [])
        self.standardizer_momentum = 0.99  # Lisser les mises à jour après l'échauffement

        
        self.standardizer = OnlineStandardizer() if standardize else None
        self.encoders = {}
        self._initialize_state()
        
        # File d'attente d'injection de caractéristiques
        self.feature_queue = []
        self.injection_interval = 2000  # Ajouter une nouvelle caractéristique toutes les 2000 étapes
        self.base_features = set()
    
    def _is_valid_feature(self, name: str) -> bool:
        """Filtre les fuites de cible et les métadonnées"""
        if name in self.feature_blacklist:
            return False
        # Détecter automatiquement les caractéristiques suspectes

          # Motifs de liste noire étendus
        blacklist_patterns = [
            'id', 'uuid', 'identifier', 'donor_id', 'user_id', 'patient_id',  # Identifiants
            'next_6m', 'target', 'label', 'donated_next',  # Fuite de cible
            'score', 'propensity', 'probability', 'predicted',  # Prédictions
            'ideal_score', 'ai_probability', 'donation_propensity',
            'date', 'timestamp', 'created_at', 'as_of_date'  # Dates (nécessitent un traitement spécifique)
        ]

        return not any(p in name.lower() for p in blacklist_patterns)

    def _initialize_state(self):
        self.t = 0
        self.weights = defaultdict(float)
        self.intercept = 0.0
        self.active_features = set()
        self.feature_start_step = {}
        self.pruned_features = set()
        self.active_interactions = set()
        self.interaction_weights = defaultdict(float)
        self.interaction_start_step = {}
        self.candidate_pairs = []
        self.residual_correlations = defaultdict(float)
        self.m = defaultdict(float)
        self.v = defaultdict(float)
        self.m_int = defaultdict(float)
        self.v_int = defaultdict(float)
        self.m_intercept = 0.0
        self.v_intercept = 0.0
        self.predictions_buffer = []
        self.actuals_buffer = []
        self.buffer_size = 1000
        self.inference_count = 0
        self.learning_count = 0
        self.last_prune_step = 0
        self.standardizer_warmup = 1000  # Ignore les mises à jour du standardiseur après l'échauffement pour la vitesse

    def setup_feature_queue(self, all_features: List[str], base_features: List[str]):
        """Filtre la file d'attente pour supprimer les fuites"""
        self.base_features = set(base_features)
        valid_queue = [f for f in all_features 
                      if f not in self.base_features 
                      and f != 'donated_next_6m'
                      and self._is_valid_feature(f)]
        
        # LISTE NOIRE DES FUITES ÉVIDENTES
        auto_blacklist = [f for f in all_features if not self._is_valid_feature(f)]
        if auto_blacklist:
            logger.info("Auto-blacklisted data-leakage features: %s", auto_blacklist)
        
        self.feature_queue = valid_queue
        np.random.shuffle(self.feature_queue)
        logger.info("Valid feature queue: %s features", len(self.feature_queue))

    def maybe_inject_feature(self, current_step: int) -> Optional[str]:
        """Injecte périodiquement une nouvelle caractéristique depuis la file d'attente"""
        if not self.feature_queue:
            return None
            
        if current_step > 0 and current_step % self.injection_interval == 0:
            new_feat = self.feature_queue.pop(0)
            if self.add_feature(new_feat, current_step):
                return new_feat
        return None

    
    def predict_proba(self, row: Dict, return_confidence: bool = False, allow_learning=True, update_stats=True):
        """Prédiction optimisée avec mise en cache des variables locales"""
        active_feats = self.active_features
        weights = self.weights
        interactions = self.active_interactions
        int_weights = self.interaction_weights
    
        # Standardiser uniquement si nécessaire
        if self.standardizer:
            if update_stats:
                row_scaled = self._preprocess(row, update_stats=True)
            else:
                # Juste transformer sans mettre à jour les statistiques
                numeric = {k: self._transform_value(k, v) for k, v in row.items()}
                row_scaled = self.standardizer.transform(numeric)
        else:
            row_scaled = {k: self._transform_value(k, v) for k, v in row.items()}
    
        z = self.intercept
        feature_hits = 0
    
        # Effets principaux
        for feat in active_feats:
            val = row_scaled.get(feat, 0.0)
            if val != 0.0:
                z += weights[feat] * val
                feature_hits += 1
    
        # Interactions
        int_list = list(interactions)[:20]
        for pair in int_list:
            f1, f2 = pair
            v1 = row_scaled.get(f1, 0.0)
            v2 = row_scaled.get(f2, 0.0)
            if v1 != 0.0 and v2 != 0.0:
                z += int_weights[pair] * v1 * v2
                feature_hits += 1
    
        prob = 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))
    
        if return_confidence:
            coverage = feature_hits / max(1, len(active_feats))
            certainty = 1.0 - 2.0 * abs(prob - 0.5)
            return float(prob), float(coverage * certainty)
        return float(prob)

    def partial_fit(self, row: Dict, target: int):
        """Apprentissage simplifié - toujours pré-traite avec mise à jour des stats"""
        self.t += 1
        self.learning_count += 1

        # Pré-traite avec mise à jour des stats (mode d'apprentissage)
        row_scaled = self._preprocess(row, update_stats=True)

        # Prédire sans mettre à jour à nouveau les stats (données déjà mises à l'échelle)
        # Reconvertir au format dict classique pour predict_proba
        pred = self.predict_proba(row_scaled, allow_learning=False, update_stats=False)
        error = pred - target

        # Mettre à jour les tampons
        self.predictions_buffer.append(pred)
        self.actuals_buffer.append(target)
        if len(self.predictions_buffer) > self.buffer_size:
            self.predictions_buffer.pop(0)
            self.actuals_buffer.pop(0)

        # Mettre à jour l'intercept avec une régularisation PLUS FORTE
        self.intercept, self.m_intercept, self.v_intercept = self._adam_update(
            self.intercept, error, self.m_intercept, self.v_intercept, 
            l1_strength=0.0, 
            l2_strength=0.5  # Régularisation PLUS FORTE - force vers 0
        )

        # Limitation plus douce pour permettre l'apprentissage mais éviter l'explosion
        self.intercept = np.clip(self.intercept, -2.0, 2.0)

        # Mettre à jour les caractéristiques en utilisant les valeurs pré-mises à l'échelle
        for feat in self.active_features:
            if feat in row_scaled:
                val = row_scaled[feat]
                if np.isfinite(val):
                    grad = error * val
                    new_w, self.m[feat], self.v[feat] = self._adam_update(
                        self.weights[feat], grad, self.m[feat], self.v[feat],
                        l1_strength=self.l1, l2_strength=self.l2
                    )
                    self.weights[feat] = new_w
          # Le prétraitement fournit des valeurs mises à l'échelle pour le modèle
        row_scaled = self._preprocess(row, update_stats=True)
    
        # Stocker séparément les valeurs brutes pour le profil idéal (avant encodage/mise à l'échelle)
        if target == 1:
            self.has_ideal_stats = True
            for feat, raw_val in row.items():
                if feat == 'donated_next_6m':
                    continue

                # Vérifier si la variable est catégorielle (a un encodeur)
                is_categorical = feat in self.encoders

                if is_categorical:
                    # Stocker la modalité catégorielle (valeur réussie la plus fréquente)
                    if feat not in self.ideal_stats['categorical']:
                        from collections import Counter
                        self.ideal_stats['categorical'][feat] = {'counter': Counter(), 'mode': None}

                    # Stocker la valeur texte brute
                    cat_stats = self.ideal_stats['categorical'][feat]
                    if isinstance(raw_val, str):
                        cat_stats['counter'][raw_val] += 1
                        cat_stats['mode'] = cat_stats['counter'].most_common(1)[0][0]

                else:
                    # Stocker les statistiques continues (médiane, percentiles)
                    if feat not in self.ideal_stats['continuous']:
                        self.ideal_stats['continuous'][feat] = {
                            'values': [],  # Garder les 1000 dernières pour calculer la médiane efficacement
                            'median': None,
                            'mean': None
                        }

                    cont_stats = self.ideal_stats['continuous'][feat]
                    if isinstance(raw_val, (int, float)) and np.isfinite(raw_val):
                        cont_stats['values'].append(float(raw_val))
                        # Conserver seulement les 1000 dernières pour éviter l'explosion mémoire
                        if len(cont_stats['values']) > 1000:
                            cont_stats['values'].pop(0)

                        # Mettre à jour les statistiques en continu
                        cont_stats['median'] = np.median(cont_stats['values'])
                        cont_stats['mean'] = np.mean(cont_stats['values'])
                        cont_stats['q25'] = np.percentile(cont_stats['values'], 25)
                        cont_stats['q75'] = np.percentile(cont_stats['values'], 75)

    
    
    def explain_prediction(self, row: Dict) -> Dict[str, float]:
        """Débogage : Affiche la contribution de chaque caractéristique"""
        row_scaled = self._preprocess(row) if self.standardizer else row
        
        logger.debug(
            "Prediction explanation: intercept=%.3f active_features=%s",
            self.intercept,
            len(self.active_features),
        )
        
        contributions = []
        z = self.intercept
        
        for feat in sorted(self.active_features):
            if feat in row_scaled:
                val = row_scaled[feat]
                weight = self.weights[feat]
                contrib = weight * val
                z += contrib
                contributions.append((feat, row.get(feat, 'N/A'), val, weight, contrib))
        
        # Trier par contribution absolue
        for feat, raw, scaled, w, c in sorted(contributions, key=lambda x: abs(x[4]), reverse=True):
            logger.debug(
                "%s | raw=%s | scaled=%.3f | weight=%.3f | contrib=%.3f",
                feat,
                raw,
                scaled,
                w,
                c,
            )
        
        prob = 1 / (1 + np.exp(-np.clip(z, -25, 25)))
        logger.debug("Final logit=%.3f probability=%.3f", z, prob)
        return {'logit': z, 'prob': prob}

    def _transform_value(self, feature_name: str, value) -> float:
        """Gère l'encodage catégoriel + la conversion de type"""
        if pd.isna(value):
            return 0.0
            
        if isinstance(value, bool):
            return 1.0 if value else 0.0
            
        if isinstance(value, (int, float)):
            return float(value)
            
        if isinstance(value, str):
            # Gestion des catégories
            if feature_name not in self.encoders:
                self.encoders[feature_name] = LabelEncoder()
                self.encoders[feature_name].fit([value, "UNK"])
                return float(self.encoders[feature_name].transform([value])[0])
            try:
                return float(self.encoders[feature_name].transform([value])[0])
            except ValueError:
                # Catégorie inconnue
                try:
                    return float(self.encoders[feature_name].transform(["UNK"])[0])
                except ValueError:
                    return 0.0
        
        return 0.0

    def _preprocess(self, row: Dict, update_stats: bool = True) -> Dict[str, float]:
        """Toujours mettre à jour le standardiseur pendant l'entraînement"""
        # Convertir les types
        numeric = {k: self._transform_value(k, v) for k, v in row.items()}
    
        # Étape 2 : Toujours mettre à jour et transformer si standardisation
        if self.standardize and self.standardizer:
            if update_stats:
                self.standardizer.partial_fit(numeric)
            return self.standardizer.transform(numeric)
    
        return numeric


    def _adam_update(self, weight, grad, m_val, v_val, l1_strength, l2_strength=0.001):
        """Adam avec décroissance des poids"""
        grad = np.clip(grad, -self.grad_clip, self.grad_clip)
        grad = grad + l2_strength * weight
        
        m_new = self.beta1 * m_val + (1 - self.beta1) * grad
        v_new = self.beta2 * v_val + (1 - self.beta2) * (grad ** 2)
        
        m_hat = m_new / (1 - self.beta1 ** self.t)
        v_hat = v_new / (1 - self.beta2 ** self.t)
        
        new_weight = weight - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
        if new_weight > 0:
            new_weight = max(0, new_weight - self.lr * l1_strength)
        elif new_weight < 0:
            new_weight = min(0, new_weight + self.lr * l1_strength)
            
        return new_weight, m_new, v_new

    def add_feature(self, feature_name: str, current_step: Optional[int] = None) -> bool:
        if feature_name in self.active_features or feature_name in self.pruned_features:
            return False
            
        step = current_step or self.t
        self.active_features.add(feature_name)
        self.weights[feature_name] = 0.0
        self.feature_start_step[feature_name] = step
        self.m[feature_name] = 0.0
        self.v[feature_name] = 0.0
        
        # Générer des paires candidates uniquement si le nombre de caractéristiques est raisonnable
        if len(self.active_features) < 50:
            for existing in self.active_features:
                if existing != feature_name:
                    pair = tuple(sorted([feature_name, existing]))
                    if pair not in self.active_interactions:
                        self.candidate_pairs.append(pair)
                        self.residual_correlations[pair] = 0.0
        return True

    def search_interactions(self, current_step: Optional[int] = None):
        if not self.candidate_pairs or len(self.active_interactions) > 20:  # Limite stricte
            return None
            
        step = current_step or self.t
        if step % self.interaction_search_interval != 0:
            return None
            
        # Vérifier uniquement les meilleurs candidats par corrélation
        top_candidates = sorted(
            self.candidate_pairs, 
            key=lambda p: abs(self.residual_correlations[p]), 
            reverse=True
        )[:50]
        
        if not top_candidates:
            return None
            
        best_pair = top_candidates[0]
        best_corr = abs(self.residual_correlations[best_pair])
        
        if best_corr > self.interaction_threshold:
            self.active_interactions.add(best_pair)
            self.interaction_weights[best_pair] = 0.0
            self.interaction_start_step[best_pair] = step
            self.m_int[best_pair] = 0.0
            self.v_int[best_pair] = 0.0
            self.candidate_pairs.remove(best_pair)
            return best_pair, best_corr
        return None

    def prune(self, current_step: Optional[int] = None):
        step = current_step or self.t
        pruned = []
        
        protected = set()
        if self.interaction_shield:
            for (f1, f2) in self.active_interactions:
                if abs(self.interaction_weights[(f1, f2)]) > self.int_threshold * 2:
                    protected.add(f1)
                    protected.add(f2)
        
        # Élaguer les caractéristiques
        to_remove = []
        for feat in list(self.active_features):
            if feat in protected:
                continue
            age = step - self.feature_start_step.get(feat, 0)
            if age > self.feat_grace and abs(self.weights[feat]) < self.feat_threshold:
                to_remove.append(feat)
        
        for feat in to_remove:
            self.active_features.remove(feat)
            self.pruned_features.add(feat)
            for d in [self.weights, self.m, self.v]:
                d.pop(feat, None)
            pruned.append(('feature', feat))
            self.candidate_pairs = [p for p in self.candidate_pairs if feat not in p]
        
        # Élaguer les interactions
        to_remove_int = []
        for pair in list(self.active_interactions):
            age = step - self.interaction_start_step.get(pair, 0)
            if age > self.int_grace and abs(self.interaction_weights[pair]) < self.int_threshold:
                to_remove_int.append(pair)
        
        for pair in to_remove_int:
            self.active_interactions.remove(pair)
            self.interaction_weights.pop(pair, None)
            self.m_int.pop(pair, None)
            self.v_int.pop(pair, None)
            pruned.append(('interaction', pair))
        
        self.last_prune_step = step
        return pruned

    def save(self, filepath: str, metadata: Optional[Dict] = None):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        encoder_states = {f: {'classes': list(e.classes_)} for f, e in self.encoders.items()}
        
        state = {
            'hyperparameters': {
                'learning_rate': self.lr, 'l1_penalty': self.l1, 'l2_penalty': self.l2,
                'feature_prune_threshold': self.feat_threshold,
                'interaction_prune_threshold': self.int_threshold,
                'feature_grace_period': self.feat_grace, 'interaction_grace_period': self.int_grace,
                'interaction_search_interval': self.interaction_search_interval,
                'interaction_threshold': self.interaction_threshold,
                'interaction_shield': self.interaction_shield,
                'beta1': self.beta1, 'beta2': self.beta2, 'grad_clip': self.grad_clip,
                'model_id': self.model_id, 'standardize': self.standardize
            },
            'state': {
                't': self.t, 'weights': dict(self.weights), 'intercept': float(self.intercept),
                'active_features': list(self.active_features),
                'feature_start_step': dict(self.feature_start_step),
                'pruned_features': list(self.pruned_features),
                'active_interactions': [list(x) for x in self.active_interactions],
                'interaction_weights': {f"{k[0]}||{k[1]}": v for k, v in self.interaction_weights.items()},
                'interaction_start_step': {f"{k[0]}||{k[1]}": v for k, v in self.interaction_start_step.items()},
                'candidate_pairs': [[a,b] for a,b in self.candidate_pairs],
                'residual_correlations': {f"{k[0]}||{k[1]}": v for k, v in self.residual_correlations.items()},
                'm': dict(self.m), 'v': dict(self.v),
                'm_int': {f"{k[0]}||{k[1]}": v for k, v in self.m_int.items()},
                'v_int': {f"{k[0]}||{k[1]}": v for k, v in self.v_int.items()},
                'm_intercept': self.m_intercept, 'v_intercept': self.v_intercept,
                'standardizer_stats': self.standardizer.get_stats() if self.standardizer else None,
                'encoders': encoder_states,
                'counts': {'inference': self.inference_count, 'learning': self.learning_count}
            },
            'metadata': metadata or {}
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(
            "Saved donor agent to %s (features=%s, intercept=%.3f)",
            filepath,
            len(self.active_features),
            self.intercept,
        )

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        agent = cls(**state['hyperparameters'])
        s = state['state']
        
        agent.t = s['t']
        agent.weights = defaultdict(float, s['weights'])
        agent.intercept = s['intercept']
        agent.active_features = set(s['active_features'])
        agent.feature_start_step = s['feature_start_step']
        agent.pruned_features = set(s['pruned_features'])
        agent.active_interactions = set(tuple(x) for x in s['active_interactions'])
        agent.interaction_weights = defaultdict(float, {tuple(k.split('||')): v for k, v in s['interaction_weights'].items()})
        agent.interaction_start_step = {tuple(k.split('||')): v for k, v in s['interaction_start_step'].items()}
        agent.candidate_pairs = [tuple(x) for x in s['candidate_pairs']]
        agent.residual_correlations = defaultdict(float, {tuple(k.split('||')): v for k, v in s['residual_correlations'].items()})
        agent.m = defaultdict(float, s['m'])
        agent.v = defaultdict(float, s['v'])
        agent.m_int = defaultdict(float, {tuple(k.split('||')): v for k, v in s['m_int'].items()})
        agent.v_int = defaultdict(float, {tuple(k.split('||')): v for k, v in s['v_int'].items()})
        agent.m_intercept = s['m_intercept']
        agent.v_intercept = s['v_intercept']
        agent.inference_count = s['counts']['inference']
        agent.learning_count = s['counts']['learning']
        
        # Restaurer le standardiseur
        if s['standardizer_stats'] and agent.standardizer:
            stats = s['standardizer_stats']
            agent.standardizer.means = defaultdict(float, stats['means'])
            agent.standardizer.counts = defaultdict(int, {k: int(v) for k, v in stats.get('counts', {}).items()})
            # Reconstruire m2s à partir des écarts-types
            for k, std in stats['stds'].items():
                n = agent.standardizer.counts[k]
                agent.standardizer.m2s[k] = (std**2) * max(1, n-1)
        
        for feat, enc_state in s.get('encoders', {}).items():
            agent.encoders[feat] = LabelEncoder()
            agent.encoders[feat].fit(enc_state['classes'])
        
        logger.info(
            "Loaded donor agent from %s (t=%s, features=%s)",
            filepath,
            agent.t,
            len(agent.active_features),
        )
        return agent
    

    def get_recommendations(self, row: Dict) -> Dict[str, Dict[str, Any]]:
        """Retourne des recommandations avec des valeurs LISIBLES PAR L'HUMAIN"""
        if not self.has_ideal_stats:
            return {"erreur": "Pas encore de statistiques idéales (besoin d'exemples positifs)"}
    
        # Obtenir la prédiction du modèle (utilise les valeurs échelonnées internement)
        current_prob = self.predict_proba(row, update_stats=False)
    
        # Traitement de l'entrée brute pour l'affichage
        recommendations = {}
    
        for feat in self.active_features:
            if feat not in row:
                continue

            current_raw = row[feat]
            weight = self.weights[feat]

            # Déterminer si la caractéristique est catégorielle
            est_categorical = feat in self.encoders and feat in self.ideal_stats['categorical']

            if est_categorical:
                # TRAITEMENT DES VALEURS CATÉGORIELLES
                stats_cat = self.ideal_stats['categorical'][feat]
                ideal_raw = stats_cat['mode']

                # verifiers si la valeur actuelle est d j optimale
                est_optimal = (str(current_raw).lower() == str(ideal_raw).lower())

                direction = 'optimal' if est_optimal else f"utiliser: {ideal_raw}"

                # Calculer l'impact (utilise les valeurs mises à l'échelle pour les calculs)
                try:
                    current_encoded = self.encoders[feat].transform([str(current_raw)])[0] if str(current_raw) in self.encoders[feat].classes_ else 0
                    ideal_encoded = self.encoders[feat].transform([ideal_raw])[0]
                    if self.standardizer and feat in self.standardizer.means:
                        mean = self.standardizer.means[feat]
                        std = np.sqrt(self.standardizer.m2s[feat] / max(1, self.standardizer.counts[feat]-1))
                        current_scaled = (current_encoded - mean) / std if std > 0 else 0
                        ideal_scaled = (ideal_encoded - mean) / std if std > 0 else 0
                    else:
                        current_scaled = current_encoded
                        ideal_scaled = ideal_encoded

                    impact = abs((ideal_scaled - current_scaled) * weight)
                except:
                    impact = 0.0

                recommendations[feat] = {
                    'type': 'categorical',
                    'current': str(current_raw),
                    'ideal': str(ideal_raw),
                    'direction': direction,
                    'priority': 'high' if impact > 1.0 else ('medium' if impact > 0.3 else 'low'),
                    'impact': round(float(impact), 3),
                    'weight': round(float(weight), 3)
                }

            else:
                # TRAITEMENT DES VALEURS CONTINUES
                if feat not in self.ideal_stats['continuous']:
                    continue

                stats_cont = self.ideal_stats['continuous'][feat]
                median_ideal = stats_cont.get('median', 0)

                # D terminer la direction en fonction du signe du poids
                if weight > 0:
                    # Plus est mieux, viser le Q75 (quartile sup rieur des donateurs ayant r ussi)
                    ideal_raw = stats_cont.get('q75', median_ideal)
                    direction = 'augmenter' if current_raw < ideal_raw else ('optimal' if current_raw >= median_ideal else 'maintenir')
                else:
                    # Moins est mieux, viser le Q25 (quartile inf rieur des donateurs ayant r ussi)
                    ideal_raw = stats_cont.get('q25', median_ideal)
                    direction = 'diminuer' if current_raw > ideal_raw else ('optimal' if current_raw <= median_ideal else 'maintenir')

                # Calculer l'impact en utilisant les valeurs mises à l'échelle
                row_scaled = self._preprocess({feat: current_raw}, update_stats=False) if self.standardizer else {feat: float(current_raw)}
                current_scaled = row_scaled.get(feat, 0)

                ideal_scaled_row = self._preprocess({feat: ideal_raw}, update_stats=False) if self.standardizer else {feat: float(ideal_raw)}
                ideal_scaled = ideal_scaled_row.get(feat, 0)

                impact = abs((ideal_scaled - current_scaled) * weight)

                # Formater les valeurs avec unités
                if feat in ['recency_days']:
                    unite = 'jours'
                    current_fmt = f"{int(current_raw)} {unite}"
                    ideal_fmt = f"{int(ideal_raw)} {unite}"
                elif feat in ['age']:
                    unite = 'ans'
                    current_fmt = f"{int(current_raw)} {unite}"
                    ideal_fmt = f"{int(ideal_raw)} {unite}"
                elif feat in ['bmi']:
                    unite = ''
                    current_fmt = f"{float(current_raw):.1f}"
                    ideal_fmt = f"{float(ideal_raw):.1f}"
                else:
                    current_fmt = str(current_raw)
                    ideal_fmt = str(int(ideal_raw)) if isinstance(ideal_raw, (int, float)) else str(ideal_raw)

                recommendations[feat] = {
                    'type': 'continuous',
                    'current': current_fmt,
                    'current_raw': float(current_raw),
                    'ideal': ideal_fmt,
                    'ideal_raw': float(ideal_raw),
                    'direction': direction,
                    'priority': 'high' if impact > 1.0 else ('medium' if impact > 0.3 else 'low'),
                    'impact': round(float(impact), 3),
                    'weight': round(float(weight), 3),
                    'unite': unite if 'unite' in locals() else ''
                }
    
        return recommendations, current_prob

    
    def explain_with_recommendations(self, row: Dict) -> str:
        """Affichage lisible avec des valeurs compréhensibles"""
        recs, current_prob = self.get_recommendations(row)
    
        if "error" in recs:
            return recs["error"]
    
        # Trier par impact
        sorted_recs = sorted(recs.items(), key=lambda x: x[1]['impact'], reverse=True)
    
        output = f"\n🎯 PROFIL IDÉAL VS ACTUEL (Score actuel: {current_prob:.1%}):\n"
        output += "-" * 90 + "\n"
        output += f"{'Caractéristique':<28} {'Valeur Actuelle':<18} {'Valeur Idéale':<18} {'Action':<15} {'Impact':<8}\n"
        output += "-" * 90 + "\n"
    
        total_potential_gain = 0.0
    
        for feat, data in sorted_recs:
            if data['priority'] == 'high':
                emoji = '🔴'
            elif data['priority'] == 'medium':
                emoji = '🟡'
            else:
                emoji = '🟢'

            direction = data['direction']
            if direction == 'optimal':
                action = '✅ OK'
            elif direction.startswith('use:'):
                action = f"→ {direction.split(':')[1].strip()}"
            else:
                action = {'increase': '↑ Augmenter', 'decrease': '↓ Diminuer', 'maintain': '~ Maintenir'}.get(direction, direction)

            current = str(data['current'])[:16]
            ideal = str(data['ideal'])[:16]

            output += f"{emoji} {feat:<25} {current:<18} {ideal:<18} {action:<15} +{data['impact']:.2f}\n"

            if data['direction'] != 'optimal':
                total_potential_gain += data['impact']
    
        # Calculer le potentiel réaliste (plafond à 95 %)
        potential_logit = np.log(current_prob / (1 - current_prob)) + total_potential_gain if current_prob > 0.01 else -4 + total_potential_gain
        potential_prob = 1 / (1 + np.exp(-np.clip(potential_logit, -10, 10)))
        potential_prob = min(potential_prob, 0.95)  # Plafond réaliste
    
        output += "-" * 90 + "\n"
        output += f"Score Actuel: {current_prob:.1%} | Score Potentiel: {potential_prob:.1%} | Gain: +{potential_prob-current_prob:.1%}\n"
        output += f"\n💡 Interprétation: En atteignant les valeursidéales ci-dessus, le donneur passerait de {current_prob:.0%} à ~{potential_prob:.0%} de probabilité de don.\n"
    
        return output

    def get_metrics(self):
        if len(self.predictions_buffer) < 100:
            return None
        from sklearn.metrics import roc_auc_score, log_loss
        preds = np.clip(np.array(self.predictions_buffer), 1e-7, 1-1e-7)
        actuals = np.array(self.actuals_buffer)
        try:
            return {
                'auc': roc_auc_score(actuals, preds),
                'logloss': log_loss(actuals, preds),
                'calibration': abs(np.mean(preds) - np.mean(actuals)),
                'n_features': len(self.active_features),
                'n_interactions': len(self.active_interactions)
            }
        except:
            return None

# ==========================================
# SERVICE RAPIDE AVEC AUTO-INJECTION
# ==========================================
class DonorPredictionService:
    def __init__(self, model_path: Optional[str] = None, 
                 auto_prune: bool = False,  # Désactivé par défaut pour conserver les caractéristiques
                 auto_interaction: bool = True,
                 injection_interval: int = 2000):
        
        self.auto_prune = auto_prune
        self.auto_interaction = auto_interaction
        
        if model_path and Path(model_path).exists():
            self.agent = PersistentAdamAgent.load(model_path)
        else:
            logger.info("Creating new donor agent with pruning disabled.")
            self.agent = PersistentAdamAgent(
                learning_rate=0.01,
                l1_penalty=0.0001,
                l2_penalty=0.01,     # Garder un L2 élevé pour éviter le surapprentissage plutôt que d'élaguer
                feature_prune_threshold=0.0,  # DÉSACTIVER L'ÉLAGAGE
                interaction_prune_threshold=0.10,
                feature_grace_period=999999,  # Pratiquement infini
                standardize=True
            )
        
        self.injection_interval = injection_interval
    
    def setup_training(self, df: pd.DataFrame, base_features: List[str]):
        """Configure la file d'attente des caractéristiques à partir des colonnes du dataframe"""
        all_features = [c for c in df.columns if c != 'donated_next_6m']
        self.agent.setup_feature_queue(all_features, base_features)
        
        # Ajouter les caractéristiques de base
        for f in base_features:
            if f in df.columns:
                self.agent.add_feature(f, 0)
    
    def train_batch(self, df: pd.DataFrame, log_interval: int = 1000):
        """Boucle d'entraînement à grande vitesse avec auto-injection"""
        records = df.to_dict('records')
        targets = df['donated_next_6m'].values
        
        logger.info("Training donor agent on %s samples.", len(records))
        logger.info("Feature injection interval: every %s steps.", self.injection_interval)
        start = datetime.now()
        
        last_inj = 0
        
        for i, (row, target) in enumerate(zip(records, targets)):
            # Injection périodique de caractéristiques
            inj = self.agent.maybe_inject_feature(i)
            if inj:
                logger.debug(
                    "Step %s injected feature %s (total features=%s)",
                    i,
                    inj,
                    len(self.agent.active_features),
                )
                last_inj = i
            
            # Entraînement
            row_dict = {k: v for k, v in row.items() if k != 'donated_next_6m'}
            self.agent.partial_fit(row_dict, int(target))
            
            # Journalisation
            if i % log_interval == 0 and i > 0:
                elapsed = (datetime.now() - start).total_seconds()
                speed = i / elapsed
                metrics = self.agent.get_metrics()
                feat_count = len(self.agent.active_features)
                recent_prob = np.mean(self.agent.predictions_buffer[-100:]) if self.agent.predictions_buffer else 0.5
                if metrics:
                    logger.info(
                        "Step %s | %.0f row/s | features=%s | prob=%.3f | auc=%.3f",
                        i,
                        speed,
                        feat_count,
                        recent_prob,
                        metrics.get('auc', 0),
                    )
            
            # Élagage optionnel (si activé)
            if self.auto_prune and i % 5000 == 0 and i > 0:
                pruned = self.agent.prune()
                if pruned:
                    logger.debug("Pruned %s items.", len(pruned))
            
            # Recherche d'interactions
            if self.auto_interaction and i % self.agent.interaction_search_interval == 0 and i > 0:
                result = self.agent.search_interactions()
                if result:
                    pair, corr = result
                    logger.debug("Interaction found: %s x %s (%.3f)", pair[0], pair[1], corr)
        
        total_time = (datetime.now() - start).total_seconds()
        logger.info(
            "Training complete in %.1fs (%.0f row/s avg). Final feature count: %s",
            total_time,
            len(records) / total_time,
            len(self.agent.active_features),
        )
        logger.debug("Active donor agent features: %s", sorted(self.agent.active_features))
    
    def predict(self, row: Dict[str, Any], learn_from: Optional[int] = None):
        if hasattr(row, 'to_dict'):
            row = row.to_dict()
        prob = self.agent.predict_proba(row, return_confidence=True)
        
        if learn_from is not None:
            self.agent.partial_fit(row, learn_from)
        
        return prob[0] if isinstance(prob, tuple) else prob
    
    def explain(self, row: Dict):
        return self.agent.explain_prediction(row)
    
    def save(self, path: str):
        self.agent.save(path, metadata={
            'features': list(self.agent.active_features),
            'samples_trained': self.agent.learning_count
        })

    def get_recommendations(self, donor_features: Dict):
        """Obtenir des recommandations d'amélioration pour un donneur"""
        return self.agent.get_recommendations(donor_features)

    def explain_with_recommendations(self, donor_features: Dict):
        """Affichage lisible de l'analyse du donneur avec recommandations"""
        return self.agent.explain_with_recommendations(donor_features)

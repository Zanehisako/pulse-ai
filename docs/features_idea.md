# 1) Catalogue de features à tester (par familles)
## A) Historique de don (comportement pur) — très fort pour la propension

donations_count_{30,90,365}d

days_since_last_donation

avg_days_between_donations_{365}d, std_days_between_donations

donation_cadence_trend (cadence en hausse/baisse)

donation_type_mix (sang total / plasma / plaquettes)

show_rate : rdv pris vs réalisés (si tu as les rdv)

cancel_rate_{90}d, no_show_rate_{90}d

preferred_time_window (matin/pm/soir) basé sur historique

weekday_preference (jour semaine le plus fréquent)

## B) Éligibilité / contraintes biologiques — fort mais attention fuite

deferral_count_{365}d (nombre de refus/ajournements)

days_since_last_deferral, last_deferral_reason_group (catégories, pas détail médical sensible)

hb_last_value, hb_trend, hb_variability (si hémoglobine suivie)

adverse_event_history_flag (malaise, etc. si dispo)

eligibility_probability_proxy (ex: historique “souvent refusé”)

Évite d’utiliser un champ “eligible_now” calculé le jour même si ça encode la cible.

## C) Valeur biologique (pour “Donneur Idéal”)

blood_group (A/B/O/AB, Rh)

rarity_score (poids métier : ex O- très haut)

compatibility_coverage_score (combien de receveurs potentiels)

cmv_status / phénotype si vous l’avez (hyper utile mais gouvernance stricte)

component_suitability (plasma/plaquettes)

donor_quality_consistency (stabilité des paramètres biologiques utiles)

## D) Relation & engagement (recrutement / fidélisation)

lifetime_donations

donor_tenure_days (ancienneté)

reactivation_count (a quitté puis revenu)

churn_risk_proxy (ex: gap > X jours vs son rythme)

communication_touch_count_{30,90}d (SMS/email/appels)

last_contact_days_ago

contact_fatigue_score (trop contacté récemment)

channel_preference (si dispo: réponse par canal)

campaign_response_rate_{365}d

## E) Géographie & accessibilité (logistique côté donneur)

home_to_nearest_center_distance_km

home_to_assigned_center_travel_time_min (idéalement temps, pas km)

public_transit_access_score (si tu as/peux approx)

car_owner_proxy (si dispo / ou inféré prudemment)

area_walkability_index / densité urbaine (niveau secteur, pas individuel)

center_density_within_10km

mobility_constraint_proxy (ex: n’accepte que centres très proches)

## F) Contexte temporel (saisonnalité / vie quotidienne)

day_of_week, month, season

is_holiday, is_school_break (selon région)

payday_week (souvent influence disponibilité)

major_event_flag (marathon, tempête annoncée, etc.)

personal_availability_proxy (créneau historique + distance)

## G) Météo (impact réel sur show/no-show + demande)

weather_temp, precipitation, snow, wind

weather_severity_index

weather_forecast_{t0+1,t0+3} (si décision future)

travel_disruption_flag (tempête, verglas)

weather_center_interaction (météo * distance)

## H) Caractéristiques centre / collecte (fixe & mobile)

center_type (fixe/mobile)

center_capacity_slots_{day} (si planif dispo)

expected_wait_time (si tu as)

open_hours_match_score (match avec préférence donneur)

center_quality_proxy (taux annulation, satisfaction si dispo)

mobile_unit_schedule_features : proximité + jour de passage + fréquence

center_distance_rank (rang du centre le plus proche pour ce donneur)

## I) Offre / demande / stocks (pour stratégie et prévision)

stock_units_{blood_group} (par site/région)

days_to_expiry_distribution (périssabilité)

incoming_supply_scheduled_{7,30}d

historical_demand_{7,30,365}d (par groupe)

demand_spike_indicator (pics historiques)

hospital_order_backlog (si vous l’avez)

criticality_score (poids métier par groupe et niveau stock)

## J) Transport & transferts (logistique réseau)

center_to_center_distance_km / time_min

transport_capacity_{day} (véhicules, slots)

transfer_cost (km, temps, froid, etc.)

route_feasibility_flag (fenêtres temps)

cold_chain_constraint_features

## K) Variables “sensibles” (culture, niveau intellectuel, etc.) — à manipuler avec précaution

Si tu veux tester :

Utilise des features au niveau zone (quartier/secteur) plutôt que “profil individuel” : area_language_mix, area_education_index, deprivation_index.

Ajoute des tests d’équité (voir section 3) et des règles “ne pas pénaliser”.

# 2) Features “spéciales” très utiles : interactions Donneur × Centre × Jour

Pour répondre à des questions du genre “qui inviter où et quand” :

distance(donor, center)

availability_match(donor_pref_time, center_open_hours)

weather(donor_area, day) × distance

center_capacity(day) × donor_propensity

expected_wait_time × donor_sensitivity_to_wait (proxy via historique)

👉 Techniquement : tu crées un dataset candidat (donor_id, center_id, day) pour recommender.

# 3) Comment tester proprement (sinon tu vas te mentir avec des features)

Checklist rapide :

Ablation par familles : ajoute/retire blocs A/B/C… et mesure.

Time-split strict (train avant, test après).

Disponibilité inference : chaque feature doit être dispo au moment où tu scores.

Stabilité : feature importance stable sur plusieurs mois.

Missingness : taux NULL, et “NULL = information” (souvent).

Leakage tests : une feature trop “parfaite” = suspect.

Fairness (si features socio) : performance + taux de ciblage par groupes/secteurs.

# 4) “Trucs” (briques) à construire pour répondre à toutes les questions plus tard
## 1) Donor 360 (profil analytique)

Une table par donneur (snapshots quotidiens) :

comportement, bio, logistique, engagement, contraintes.
→ Sert à tout : churn, propension, personnalisation.

## 2) Feature Store (offline + online)

Offline : historique training

Online : dernières valeurs pour inference < 1s
→ Garantie “mêmes features training/production”.

## 3) 3 modèles + 1 moteur de décision

Propension (qui est susceptible)

Demande (de quoi on aura besoin)

Donneur idéal (valeur + logistique + propension)

Politique (attendre vs agir / contacter vs transférer / planifier collecte)

## 4) Simulateur “what-if”

Tu peux répondre à :

“Si tempête demain ?”

“Si on ajoute une collecte mobile à X ?”

“Si capacité centre +20% ?”

“Si on contacte top-500 O- ?”
→ Sorties : risque rupture, gaspillage, coût, service level.

## 5) Système d’alertes + recommandations (Kafka)

alerts.shortage_risk

alerts.weather_disruption

recommendations.contact_list

recommendations.transfer_plan
→ Plug direct dashboards & ops.

# 5) Les questions “types” auxquelles tu pourras répondre

Qui contacter aujourd’hui pour O- à moins de 10 km, avec probabilité > X ?

Quel centre risque une rupture dans 7 jours et pourquoi ?

Est-ce que je dois attendre (les dons planifiés suffisent) ou agir (campagne / transfert) ?

Où placer une collecte mobile pour maximiser unités collectées avec min coût ?

Quel est le ROI d’une campagne (invitations) et la fatigue donneur ?

Quels facteurs font chuter le show-rate (météo, distance, attentes) ?
-- ============================================
-- PIOS — Création complète de la base de données
-- Compatible: PostgreSQL
-- ============================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS postgis;  -- pour geometry
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- MODULE A : AUTHENTIFICATION & UTILISATEURS
-- ============================================

CREATE TABLE ROLE (
    id_role         SERIAL PRIMARY KEY,
    nom_role        VARCHAR(50)  UNIQUE NOT NULL,
    description     TEXT,
    permissions     JSONB,
    niveau_acces    INT CHECK (niveau_acces BETWEEN 1 AND 5)
);
CREATE INDEX idx_role_nom ON ROLE(nom_role);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE CENTRE_DON (
    id_centre           SERIAL PRIMARY KEY,
    code_centre         VARCHAR(20)  UNIQUE NOT NULL,
    nom                 VARCHAR(150) NOT NULL,
    type                VARCHAR(20)  CHECK (type IN ('fixe','mobile','hopital','clinique')),
    adresse             TEXT,
    code_postal         VARCHAR(10),
    ville               VARCHAR(100),
    region              VARCHAR(100),
    pays                VARCHAR(50)  DEFAULT 'France',
    latitude            DECIMAL(10,8),
    longitude           DECIMAL(11,8),
    telephone           VARCHAR(15),
    email               VARCHAR(100),
    capacite_journaliere INT,
    horaires            JSONB,
    statut              VARCHAR(20)  DEFAULT 'actif'
                        CHECK (statut IN ('actif','ferme_temporaire','en_maintenance','ferme_definitif')),
    equipements         JSONB,
    polygone_secteur    GEOMETRY,
    date_ouverture      DATE
);
CREATE INDEX idx_centre_ville     ON CENTRE_DON(ville, region);
CREATE INDEX idx_centre_statut    ON CENTRE_DON(statut);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE UTILISATEUR (
    id_utilisateur      SERIAL PRIMARY KEY,
    email               VARCHAR(100) UNIQUE NOT NULL,
    mot_de_passe_hash   VARCHAR(255) NOT NULL,
    nom                 VARCHAR(100),
    prenom              VARCHAR(100),
    telephone           VARCHAR(15),
    photo_profil        VARCHAR(255),
    id_role             INT REFERENCES ROLE(id_role),
    id_centre           INT REFERENCES CENTRE_DON(id_centre),
    statut              VARCHAR(20)  DEFAULT 'actif'
                        CHECK (statut IN ('actif','inactif','suspendu')),
    derniere_connexion  TIMESTAMP,
    langue              VARCHAR(2)   DEFAULT 'FR' CHECK (langue IN ('FR','EN','AR')),
    theme               VARCHAR(20)  DEFAULT 'clair'
                        CHECK (theme IN ('clair','sombre','haut_contraste')),
    date_creation       TIMESTAMP    DEFAULT NOW(),
    token_refresh       VARCHAR(500),
    token_expiration    TIMESTAMP
);
CREATE INDEX idx_utilisateur_email   ON UTILISATEUR(email);
CREATE INDEX idx_utilisateur_role    ON UTILISATEUR(id_role);
CREATE INDEX idx_utilisateur_centre  ON UTILISATEUR(id_centre);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE JOURNAL_AUDIT (
    id_audit        SERIAL PRIMARY KEY,
    id_utilisateur  INT REFERENCES UTILISATEUR(id_utilisateur),
    action          VARCHAR(100),
    module          VARCHAR(50),
    details         JSONB,
    ip_adresse      VARCHAR(45),
    timestamp       TIMESTAMP DEFAULT NOW(),
    niveau_gravite  VARCHAR(20) CHECK (niveau_gravite IN ('info','warning','critique'))
);
CREATE INDEX idx_audit_user      ON JOURNAL_AUDIT(id_utilisateur);
CREATE INDEX idx_audit_timestamp ON JOURNAL_AUDIT(timestamp);
CREATE INDEX idx_audit_module    ON JOURNAL_AUDIT(module);

-- ============================================
-- MODULE B : GESTION DES DONNEURS
-- ============================================

CREATE TABLE DONNEUR (
    id_donneur              SERIAL PRIMARY KEY,
    numero_donneur          VARCHAR(20)  UNIQUE NOT NULL,
    nom                     VARCHAR(100) NOT NULL,
    prenom                  VARCHAR(100) NOT NULL,
    date_naissance          DATE         NOT NULL,
    sexe                    VARCHAR(1)   CHECK (sexe IN ('M','F')),
    groupe_sanguin          VARCHAR(3)   CHECK (groupe_sanguin IN ('A+','A-','B+','B-','AB+','AB-','O+','O-')),
    rheusus                 VARCHAR(1)   CHECK (rheusus IN ('+','-')),
    email                   VARCHAR(100) UNIQUE,
    telephone               VARCHAR(15),
    telephone_urgence       VARCHAR(15),
    adresse                 TEXT,
    code_postal             VARCHAR(10),
    ville                   VARCHAR(100),
    pays                    VARCHAR(50)  DEFAULT 'France',
    latitude                DECIMAL(10,8),
    longitude               DECIMAL(11,8),
    poids                   DECIMAL(5,2),
    taille                  INT,
    profession              VARCHAR(100),
    statut                  VARCHAR(20)  DEFAULT 'eligible'
                            CHECK (statut IN ('eligible','non_eligible','suspendu','actif','inactif')),
    raison_suspension       TEXT,
    date_inscription        TIMESTAMP    DEFAULT NOW(),
    dernier_don             DATE,
    prochain_don_eligible   DATE,
    nombre_total_dons       INT          DEFAULT 0,
    consentement_sms        BOOLEAN      DEFAULT FALSE,
    consentement_email      BOOLEAN      DEFAULT FALSE,
    consentement_geoloc     BOOLEAN      DEFAULT FALSE,
    photo_profil            VARCHAR(255),
    qr_code                 VARCHAR(255)
);
CREATE INDEX idx_donneur_numero        ON DONNEUR(numero_donneur);
CREATE INDEX idx_donneur_email         ON DONNEUR(email);
CREATE INDEX idx_donneur_groupe        ON DONNEUR(groupe_sanguin);
CREATE INDEX idx_donneur_ville         ON DONNEUR(ville, code_postal);
CREATE INDEX idx_donneur_statut        ON DONNEUR(statut);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE ANTECEDENT_MEDICAL (
    id_antecedent           SERIAL PRIMARY KEY,
    id_donneur              INT  REFERENCES DONNEUR(id_donneur),
    type                    VARCHAR(20) CHECK (type IN ('maladie','chirurgie','allergie','medicament','voyage')),
    description             TEXT,
    date_debut              DATE,
    date_fin                DATE,
    contre_indication       BOOLEAN DEFAULT FALSE,
    duree_suspension_jours  INT
);
CREATE INDEX idx_antecedent_donneur ON ANTECEDENT_MEDICAL(id_donneur);
CREATE INDEX idx_antecedent_type    ON ANTECEDENT_MEDICAL(type);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE CONSENTEMENT_DONNEE (
    id_consentement         SERIAL PRIMARY KEY,
    id_donneur              INT REFERENCES DONNEUR(id_donneur),
    type_consentement       VARCHAR(100),
    accepte                 BOOLEAN   DEFAULT FALSE,
    date_consentement       TIMESTAMP DEFAULT NOW(),
    version_document        VARCHAR(20),
    signature_electronique  TEXT
);
CREATE INDEX idx_consentement_donneur ON CONSENTEMENT_DONNEE(id_donneur);
CREATE INDEX idx_consentement_type    ON CONSENTEMENT_DONNEE(type_consentement);

-- ============================================
-- MODULE C : GESTION DES DONS
-- ============================================

CREATE TABLE PERSONNEL_MEDICAL (
    id_personnel    SERIAL PRIMARY KEY,
    id_utilisateur  INT UNIQUE REFERENCES UTILISATEUR(id_utilisateur),
    matricule       VARCHAR(20) UNIQUE NOT NULL,
    specialite      VARCHAR(50) CHECK (specialite IN ('medecin','infirmier','technicien_lab','preleveur')),
    certifications  JSONB,
    date_embauche   DATE,
    statut          VARCHAR(20) DEFAULT 'actif'
                    CHECK (statut IN ('actif','conge','formation','inactif'))
);
CREATE INDEX idx_personnel_matricule  ON PERSONNEL_MEDICAL(matricule);
CREATE INDEX idx_personnel_specialite ON PERSONNEL_MEDICAL(specialite);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE VEHICULE (
    id_vehicule                 SERIAL PRIMARY KEY,
    immatriculation             VARCHAR(20) UNIQUE NOT NULL,
    type                        VARCHAR(30) CHECK (type IN ('ambulance','camionnette','vehicule_leger','moto')),
    marque                      VARCHAR(50),
    modele                      VARCHAR(50),
    annee                       INT,
    capacite_poches             INT,
    equipement_refrigeration    BOOLEAN     DEFAULT FALSE,
    temperature_min             DECIMAL(5,2),
    temperature_max             DECIMAL(5,2),
    statut                      VARCHAR(20) DEFAULT 'disponible'
                                CHECK (statut IN ('disponible','en_mission','maintenance','hors_service')),
    latitude                    DECIMAL(10,8),
    longitude                   DECIMAL(11,8),
    date_derniere_position      TIMESTAMP,
    gps_device_id               VARCHAR(100),
    kilometrage                 INT,
    date_assurance              DATE,
    date_controle_technique     DATE
);
CREATE INDEX idx_vehicule_statut ON VEHICULE(statut);
CREATE INDEX idx_vehicule_gps    ON VEHICULE(gps_device_id);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE COLLECTE_MOBILE (
    id_collecte             SERIAL PRIMARY KEY,
    id_vehicule             INT REFERENCES VEHICULE(id_vehicule),
    id_coordinateur         INT REFERENCES PERSONNEL_MEDICAL(id_personnel),
    lieu_collecte           VARCHAR(200),
    adresse                 TEXT,
    latitude                DECIMAL(10,8),
    longitude               DECIMAL(11,8),
    date_debut              TIMESTAMP,
    date_fin                TIMESTAMP,
    statut                  VARCHAR(20) CHECK (statut IN ('planifiee','en_cours','terminee','annulee','reportee')),
    objectif_dons           INT,
    nombre_dons_realises    INT DEFAULT 0,
    rayon_geofence_m        INT DEFAULT 100,
    qr_code_acces           VARCHAR(255)
);
CREATE INDEX idx_collecte_vehicule ON COLLECTE_MOBILE(id_vehicule);
CREATE INDEX idx_collecte_statut   ON COLLECTE_MOBILE(statut);
CREATE INDEX idx_collecte_date     ON COLLECTE_MOBILE(date_debut);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE DON (
    id_don                  SERIAL PRIMARY KEY,
    numero_poche            VARCHAR(50)  UNIQUE NOT NULL,
    id_donneur              INT  REFERENCES DONNEUR(id_donneur),
    id_centre               INT  REFERENCES CENTRE_DON(id_centre),
    id_medecin              INT  REFERENCES PERSONNEL_MEDICAL(id_personnel),
    id_collecte             INT  REFERENCES COLLECTE_MOBILE(id_collecte),
    date_don                TIMESTAMP   DEFAULT NOW(),
    type_don                VARCHAR(30) CHECK (type_don IN ('sang_total','plasma','plaquettes','double_globules','aferese')),
    volume_ml               INT,
    duree_minutes           INT,
    statut                  VARCHAR(20) DEFAULT 'planifie'
                            CHECK (statut IN ('planifie','en_cours','termine','annule','refuse','valide','detruit')),
    raison_refus            TEXT,
    code_barre              VARCHAR(100) UNIQUE,
    nfc_tag                 VARCHAR(100),
    commentaire             TEXT,
    prochaine_eligibilite   DATE,
    created_at              TIMESTAMP   DEFAULT NOW(),
    updated_at              TIMESTAMP   DEFAULT NOW()
);
CREATE INDEX idx_don_poche    ON DON(numero_poche);
CREATE INDEX idx_don_barre    ON DON(code_barre);
CREATE INDEX idx_don_donneur  ON DON(id_donneur);
CREATE INDEX idx_don_centre   ON DON(id_centre);
CREATE INDEX idx_don_date     ON DON(date_don);
CREATE INDEX idx_don_statut   ON DON(statut);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE EXAMEN_MEDICAL (
    id_examen           SERIAL PRIMARY KEY,
    id_don              INT UNIQUE REFERENCES DON(id_don),
    id_medecin          INT REFERENCES PERSONNEL_MEDICAL(id_personnel),
    date_examen         TIMESTAMP    DEFAULT NOW(),
    poids_kg            DECIMAL(5,2),
    tension_systolique  INT,
    tension_diastolique INT,
    pouls               INT,
    temperature         DECIMAL(4,2),
    hemoglobine         DECIMAL(4,2),
    ferritine           DECIMAL(6,2),
    hematocrite         DECIMAL(4,2),
    resultat            VARCHAR(30)  CHECK (resultat IN ('apte','inapte_temporaire','inapte_definitif')),
    observations        TEXT,
    tests_supplementaires JSONB
);
CREATE INDEX idx_examen_resultat ON EXAMEN_MEDICAL(resultat);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE LABORATOIRE (
    id_laboratoire      SERIAL PRIMARY KEY,
    nom                 VARCHAR(150) NOT NULL,
    adresse             TEXT,
    telephone           VARCHAR(15),
    accreditations      JSONB,
    capacite_tests_jour INT,
    equipements         JSONB
);
CREATE INDEX idx_labo_nom ON LABORATOIRE(nom);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE TEST_SANGUIN (
    id_test         SERIAL PRIMARY KEY,
    id_don          INT REFERENCES DON(id_don),
    id_laboratoire  INT REFERENCES LABORATOIRE(id_laboratoire),
    date_test       TIMESTAMP   DEFAULT NOW(),
    type_test       VARCHAR(30) CHECK (type_test IN ('VIH','hepatite_B','hepatite_C','syphilis','HTLV','paludisme')),
    resultat        VARCHAR(20) CHECK (resultat IN ('negatif','positif','douteux','en_attente')),
    valeur          VARCHAR(50),
    technicien      VARCHAR(100),
    commentaire     TEXT,
    conforme        BOOLEAN     DEFAULT TRUE
);
CREATE INDEX idx_test_don      ON TEST_SANGUIN(id_don);
CREATE INDEX idx_test_type     ON TEST_SANGUIN(type_test);
CREATE INDEX idx_test_resultat ON TEST_SANGUIN(resultat);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE RENDEZ_VOUS (
    id_rdv          SERIAL PRIMARY KEY,
    id_donneur      INT REFERENCES DONNEUR(id_donneur),
    id_centre       INT REFERENCES CENTRE_DON(id_centre),
    date_rdv        TIMESTAMP,
    duree_estimee   INT,
    type_don_prevu  VARCHAR(30) CHECK (type_don_prevu IN ('sang_total','plasma','plaquettes','double_globules')),
    statut          VARCHAR(20) DEFAULT 'planifie'
                    CHECK (statut IN ('planifie','confirme','en_cours','termine','annule','no_show')),
    rappel_envoye   BOOLEAN     DEFAULT FALSE,
    date_rappel     TIMESTAMP,
    canal_rappel    VARCHAR(20) CHECK (canal_rappel IN ('email','sms','appel','notification')),
    commentaire     TEXT
);
CREATE INDEX idx_rdv_donneur ON RENDEZ_VOUS(id_donneur);
CREATE INDEX idx_rdv_centre  ON RENDEZ_VOUS(id_centre);
CREATE INDEX idx_rdv_date    ON RENDEZ_VOUS(date_rdv);
CREATE INDEX idx_rdv_statut  ON RENDEZ_VOUS(statut);

-- ============================================
-- MODULE D : STOCK & INVENTAIRE
-- ============================================

CREATE TABLE STOCK (
    id_stock                SERIAL PRIMARY KEY,
    id_centre               INT REFERENCES CENTRE_DON(id_centre),
    groupe_sanguin          VARCHAR(3)  CHECK (groupe_sanguin IN ('A+','A-','B+','B-','AB+','AB-','O+','O-')),
    type_produit            VARCHAR(30) CHECK (type_produit IN ('sang_total','globules_rouges','plasma','plaquettes','cryoprecipite')),
    quantite_unites         INT         DEFAULT 0,
    quantite_ml             INT         DEFAULT 0,
    seuil_critique          INT,
    seuil_alerte            INT,
    date_maj                TIMESTAMP   DEFAULT NOW(),
    consommation_moy_jour   DECIMAL(8,2),
    jours_couverture        DECIMAL(5,2),
    UNIQUE (id_centre, groupe_sanguin, type_produit)
);
CREATE INDEX idx_stock_couverture ON STOCK(jours_couverture);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE UNITE_SANGUINE (
    id_unite                SERIAL PRIMARY KEY,
    numero_unite            VARCHAR(50)  UNIQUE NOT NULL,
    id_don                  INT REFERENCES DON(id_don),
    id_centre_stockage      INT REFERENCES CENTRE_DON(id_centre),
    groupe_sanguin          VARCHAR(3)  CHECK (groupe_sanguin IN ('A+','A-','B+','B-','AB+','AB-','O+','O-')),
    type_produit            VARCHAR(30) CHECK (type_produit IN ('sang_total','globules_rouges','plasma','plaquettes','cryoprecipite')),
    volume_ml               INT,
    date_prelevement        TIMESTAMP,
    date_peremption         TIMESTAMP,
    statut                  VARCHAR(20) DEFAULT 'disponible'
                            CHECK (statut IN ('disponible','reserve','distribue','perime','detruit','quarantaine')),
    temperature_stockage    DECIMAL(5,2),
    code_barre              VARCHAR(100) UNIQUE,
    nfc_tag                 VARCHAR(100),
    localisation_frigo      VARCHAR(50),
    historique_temperature  JSONB
);
CREATE INDEX idx_unite_centre      ON UNITE_SANGUINE(id_centre_stockage);
CREATE INDEX idx_unite_groupe      ON UNITE_SANGUINE(groupe_sanguin, type_produit);
CREATE INDEX idx_unite_statut      ON UNITE_SANGUINE(statut);
CREATE INDEX idx_unite_peremption  ON UNITE_SANGUINE(date_peremption);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE MOUVEMENT_STOCK (
    id_mouvement            SERIAL PRIMARY KEY,
    id_unite                INT REFERENCES UNITE_SANGUINE(id_unite),
    id_centre_source        INT REFERENCES CENTRE_DON(id_centre),
    id_centre_destination   INT REFERENCES CENTRE_DON(id_centre),
    type_mouvement          VARCHAR(20) CHECK (type_mouvement IN ('entree','sortie','transfert','destruction','peremption','distribution')),
    quantite                INT,
    date_mouvement          TIMESTAMP DEFAULT NOW(),
    id_utilisateur          INT REFERENCES UTILISATEUR(id_utilisateur),
    motif                   TEXT,
    numero_bon              VARCHAR(50),
    signature_electronique  TEXT
);
CREATE INDEX idx_mouvement_unite ON MOUVEMENT_STOCK(id_unite);
CREATE INDEX idx_mouvement_type  ON MOUVEMENT_STOCK(type_mouvement);
CREATE INDEX idx_mouvement_date  ON MOUVEMENT_STOCK(date_mouvement);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE EQUIPEMENT_STOCKAGE (
    id_equipement               SERIAL PRIMARY KEY,
    id_centre                   INT REFERENCES CENTRE_DON(id_centre),
    type                        VARCHAR(30) CHECK (type IN ('refrigerateur','congelateur','agitateur_plaquettes','centrifugeuse')),
    marque                      VARCHAR(100),
    modele                      VARCHAR(100),
    numero_serie                VARCHAR(100) UNIQUE,
    capacite_litres             INT,
    temperature_min             DECIMAL(5,2),
    temperature_max             DECIMAL(5,2),
    temperature_actuelle        DECIMAL(5,2),
    statut                      VARCHAR(20) DEFAULT 'operationnel'
                                CHECK (statut IN ('operationnel','alarme','maintenance','hors_service')),
    date_derniere_maintenance   DATE,
    prochaine_maintenance       DATE,
    capteur_iot_id              VARCHAR(100)
);
CREATE INDEX idx_equip_centre ON EQUIPEMENT_STOCKAGE(id_centre);
CREATE INDEX idx_equip_statut ON EQUIPEMENT_STOCKAGE(statut);

-- ============================================
-- MODULE E : LOGISTIQUE & TRANSPORT
-- ============================================

CREATE TABLE TRAJET (
    id_trajet               SERIAL PRIMARY KEY,
    id_vehicule             INT REFERENCES VEHICULE(id_vehicule),
    id_chauffeur            INT REFERENCES UTILISATEUR(id_utilisateur),
    id_centre_depart        INT REFERENCES CENTRE_DON(id_centre),
    id_centre_arrivee       INT REFERENCES CENTRE_DON(id_centre),
    date_depart_planifie    TIMESTAMP,
    date_arrivee_planifie   TIMESTAMP,
    date_depart_reel        TIMESTAMP,
    date_arrivee_reel       TIMESTAMP,
    statut                  VARCHAR(20) CHECK (statut IN ('planifie','en_cours','termine','annule','retarde')),
    distance_km             DECIMAL(8,2),
    duree_estimee_min       INT,
    duree_reelle_min        INT,
    itineraire_geojson      JSONB,
    unites_transportees     INT,
    temperature_moyenne     DECIMAL(5,2),
    incidents               JSONB
);
CREATE INDEX idx_trajet_vehicule ON TRAJET(id_vehicule);
CREATE INDEX idx_trajet_statut   ON TRAJET(statut);
CREATE INDEX idx_trajet_date     ON TRAJET(date_depart_planifie);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE POSITION_VEHICULE (
    id_position                 SERIAL PRIMARY KEY,
    id_vehicule                 INT REFERENCES VEHICULE(id_vehicule),
    latitude                    DECIMAL(10,8),
    longitude                   DECIMAL(11,8),
    altitude                    DECIMAL(8,2),
    vitesse_kmh                 DECIMAL(5,2),
    cap_degres                  INT,
    precision_m                 DECIMAL(6,2),
    timestamp                   TIMESTAMP DEFAULT NOW(),
    temperature_compartiment    DECIMAL(5,2)
);
CREATE INDEX idx_position_vehicule   ON POSITION_VEHICULE(id_vehicule);
CREATE INDEX idx_position_timestamp  ON POSITION_VEHICULE(timestamp);

-- ============================================
-- MODULE F : ALERTES & NOTIFICATIONS
-- ============================================

CREATE TABLE ALERTE (
    id_alerte                       SERIAL PRIMARY KEY,
    type                            VARCHAR(50) CHECK (type IN (
                                        'stock_critique','peremption_proche','rupture_chaine_froid',
                                        'collecte_retard','prediction_rupture','anomalie_test','equipement_defaillant')),
    niveau                          VARCHAR(20) CHECK (niveau IN ('info','warning','critique','urgence')),
    titre                           VARCHAR(200),
    message                         TEXT,
    id_centre                       INT REFERENCES CENTRE_DON(id_centre),
    id_unite                        INT REFERENCES UNITE_SANGUINE(id_unite),
    id_equipement                   INT REFERENCES EQUIPEMENT_STOCKAGE(id_equipement),
    donnees_contexte                JSONB,
    date_declenchement              TIMESTAMP DEFAULT NOW(),
    date_acquittement               TIMESTAMP,
    id_utilisateur_acquittement     INT REFERENCES UTILISATEUR(id_utilisateur),
    statut                          VARCHAR(20) DEFAULT 'active'
                                    CHECK (statut IN ('active','acquittee','resolue','ignoree')),
    actions_entreprises             TEXT,
    seuil_declenche                 DECIMAL(10,2),
    valeur_actuelle                 DECIMAL(10,2)
);
CREATE INDEX idx_alerte_type   ON ALERTE(type);
CREATE INDEX idx_alerte_niveau ON ALERTE(niveau);
CREATE INDEX idx_alerte_statut ON ALERTE(statut);
CREATE INDEX idx_alerte_date   ON ALERTE(date_declenchement);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE NOTIFICATION (
    id_notification SERIAL PRIMARY KEY,
    id_utilisateur  INT REFERENCES UTILISATEUR(id_utilisateur),
    id_alerte       INT REFERENCES ALERTE(id_alerte),
    type            VARCHAR(30) CHECK (type IN ('alerte','rappel_rdv','info','message','systeme')),
    titre           VARCHAR(200),
    contenu         TEXT,
    canal           VARCHAR(20) CHECK (canal IN ('push','email','sms','in_app')),
    priorite        VARCHAR(20) CHECK (priorite IN ('basse','normale','haute','urgente')),
    date_envoi      TIMESTAMP   DEFAULT NOW(),
    date_lecture    TIMESTAMP,
    statut          VARCHAR(20) DEFAULT 'en_attente'
                    CHECK (statut IN ('en_attente','envoyee','lue','echec')),
    erreur          TEXT
);
CREATE INDEX idx_notif_user   ON NOTIFICATION(id_utilisateur);
CREATE INDEX idx_notif_statut ON NOTIFICATION(statut);
CREATE INDEX idx_notif_date   ON NOTIFICATION(date_envoi);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE REGLE_ALERTE (
    id_regle                SERIAL PRIMARY KEY,
    nom_regle               VARCHAR(100) UNIQUE,
    description             TEXT,
    type_metrique           VARCHAR(50),
    condition               VARCHAR(50),
    seuil                   DECIMAL(10,2),
    niveau_alerte           VARCHAR(20) CHECK (niveau_alerte IN ('info','warning','critique','urgence')),
    actif                   BOOLEAN     DEFAULT TRUE,
    destinataires_roles     JSONB,
    canaux_notification     JSONB,
    delai_escalade_heures   INT,
    frequence_max_heures    INT
);
CREATE INDEX idx_regle_actif ON REGLE_ALERTE(actif);

-- ============================================
-- MODULE G : DASHBOARD & KPI
-- ============================================

CREATE TABLE KPI (
    id_kpi              SERIAL PRIMARY KEY,
    code_kpi            VARCHAR(50) UNIQUE NOT NULL,
    nom                 VARCHAR(100),
    description         TEXT,
    formule_calcul      TEXT,
    unite               VARCHAR(20),
    frequence_calcul    VARCHAR(20) CHECK (frequence_calcul IN ('temps_reel','horaire','quotidien','hebdomadaire','mensuel')),
    seuil_vert          DECIMAL(10,2),
    seuil_orange        DECIMAL(10,2),
    seuil_rouge         DECIMAL(10,2)
);

CREATE TABLE VALEUR_KPI (
    id_valeur       SERIAL PRIMARY KEY,
    id_kpi          INT REFERENCES KPI(id_kpi),
    id_centre       INT REFERENCES CENTRE_DON(id_centre),
    date_calcul     TIMESTAMP   DEFAULT NOW(),
    valeur          DECIMAL(15,4),
    statut_seuil    VARCHAR(10) CHECK (statut_seuil IN ('vert','orange','rouge')),
    details         JSONB
);
CREATE INDEX idx_valeur_kpi    ON VALEUR_KPI(id_kpi);
CREATE INDEX idx_valeur_centre ON VALEUR_KPI(id_centre);
CREATE INDEX idx_valeur_date   ON VALEUR_KPI(date_calcul);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE WIDGET_DASHBOARD (
    id_widget           SERIAL PRIMARY KEY,
    id_utilisateur      INT REFERENCES UTILISATEUR(id_utilisateur),
    type_widget         VARCHAR(30) CHECK (type_widget IN (
                            'graphique_courbe','carte_thermique','jauge',
                            'tableau','carte_geo','compteur','gantt')),
    titre               VARCHAR(150),
    configuration       JSONB,
    requete_donnees     TEXT,
    position_x          INT,
    position_y          INT,
    largeur             INT,
    hauteur             INT,
    ordre               INT,
    actif               BOOLEAN DEFAULT TRUE,
    partage             BOOLEAN DEFAULT FALSE,
    rafraichissement_sec INT    DEFAULT 60
);
CREATE INDEX idx_widget_user  ON WIDGET_DASHBOARD(id_utilisateur);
CREATE INDEX idx_widget_actif ON WIDGET_DASHBOARD(actif);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE DASHBOARD_PERSONNALISE (
    id_dashboard    SERIAL PRIMARY KEY,
    id_utilisateur  INT REFERENCES UTILISATEUR(id_utilisateur),
    nom             VARCHAR(100),
    description     TEXT,
    widgets         JSONB,
    layout          JSONB,
    filtres_defaut  JSONB,
    par_defaut      BOOLEAN   DEFAULT FALSE,
    partage         BOOLEAN   DEFAULT FALSE,
    date_creation   TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_dashboard_user ON DASHBOARD_PERSONNALISE(id_utilisateur);

-- ============================================
-- MODULE I : APPLICATION MOBILE
-- ============================================

CREATE TABLE MISSION_TERRAIN (
    id_mission              SERIAL PRIMARY KEY,
    id_utilisateur          INT REFERENCES UTILISATEUR(id_utilisateur),
    id_collecte             INT REFERENCES COLLECTE_MOBILE(id_collecte),
    id_vehicule             INT REFERENCES VEHICULE(id_vehicule),
    titre                   VARCHAR(150),
    description             TEXT,
    type                    VARCHAR(30) CHECK (type IN ('collecte','livraison','inventaire','maintenance','urgence')),
    date_debut_planifie     TIMESTAMP,
    date_fin_planifie       TIMESTAMP,
    date_debut_reel         TIMESTAMP,
    date_fin_reel           TIMESTAMP,
    statut                  VARCHAR(20) DEFAULT 'assignee'
                            CHECK (statut IN ('assignee','acceptee','en_cours','terminee','annulee')),
    priorite                VARCHAR(20) CHECK (priorite IN ('basse','normale','haute','urgente')),
    itineraire_optimise     JSONB,
    checklist               JSONB,
    photos                  JSONB,
    signature_electronique  TEXT,
    commentaires            TEXT
);
CREATE INDEX idx_mission_user     ON MISSION_TERRAIN(id_utilisateur);
CREATE INDEX idx_mission_statut   ON MISSION_TERRAIN(statut);
CREATE INDEX idx_mission_priorite ON MISSION_TERRAIN(priorite);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE SCAN_POCHE (
    id_scan                 SERIAL PRIMARY KEY,
    id_mission              INT REFERENCES MISSION_TERRAIN(id_mission),
    id_unite                INT REFERENCES UNITE_SANGUINE(id_unite),
    id_utilisateur          INT REFERENCES UTILISATEUR(id_utilisateur),
    type_scan               VARCHAR(20) CHECK (type_scan IN ('qr_code','code_barre','nfc','manuel')),
    date_scan               TIMESTAMP   DEFAULT NOW(),
    latitude                DECIMAL(10,8),
    longitude               DECIMAL(11,8),
    temperature_relevee     DECIMAL(5,2),
    photo_preuve            VARCHAR(255),
    anomalie_detectee       BOOLEAN     DEFAULT FALSE,
    description_anomalie    TEXT
);
CREATE INDEX idx_scan_mission ON SCAN_POCHE(id_mission);
CREATE INDEX idx_scan_unite   ON SCAN_POCHE(id_unite);
CREATE INDEX idx_scan_date    ON SCAN_POCHE(date_scan);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE SYNCHRONISATION_MOBILE (
    id_sync             SERIAL PRIMARY KEY,
    id_utilisateur      INT REFERENCES UTILISATEUR(id_utilisateur),
    device_id           VARCHAR(100),
    date_sync           TIMESTAMP   DEFAULT NOW(),
    type_sync           VARCHAR(20) CHECK (type_sync IN ('pull','push','bidirectionnel')),
    donnees_envoyees    JSONB,
    donnees_recues      JSONB,
    taille_donnees_ko   INT,
    duree_sync_ms       INT,
    statut              VARCHAR(20) CHECK (statut IN ('reussi','partiel','echec')),
    conflits_detectes   JSONB,
    conflits_resolus    JSONB,
    erreurs             TEXT
);
CREATE INDEX idx_sync_user   ON SYNCHRONISATION_MOBILE(id_utilisateur);
CREATE INDEX idx_sync_device ON SYNCHRONISATION_MOBILE(device_id);
CREATE INDEX idx_sync_date   ON SYNCHRONISATION_MOBILE(date_sync);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE GEOFENCE (
    id_geofence     SERIAL PRIMARY KEY,
    id_centre       INT REFERENCES CENTRE_DON(id_centre),
    nom             VARCHAR(100),
    geometrie       GEOMETRY,
    rayon_m         INT,
    actif           BOOLEAN DEFAULT TRUE,
    declencheurs    JSONB,
    actions         JSONB
);
CREATE INDEX idx_geofence_centre ON GEOFENCE(id_centre);
CREATE INDEX idx_geofence_actif  ON GEOFENCE(actif);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE EVENEMENT_GEOFENCE (
    id_evenement            SERIAL PRIMARY KEY,
    id_geofence             INT REFERENCES GEOFENCE(id_geofence),
    id_utilisateur          INT REFERENCES UTILISATEUR(id_utilisateur),
    id_vehicule             INT REFERENCES VEHICULE(id_vehicule),
    type                    VARCHAR(20) CHECK (type IN ('entree','sortie','sejour')),
    date_evenement          TIMESTAMP   DEFAULT NOW(),
    latitude                DECIMAL(10,8),
    longitude               DECIMAL(11,8),
    notification_envoyee    BOOLEAN     DEFAULT FALSE
);
CREATE INDEX idx_event_geofence ON EVENEMENT_GEOFENCE(id_geofence);
CREATE INDEX idx_event_date     ON EVENEMENT_GEOFENCE(date_evenement);

-- ============================================
-- MODULE J : COMMUNICATION
-- ============================================

CREATE TABLE CONVERSATION (
    id_conversation     SERIAL PRIMARY KEY,
    titre               VARCHAR(150),
    type                VARCHAR(20) CHECK (type IN ('individuelle','groupe','equipe','mission')),
    id_mission          INT REFERENCES MISSION_TERRAIN(id_mission),
    participants        JSONB,
    date_creation       TIMESTAMP DEFAULT NOW(),
    derniere_activite   TIMESTAMP,
    archivee            BOOLEAN   DEFAULT FALSE
);
CREATE INDEX idx_conv_type     ON CONVERSATION(type);
CREATE INDEX idx_conv_activite ON CONVERSATION(derniere_activite);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE MESSAGE (
    id_message          SERIAL PRIMARY KEY,
    id_expediteur       INT REFERENCES UTILISATEUR(id_utilisateur),
    id_conversation     INT REFERENCES CONVERSATION(id_conversation),
    contenu             TEXT,
    fichiers_attaches   JSONB,
    date_envoi          TIMESTAMP   DEFAULT NOW(),
    priorite            VARCHAR(20) DEFAULT 'normale'
                        CHECK (priorite IN ('normale','haute','urgente')),
    lu                  BOOLEAN     DEFAULT FALSE,
    date_lecture        TIMESTAMP
);
CREATE INDEX idx_message_conv ON MESSAGE(id_conversation);
CREATE INDEX idx_message_date ON MESSAGE(date_envoi);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE CAMPAGNE_COMMUNICATION (
    id_campagne             SERIAL PRIMARY KEY,
    nom_campagne            VARCHAR(150),
    description             TEXT,
    type                    VARCHAR(30) CHECK (type IN ('recrutement','rappel_don','urgence_stock','evenement','information')),
    canal                   VARCHAR(20) CHECK (canal IN ('email','sms','push','multi_canal')),
    date_debut              TIMESTAMP,
    date_fin                TIMESTAMP,
    cible                   JSONB,
    contenu                 TEXT,
    template                TEXT,
    statut                  VARCHAR(20) CHECK (statut IN ('brouillon','planifiee','en_cours','terminee','annulee')),
    nombre_destinataires    INT,
    nombre_envoyes          INT         DEFAULT 0,
    nombre_ouverts          INT         DEFAULT 0,
    nombre_clics            INT         DEFAULT 0,
    taux_conversion         DECIMAL(5,2)
);
CREATE INDEX idx_campagne_type   ON CAMPAGNE_COMMUNICATION(type);
CREATE INDEX idx_campagne_statut ON CAMPAGNE_COMMUNICATION(statut);
CREATE INDEX idx_campagne_date   ON CAMPAGNE_COMMUNICATION(date_debut);

-- ============================================
-- MODULE K : REPORTING
-- ============================================

CREATE TABLE RAPPORT (
    id_rapport              SERIAL PRIMARY KEY,
    titre                   VARCHAR(200),
    type                    VARCHAR(30) CHECK (type IN ('journalier','hebdomadaire','mensuel','annuel','personnalise')),
    categorie               VARCHAR(30) CHECK (categorie IN ('dons','stock','logistique','qualite','financier','predictions')),
    id_utilisateur_createur INT REFERENCES UTILISATEUR(id_utilisateur),
    date_generation         TIMESTAMP   DEFAULT NOW(),
    periode_debut           DATE,
    periode_fin             DATE,
    filtres                 JSONB,
    donnees                 JSONB,
    format                  VARCHAR(10) CHECK (format IN ('PDF','CSV','Excel','JSON')),
    path_fichier            VARCHAR(255),
    taille_ko               INT,
    partage                 BOOLEAN     DEFAULT FALSE
);
CREATE INDEX idx_rapport_type     ON RAPPORT(type);
CREATE INDEX idx_rapport_categorie ON RAPPORT(categorie);
CREATE INDEX idx_rapport_date     ON RAPPORT(date_generation);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE EXPORT_DONNEES (
    id_export       SERIAL PRIMARY KEY,
    id_utilisateur  INT REFERENCES UTILISATEUR(id_utilisateur),
    type_donnees    VARCHAR(100),
    filtres         JSONB,
    format          VARCHAR(10) CHECK (format IN ('CSV','Excel','JSON','XML','PDF')),
    date_demande    TIMESTAMP   DEFAULT NOW(),
    date_generation TIMESTAMP,
    statut          VARCHAR(20) DEFAULT 'en_attente'
                    CHECK (statut IN ('en_attente','en_cours','termine','echec')),
    path_fichier    VARCHAR(255),
    taille_ko       INT,
    expiration      TIMESTAMP
);
CREATE INDEX idx_export_user   ON EXPORT_DONNEES(id_utilisateur);
CREATE INDEX idx_export_statut ON EXPORT_DONNEES(statut);

-- ============================================
-- MODULE L : CONFIGURATION
-- ============================================

CREATE TABLE PARAMETRE_SYSTEME (
    id_parametre                SERIAL PRIMARY KEY,
    cle                         VARCHAR(100) UNIQUE NOT NULL,
    valeur                      TEXT,
    type                        VARCHAR(20) CHECK (type IN ('string','number','boolean','json')),
    categorie                   VARCHAR(50),
    description                 TEXT,
    modifiable                  BOOLEAN   DEFAULT TRUE,
    date_modification           TIMESTAMP,
    id_utilisateur_modification INT REFERENCES UTILISATEUR(id_utilisateur)
);
CREATE INDEX idx_param_cle       ON PARAMETRE_SYSTEME(cle);
CREATE INDEX idx_param_categorie ON PARAMETRE_SYSTEME(categorie);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE HISTORIQUE_MODIFICATION (
    id_historique       SERIAL PRIMARY KEY,
    table_cible         VARCHAR(50),
    id_enregistrement   INT,
    action              VARCHAR(10) CHECK (action IN ('INSERT','UPDATE','DELETE')),
    anciennes_valeurs   JSONB,
    nouvelles_valeurs   JSONB,
    id_utilisateur      INT REFERENCES UTILISATEUR(id_utilisateur),
    date_modification   TIMESTAMP DEFAULT NOW(),
    ip_adresse          VARCHAR(45)
);
CREATE INDEX idx_historique_cible ON HISTORIQUE_MODIFICATION(table_cible, id_enregistrement);
CREATE INDEX idx_historique_date  ON HISTORIQUE_MODIFICATION(date_modification);

-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE SALLE_PRELEVEMENT (
    id_salle    SERIAL PRIMARY KEY,
    id_centre   INT REFERENCES CENTRE_DON(id_centre),
    nom_salle   VARCHAR(50),
    capacite_lits INT,
    statut      VARCHAR(20) DEFAULT 'disponible'
                CHECK (statut IN ('disponible','occupee','maintenance','desinfection')),
    equipements JSONB,
    temperature DECIMAL(4,2),
    humidite    DECIMAL(5,2)
);
CREATE INDEX idx_salle_centre ON SALLE_PRELEVEMENT(id_centre);
CREATE INDEX idx_salle_statut ON SALLE_PRELEVEMENT(statut);

-- ============================================
-- DATA VERSION — suivi des mises à jour
-- ============================================

CREATE TABLE DATA_VERSION (
    id              SERIAL PRIMARY KEY,
    key             VARCHAR(50)  UNIQUE NOT NULL,  -- ex: 'quebec_donneurs'
    last_updated    TIMESTAMP    NOT NULL DEFAULT NOW(),
    version_hash    VARCHAR(64),                   -- SHA256 des données (optionnel)
    description     TEXT,                          -- note sur la mise à jour
    updated_by      VARCHAR(100)                   -- 'cron', 'admin', 'api', etc.
);

CREATE INDEX idx_dataversion_key ON DATA_VERSION(key);

-- Valeur initiale
INSERT INTO DATA_VERSION (key, last_updated, description, updated_by)
VALUES ('quebec_donneurs', NOW(), 'Initialisation', 'system');
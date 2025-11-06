# 🚘 Système de Lecture Automatique de Plaques d’Immatriculation

### _(Automatic License Plate Recognition System with CNN Fine-Tuning)_

---

## 🧭 Overview / Aperçu

This project aims to develop an **Automatic License Plate Recognition (ALPR)** system using **Deep Learning** and **Computer Vision** techniques.  
It combines:

- **YOLOv8** for plate **detection**
- **CNN / CRNN fine-tuned models** for **character recognition (OCR)**
- and a **FastAPI** backend for **real-time inference**

Le système détecte la plaque, la recadre, lit les caractères, applique des corrections et renvoie le texte final avec le score de confiance.

---

## 🧠 Objectifs du Projet

1. Détecter automatiquement la plaque d’immatriculation sur un véhicule.
2. Extraire et normaliser la zone détectée (recadrage et prétraitement).
3. Lire le texte de la plaque à l’aide d’un modèle CNN/CRNN fine-tuné.
4. Corriger les confusions (O↔0, I↔1, B↔8) avec des règles métiers.
5. Fournir une API REST pour permettre une utilisation temps réel.

---

## 🏗️ Architecture Générale

Image → Détection (YOLOv8)
→ Recadrage + Prétraitement (OpenCV)
→ Lecture OCR (CNN/CRNN fine-tuné)
→ Post-Traitement & Validation
→ Résultat : Texte + Confiance

---

## ⚙️ Technologies Utilisées

| Catégorie     | Librairies                       | Utilisation                |
| ------------- | -------------------------------- | -------------------------- |
| Deep Learning | PyTorch, TorchVision             | Entraînement CNN/CRNN      |
| Détection     | Ultralytics YOLOv8               | Détection de plaques       |
| Vision        | OpenCV, Albumentations           | Prétraitement d’images     |
| OCR           | EasyOCR, CRNN                    | Lecture de caractères      |
| API           | FastAPI, Uvicorn                 | Déploiement du service     |
| Analyse       | Pandas, Matplotlib, Scikit-learn | Évaluation & visualisation |
| Déploiement   | ONNX, ONNXRuntime                | Inférence multiplateforme  |

---

## 📊 Métriques d’Évaluation

- **mAP@0.5** : précision de détection (YOLO)
- **CER (Character Error Rate)** : taux d’erreur OCR
- **Accuracy** : plaques correctement lues
- **FPS** : performance temps réel

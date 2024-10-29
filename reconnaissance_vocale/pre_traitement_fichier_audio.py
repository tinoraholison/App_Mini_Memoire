import librosa
import numpy as np
from pathlib import Path
import os

# Dossier contenant les fichiers audio segmentés
input_folder = Path('reconnaissance_vocale/fichier_audio/fichier_audio_par_mot')

# Dossier pour enregistrer les caractéristiques extraites
output_features_folder = Path('reconnaissance_vocale/audio_features')
output_features_folder.mkdir(parents=True, exist_ok=True)

# Paramètres de pré-traitement
sample_rate = 22050  # Fréquence d'échantillonnage (ex: 22050 Hz)
n_mfcc = 13          # Nombre de coefficients MFCC pour la représentation

# Parcourir tous les fichiers audio dans le dossier 'fichier_audio_par_mot'
for file_path in input_folder.iterdir():
    if file_path.suffix == '.wav':  # Traiter uniquement les fichiers wav
        # Charger le fichier audio et rééchantillonner
        audio, sr = librosa.load(file_path, sr=sample_rate)

        # Normaliser le volume
        audio = librosa.util.normalize(audio)

        # Extraire les caractéristiques : MFCC (Mel Frequency Cepstral Coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # Stocker les caractéristiques extraites en fichier .npy
        feature_file = output_features_folder / f'{file_path.stem}_mfcc.npy'
        np.save(feature_file, mfccs)

        print(f"Les caractéristiques MFCC de '{file_path.name}' ont été enregistrées dans '{feature_file}'")

print("Pré-traitement terminé. Toutes les caractéristiques sont sauvegardées.")

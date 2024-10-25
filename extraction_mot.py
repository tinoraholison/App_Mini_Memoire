import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

# Dossier contenant les fichiers audio à segmenter
input_folder = Path('fichier_audio')

# Créer le sous-dossier 'fichier_audio_par_mot' pour stocker les segments
output_folder = input_folder / 'fichier_audio_par_mot'
output_folder.mkdir(parents=True, exist_ok=True)

# Parcourir tous les fichiers audio dans le dossier 'fichier_audio'
for file_path in input_folder.iterdir():
    if file_path.suffix in ['.mp3', '.wav']:  # traiter uniquement les fichiers audio
        # Charger le fichier audio
        audio, sr = librosa.load(file_path)
        
        # Détecter les segments basés sur les silences
        intervals = librosa.effects.split(audio, top_db=30)
        
        # Extraire chaque segment et l'enregistrer dans le sous-dossier
        for i, interval in enumerate(intervals):
            start, end = interval
            word_audio = audio[start:end]
            
            # Créer un nom de fichier basé sur le fichier original et l'index du segment
            base_filename = file_path.stem  # Nom sans extension
            output_file = output_folder / f'{base_filename}_mot_{i}.wav'
            
            # Sauvegarder le segment dans le sous-dossier
            sf.write(output_file, word_audio, sr)

print(f"Les fichiers segmentés ont été enregistrés dans le dossier '{output_folder}'")

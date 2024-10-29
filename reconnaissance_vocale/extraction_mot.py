import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

# Dossier contenant les fichiers audio à segmenter
input_folder = Path('fichier_audio')

# Créer le sous-dossier 'fichier_audio_par_mot' pour stocker les segments
output_folder = input_folder / 'fichier_audio_par_mot'
output_folder.mkdir(parents=True, exist_ok=True)

# Durée maximale du silence à tolérer entre les segments pour fusionner (en secondes)
silence_threshold = 0.5

# Parcourir tous les fichiers audio dans le dossier 'fichier_audio'
for file_path in input_folder.iterdir():
    if file_path.suffix in ['.mp3', '.wav']:  # traiter uniquement les fichiers audio
        # Charger le fichier audio
        audio, sr = librosa.load(file_path)
        
        # Détecter les segments basés sur les silences
        intervals = librosa.effects.split(audio, top_db=30)
        
        # Fusionner les segments trop proches les uns des autres
        merged_intervals = []
        current_start, current_end = intervals[0]
        
        for start, end in intervals[1:]:
            # Si le silence entre les segments est inférieur au seuil, on les fusionne
            if (start - current_end) / sr < silence_threshold:
                current_end = end  # Prolonge le segment en cours
            else:
                merged_intervals.append((current_start, current_end))
                current_start, current_end = start, end  # Commence un nouveau segment

        # Ajouter le dernier segment en cours
        merged_intervals.append((current_start, current_end))

        # Sauvegarder chaque segment fusionné dans le sous-dossier
        for i, (start, end) in enumerate(merged_intervals):
            word_audio = audio[start:end]
            
            # Créer un nom de fichier basé sur le fichier original et l'index du segment
            base_filename = file_path.stem  # Nom sans extension
            output_file = output_folder / f'{base_filename}_mot_{i}.wav'
            
            # Sauvegarder le segment dans le sous-dossier
            sf.write(output_file, word_audio, sr)

print(f"Les fichiers segmentés et fusionnés ont été enregistrés dans le dossier '{output_folder}'")

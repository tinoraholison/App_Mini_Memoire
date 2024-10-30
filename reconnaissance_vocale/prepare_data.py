import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Dossier des caractéristiques MFCC
features_folder = Path('reconnaissance_vocale/audio_features')

def prepare_data(features_folder, maxlen=38):
    # Chargement des caractéristiques et des étiquettes
    X = []
    y = []
    label_map = {}
    label_index = 0

    for file_path in features_folder.glob('*.npy'):
        mfcc_features = np.load(file_path)
        
        # Vérification et ajustement de la forme
        if mfcc_features.shape[0] != 13:
            print(f"Avertissement : {file_path} a une forme incompatible {mfcc_features.shape}, fichier ignoré.")
            continue
        
        # Tronquer ou remplir pour atteindre la longueur fixe (maxlen)
        if mfcc_features.shape[1] < maxlen:
            # Remplir à droite pour atteindre maxlen
            padded_features = np.pad(mfcc_features, ((0, 0), (0, maxlen - mfcc_features.shape[1])), mode='constant')
        else:
            # Tronquer à maxlen
            padded_features = mfcc_features[:, :maxlen]
        
        X.append(padded_features)
        
        # Extraire le nom de la coutume (ignorer le nom de la personne)
        parts = file_path.stem.split('_')
        label = parts[2]
        if label not in label_map:
            label_map[label] = label_index
            label_index += 1
        y.append(label_map[label])

    # Convertir X et y en tableaux numpy
    X = np.array(X)[..., np.newaxis]  # Ajouter une dimension pour les canaux
    y = np.array(y)
    
    # Diviser les données en ensembles d'entraînement et de validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convertir les étiquettes en format one-hot pour la classification
    y_train = to_categorical(y_train, num_classes=len(label_map))
    y_val = to_categorical(y_val, num_classes=len(label_map))
    
    y_train = np.array(y_train, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)

    # À la fin de la fonction prepare_data, juste avant le `return`:
    print("Type de X :", type(X), ", forme :", X.shape)
    print("Type de y :", type(y), ", forme :", y.shape)

    # Après avoir fait le train-test split et le one-hot encoding:
    print("Type de X_train :", type(X_train), ", forme :", X_train.shape)
    print("Type de y_train :", type(y_train), ", forme :", y_train.shape)
    print("Type de X_val :", type(X_val), ", forme :", X_val.shape)
    print("Type de y_val :", type(y_val), ", forme :", y_val.shape)

    
    return X_train, X_val, y_train, y_val, label_map

# Appel de la fonction
#X_train, X_val, y_train, y_val, label_map = prepare_data(features_folder)

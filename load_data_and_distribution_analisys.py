import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import umap.umap_ as umap

def load_data():
    # Download latest version
    path = kagglehub.dataset_download("sooyoungher/smoking-drinking-dataset")
    print("Path to dataset files:", path)

    file_name = "smoking_driking_dataset_Ver01.csv"
    file_path = f"{path}/{file_name}"
    dataset = pd.read_csv(file_path)

    return dataset

def print_summary(dataset):
    # Analisi iniziale del dataset
    print("\nPrime righe del dataset:")
    print(dataset.head())

    print("\nDimensioni del dataset:")
    print(dataset.shape)

    print("\nTipi di dati:")
    print(dataset.info())

    print("\nStatistiche descrittive di base:")
    print(dataset.describe())

def plot_classes_distribution(dataset):
    smoke = dataset["SMK_stat_type_cd"].value_counts()
    drink = dataset["DRK_YN"].value_counts()

    # Distribuzione SMK_stat_type_cd
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1) # Num righe, num colonne, posizione 
    sns.barplot(x=smoke.index, y=smoke.values, palette="Blues_d")
    plt.title("Distribuzione delle classi relative al fumo")
    plt.xlabel("Tipo di fumatore")
    plt.ylabel("Frequenza")

    # Distribuzione DRK_YN
    plt.subplot(1, 2, 2)
    sns.barplot(x=drink.index, y=drink.values, palette="Greens_d")
    plt.title("Distribuzione delle classi relative al bere")
    plt.xlabel("Bevitore (Y/N)")
    plt.ylabel("Frequenza")

    plt.tight_layout()
    plt.show()

def boxplot(dataset):
    dataset_mod = dataset.drop(columns=["sex", "SMK_stat_type_cd", "DRK_YN"])

    # Boxplot per le principali feature numeriche
    plt.figure(figsize=(25, 20))
    for i, col in enumerate(dataset_mod.columns):
        plt.subplot(5, 5, i + 1)
        sns.boxplot(y=dataset_mod[col], color="skyblue")
        plt.title(col)
        plt.tight_layout()
    plt.show()

def print_max_values(dataset):
    # Verifica dei valori massimi e distanza dal secondo massimo e dalla media
    dataset_mod = dataset.drop(columns=["sex", "SMK_stat_type_cd", "DRK_YN"])

    for col in dataset_mod.columns:
        max = dataset_mod[col].max()
        max_count = (dataset_mod[col] == max).sum()
        second_max = dataset_mod[col][dataset_mod[col] < max].max()
        distance_max = max - second_max
        mean_value = dataset[col].mean()
        distance_mean = max - mean_value

        print(f"Colonna: {col}")
        print(f"    -Valore massimo: {max}")
        print(f"    -Occorrenze del massimo: {max_count}")
        print(f"    -Secondo massimo: {second_max}")
        print(f"    -Distanza tra massimo e secondo massimo: {distance_max}")
        print(f"    -Media: {mean_value}")
        print(f"    -Distanza tra massimo e media: {distance_mean}")
        print()

def print_outliers(dataset):
    # Determinazione del numero di valori "fuori scala" (outliers) per ogni feature
    dataset_mod = dataset.drop(columns=["sex", "SMK_stat_type_cd", "DRK_YN"])

    for col in dataset_mod.columns:
        Q1 = dataset_mod[col].quantile(0.25)
        Q3 = dataset_mod[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = dataset_mod[(dataset_mod[col] < lower_bound) | (dataset_mod[col] > upper_bound)]
        print(f"Colonna: {col}")
        print(f"    -Valori fuori scala: {len(outliers)}")
        print(f"    -Limiti: {lower_bound} - {upper_bound}")
        print()

def categorical_encoding(dataset):
    categorical_cols = ["sex", "DRK_YN"]

    for col in categorical_cols:
        encoder = LabelEncoder()
        dataset[col] = encoder.fit_transform(dataset[col])

    print(dataset.head())

    return dataset

def split_dataset_v1(dataset):
    # Divisione del dataset in feature e target (X e Y)
    target_smoke = "SMK_stat_type_cd"
    target_drink = "DRK_YN"
    X_smoke = dataset.drop(columns=["SMK_stat_type_cd", "DRK_YN"])
    Y_smoke = dataset[target_smoke]
    X_drink = dataset.drop(columns=["SMK_stat_type_cd", "DRK_YN"])
    Y_drink = dataset[target_drink]

    return X_smoke, Y_smoke, X_drink, Y_drink

def split_dataset_v2(dataset):
    # Divisione del dataset in feature e target (X e Y)
    target_smoke = "SMK_stat_type_cd"
    target_drink = "DRK_YN"
    X_smoke = dataset.iloc[:, dataset.columns != target_smoke]
    Y_smoke = dataset[target_smoke]
    X_drink = dataset.iloc[:, dataset.columns != target_drink]
    Y_drink = dataset[target_drink]

    return X_smoke, Y_smoke, X_drink, Y_drink

def plot_confusion_matrix(y_true, y_pred, display_labels, title, cmap='viridis'):
    # Compute normalized confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    # Create a figure and an axis
    _, ax = plt.subplots(figsize=(6, 6))
    
    # Create and plot the confusion matrix display
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    cm_display.plot(ax=ax, cmap=cmap, colorbar=False)
    
    # Set plot title and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    # Rotate y-tick labels for better readability
    plt.setp(ax.get_yticklabels(), rotation=45)
    
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

def normalize_data(X_smoke, X_drink):
    scaler = StandardScaler()
    X_smoke_scaled = scaler.fit_transform(X_smoke)
    X_drink_scaled = scaler.fit_transform(X_drink)
    X_smoke_scaled = pd.DataFrame(X_smoke_scaled, columns=X_smoke.columns)
    X_drink_scaled = pd.DataFrame(X_drink_scaled, columns=X_drink.columns)

    return X_smoke_scaled, X_drink_scaled

def remove_outliers(dataset):
    
    thresholds = {
    "waistline": 200,
    "sight_left": 4,
    "sight_right": 4,
    "SBP": 240,
    "DBP": 160,
    "BLDS": 600,
    "tot_chole": 1000,
    "HDL_chole": 700,
    "LDL_chole": 2000,
    "triglyceride": 3500,
    "serum_creatinine": 30,
    "SGOT_AST": 2000,
    "SGOT_ALT": 2000,
    "gamma_GTP": 900,
    }

    for col, threshold in thresholds.items():
        dataset.loc[dataset[col] > threshold, col] = None

    # Rimozione delle righe con valori mancanti
    dataset_cleaned = dataset.dropna(subset=thresholds.keys())

    return dataset_cleaned

def load_dataset_cleaned():
    dataset = load_data()
    categorical_encoding(dataset)
    dataset_cleaned = remove_outliers(dataset)

    return dataset_cleaned

def sample_visualization(X_drink, X_smoke, y_drink, y_smoke):
    X_smoke, X_drink = normalize_data(X_smoke, X_drink)

    X_smoke, y_smoke = X_smoke.sample(100000, random_state=42), y_smoke.loc[X_smoke.sample(100000, random_state=42).index]
    print(X_smoke.shape, y_smoke.shape)
    X_drink, y_drink = X_drink.sample(100000, random_state=42), y_drink.loc[X_drink.sample(100000, random_state=42).index]

    reducer = umap.UMAP(n_components=2, random_state=42)
    X_smoke = reducer.fit_transform(X_smoke)
    X_drink = reducer.fit_transform(X_drink)
    df_smoke = pd.DataFrame(X_smoke, columns=["Componente 1", "Componente 2"])
    df_smoke["Target"] = y_smoke.values
    df_drink = pd.DataFrame(X_drink, columns=["Componente 1", "Componente 2"])
    df_drink["Target"] = y_drink.values

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_smoke, x="Componente 1", y="Componente 2", hue="Target", palette="viridis", alpha=0.7)
    plt.title("Visualizzazione UMAP del dataset Smoke")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_drink, x="Componente 1", y="Componente 2", hue="Target", palette="viridis", alpha=0.7)
    plt.title("Visualizzazione UMAP del dataset Drink")
    plt.show()
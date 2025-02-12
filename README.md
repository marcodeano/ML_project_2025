# Analisi e classificazione di abitudini di consumo di alcool e sigarette, tramite tecniche di Machine Learning

<p align="center">
<strong>Laurea magistrale in "Ingegneria e scienze informatiche"</strong><br>
<strong>Anno accademico 2024/2025</strong><br>
<strong>Corso di "Fondamenti di Machine Learning"</strong><br>
<strong>Autori:</strong><br>
Marco Deano, matricola VR503057<br>
Marco Giacopuzzi, matricola VR509643
</p>

## Indice

- **Dataset preso in analisi ed obiettivi del progetto**
- **State Of The Art**
- **Metodologie, modelli ed algoritmi utilizzati**
- **Procedure seguite e risultati ottenuti**
- **Conclusioni**
- **Biliografia**

## Dataset preso in analisi ed obiettivi del progetto

### Presentazione del dataset

Il dataset utilizzato in questo progetto[1], è un dataset tabellare composto da circa 1 milione di righe e 24 features, contenenti informazioni su vari parametri clinici e/o anagrafici degli individui; le features presenti includono sia dati numerici (misure biometriche) sia variabili categoriali (sesso).

Le label di classificazione considerate sono in tutto 2:

- **Bevitore / Non bevitore** → problema di classificazione binaria
- **Fumatore / Ex fumatore / Non fumatore** → problema di classificazione multiclasse

Il dataset presenta sbilanciamento per quanto riguarda le classi inerenti il fumo, con una distribuzione di circa 60% non fumatori, 15% ex-fumatori e 25% fumatori attuali, mentre la distribuzione delle classi inerenti il bere è bilanciata (50%-50%).

Abbiamo deciso di lavorare con questo dataset per diversi motivi: 
1. **Dimensioni del dataset e sua struttura:** essendo un dataset contenente quasi 1 milione di sample ed essendo ben strutturato con delle feature chiare ed esplicite, abbiamo ritenuto fosse adatto per simulare una situazione reale in cui viene richiesto di risolvere una task di classificazione dati dei parametri clinici di vari pazienti (per di più il dataset proviene da un sito governativo Coreano, dunque è effettivamente un dataset reale).  
Inoltre non abbiamo dovuto spendere molto tempo per comprenderne il contenuto e non si sono rese necessarie operazioni di pre-processing eccessive.   

2. **Tipologia di dataset:**: entrambi crediamo che l'utilizzo delle più svariate tecniche di Machine Learning sia uno strumento molto potente da poter utilizzare nei più svariati ambiti, e uno di questi è sicuramente quello medico; proprio per questo nelle nostre ricerche per trovare il dataset da utilizzare per il progetto, abbiamo cercato per quanto possibile di riuscire a lavorare con un dataset che centrasse proprio con questo tipo di dominio.  
Anche se lo scopo principale di questo dataset non è quello di individuare la presenza o meno di una certa malattia nel corpo di un individuo, ci è sembrato interessante cercare un modello di Machine Learning che possa indicare in maniera più o meno confidente se un individuo è un fumatore, non lo è, lo era oppure se è un bevitore oppure no.  
D'altronde in un caso di ricovero di un individuo per motivi di salute, potrebbe essere importante sapere il rapporto che il paziente ha con l'alcool e con il fumo, e non è da escludere che il paziente stesso possa anche mentire o non voglia proprio rispondere a riguardo: avere a disposizione uno strumento di questo tipo che ci aiuta a rispondere a delle domande a cui il paziente può anche decidere di non rispondere o mentire, è sicuramente utile.

3. **Task diverse per un solo datset:** proprio per il motivo appena citato, il dataset si presta bene oltre che per un problema di classificazione binaria e multiclasse, anche ad un problema di clustering: infatti non sappiamo di preciso se le groud truth presenti nel dataset sono effettivamente reali oppure no, quindi abbiamo pensato che fosse una buona idea utilizzare il dataset per una task di questo tipo.  
In questo modo potremmo eventualmente individuare gli individui che presumibilmente hanno mentito a riguardo del consumo di alcool o fumo.

### Obiettivi del progetto

Come è facilamente intuibile dal paragrafo precedente, gli obiettivi del nostro progetto sono essenzialmente 2:

- Creare un modello che cerchi di classificare correttamente il maggior numero possibile di sample.
- Creare un modello che divida il dataset in cluster, in modo da darci una stima del numero di individui che hanno mentito a riguardo del consumo di alcool o fumo.

Cercheremo di raggiungere questi obiettivi per step iniziando con dei modelli più semplici e proseguendo con degli altri più complessi o comunque diversi come filosofia; durante tutto lo svolgimento del progetto verranno comunque monitorati costantemente i risultati delle predizioni dei vari modelli e cercheremo di volta in volta di migliorarli sempre di più.

## State Of The Art (accenni)

L'analisi di dataset inerenti l'ambito medico e la creazione di modelli utili per la di malattie e non solo, è un tema ben studiato nell'ambito del Machine Learning.  
Attualmente, i modelli più utilizzati per problemi di classificazione in questo contesto sono sicuramente[2]:

- **Decision Tree e Random Forest:** note per la loro robustezza agli outlier e per la capacità di gestire feature categoriali e feature numeriche senza necessità di encoding e scalatura.
- **Support Vector Machine:** efficaci in scenari con alta dimensionalità, specialmente con l’uso di kernel non lineari.
- **Reti neurali:** modelli molto più complessi rispetto a quelli più tradizionali di Machine Learning, ma spesso richiedono dataset molto grandi.

Tuttavia, questi modelli non possono essere applicati senza una selezione ragionata e sensata degli iperparametri e delle feature da utilizzare e le criticità a cui si rischia di andare incontro sono:

- **Sbilanciamento delle classi:** nei dataset reali come quello preso in esame, alcune classi (es. ex-fumatori) possono essere sottorappresentate, portando a bias nei modelli.
- **Feature selection:** non tutte le feature presenti nel dataset sono necessariamente informative ed importanti per la classificazione e l’inclusione di quelle più irrilevanti può peggiorare le performance.
- **Scelta degli iperparametri:** la selezione ottimale di parametri come profondità degli alberi in Random Forest o il tipo di kernel in SVM o ancora il numero di "neighbours" in modelli tipo K-NN, influisce significativamente sulla qualità del modello.

Per affrontare queste problematiche, nel progetto si adotteranno alcuni di quegli accorgimenti (tecniche) studiate a lezione:

- **Feature selection:** per ridurre la dimensionalità del dataset e migliorare l'efficienza del modello.
- **Ottimizzazione degli iperparametri:** tramite Grid Search verranno testati i vari modelli con combinazioni di iperparametri differenti, con l'obiettivo di trovare quella che più si adatta al nostro problema.
- **Principal Component Analysis:** per ridurre la dimensionalità mantenendo la maggior parte della varianza, per molti modelli predittivi sarà necessario apllicare la PCA.

## Metodologie, modelli ed algoritmi utilizzati



## Bibliografia

[1]https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset

[2] https://www.digital4.biz/executive/medicina-e-intelligenza-artificiale-come-le-macchine-possono-migliorare-la-nostra-salute/

# DA FARE 

- Commentare come mai con feature engeneering non migliora la situa
- Fare pca dopo feature engeneering
- Fare stacking
- Fare predittore ad-hoc
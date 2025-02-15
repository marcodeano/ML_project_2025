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

2. **Tipologia di dataset:** entrambi crediamo che l'utilizzo delle più svariate tecniche di Machine Learning sia uno strumento molto potente da poter utilizzare nei più svariati ambiti, e uno di questi è sicuramente quello medico; proprio per questo nelle nostre ricerche per trovare il dataset da utilizzare per il progetto, abbiamo cercato per quanto possibile di riuscire a lavorare con un dataset che centrasse proprio con questo tipo di dominio.  
Anche se lo scopo principale di questo dataset non è quello di individuare la presenza o meno di una certa malattia nel corpo di un individuo, ci è sembrato interessante cercare un modello di Machine Learning che possa indicare in maniera più o meno confidente se un individuo è un fumatore, non lo è, lo era oppure se è un bevitore oppure no.  
D'altronde in un caso di ricovero di un individuo per motivi di salute, potrebbe essere importante sapere il rapporto che il paziente ha con l'alcool e con il fumo, e non è da escludere che il paziente stesso possa anche mentire o non voglia proprio rispondere a riguardo: avere a disposizione uno strumento di questo tipo che ci aiuta a rispondere a delle domande a cui il paziente può anche decidere di non rispondere o mentire, è sicuramente utile.

3. **Obiettivo del progetto non banale:** nonostente il dataset preveda per il caso del fumo di fare una classificazione con sole 3 classi e per il caso dell'alcool di fare una classificazione binaria, riteniamo che classificare un paziente come "Fumatore", "Ex-fumatore" o "Non fumatore" non sia per nulla banale; non sappiamo infatti da quanto tempo eventualmente i pazienti hanno smesso di fumare, con che frequenza fumano oppure se hanno dei parametri clinici anomali per via di altre malattie.  
Difatti, come si evincerà meglio nel proseguio del progetto, abbiamo concentrato molte delle nostre energie nel tentare di migliorare il più possibile la predizione per il caso del fumo e questo proprio perchè si è rivelata una task per nulla banale.

<!-- 3. **Task diverse per un solo datset:** proprio per il motivo appena citato, il dataset si presta bene oltre che per un problema di classificazione binaria e multiclasse, anche ad un problema di clustering: infatti non sappiamo di preciso se le groud truth presenti nel dataset sono effettivamente reali oppure no, quindi abbiamo pensato che fosse una buona idea utilizzare il dataset per una task di questo tipo.  
In questo modo potremmo eventualmente individuare gli individui che presumibilmente hanno mentito a riguardo del consumo di alcool o fumo. -->

### Obiettivo del progetto

Come è facilamente intuibile dal paragrafo precedente, l'obiettivo principale del nostro progetto è:

- Creare un modello che cerchi di classificare correttamente il maggior numero possibile di sample sia per il caso del fumo che dell'alcool.
- ALTRO???
<!-- - Creare un modello che divida il dataset in cluster, in modo da darci una stima del numero di individui che hanno mentito a riguardo del consumo di alcool o fumo. -->

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
- **Tecniche di ensemble prediction:** spesso utilizzare solo un modello per fare predizioni su un dataset molto grande, non è la scelta migliore ed il rischio è che vengano fatte predizioni corrette per certe classi e per altre meno; proprio per questo sfrutteremo tecniche di ensemble prediction (come bagging, boosting e stacking) per cercare di predirre in maniera più robusta le classi dei vari sample.

## Metodologie, modelli ed algoritmi utilizzati

Per raggiungere gli obiettivi che ci siamo prefissati, abbiamo eseguito in un ordine ben preciso determinati step che verranno descritti di seguito.

### Preprocessing e Pulizia del Dataset

Prima di applicare qualsiasi modello di Machine Learning, è fondamentale preparare il dataset in modo che si possano sfruttare al meglio le potenzialità dei vari modelli di apprendimento; abbiamo dunque nell'ordine:  
1. Identificato i valori fuori scala (outlier) utilizzando distribuzioni statistiche (medie, percentili, IQR e deviazione standard).
2. Rimosso i sample con valori estremamente anomali.
3. Fatto encoding delle variabili categoriali, più nel dettaglio Label Encoding (non si è reso necessario il One-Hot Encoding visto che le uniche feature categoriali somo "sex" e le label delle classi).

### Un primo esempio con un modello semplice

Per partire abbiamo implementato un semplice modello di Logistic Regression per vedere che lower bound abbiamo per quanto riguarda le metriche di valutazione (accuratezza, recall, precisione, f1_score).

### 

## Bibliografia

[1]https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset

[2] https://www.digital4.biz/executive/medicina-e-intelligenza-artificiale-come-le-macchine-possono-migliorare-la-nostra-salute/

# DA FARE 

- Commentare come mai con feature engeneering non migliora la situa
- Commentare come mai con pca non migliora la situa
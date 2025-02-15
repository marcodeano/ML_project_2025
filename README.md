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

- Creare un modello che cerchi di classificare correttamente il maggior numero possibile di sample sia per il caso del fumo che dell'alcool; nonostante però durante tutto lo svolgimento del progetto si sia cercato di migliorare i risultati per entrambe le task di classificazione, alla fine i nostri sforzi principali sono stati dedicati alla classificazione multiclasse del caso del fumo (per tutti i motivi descritti sopra).
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

## Metodologie, modelli ed algoritmi utilizzati

Per raggiungere gli obiettivi che ci siamo prefissati, abbiamo eseguito in un ordine ben preciso determinati step che verranno descritti di seguito; durante tutti i test effettuati i risultati e le performance dei vari modelli/teniche utilizzate, sono stati monitorati tenendo in considerazione le classiche metriche di `Accuratezza`, `Precisione`, `Recall` ed `F1-score` ma dando più importanza all'`Accuratezza` per il caso della classificazione binaria dell'acool e all'`F1-score` per il caso della classificazione multiclasse del fumo. Quest'ultima scelta è stata fatta perchè l'F1-score risulta essere una metrica particolarmente utile nei problemi di classificazione multiclasse sbilanciata perché fornisce un bilanciamento tra precision e recall; inoltre evita che un modello sembri performante solo perché predice bene la classe più rappresentata e quindi un buon F1-score indica che il modello non solo riconosce le classi meno rappresentate, ma lo fa anche in modo preciso.  Oltre ciò, si è rivelato essenziale durante tutto il progetto l'utilizzo e la visualizzazione delle `Confusion Matrix`.

### Preprocessing e pulizia del dataset

Prima di applicare qualsiasi modello di Machine Learning, è fondamentale preparare il dataset in modo che si possano sfruttare al meglio le potenzialità dei vari modelli di apprendimento; abbiamo dunque nell'ordine:  

1. Fatto una prima analisi dettagliata del dataset protagonista del progetto di Machine Learning ed estrapolato quindi tutte le informazioni utili per comprenderne la struttura; più nello specifico abbiamo ricavato, tramite dei semplici script, la distribuzione dei valori per le varie feature, la distribuzione delle classi in tutto il dataset e tracciato dei "boxplot" per individuare la presenza di eventuali outlier che possono interferire con il training dei vari modelli che verranno utilizzati nel progetto.
2. Identificato i valori fuori scala (outlier): l'analisi dettagliata del dataset fatta al passo precedente, ci ha permesso di individuare facilmente molti degli outlier più evidenti (che presentavano valori estremamente distanti dalla media, es: waistline = 999) ed eliminarli dal dataset, ma ci ha anche spinti ad utilizzare uno strumento statistico per individuare gli outlier un po' meno evidenti ma comunque anomali rispetto alla distribuzione generale del dataset. A tale scopo abbiamo utilizzato lo Z-score: una misura statistica che indica di quante deviazioni standard un valore di una feature si discosta dalla media; per definire la soglia di esclusione dei sample, abbiamo stabilito un valore limite di Z-score pari a 5, oltre il quale i campioni vengono considerati anomali ed esclusi dal dataset. In questo modo abbiamo escluso da tutte le operazioni di predizione future, tutti quei sample che o hanno valori errati oppure rappresentano casi clinici particolari che andrebbero ad influire negativamente sui risultati del progetto.
3. Fatto encoding delle variabili categoriali: più nel dettaglio Label Encoding (non si è reso necessario il One-Hot Encoding visto che le uniche feature categoriali sono "sex" e le label delle classi). 

Abbiamo concordato, terminati questi passaggi, non fosse necessario eseguire subito uno scaling dei dati (visto che sono presenti features con valori molto grandi e altre con valori più piccoli), perchè abbiamo ritenuto essere più opportuno fare queste operazioni eventualmente in un secondo momento a seconda dei modelli di Machine Learning che avremmo utilizzato.

#### Script e notebook di riferimento:
- `load_data_and_distribution_analisys.py`
- `distribution_analisys_and_outlier_remove.ipynb`

### Un primo esempio con un modello semplice

Una volta terminata la fase di analisi e pulizia dei dati, abbiamo testato un modello base, utilizzando un classificatore semplice come un `Logistic Regressor`, per ottenere un primo benchmark sulle performance e dopo aver analizzato questi primissimi risultati, abbiamo gradualmente introdotto modelli più complessi e implementato diverse tecniche di ottimizzazione per migliorare le prestazioni.

### Random Forest

Il modello che si è rivelato il migliore per quanto rifuarda la task di classificare correttamente le classi del nostro dataset, è stato proprio il primo che abbiamo testato: la Random Forest. Sapevamo che questo fosse un modello molto diffuso e adatto all'ambito medico, ma non avremmo pensato che ci avrebbe fatto ottenere risultati migliori anche di modelli come SVM non lineari o tecniche come il Boosting, il Bagging e lo Stacking (il perchè lo abbiamo ritenuto il miglior modello per il nostro progetto, lo riportiamo nella parte di conclusioni e risultati finali).

Abbiamo inizialmente testato un modello base di Random Forest, senza esplicitare i valori dei vari iperparametri, per ottenere dei primi risultati di riferimento e comprendere il comportamento del modello sui nostri dati; dopo aver valutato le metriche di performance, ci siamo resi conto che c'erano margini di miglioramento e abbiamo deciso di affinare il modello seguendo un approccio progressivo.

Per prima cosa, abbiamo proceduto con un’ottimizzazione degli iperparametri sfruttando una "Grid Search" per affinare ulteriormente la ricerca; tra i parametri più rilevanti abbiamo considerato il numero di alberi nella foresta, la profondità massima degli alberi e il numero minimo di campioni richiesti per effettuare uno split, tenendo sempre monitorati tutte le metriche citate prima ma prestando particolare attenzione all'F1-score. Ci siamo subito accorti che, solo facendo una ricerca degli iperparametri ottimali, le metriche cominciavano a migliorare e la confusion matrix diventava sempre più bilanciata, con una riduzione degli errori di classificazione soprattutto nelle classi meno rappresentate; inoltre la confusion matrix ci ha permesso di individuare quali classi venivano maggiormente confuse tra loro, aiutandoci a comprendere dove il modello faticava maggiormente

A questo punto, abbiamo provato ad ottimizzare la RandomForest sfruttando la tecnica della feature selection: avendo a che fare con un modello predittivo di questo tipo, ovvero abbastanza robusto alle feature irrilevanti, non ci aspettavamo un grosso miglioramento delle performance ma nonostante ciò rimane comunque un passaggio utile per poter eventualmente semplificare il modello andando a togliere anche poche feature e rendere il modello predittivo più veloce. Inoltre, non è stato indicato a priori un numero fissato di feature da selezionare, in quanto non certi di quelle che potessero essere le performance selezionando solo il 10%, il 20%, il 50%, il 75%... delle feature; quindi abbiamo implementato un algoritmo di feature selection di tipo Forward, monitorando la F1-score ad ogni "best_feature" aggiunta, e salvando la combinazione di feature migliore tra tutte quelle testate.

Per validare ulteriormente il nostro approccio, abbiamo applicato la cross-validation utilizzando la tecnica StratifiedKFold, in modo da avere una valutazione più affidabile delle performance su diverse suddivisioni del dataset: siccome nello step precedente in cui abbiamo effettuato feature selection abbiamo allenato il nostro modello tenendo costante il set di train e di validazione, potremmo aver "overfittato" in base alla suddivisione specifica; con la cross-validation, testiamo il modello su diverse porzioni del dataset e possiamo valutare se le feature scelte hanno migliorato davvero il modello.

Dopo tutte queste analisi, la Random Forest si è dimostrata un modello solido, in grado di ottenere buoni risultati con un tempo di addestramento contenuto; inoltre, come si è potuto evincere, non si sono rese necessarie operazioni come Scaling dei dati o RIPARTI DA QUA 


Abbiamo inoltre sperimentato l'uso della PCA (Principal Component Analysis) per verificare se la riduzione dimensionale potesse contribuire a migliorare l'efficienza e l'accuratezza dei modelli. Infine, abbiamo condotto una ricerca degli iperparametri ottimali attraverso tecniche di ottimizzazione come la grid search e la cross-validation, con l’obiettivo di massimizzare le metriche di valutazione e ottenere il miglior compromesso tra accuratezza e generalizzazione del modello.

- **Feature selection:** per ridurre la dimensionalità del dataset e migliorare l'efficienza del modello.
- **Ottimizzazione degli iperparametri:** tramite Grid Search verranno testati i vari modelli con combinazioni di iperparametri differenti, con l'obiettivo di trovare quella che più si adatta al nostro problema.
- **Principal Component Analysis:** per ridurre la dimensionalità mantenendo la maggior parte della varianza, per molti modelli predittivi sarà necessario apllicare la PCA.
- **Tecniche di ensemble prediction:** spesso utilizzare solo un modello per fare predizioni su un dataset molto grande, non è la scelta migliore ed il rischio è che vengano fatte predizioni corrette per certe classi e per altre meno; proprio per questo sfrutteremo tecniche di ensemble prediction (come bagging, boosting e stacking) per cercare di predirre in maniera più robusta le classi dei vari sample.

### 

## Bibliografia

[1]https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset

[2] https://www.digital4.biz/executive/medicina-e-intelligenza-artificiale-come-le-macchine-possono-migliorare-la-nostra-salute/

# DA FARE 

- Commentare come mai con feature engeneering non migliora la situa
- Commentare come mai con pca non migliora la situa
- Aggiungere i link bibliografici
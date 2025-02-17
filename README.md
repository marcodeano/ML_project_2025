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
- **Conclusioni**
- **Biliografia**

## Dataset preso in analisi ed obiettivi del progetto

### Presentazione del dataset

Il dataset utilizzato in questo progetto[1], è un dataset tabellare composto da circa 1 milione di righe e 24 features, contenenti 
informazioni su vari parametri clinici e/o anagrafici degli individui; le features presenti includono sia dati numerici 
(misure biometriche) sia variabili categoriali (sesso).

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

### Obiettivo del progetto

Come è facilamente intuibile dal paragrafo precedente, l'obiettivo principale del nostro progetto è:

- Creare un modello che cerchi di classificare correttamente il maggior numero possibile di sample sia per il caso del fumo che dell'alcool; nonostante però durante tutto lo svolgimento del progetto si sia cercato di migliorare i risultati per entrambe le task di classificazione, alla fine i nostri sforzi principali sono stati dedicati alla classificazione multiclasse del caso del fumo (per tutti i motivi descritti sopra).

Cercheremo di raggiungere questo obiettivo per step iniziando con dei modelli più semplici e proseguendo con degli altri più complessi o comunque diversi come filosofia; durante tutto lo svolgimento del progetto verranno comunque monitorati costantemente i risultati delle predizioni dei vari modelli e cercheremo di volta in volta di migliorarli sempre di più.

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

Come si evincerà di seguito, nel nostro progetto abbiamo testato modelli di Machine Learning diversi cercando di valutare quale fosse più adatto al nostro problema di classificazione; ovviamente non ci siamo limitati a testare semplicemente i modelli base, ma abbiamo accompagnato ogni fase con diverse tecniche per migliorare la qualità dei dati da presentare in input ai modelli e ottimizzare le prestazioni dei modelli stessi. Abbiamo quindi adottato, a seconda anche del modello con cui stavamo lavorando, le seguenti tecniche: 

- **Feature selection:** per ridurre la dimensionalità del dataset e migliorare l'efficienza dei modelli.
- **Feature engeneering:** parallelamente alla feature selection, abbiamo anche cercato di creare, trasformare ed in generale ottimizzare le feature del dataset per migliorare le performance dei nostri modelli; più nel dettaglio abbiamo fatto delle ricerche inerenti il dominio di cui ci siamo occupati in questo progetto e abbiamo creato delle nuove feature in modo da ottenere un dataset con dei parametri differenti da quelli che avevamo a disposizione all'inizio e tutto ciò ovviamente con l'obiettivo di cercare di migliorare le performance dei nostri modelli predittivi.
- **Ottimizzazione degli iperparametri:** tramite "Grid Search" sono stati testati i vari modelli con combinazioni di iperparametri differenti, con l'obiettivo di trovare quella che più si adatta al nostro problema.
- **Principal Component Analysis:** per ridurre la dimensionalità mantenendo la maggior parte della varianza, per molti modelli predittivi risulta sempre utile apllicare la PCA.
- **Tecniche di ensemble prediction:** spesso utilizzare solo un modello per fare predizioni su un dataset molto grande, non è la scelta migliore ed il rischio è che vengano fatte predizioni corrette per certe classi e per altre meno; proprio per questo sfrutteremo tecniche di ensemble prediction (come bagging, boosting e stacking) per cercare di predirre in maniera più robusta le classi dei vari sample.
- **Classificazione gerarchica:** un ulteriore passo è stato l’introduzione di una strategia di classificazione gerarchica, particolarmente utile per gestire il problema dell’imbalanciamento delle classi soprattutto per la task sul fumo.

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

Per prima cosa, abbiamo proceduto con un’ottimizzazione degli iperparametri sfruttando una "Grid Search" per affinare ulteriormente la ricerca; tra i parametri più rilevanti abbiamo considerato il numero di alberi nella foresta, la profondità massima degli alberi e il numero minimo di campioni richiesti per effettuare uno split dei nodi, tenendo sempre monitorati tutte le metriche citate prima ma prestando particolare attenzione all'F1-score. Ci siamo subito accorti che, solo facendo una ricerca degli iperparametri ottimali, le metriche cominciavano a migliorare e la confusion matrix diventava sempre più bilanciata, con una riduzione degli errori di classificazione soprattutto nelle classi meno rappresentate; inoltre la confusion matrix ci ha permesso di individuare quali classi venivano maggiormente confuse tra loro, aiutandoci a comprendere dove il modello faticava maggiormente

A questo punto, abbiamo provato ad ottimizzare la RandomForest sfruttando la tecnica della feature selection: avendo a che fare con un modello predittivo di questo tipo, ovvero abbastanza robusto alle feature irrilevanti, non ci aspettavamo un grosso miglioramento delle performance ma nonostante ciò rimane comunque un passaggio utile per poter eventualmente semplificare il modello andando a togliere anche poche feature e rendere il modello predittivo più veloce. Inoltre, non è stato indicato a priori un numero fissato di feature da selezionare, in quanto non certi di quelle che potessero essere le performance selezionando solo il 10%, il 20%, il 50%, il 75%... delle feature; quindi abbiamo implementato un algoritmo di feature selection di tipo Forward, monitorando la F1-score ad ogni "best_feature" aggiunta, e salvando la combinazione di feature migliore tra tutte quelle testate.

Per validare ulteriormente il nostro approccio, abbiamo infine applicato la cross-validation utilizzando la tecnica `StratifiedKFold`, in modo da avere una valutazione più affidabile delle performance su diverse suddivisioni del dataset: siccome nello step precedente in cui abbiamo effettuato feature selection abbiamo allenato il nostro modello tenendo costante il set di train e di validazione, potremmo aver "overfittato" in base alla suddivisione specifica; con la cross-validation, testiamo il modello su diverse porzioni del dataset e possiamo valutare se le feature scelte hanno migliorato davvero il modello.

Dopo tutte queste analisi, la Random Forest si è dimostrata un modello solido, in grado di ottenere buoni risultati con un tempo di addestramento contenuto; inoltre, come si è potuto evincere, non si sono rese necessarie operazioni come Scaling dei dati o Dimensionality Reduction e questo perchè il modello Random Forest non è influenzato da grandezze diverse tra le feature e gestisce bene dati ad alta dimensionalità.

#### Notebook di riferimento:

- `random_forest_classifier_base_model.ipynb`
- `random_forest_classifier_with_feature_selection.ipynb`
- `random_forest_classifier_cross_validation.ipynb`

### Altri classificatori

# Altri classificatori testati

Ovviamente non ci siamo limitati ad utilizzare ed ottimizzare solamente il modello di Random Forest, anzi abbiamo effettuato le stesse procedure fatte per tale modello anche con modelli totalmente differenti come filosofia; nel progetto non sono stati riportati tutti questi passaggi anche per questi modelli per non appesantire troppo il progetto, però abbiamo comunque riportato tutti i risultati per mettere in evidenza le differenze di performance tra i vari approcci e giustificare le scelte effettuate. 
I modelli allenati e testati sono stati:

- **KNN:** rappresenta un metodo basato sulla similarità tra i dati, il che ci ha permesso di analizzare il comportamento del modello quando la decisione dipende strettamente dalla vicinanza ai punti nel dataset; questo modello è sicuramente utile perchè non ha bisogno di nessun tipo di assunzione per quanto riguarda i dati e permette di costruire dei "bound" tra le classi di tipo non lineare e di conseguenza permette di capire se le classi siano separabili nello spazio delle feature a disposizione.
- **AdaBoost (con base_estimator Albero Decisionale):** un modello di ensemble basato sul boosting e noto per la sua capacità di concentrarsi sugli errori commessi dai modelli più deboli nelle iterazioni precedenti, migliorando progressivamente la qualità della classificazione; abbiamo deciso di testare anche questo tipo di "modello" in modo da verificare se un metodo di ensemble focalizzato su un raffinamento iterativo potesse fornire risultati migliori rispetto a un metodo più stabile come Random Forest.
- **Support Vector Machine (kernel lineare):** è un modello molto efficace nei problemi di classificazione, in particolare quando i dati sono ben separabili in uno spazio ad alta dimensione; nonostante sapessimo che difficilmente i sample del nostro dataset lo sarebbero stati, abbiamo deciso di testare anche questo tipo di modello per verificare empiricamente se una SVM con kernel lineare potesse comunque individuare un iperpiano di separazione efficace, magari grazie all’uso di margini ampi e alla capacità di gestire outlier con il parametro di penalizzazione.

Ovviamente prima di allenare questi modelli, ci siamo assicurati di normalizzare i dati di train e test per ottenere le migliori performance possibili.

Oltre a testare questi modelli nella loro versione base e con iperparametri ottimi, abbiamo anche testato l'applicazione di PCA come tecnica di Dimensionality Reduction (ovviamente solo per il modello KNN ed il modello SVM, per il modello AdaBoost con base_estimator Albero Decisionale abbiamo ritenuto fosse non necessario): difatto avendo a che fare con un dataset con 22 feature e avendo gia testato la tecnica della feature selection, volevamo valutare se la riduzione della dimensionalità tramite PCA potesse migliorare le prestazioni dei modelli.  Anche per quanto riguarda l'utilizzo di questa tecnica con i modelli sopra citati, non abbiamo creato un notebook contenente tutti i passaggi per selezionare il miglior numero di componenti da passare come parametro alla PCA, ma abbiamo lasciato solamente la versione che ci ha permesso di ottenere i migliori risultati.

#### Notebook di riferimento:

- `other_classifiers.ipynb`
- `other_classifiers_with_pca.ipynb`

### Feature engeneering

Una volta testati tutti questi modelli predittivi di Machine Learning, abbiamo fatto delle ricerche inerenti il dominio di cui ci siamo occupati in questo progetto e abbiamo creato delle nuove feature in modo da creare un dataset con dei parametri differenti da quelli che avevamo a disposizione all'inizio; ovviamente l'obiettivo di tutto ciò è stato cercare di migliorare le performance dei nostri modelli predittivi estraendo informazioni più significative dalle feature esistenti o crearne di nuove che potessero contribuire a migliorare le performance dei classificatori.  

#### Caratteristiche antropometriche

Queste caratteristiche offrono buoni indicatori circa lo stato fisico di un paziente, elementi che possono essere influenzati dal fumo e dal consumo di alcol.
L'obesità e la distribuzione anomala del grasso sono condizioni che sono spesso correlate a comportamenti a rischio e possono essere identificate utilizzando questi parametri.

- **BMI**: Misura l'obesità generale.
- **wth_ratio**: Rapporto vita/altezza, indica l'obesità addominale.
- **wtw_ratio**: Rapporto vita/peso, evidenzia la distribuzione del grasso.
- **obesity_flag**: Flag per identificare rapidamente l'obesità.

#### Caratteristiche Cardiovascolari

Le caratteristiche cardiovascolari offrono indicazioni chiave sullo stato del sistema circolatorio, il quale può essere compromesso dal fumo e dal consumo di alcol.
Questi parametri evidenziano anomalie pressorie e rigidità arteriosa, segnali indiretti di comportamenti a rischio.

- **pulse_pressure**: Differenza tra SBP e DBP, indice di rigidità arteriosa.
- **MAP**: Misura globale della pressione arteriosa.
- **bp_category**: Classifica la pressione, rilevando eventuali anomalie.

#### Profilo Lipidico e Rapporti Metabolici

Queste caratteristiche metaboliche forniscono informazioni utili sullo stato cardiovascolare e metabolico dei pazienti.
Forniscono anche informazioni sul modo in cui il fumo e il consumo di alcol possono alterare queste caratteristiche.
Di conseguenza, un profilo lipidico instabile e un disequilibrio nei rapporti metabolici possono essere indicazioni indirette di un modo in cui uno vive uno stile di vita malsano.

- **TC_HDL_ratio**: Indica il rischio cardiovascolare confrontando il colesterolo totale con l'HDL.
- **LDL_HDL_ratio**: Evidenzia lo squilibrio tra il colesterolo “cattivo” e quello “buono”.
- **non_HDL_chole**: Misura le particelle aterogeniche, segnale di potenziali patologie.
- **triglyceride_hdl_ratio**: Riflette il rischio metabolico influenzato da dieta e stile di vita.
- **AIP**: Valuta sinteticamente il rischio cardiovascolare tramite il logaritmo di trigliceridi/HDL.
- **TyG**: Indica resistenza insulinica e rischio metabolico calcolando trigliceridi e glucosio.

#### Funzione Epatica e Renale

L'inclusione di indicatori per la funzione epatica e renale è vantaggiosa perché cambiamenti in questi parametri possono essere segnali indiretti di danni organici causati dal fumo e dal consumo di alcol.
Queste azioni aiutano a identificare stili di vita a rischio fornendo informazioni sullo stato degli organi.

- **AST_ALT_ratio**: Valori elevati possono indicare danni epatici da alcol.
- **liver_enzyme_interaction**: Segnala anomalie enzimatiche legate all'abuso di alcol.
- **liver_enzyme_avg**: Fornisce una panoramica rapida della funzione del fegato.
- **eGFR**: Utile per rilevare danni renali associati a stili di vita non salutari.

Visto che tramite l'aggiunta di queste nuove feature abbiamo incrementato la dimensionalità del dataset, abbiamo ritenuto opportuno fare dei test applicando anche la tecnica della PCA; c'è da precisare che abbiamo testato solo questa tecnica e non la Feature Reduction dopo l'aggiunta di queste nuove feature, per una questione di tempo per ottenere i risultati.  Abbiamo ritenuto che l'utilizzo della PCA fosse un test necessario da fare per provare ad estrarre pattern più significativivisto visto che tramite feature engeneering abbiamo aumentato il numero delle feature totali all'interno del dataset.

#### Notebook di riferimento:

- `feature_engeneering.ipynb`
- `feature_engeneering_with_pca.ipynb`

### Bagging ensemble

Arrivati a questo punto del progetto non ci rimanevano molti modelli/tecniche da provare per migliorare i risultati già ottenuti, quindi una delle ultime tecniche di predizione che abbiamo testato è stata quella del bagging ensemble: una tecnica che consiste nel combinare i risultati ottenuti da più modelli per ottenere una predizione più accurata. I modelli che abbiamo utilizzato per fare bagging ensemble, sono stati AdaBoost e Support Vector Machine (kernel rbf), abbiamo scelto questi 2 modelli per le seguenti ragioni:

- Volevamo provare a fondere la tecnica del boosting (caratteristica di AdaBoost) con la tecnica del bagging ensemble
- Siccome il tempo per il training di un modello di SVM non lineare scala in maniera quadratica con il numero di sample, era proibitivo allenare un modello singolo di questo tipo anche solo utilizzando il 50% del dataset (500.000 sample); dunque abbiamo optato per utilizzare la tecnica dell'ensemble utilizzando più modelli SVM non lineari ma allenati ognuno su una porzione ristretta del dataset.

Non abbiamo testato la tecnica del bagging ensemble per altri modelli come Random Forest perchè secondo il nostro parere non avrebbe apportato un beneficio significativo: nel caso della Random Forest il bagging è già un componente fondamentale del modello stesso in cui vengono costruiti più alberi su sottocampioni del dataset e poi aggregati tramite voto maggioritario, mentre per quanto riguarda KNN, il bagging non risulta particolarmente efficace perchè il suo comportamento dipende essenzialmente dalla scelta dei vicini più prossimi e dalla distanza utilizzata.

#### Notebook di riferimento:

- `bagging_ensemble.ipynb`

### Stacking Classifier e Voting Classifier

Altre 2 tecniche di ensemble che abbiamo testato, sono state lo Stacking e il Voting Classifier: entrambi i metodi mirano a combinare più modelli per ottenere una predizione finale più robusta come nel caso del bagging e del boosting, ma con approcci differenti.

- Lo Stacking sfrutta un livello aggiuntivo di apprendimento rispetto al bagging, in quanto i risultati dei singoli modelli base utilizzati vengono sfruttati come input per un meta-modello, che impara a combinare le loro predizioni in modo ottimale; abbiamo selezionato come modelli base tutti quelli che abbiamo sfruttato nei test precedenti (escluso il SVM lineare) e abbiamo utilizzato come meta-modello un SVM con kernel lineare.  Questa scelta è stata fatta per un motivo semplice: il modello Random Forest fino a questo momento è risultato il migliore e abbiamo pensato innanzitutto che fosse corretto sfruttarlo come modello base all'interno dello stacking piuttosto che come meta-modello finale e che fosse invece sbagliato inserirlo sia come modello base che come meta-modello, perchè avrebbe ridotto la diversità del sistema di ensemble. La scelta del meta-modello è dunque ricaduta sul SVM lineare vista la sua velocità di training e i comunque buoni risultati che ci aveva garantito fino a questo punto del progetto.
- Il Voting Classifier, invece, segue un approccio più semplice: aggrega le predizioni dei vari modelli base attraverso una strategia di voto, che può essere di tipo hard voting (sceglie la classe con più voti tra i modelli) o soft voting (media delle probabilità predette).

#### Notebook di riferimento:

- `model_stacking.ipynb`

### Classificatore gerarchico

Arrivati a questo punto del progetto, ci siamo resi conto che l'obiettivo di separare in maniera sempre più precisa la classe dei Fumatori da quella degli Ex-fumatori e da quella dei Non-fumatori risultava particolarmente complesso a causa dello sbilanciamento delle classi e della possibile sovrapposizione nei pattern dei dati; ci siamo accorti però che tutti i modelli allenati e perfezionati fino ad ora, avevano una cosa in comune: non riuscivano a separare correttamente la classe degli Ex-fumatori da quella dei Fumatori. Facendo poi un'analisi ancora più approfondita dei risultati, in particol modo delle matrici dei confusione ottenute, ci siamo però accorti che uno dei modelli testati nel progetto riusciva piuttosto bene a non classificare i Non Fumatori rispetto ai Fumatori o Ex-fumatori: si tratta del modello Random Forest. 

[...Screen matrice di confusione per far capire e giustificare il perchè abbiamo usato un classificatore gerarchico...]

Abbiamo quindi deciso di fare un ultimo tentativo per cercare di migliorare i risultati ottenuti fino a questo punto del progetto, sperimentando un approccio gerarchico alla classificazione e suddividento il problema in 2 fasi distinte:

- `Primo classificatore:` un modello iniziale per distinguere tra Non-fumatori e la classe unificata Ex-fumatori + Fumatori.
- `Secondo classificatore:` un modello specifico addestrato per separare i campioni precedentemente classificati come Ex-fumatori + Fumatori, suddividendoli nelle due classi originali.

L'idea alla base di questa strategia è che un primo modello possa apprendere meglio le caratteristiche distintive tra chi non ha mai fumato e chi, invece, ha avuto un'esposizione al fumo e successivamente il secondo classificatore può concentrarsi sulla separazione tra Ex-fumatori e Fumatori attuali, che presentano differenze più sottili e meno evidenti rispetto alla distinzione con i Non-fumatori.  Abbiamo dunque fatto una ricerca su quello che potesse essere un primo modello ottimale che potesse distinguere la classe dei Non-fumatori e la classe unificata Ex-fumatori + Fumatori e di seguito una ricerca analoga per selezionare il secondo modello. I nostri test hanno evidenziato come il miglior modello per riuscire a separare i Non-fumatori dalla classe unificata Ex-fumatori + Fumatori è lo Stacking Classifier, mentre per quanto riguarda il secondo modello il migliore i grado di separare Ex-fumatori e Fumatori è la Random Forest (anche lo Stacking Classifier si avvicina molto come performance).

Nonostante l'idea di partenza ci sembrasse buona, ci siamo però accorti di quanto non fosse banale costruire un classificatore di questo tipo: il fatto che il primo modello potesse classificare erroneamente dei sample con label "Non-fumatore" come "Ex-fumatore o Fumatore" e che poi dovessere essere classificati come "Ex-fumatore" o "Fumatore" ci ha creato dei grossi problemi nella creazione del modello che avevamo in mente noi. Difatti alla fine siamo stati costretti ad allenare il secondo modello con tutte e 3 le label iniziali e non solo sulle label di "Ex-fumatore" e "Fumatore" come avevamo in mente all'inizio.  Un'altra alternativa che ci era venuta in mente, è stata quella di allenare due modelli totalmente separati (il primo con label "Non-fumatore" e "Ex-fumatore o Fumatore" ed il secondo con label "Ex-fumatore" e "Fumatore") con una porzione del dataset, e in un secondo momento utilizzare il restante dataset come test; non abbiamo implementato questa alternativa per il semplice motivo che non sarebbe stato possibile avere dei risultati direttamente comparabili con quelli ottenuti fino a questo punto.

#### Notebook di riferimento:

- `separation_smokers_non_smokers.ipynb`
- `separation_ex_smokers_non_smokers.ipynb`
- `multistep_smokers_test.ipynb`

### Visualizzazione dati tramite t-SNE

Per farci un'idea più generale di quello che potesse essere la distribuzione nello spazio dei dati presenti nel dataset, abbiamo anche sfruttato la tecnica di t-SNE; in questo modo abbiamo potuto visualizzare quanto effettivamente le classi del nostro dataset fossero sovrapposte tra di loro. Ovviamente per rendere il tutto più comprensibile, abbiamo "plottato" il tutto in uno spazio a 2 dimensioni e considerando in un primo test le label "Non-fumatore" e "Ex-fumatore o Fumatore" accorpate in una unica classe ed in un secondo test le classi "Ex-fumatore" e "Fumatore".

#### Notebook di riferimento:

- `data_visualization.ipynb`

## Risultati

## Conclusione

## Bibliografia

[1]https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset

[2] https://www.digital4.biz/executive/medicina-e-intelligenza-artificiale-come-le-macchine-possono-migliorare-la-nostra-salute/

# DA FARE 

- Commentare come mai con feature engeneering non migliora la situa
- Commentare come mai con pca non migliora la situa
- Aggiungere i link bibliografici
- Se i risultati di model stacking sono migliori con combinazioni diverse, cambia anche dentro i notebook "separation"
- Sistemare il notebook di datavisualization
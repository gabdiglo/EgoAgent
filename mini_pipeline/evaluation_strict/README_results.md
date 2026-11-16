# Esperimento locale ispirato a EgoAgent

## Collocazione nella presentazione

Dopo la descrizione dei risultati degli autori, introdurre una sezione distinta: **Esperimento locale su Mac**. Usare tre slide: (1) obiettivo e semplificazioni, (2) protocollo sperimentale, (3) risultati e limiti. Non sostituire i dettagli sperimentali del paper originale con quelli del progetto locale.

Per mantenere la durata di 30 minuti, una possibile ripartizione è: introduzione 5 minuti, architettura e InternLM 8, loss 4, dati e training degli autori 2, risultati del paper 4, esperimento locale 4, limiti e conclusioni 3. Le tre slide pratiche vanno dopo i risultati degli autori e prima delle conclusioni.

## Obiettivo e architettura

È stata realizzata una pipeline ridotta ispirata alla previsione degli stati futuri nello spazio delle rappresentazioni di EgoAgent. L'esperimento considera coppie di frame di WalkingTours campionati a 1 FPS e distanziati di cinque secondi. Un encoder ResNet-18, seguito da una proiezione MLP, produce embedding di dimensione 256 normalizzati a norma unitaria. Un secondo MLP, con dimensioni 256–512–256, predice l'embedding futuro a partire da quello corrente.

La pipeline locale non implementa InternLM, l'attenzione causale su sequenze interlacciate, la componente di azione 3D o l'architettura completa JEAP. Il collegamento con il paper consiste nell'anticipazione del futuro in uno spazio latente, confrontata con una baseline esplicita.

## Addestramento descritto dagli script

La prima fase usa ResNet-18 inizializzato senza pesi preaddestrati e una loss contrastiva temporale simmetrica, con temperatura 0,1. Nella seconda fase l'encoder viene congelato e si ottimizza il predittore minimizzando uno meno la similarità coseno con l'embedding del frame futuro. Non è presente un observer aggiornato con EMA: l'encoder che estrae il target è congelato.

Gli script forniti configurano immagini 160 × 160, batch size 16, AdamW con learning rate 10⁻⁴ e tre epoche per fase. Selezionano al massimo 20.000 coppie per il training e 3.000 per la validazione, con random_state=42 per il campionamento. Le dimensioni dei CSV prima del campionamento sono 32.432 coppie di training, 4.844 di validazione e 10.489 di test. I checkpoint vengono selezionati in base alla loss di validazione.

La configurazione locale non usa i tre pesi λrep=2, λpred=1 e λact=3 del paper, perché le loss vengono applicate in due fasi separate e non è presente la previsione delle azioni. Gli script non definiscono il warm-up lineare o il cosine decay del paper: AdamW usa il learning rate impostato a 10⁻⁴.

Indicando con f l'encoder e con g il predittore, la seconda fase produce ẑ(t+5) = g(f(I(t))) e minimizza 1 − cos(ẑ(t+5), f(I(t+5))). Durante questa fase f rimane congelato.

Questi dettagli descrivono il codice disponibile: in questa verifica sono stati riutilizzati i checkpoint esistenti, senza ripetere l'addestramento. Gli script non salvano un log completo di durata, epoche effettivamente completate e hardware di training nei checkpoint. Il tempo della nuova valutazione non deve essere presentato come tempo di addestramento.

## Dati e protocollo di valutazione

La separazione dei dati è per video: sette video per il training (Chiang Mai, Istanbul, Kuala Lumpur, Singapore, Stockholm, Venice e Wildlife), Amsterdam per la validazione e Bangkok per il test. I video non si sovrappongono tra le partizioni. Il risultato di test riguarda quindi un solo video.

Ogni immagine del test viene decodificata completamente con la lettura permissiva delle immagini troncate disabilitata. Se uno dei due frame non è leggibile, l'intera coppia viene esclusa e registrata. Le coppie duplicate vengono escluse. Nessun errore viene compensato sostituendo la coppia con quella successiva. Ogni coppia ammessa contribuisce una sola volta alle medie, con lo stesso insieme di esempi per modello e baseline.

La baseline di persistenza usa l'embedding corrente come previsione del futuro. Il modello usa invece l'output dell'MLP. Il riferimento è sempre l'embedding del frame realmente osservato dopo cinque secondi, estratto dallo stesso encoder congelato. Questa procedura misura l'allineamento nello spazio latente: non valuta la generazione di un'immagine e non usa il retrieval Top-1/mAP del paper.

Per ridurre il calcolo, ogni frame condiviso fra più coppie viene codificato una sola volta con l'encoder in modalità eval. Gli embedding vengono memorizzati in RAM. Non vengono modificate le coppie o le metriche per effetto di questo riuso.

## Risultati della nuova valutazione completa

Controllo su tutte le 10489 coppie del CSV di test e su 10497 immagini distinte.

| Quantità | Valore |
|---|---:|
| Coppie effettivamente valutate, tutte uniche | 8701 |
| Coppie escluse | 1788 (17,05%) |
| Immagini illeggibili | 1253 |
| Tempo di controllo, codifica e valutazione | 894,2 s, circa 14,9 min |
| Dispositivo / precisione | CPU / FP32 |
| Parametri encoder | 11570496 |
| Parametri predittore | 262912 |

| Metrica | Persistenza | Predittore |
|---|---:|---:|
| Similarità coseno media (maggiore è meglio) | 0,725515 | 0,731794 |
| MSE media (minore è meglio) | 0,002144 | 0,002095 |

Differenza modello meno baseline nel coseno: **0,006279**. Quota di coppie sulle quali il modello ottiene un coseno maggiore della baseline: **50,68%**. Questa quota non è un'accuracy di classificazione o di retrieval.

Il predittore ottiene un piccolo miglioramento medio rispetto alla persistenza nel campione valutato. Non si formula una conclusione di significatività statistica: le coppie sono temporalmente correlate e provengono da un solo video. I valori precedentemente salvati nel progetto vanno sostituiti con questi nella sezione dedicata al nuovo protocollo, riportando sempre le esclusioni.

La misurazione del tempo inizia dopo il caricamento dei checkpoint e comprende decodifica delle immagini, calcolo degli embedding e confronto delle coppie; non comprende l'addestramento. PyTorch 2.12.1, batch size 16, quattro thread CPU e quattro thread per la decodifica. MPS non era disponibile nella sessione di esecuzione.

## Interpretazione delle metriche

La similarità coseno è migliore quando cresce, la MSE quando diminuisce. Poiché i vettori sono normalizzati e hanno dimensione 256, MSE = 2(1 − coseno)/256, salvo arrotondamenti numerici. Le due metriche esprimono dunque lo stesso confronto geometrico e non costituiscono due evidenze indipendenti.

## Limiti da dichiarare

- Il test considera un singolo video e coppie temporalmente correlate. Un guadagno medio non dimostra da solo significatività statistica o generalizzazione ad altri ambienti.
- L'esclusione di immagini illeggibili modifica l'insieme effettivamente valutato. Le percentuali e i motivi delle esclusioni devono essere riportati.
- L'encoder è stato addestrato per avvicinare frame temporalmente vicini: una similarità assoluta elevata non dimostra, da sola, un'accurata comprensione delle dinamiche. Il confronto con la persistenza è necessario.
- La correzione riguarda la valutazione. Gli script di training originali conservano la gestione permissiva e la sostituzione delle coppie illeggibili; non è stata condotta una nuova verifica completa dei dati di training e validazione.
- Non si tratta di una riproduzione quantitativa di EgoAgent: cambiano architettura, obiettivi, dati utilizzati e protocollo di valutazione.

## Riferimenti

Chen et al., *EgoAgent: A Joint Predictive Agent Model in Egocentric Worlds*, ICCV 2025, sezioni 3.1–3.3 e 4.2. Per l'implementazione locale, gli script stage1_train_encoder.py, stage2_train_future_predictor.py ed eval_future_prediction_strict.py nel progetto fornito.

## Paragrafo pronto per la relazione

Per verificare la previsione nello spazio latente con risorse locali, è stata valutata una pipeline ridotta ispirata a EgoAgent, composta da un encoder ResNet-18 e da un predittore MLP. Il modello riceve un frame di WalkingTours e predice l'embedding del frame a cinque secondi di distanza. La valutazione ha riutilizzato i checkpoint esistenti su un MacBook Air M2 con 8 GB di memoria, usando la CPU in FP32. Delle 10.489 coppie elencate nel test di Bangkok, 8701 sono state valutate e 1788 escluse (17,05%) dopo il controllo dei file e delle duplicazioni. Nessuna coppia esclusa è stata sostituita. La similarità coseno media è 0,731794 per il modello e 0,725515 per la baseline di persistenza, con una differenza di 0,006279. Il predittore ottiene un piccolo miglioramento medio rispetto alla persistenza nel campione valutato. Il test riguarda un unico video e non costituisce una riproduzione quantitativa dei risultati del modello completo EgoAgent.

## Testo per tre slide, circa 4 minuti complessivi

### Slide 1 — Esperimento locale ispirato a EgoAgent

- MacBook Air M2, memoria 8 GB.
- Encoder ResNet-18, embedding normalizzato di dimensione 256.
- MLP per prevedere la rappresentazione del frame a +5 secondi.
- Input solo visivo. Architettura ridotta senza InternLM o azioni 3D.

### Slide 2 — Protocollo sperimentale

- Due fasi negli script: apprendimento contrastivo dell'encoder e training del predittore con encoder congelato.
- Training, validazione e test separati per video; test su Bangkok.
- Baseline: mantenere invariato l'embedding del frame corrente.
- Valutazione rigorosa: 8701 coppie uniche ammesse, 1788 escluse, nessuna sostituzione.

### Slide 3 — Risultati e limiti

- Coseno medio: persistenza 0,7255, modello 0,7318.
- Differenza media: 0,0063.
- Il predittore ottiene un piccolo miglioramento medio rispetto alla persistenza nel campione valutato.
- Risultato su un solo video, con esclusioni dovute alla leggibilità dei dati.
- Nessuna conclusione sull'efficacia della JEAP completa o sul ruolo delle azioni.

## Come rieseguire

Con la chiavetta collegata, da Terminale:

```sh
cd /Users/gabrielladigloria/Documents/EgoAgent-1
./.venv/bin/python mini_pipeline/eval_future_prediction_strict.py
```

Per una dimostrazione breve di funzionamento, separata dalla valutazione completa:

```sh
./.venv/bin/python mini_pipeline/eval_future_prediction_strict.py --limit 32 --output mini_pipeline/evaluation_strict/demo
```

Il comando completo valuta l'intero CSV e salva results.json, per_sample.csv, excluded_pairs.csv e invalid_frames.csv in mini_pipeline/evaluation_strict. La modalità breve usa le prime 32 righe e non va impiegata per selezionare risultati favorevoli. L'opzione --device cpu forza la CPU; il valore predefinito auto seleziona MPS se disponibile.

## Fonti bibliografiche da inserire

- [Chen et al., EgoAgent, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_EgoAgent_A_Joint_Predictive_Agent_Model_in_Egocentric_Worlds_ICCV_2025_paper.html): riferimento per il modello originale e l'idea di previsione degli stati futuri.
- [He et al., Deep Residual Learning for Image Recognition, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html): riferimento teorico per ResNet, usata nell'esperimento locale.
- InternLM Team, 2023: mantenere il riferimento nella sezione teorica su EgoAgent, senza attribuirne l'uso alla pipeline locale.

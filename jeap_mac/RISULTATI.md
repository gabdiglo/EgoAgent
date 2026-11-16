# Esperimento locale JEAP con InternLM — 15 settembre 2026

## Cosa è stato eseguito

Reimplementazione ridotta della struttura JEAP, con il codice ufficiale InternLMModel inizializzato da zero, observer EMA e loss di rappresentazione, previsione dello stato e azione pesate 2:1:3. Non è l’esecuzione integrale del codice EgoAgent né una replica del benchmark originale.

- 572.915 parametri addestrabili; immagini 64 × 64; due layer, quattro teste, hidden size 128.
- Ego-Exo4D: cmu_soccer03_3 e sfu_basketball_11_44 per training; georgiatech_bike_06_8 per validazione. Selezione per dimensione dei file e disponibilità delle annotazioni, non per prestazioni.
- 82 clip di training e 128 di validazione, da take distinti. Le clip si sovrappongono temporalmente e non sono campioni indipendenti.
- 30 FPS, intervallo di tre frame (0,1 s), tre istanti osservati più un frame futuro; pose di 17 joint trasformate in coordinate della camera. Joint mancanti mascherati.
- CPU, FP32, batch 2, AdamW, learning rate 0,0001, seed 42, tre epoche (123 aggiornamenti). Tempo del ciclo training e validazione: 11,72 secondi; esclusi preparazione, caricamento iniziale e valutazione finale.
- Checkpoint scelto per loss totale minima di validazione: epoca 3. Nessun test indipendente.

## Risultati esplorativi sulla validazione

| Metrica | Modello | Persistenza |
|---|---:|---:|
| Coseno dei logits della testa DINO rispetto al futuro (maggiore è meglio) | 0,198355 | 0,998501 |
| MPJPE delle pose, cm (minore è meglio) | 28,51 | 5,06 |

384 previsioni di stato e 4.333 confronti di joint validi. Modello e baseline delle pose usano gli stessi joint visibili, escludendo il primo istante per il quale manca la posa precedente. La baseline dello stato usa l’immagine corrente; quella delle pose usa la posa precedente in coordinate della rispettiva camera. Il confronto è condizionato sulle azioni reali del contesto, non un rollout autonomo.

Il modello non supera la persistenza. Il coseno molto elevato della baseline su frame vicini non dimostra rappresentazioni semantiche di qualità e non esclude un’insufficiente discriminazione dell’observer. Non sono stati misurati Top-1/mAP né la generalizzazione su test. La loss totale di validazione passa da 17,5544 a 16,9011, ma la loss azione è migliore all’epoca 2; il teacher cambia durante il training, quindi le loss non sono una misura su target fissi.

## Testo da inserire nella relazione o nelle slide

«Per verificare l’eseguibilità del metodo su hardware personale, abbiamo implementato una versione ridotta di JEAP con il codice ufficiale InternLM, inizializzato da zero. Abbiamo effettuato un training congiunto su immagini RGB e pose 3D di Ego-Exo4D: 82 clip di training e 128 di validazione, per tre epoche su CPU. Il modello include un observer EMA e i tre obiettivi del metodo. In questa prova la persistenza ottiene risultati migliori, con MPJPE di 5,06 cm contro 28,51 cm del modello. L’esperimento verifica il funzionamento della pipeline, ma non riproduce la scala o le prestazioni di EgoAgent.»

Dedica circa 2–3 minuti a configurazione locale, tabella e limiti, dopo aver presentato i risultati originali degli autori. Mantieni la configurazione A100 del paper nella sezione dedicata agli autori e presenta questa prova come esperimento separato.

Semplificazioni: modello molto più piccolo di 300M/1B, una posa per osservazione invece di cinque, due viste globali anziché multi-crop completo, nessuna rettifica fisheye, campione minimo e addestramento brevissimo. I risultati precedenti ResNet-18 + MLP appartengono a un altro esperimento e non vanno attribuiti a InternLM/JEAP.

## Artefatti

Checkpoint: `jeap_mac/runs/joint/best.pt`. Log, configurazione, audit e metriche nella stessa cartella. Codice e istruzioni: `jeap_mac/README.md`. Manifest e frame locali: `jeap_mac/data/prepared/`.

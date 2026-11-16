# Demo per esame — mini-JEAP su Mac

Questa demo usa il checkpoint **joint reale** già presente nel progetto per produrre una figura PNG e un JSON relativi a un clip del manifest. Non riaddestra il modello e non ricrea i benchmark degli autori.

## Esecuzione

Dalla directory principale del repository:

```bash
source .venv/bin/activate
python jeap_mac/exam_demo.py
```

L'output predefinito è `jeap_mac/runs/exam_demo/visualizzazione/clip_000.png` e il corrispondente `clip_000.json`.

Si può scegliere un indice preciso in modo riproducibile:

```bash
python jeap_mac/exam_demo.py --index 5
```

L'indice 0 è la scelta predefinita **per non selezionare a posteriori il risultato migliore**. Non scegliere un esempio a partire dall'accuratezza per poi presentarlo come rappresentativo.

## Cosa mostra

1. Tre frame osservati e il quarto frame realmente acquisito dal dataset; **il quarto non è generato dal modello**.
2. Similarità coseno tra i logit DINO predetti e quelli estratti dal futuro reale tramite Observer, rispetto alla baseline di persistenza.
3. MPJPE sulle pose per il clip selezionato, sulla stessa intersezione di giunti visibili usata nella valutazione locale.
4. Confronto grafico dei giunti 3D previsti e reali sul piano XY della camera. Non è un overlay in pixel dell'immagine.

## Cosa NON dimostra

- Non è una riproduzione della valutazione **Top-1/mAP** del paper EgoAgent.
- Non è una generazione di video/fotogrammi del futuro.
- Non è un benchmark su test indipendente: il manifest predefinito è **validazione**, già usata per scegliere il checkpoint.
- Il modello è la **mini-JEAP**: 64 × 64, due layer InternLM, hidden size 128, 1 posa per osservazione; non è EgoAgent-300M/1B.
- La valutazione aggregata esistente (`jeap_mac/runs/joint/evaluation.json`) mostra che **la persistenza supera il modello**; il grafico di un singolo clip non può sostituirla.

## Presentazione orale

«Ho scelto di visualizzare la previsione dello stato futuro perché collega i frame egocentrici, le pose 3D e la rappresentazione appresa dal Transformer. La figura mostra il futuro realmente osservato come riferimento; il sistema predice un embedding, che confronto con il target dell'Observer e con una baseline di persistenza. Ho implementato una versione ridotta per verificarne la pipeline sul Mac. Non riproduco le accuratezze del paper e dichiaro i limiti dell'esperimento locale.»

## Sicurezza e dati

Questo repository pubblico contiene file immagine e manifest derivati da Ego-Exo4D; **verificare i termini di accesso e redistribuzione del dataset** prima di condividerlo pubblicamente. Se necessario, rimuovere i file dal repository pubblico e dalla cronologia Git, non solo dal tracking corrente. Non pubblicare credenziali AWS, token o file `.env`. Anche i file `.DS_Store` presenti nel repository possono essere rimossi in una pulizia separata.

Per il contesto e le metriche generali, leggere `jeap_mac/README.md` e `jeap_mac/RISULTATI.md`.

# Rettifica geometrica Aria fisheye → pinhole (opzionale)

La cattura RGB egocentrica di Ego-Exo4D, effettuata con camere Aria, può essere soggetta a distorsione fisheye. Nel preprocessing ufficiale EgoAgent i fotogrammi vengono rettificati usando i parametri di calibrazione del sensore per ottenere una geometria di camera pinhole.

**Questa è una correzione geometrica calibrata, non un miglioramento neurale, denoising o generazione di dettagli.**

## Implementazione aggiunta

`jeap_mac/undistort_frames.py`:
- legge i fotogrammi estratti da un video MP4 *frame-aligned* senza modificarne l'indice;
- legge la calibrazione RGB dal file VRS dello **stesso take** usando `projectaria-tools`;
- verifica che la risoluzione dei frame sia coerente con la calibrazione del sensore;
- costruisce una calibrazione lineare/pinhole alla stessa risoluzione;
- usa il warping geometrico calibrato con interpolazione bilineare;
- salva i fotogrammi rettificati senza cambiare gli ID, un confronto `before_after.jpg` e `rectification.json` con le informazioni di provenienza.

Non sostituisce i frame originali e rifiuta una cartella output non vuota.

## Dipendenza opzionale (Mac)

L'ambiente per il modello non richiede Project Aria. Per abilitare la rettifica:

```bash
source .venv/bin/activate
python -m pip install projectaria-tools
```

L'installazione dipende dalla compatibilità tra la tua versione di Python, macOS e i wheel disponibili. Non è stata verificata su questo Mac.

## Esempio completo per un singolo take

Sostituisci i tre percorsi con quelli REALI del medesimo take, senza pubblicare dati soggetti a licenza o credenziali.

```bash
# 1. Estrai i frame preservando la numerazione originale (se non già presenti).
python jeap_mac/extract_frames.py \
  --video /PATH/TO/TAKE/frame_aligned_videos/downscaled/448/aria01_214-1.mp4 \
  --output /PATH/TO/TAKE/raw_frames \
  --max-frames 1024

# 2. Rettifica pochi frame come prova, in una directory nuova.
python jeap_mac/undistort_frames.py \
  --frames /PATH/TO/TAKE/raw_frames \
  --vrs /PATH/TO/TAKE/aria01_noimagestreams.vrs \
  --output /PATH/TO/TAKE/rectified_test \
  --limit 10

open /PATH/TO/TAKE/rectified_test/before_after.jpg

# 3. Poi esegui la rettifica completa su una directory diversa:
python jeap_mac/undistort_frames.py \
  --frames /PATH/TO/TAKE/raw_frames \
  --vrs /PATH/TO/TAKE/aria01_noimagestreams.vrs \
  --output /PATH/TO/TAKE/rectified_frames
```

`--limit 10` è solo una prova: NON usare un manifest di training completo con 10 frame rettificati. Non usare `rectified_test` come dataset di training.

## Integrazione con mini-JEAP

La pipeline esistente `prepare_clips.py` accetta una directory di frame tramite `--frames`. È sufficiente passare **la directory rettificata**, mantenendo gli stessi parametri e le stesse annotazioni sincronizzate:

```bash
python jeap_mac/prepare_clips.py \
  --body /PATH/TO/TAKE/body_annotations.json \
  --camera /PATH/TO/TAKE/camera_annotations.json \
  --frames /PATH/TO/TAKE/rectified_frames \
  --take-uid TAKE_UID \
  --fps 30 \
  --output /PATH/TO/TAKE/rectified_clips.jsonl
```

Ripeti per tutti i take scelti per training e validazione. Mantieni separati i take tra train e val e aggrega i nuovi manifest. Preserva la sincronizzazione temporale: NON rinominare i file `frame_XXXXXXX` e NON ricampionare il video.

**Il vecchio checkpoint `jeap_mac/runs/joint/best.pt` è stato addestrato con immagini non rettificate.** La nuova rettifica NON cambia retroattivamente quel training e NON rende confrontabili automaticamente metriche vecchie e nuove. Per una valutazione rigorosa della scelta di preprocessing occorre:
1. costruire un nuovo dataset rettificato, conservando i medesimi take/split/parametri della versione precedente;
2. addestrare nuovamente e salvare il nuovo checkpoint in una directory diversa;
3. valutarlo con gli stessi criteri e una separazione di test indipendente (non disponibile nella prova precedente);
4. confrontare risultati e costi, indicando chiaramente l'eventuale cambiamento di dominio visivo.

Il codice non modifica `train.py`, `prepare_clips.py`, i manifest originali, le immagini originali, il checkpoint precedente o i risultati salvati.

## Verifiche

```bash
python -m unittest discover -s jeap_mac -p 'test_undistort_frames.py'
```

Il test controlla ordinamento degli ID, rifiuto di ID duplicati e di calibrazioni geometricamente incoerenti e il contratto shape/dtype. È un test simulato, non dimostra la qualità della rettifica su un VRS reale: per quello occorre verificare `before_after.jpg`, `rectification.json` e i frame con il file VRS associato.

## Nota per l'esame (IT / EN)

**IT:** «EgoAgent originale rettifica geometricamente i frame fisheye di Project Aria in una vista pinhole, usando la calibrazione intrinseca della camera. Ho aggiunto un preprocessing opzionale equivalente come procedura nel mio progetto; i risultati già salvati della mini-JEAP non includono ancora questo nuovo passaggio e devono essere distinti da eventuali esperimenti futuri con immagini rettificate.»

**EN:** “The original EgoAgent pipeline geometrically rectifies Project Aria fisheye frames into a pinhole view using camera calibration. I added an optional calibrated preprocessing step to my local project; the previously saved mini-JEAP results do not include this new step and must be kept separate from any future experiments on rectified images.”

Fonti: `dataset_preprocess/undistort_egovideos.py` e README del repository EgoAgent; API di Project Aria Tools.

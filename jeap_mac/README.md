# JEAP piccola con InternLM, per Mac

## Stato verificato

Implementazione locale della struttura JEAP descritta nel paper EgoAgent, con il **vero `InternLMModel` del codice ufficiale InternLM**, ridotto a 2 layer, hidden size 128 e 4 teste. Inizializzazione da zero. Non vengono caricati pesi linguistici.

Il repository EgoAgent indicato dall'utente è https://github.com/zju3dv/EgoAgent/blob/main/README.md. La revisione ispezionata è `99a3c4a1625689db64bda9b4c408af4e5738e53f`: il README descrive il training completo, ma il relativo albero Git non include il modello JEAP, le configurazioni joint_train e alcuni degli script citati. **Questo pacchetto è quindi una reimplementazione ridotta del metodo, non il training ufficiale EgoAgent eseguito senza modifiche.**

I due file Python in `vendor/` provengono da https://huggingface.co/internlm/internlm-7b alla revisione `96e127d08d851a88cac736a9b091dd953ae1b873`. Sono conservati senza modifiche con licenza Apache 2.0 e hash SHA-256 in `vendor/source.json`. Il nome del repository “7b” identifica la fonte del codice: non sono stati scaricati o creati pesi da 7 miliardi.

## Componenti implementati

- Proiezione convoluzionale delle immagini in patch token.
- InternLM ufficiale: RMSNorm, rotary positional embeddings, MLP gated SiLU, attenzione causale eager.
- Sequenza interlacciata per ogni istante: patch dell'immagine, action query, action token, state query.
- Action query prima dell'azione corrente, per impedire che il target trapeli nella previsione.
- State query dopo l'azione corrente, per consentire il condizionamento della previsione dello stato.
- Pose corporee di 17 joint in coordinate della camera, in metri. Mask dei joint mancanti.
- Observer che vede soltanto immagini, senza gradiente, aggiornato con EMA delle componenti visive condivise.
- Loss di rappresentazione DINO su due viste, loss DINO per lo stato futuro e loss L1 per le pose, pesate 2, 1 e 3.
- Centratura e temperatura dei target, normalizzazione del bottleneck e proiezione finale con weight normalization.

Il predictor richiede 572.915 parametri addestrabili; il totale incluso l'observer è 1.118.259. L'attenzione non richiede CUDA né FlashAttention. CPU è il dispositivo predefinito verificato; l'opzione MPS è disponibile nel training ma non è stata verificata nella sessione Codex.

## Semplificazioni rispetto al paper

Immagini 64 × 64, patch 16 × 16, due blocchi Transformer, 256 prototipi nella testa DINO e soltanto due viste globali. Non viene riprodotta la configurazione da 300M/1B, il multi-crop completo, la durata del training o il benchmark originale. Ogni azione contiene una posa 3D per osservazione, anziché un gruppo di cinque pose. La preparazione usa frame-aligned RGB senza implementare la rettifica fisheye del paper: la scelta deve essere dichiarata. Il backbone e la struttura causale sono reali, ma la configurazione è sperimentale.

## Prove già eseguite

`test_model.py` verifica su input sintetici: causalità delle query, dipendenza dello stato dall'azione, assenza di informazione dai frame futuri, stop-gradient dell'observer, gradienti del backbone e dei moduli d'azione, formula EMA e caricamento dei checkpoint.

`test_data.py` verifica, sempre su una fixture sintetica separata: preparazione dei manifest sincronizzati, trasformazione world-to-camera, caricamento delle pose, una fase di training con tutte e tre le loss, checkpoint ed esecuzione della valutazione.

`runs/visual_probe/` contiene una prova reale su 16 coppie di immagini WalkingTours del training set: 20 passi con obiettivi visivi, senza azioni. **Il checkpoint è marcato visual-only; non è un modello JEAP addestrato congiuntamente e non va usato per dichiarare prestazioni su pose o stati futuri.** `evaluate.py` lo rifiuta esplicitamente.

Il 15 settembre 2026 è stato completato un training congiunto su RGB e pose reali: 82 clip di training, 128 di validazione, tre epoche su CPU. Risultati e limiti sono in `RISULTATI.md`; checkpoint e log in `runs/joint/`. I manifest portabili e i fotogrammi locali sono in `data/prepared/`.

## Ripristinare l'accesso al dataset

La documentazione ufficiale indica che le credenziali durano 14 giorni e possono essere rinnovate tramite la richiesta di accesso: https://docs.ego-exo4d-data.org/getting-started/ e https://ego4d.dev/request/ego-exo4d . L'utente deve gestire personalmente l'accordo e il rinnovo.

Una volta ricevute credenziali valide, configurarle nel Terminale senza incollarle nella chat:

```sh
aws configure --profile egoexo
```

## Eseguire i test sul Mac

Dalla cartella del progetto:

```sh
./.venv/bin/python jeap_mac/test_model.py
./.venv/bin/python jeap_mac/test_data.py
```

L'ambiente esistente contiene già le dipendenze usate dai test. `ffmpeg` e `ffprobe` servono soltanto per l'estrazione delle immagini; boto3 serve per il download.

## Scaricare solo pochi video

Prima ispezionare il piano. Il selettore cerca due take di training e una di validazione, con RGB egocentrico downscaled a 448 pixel e annotazioni body/camera locali leggibili. Impone un limite complessivo di 500 MB. Non scarica l'intero dataset. Non esclude un file solo per ridurre artificialmente le loss.

```sh
./.venv/bin/python jeap_mac/download_subset.py --annotations '/Volumes/BIWIN UFD/EgoExo4D/annotations'
```

Per scaricare gli stessi tipi di file con accesso autorizzato:

```sh
./.venv/bin/python jeap_mac/download_subset.py --annotations '/Volumes/BIWIN UFD/EgoExo4D/annotations' --download
```

La selezione usa le annotazioni esistenti per scegliere i take, non il takes.json locale danneggiato. Il download è stato completato dopo la correzione delle credenziali; i video Aria RGB sono riconosciuti dal suffisso `_214-1.mp4`, perché il campo views del catalogo è nullo. Se nessun candidato soddisfa i requisiti, lo script si ferma con un errore esplicito. Dopo il download consultare `data/egoexo/download_plan.json` per UID e percorsi effettivi.

## Preparare clip sincronizzate

Estrarre le immagini da ciascun MP4 frame-aligned con `extract_frames.py`. Non modificare il frame rate: il numero del file deve corrispondere all'indice del frame nel video allineato e alle chiavi delle annotazioni. Lo script usa numerazione a partire da zero e salva i dati di ffprobe. Esempio schematico, da compilare con i percorsi del piano:

```sh
./.venv/bin/python jeap_mac/extract_frames.py --video PERCORSO_MP4 --output CARTELLA_FRAME --max-frames 1024
./.venv/bin/python jeap_mac/prepare_clips.py --body ANNOTAZIONE_BODY_JSON --camera ANNOTAZIONE_CAMERA_JSON --frames CARTELLA_FRAME --take-uid UID --fps FPS_VERIFICATO --output MANIFEST_JSONL
```

Le azioni si ricavano dalle annotazioni 3D dello stesso take e dello stesso frame. Si applica la trasformazione world-to-camera delle annotazioni ufficiali. I joint mancanti sono mascherati nella loss e impostati a zero nell'input. Gli errori non vengono sostituiti con altri campioni. Usare take distinti per train e val; il loader lo verifica.

Le impostazioni predefinite sono tre istanti di input più un'immagine futura e gap di tre frame. A 30 FPS il passo è 0,1 secondi. Se le annotazioni hanno un altro campionamento, impostare `--gap` coerentemente e verificare l'esito. Questi intervalli sono una semplificazione dichiarata della configurazione del paper.

## Addestramento e valutazione congiunti

```sh
./.venv/bin/python jeap_mac/train.py --train TRAIN_JSONL --val VAL_JSONL --epochs 3
./.venv/bin/python jeap_mac/evaluate.py --manifest VAL_JSONL --checkpoint jeap_mac/runs/joint/best.pt --output jeap_mac/runs/joint/evaluation.json
```

Questi sono esempi con segnaposto. Il training rifiuta manifest assenti, immagini illeggibili, pose non allineate e take sovrapposti tra train e val. L'optimizer è AdamW con learning rate 10⁻⁴, clipping a 1 e batch size 2. Le pose sono indispensabili per il training congiunto; WalkingTours da solo non le fornisce.

La valutazione locale riporta coseno tra output delle teste DINO, confronto con persistenza dello stato e MPJPE delle pose, in centimetri, rispetto alla posa precedente. Per il confronto delle pose si esclude il primo istante e si usa la stessa intersezione dei joint visibili per modello e baseline. Non è il benchmark retrieval Top-1/mAP del paper. I risultati su val usata per selezionare il checkpoint sono esplorativi: per misurare generalizzazione occorre un test separato.

## Cosa dire nella presentazione, allo stato attuale

«Abbiamo eseguito sul Mac un breve addestramento congiunto di una reimplementazione ridotta di JEAP, con il codice ufficiale InternLM inizializzato da zero. Usiamo RGB e pose 3D sincronizzati di Ego-Exo4D, un observer EMA e tre obiettivi di apprendimento. Su questo piccolo campione il modello non supera la persistenza: l’esperimento dimostra l’eseguibilità della pipeline, non replica le prestazioni del paper.»

I risultati 0,731794 contro 0,725515 appartengono alla precedente pipeline ResNet-18 + MLP. Non attribuirli a questa implementazione InternLM/JEAP. Anche i valori delle loss sintetiche e la prova visual-only non sono risultati scientifici della JEAP.

Fonti: Chen et al., EgoAgent, ICCV 2025, §3.1–3.2; repository ufficiale EgoAgent indicato sopra; codice ufficiale InternLM indicato in vendor/source.json; DINO https://github.com/facebookresearch/dino ; downloader ufficiale https://docs.ego-exo4d-data.org/download/ .

## Riprodurre la prova reale completata

```sh
./.venv/bin/python jeap_mac/train.py --train jeap_mac/data/prepared/train.jsonl --val jeap_mac/data/prepared/val.jsonl --epochs 3 --output jeap_mac/runs/reproduction
./.venv/bin/python jeap_mac/evaluate.py --manifest jeap_mac/data/prepared/val.jsonl --checkpoint jeap_mac/runs/joint/best.pt --output jeap_mac/runs/joint/evaluation.json
```

I dati locali restano soggetti alla licenza Ego-Exo4D; il pacchetto con i fotogrammi non è un artefatto pubblico.

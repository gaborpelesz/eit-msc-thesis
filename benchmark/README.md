# Assessing the State of the Art

## Docker

Build docker image
```bash
docker buildx build -t sota-mvs .
```

Run docker container
```bash
docker run --rm --gpus all -it sota-mvs:latest
```

## Evaluation

```bash
eval-cli --datasets courtyard --methods ACMH --width 1300 2>&1 | tee evaluation.log
```

## Layout

- [ETH3D dataset downloader](./eth3d)
- evaluation: `src/eval.py`

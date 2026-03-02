# CS 199 Project Repository
Aisha Go
Paolo Lapira

## Project Structure
```
├── PCA
│   ├── Dockerfile
│   ├── data
│   ├── docker-compose.yml
│   ├── requirements.txt
│   └── src
├── README.md
├── classification
│   ├── classification_results.csv
│   ├── classifier.py
│   ├── image_preprocess.py
│   ├── inverted_images
│   └── models
└── image_extraction
    ├── extract_images.py
    ├── output_pngs
    └── requirements.txt
```

## How to run

#### Requirements
- Docker

### PCA (Docker)

```
cd PCA
docker compose up
```

Edit `INPUT_CSV` in `docker-compose.yml` to point to specific features CSV before running contianer.

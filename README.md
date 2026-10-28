# Breast Cancer Classification — MLOps Pipeline

A machine learning API that classifies breast cancer data into **benign** and **malignant** categories using a trained Scikit-learn model. The project demonstrates model serving, automated testing, containerization, and continuous integration.

> **Disclaimer:** This project is for educational purposes only. It is not intended for medical diagnosis or clinical decision-making.

## Features

- **Machine Learning:** Scikit-learn classification pipeline.
- **REST API:** FastAPI endpoints for health checks and predictions.
- **Input Validation:** Validates prediction requests using a defined input schema.
- **Automated Testing:** Pytest tests for API behavior and invalid inputs.
- **Docker:** Containerized application for consistent deployment.
- **CI/CD:** GitHub Actions workflow that runs automated API tests.

## Tech Stack

- Python
- FastAPI
- Scikit-learn
- Pandas / NumPy (if used by your implementation)
- Pytest
- Docker
- GitHub Actions

## Project Structure

```text
breast-cancer-mlops-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml
├── tests/
│   └── test_app.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── app.py
├── model.joblib
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/L0VEKUSH/CA2-GenAI.git
cd CA2-GenAI
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Run the API

```powershell
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | API home |
| `/health` | GET | Health check |
| `/predict` | POST | Predict the class for supplied measurements |
| `/docs` | GET | Interactive Swagger API documentation |

### Example prediction request

Send a JSON object containing the 30 numeric features required by the `CancerInput` schema to `POST /predict`. Use a sample from the dataset for meaningful software testing.

Example response:

```json
{
  "prediction": "benign",
  "class_id": 0,
  "class_probabilities": {
    "benign": 1,
    "malignant": 0
  },
  "note": "Educational prediction only; not for medical diagnosis."
}
```

The response above is illustrative. Actual predictions depend on the supplied measurements and model.

## Run Tests

```powershell
python -m pytest -v
```

The current test suite contains five tests covering the home endpoint, health endpoint, prediction endpoint, missing features, and invalid measurements.

## Run with Docker

Build the image:

```powershell
docker build -t breast-cancer-mlops .
```

Start a container:

```powershell
docker run --rm -d --name breast-cancer-api -p 8001:8000 breast-cancer-mlops
```

Check the health endpoint:

```powershell
Invoke-RestMethod http://localhost:8001/health
```

Open Swagger UI at `http://localhost:8001/docs`.

Stop the container when finished:

```powershell
docker stop breast-cancer-api
```

## Continuous Integration

GitHub Actions automatically runs the API test suite on the configured repository events. The latest observed workflow completed successfully.

Workflow: `.github/workflows/ci.yml`

## Model Compatibility

The serialized model was trained with Scikit-learn **1.6.1**. Use a compatible version when loading the model; changing library versions may cause compatibility warnings or unexpected behavior.

## Limitations

- Predictions are educational and are not clinically validated.
- Model outputs depend on the input features and the quality of the training data.
- A model probability is not a guarantee of certainty.
- Clinical decisions must be made by qualified healthcare professionals using validated diagnostic methods.

## Author

**Lovekush Kumar**

GitHub: [L0VEKUSH](https://github.com/L0VEKUSH)

# NLP Abbreviation Detection Web Service

## Project Overview
This project is part of the Natural Language Processing (NLP) coursework for COMM061 at the University of Surrey. The goal of the project was to develop a web service capable of detecting abbreviations in scientific texts using machine learning models. After extensive experimentation with various models, **BERT** was selected as the best-performing model for the task.

The web service is implemented using **FastAPI**, enabling high-performance asynchronous request handling. The project also incorporates **logging, monitoring, CI/CD pipeline**, and **stress testing** to ensure robustness and scalability.

---

## Features
- **Abbreviation Detection**: Identifies abbreviations in scientific texts using a fine-tuned BERT model.
- **FastAPI-Based Web Service**: Provides an API endpoint to submit text for analysis.
- **Logging and Monitoring**: Logs user requests and model predictions for debugging and analysis.
- **Continuous Integration/Continuous Deployment (CI/CD)**: Automates model training, validation, and deployment.
- **Stress Testing**: Evaluates service performance under high loads.

---

## Model Selection and Experimentation
Each team member conducted independent experiments to evaluate different models and preprocessing techniques. Key findings include:

- **BERT consistently outperformed** SVM, BiLSTM, and Transformer-based models.
- **SMOTE improved recall** for minority classes.
- **Bayesian optimization fine-tuned BERT** to an F1-score of **0.95**.
- **FastAPI was chosen** for deployment due to its high performance and scalability.

---

## Technologies Used
- **Programming Language**: Python
- **Libraries & Frameworks**:
  - Transformers (Hugging Face)
  - FastAPI
  - PyTorch
  - TensorFlow
  - Scikit-learn
  - Locust (for stress testing)
- **Deployment & Logging**:
  - FastAPI for API development
  - Python logging module for monitoring
  - CI/CD pipeline for automation

---

## Installation
### Prerequisites
Ensure you have **Python 3.8+** installed on your machine.

### Steps
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the FastAPI server:
   ```bash
   uvicorn scripts.api:app --host 0.0.0.0 --port 8000
   ```
5. Access the API at:
   - **Swagger UI**: `http://localhost:8000/docs`
   - **Redoc**: `http://localhost:8000/redoc`

---

## Usage
### Making Predictions
Send a POST request to the `/predict` endpoint with a JSON payload:
```json
{
  "text": "We developed a variant of gene set enrichment analysis (GSEA) to determine..."
}
```
Response:
```json
{
  "tokens": ["We", "developed", "a", "variant", "of", "gene", "set", "enrichment", "analysis", "(", "GSEA", ")"],
  "labels": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-AC", "O"],
  "confidence_scores": [0.95, 0.98, 0.99, 0.96, 0.97, 0.99, 0.94, 0.98, 0.99, 0.92, 0.97, 0.91]
}
```

---

## Performance Evaluation
A **stress test** was conducted using **Locust**, simulating up to **10,000 concurrent users**. Key results:
- **Average response time**: 14,328 ms
- **99th percentile response time**: 46,000 ms
- **Maximum response time**: 57,959 ms
- **No request failures**, proving service stability

### Recommendations for Optimization:
- **Model Optimization**: Apply quantization, pruning, or distillation to improve efficiency.
- **Infrastructure Scaling**: Implement horizontal scaling using Kubernetes.
- **Asynchronous Processing**: Optimize API request handling to reduce delays.
- **Caching**: Implement caching for frequently queried responses.

---

## Monitoring and Logging
Logging is enabled using Python’s `logging` module. Logs are stored in `logs/service.log`, capturing:
- User inputs
- Model predictions
- Errors and warnings

---

## Continuous Integration/Continuous Deployment (CI/CD)
The CI/CD pipeline automates model training and deployment by:
1. Loading new training data
2. Retraining BERT and evaluating performance
3. Saving the best model
4. Updating the configuration file with the latest model path

---

## Team Members
- **Saksham Ashwini Rai**
- **Raj Vinod Mistry**
- **Suhas Trimbak Barapatre**
- **Aditya Narayan Sawant**
- **Kunal Vinayshankar Singh**

---

## Additional Resources
- **Project Report**: Available in the repository for in-depth details.

---

## License
This project is for educational purposes and is licensed under **MIT License**.

---


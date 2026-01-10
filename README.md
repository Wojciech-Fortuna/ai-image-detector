# AI Image Detector

Web-based application for detecting AI-generated images using multiple analysis methods.

The project consists of a backend API with batch CLI support and a web frontend.

---

## Project Structure

```text
.
├── backend/     # Backend API and batch CLI
├── frontend/    # Web frontend application
├── scripts/     # Non-production scripts for training, data generation, evaluation and utility tasks
│
├── LICENSE
├── .gitignore
└── README.md
```

- **backend/** – production backend responsible for inference and offline batch analysis  
- **frontend/** – web-based user interface  
- **scripts/** – non-production scripts for training, data generation, evaluation and utility tasks

---

## Backend

Navigate to the `backend` directory and run:

```bash
docker build -t ai-image-detector-backend .
docker compose up
```

The backend will be available at:

```
http://localhost:8000
```

> Run `docker build` **only on the first startup**  

---

## Frontend

Navigate to the `frontend` directory and run:

```bash
npm install
npm run dev
```

The frontend will be available at:

```
http://localhost:5173
```

> Run `npm install` **only on the first startup**  

---

## Stopping the application

To stop the application:

- **Backend (Docker)** – press:
  ```bash
  CTRL + C
  ```

- **Frontend (npm dev server)** – press:
  ```bash
  CTRL + C
  ```

If you want to remove the Docker containers, run:

```bash
docker compose down
```

---

## Batch CLI

Batch CLI is used for offline analysis of image folders.

### Usage

Navigate to the `backend` directory and run:

```bash
python app.py <input_dir> --out <output_dir>
```

Example:

```bash
python app.py ./images --out ./out
```

### Optional arguments

```bash
--threshold 0.5
```
Sets the AI / REAL decision threshold (default: `0.5`).

```bash
--methods ela,fft,attrib_generator
```
Comma-separated list of methods to run  
(default: `combined_methods`).

---

## Testing

### Backend tests

To run backend tests locally, navigate to the `backend/tests` directory and run:

```bash
pytest -q
```

All tests should pass:

```
50 passed
```

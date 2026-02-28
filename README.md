Neural Movie Clusters
Αλγόριθμοι Σύστασης με Χρήση Τεχνητών Νευρωνικών Δικτύων και Τεχνικών Ομαδοποίησης Δεδομένων
This project explores movie recommendation algorithms using Neural Networks and Clustering techniques on IMDB datasets.

---

🛠 Setup Instructions
For Conda Users (Recommended)

This method works on Windows, macOS, and Linux. Conda will automatically handle OS-specific dependencies.

1. Create the environment:

```

conda env create -f environment.yml

```

2. Activate the environment:

```

conda activate movie_ml

```


For Non-Conda Users (Pip)

If you are using a standard Python virtual environment (`venv`).

1. Create and Activate Virtual Env:

• Windows:

```

python -m venv venv

.\venv\Scripts\activate

```

• macOS / Linux:

```

python3 -m venv venv

source venv/bin/activate

```


2. Install Dependencies:

```

pip install -r requirements.txt

```

Note: If you encounter errors with system-specific packages (like MKL), try installing the main libraries manually: `pip install numpy pandas scikit-learn matplotlib seaborn torch`.
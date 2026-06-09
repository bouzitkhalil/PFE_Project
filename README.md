# Smart Workplace Occupancy & Employee Flow Analytics Platform

Plateforme web locale Flask pour la detection d'entrees/sorties, la reconnaissance faciale par webcam, l'analyse d'occupation et la generation de rapports.

## Clonage du dépôt

```powershell
git clone https://github.com/bouzitkhalil/PFE_Project.git
cd PFE_Project
```


## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Puis ouvrir :

```text
http://127.0.0.1:5000
```

## Compte administrateur

- Email : `admin@smartwork.local`
- Mot de passe : `admin123`

Changez ce mot de passe dans une vraie utilisation.

## Fonctionnement

- La webcam integree du PC est utilisee via OpenCV.
- Les photos de reference des employes sont encodees avec `face_recognition`.
- Les presences sont stockees dans une base SQLite persistante.
- Les doublons de detection continue sont limites par une fenetre anti-spam configurable.
- Les rapports PDF et Excel sont generes depuis l'historique de presence.

## Structure

- `app.py` : application Flask, base de donnees, routes, video streaming, rapports.
- `templates/` : pages HTML Bootstrap.
- `static/css/style.css` : interface dashboard premium.
- `static/js/dashboard.js` : graphiques Chart.js.
- `data/smartwork.db` : base SQLite creee automatiquement.
- `uploads/employees/` : photos de reference employees.

# 🍏 FWIS: Food Waste Intelligence System

An AI-powered inbound shipment prediction and logistics optimization engine. FWIS helps warehouse operators and supply chain managers automatically assess the quality of inbound produce shipments, predict batch waste percentages, calculate remaining shelf life (RSL), and instantly route stock using intelligent action directives.

---

## 🚀 System Architecture

FWIS is split into a decoupled, fast, and scalable two-part architecture:
1. **Backend (FastAPI):** A high-performance Python API that runs predictive machine learning models in real-time.
2. **Frontend (Streamlit):** A clean, user-friendly dashboard for manual manifests, letting operators run diagnostics instantly.

---

## 📦 Project Structure

```text
Fwis-Fastapi/
│
├── backend/
│   ├── main.py            # FastAPI backend server
│   ├── model_waste.pkl    # Regressor for batch waste %
│   ├── model_rsl.pkl      # Regressor for remaining shelf life (days)
│   ├── model_reason.pkl   # Classifier for primary degradation risk
│   └── columns_layout.pkl # Training schema column alignment mapping
│
├── frontend/
│   └── app.py             # Streamlit dashboard interface
│
├── requirements.txt       # Unified Python dependencies
└── README.md              # Documentation

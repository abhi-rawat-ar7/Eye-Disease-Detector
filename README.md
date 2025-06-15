Steps to run project :-

Step 1: Navigate to backend :-
cd backend/

Create Virtual Environment:
python -m venv venv

Activate Virtual Environment:
Windows:
source .\venv\Scripts\activate

macOS/Linux:
source venv/bin/activate

Step 2: Run the Backend
Open Terminal: Ensure your terminal in VS Code is still running in the backend directory and your virtual environment is activated.

Run Uvicorn:
uvicorn main:app --reload

Step 3: Run Streamlit App:
Open new terminal, run below command :-
streamlit run streamlit_app.py
This will open a new tab in your web browser, typically at http://localhost:8501.

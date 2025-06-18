from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from rag_pipeline import add_to_vector_db, query_vector_db
from pdf_load import extract_text_from_pdf
import uuid
import os, logging
import ollama

app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def process_pdf_sync(file_path: str, task_id: str):
    logging.info(f"Task {task_id}: Started processing {file_path}")
    try:
        text = extract_text_from_pdf(file_path)
        logging.info(f"Task {task_id}: Extracted text from PDF")

        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        ids = [str(uuid.uuid4()) for _ in chunks]
        add_to_vector_db(chunks, ids)

        logging.info(f"Task {task_id}: Successfully added chunks to vector DB")
    except Exception as e:
        logging.error(f"Task {task_id}: Failed with error: {e}")

@app.post("/upload-pdf/")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_path = f"./data/{file.filename}"
    os.makedirs("./data", exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    task_id = str(uuid.uuid4())
    logging.info(f"Received upload, assigned task_id: {task_id}")

    background_tasks.add_task(process_pdf_sync, file_path, task_id)

    return {"message": f"{file.filename} is being processed.", "task_id": task_id}

@app.get("/ask/")
async def ask_question(q: str):
    context_docs = query_vector_db(q)
    context = "\n".join(context_docs)

    prompt = f"Answer the question using the context below.\n\nContext:\n{context}\n\nQuestion: {q}"

    response = ollama.chat(model="llama2-uncensored:7b", messages=[
        {"role": "user", "content": prompt}
    ])

    return {"response": response['message']['content']}

@app.get("/ping")
async def ping():
    return {"status": "pong",}
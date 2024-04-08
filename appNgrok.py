from fastapi import FastAPI, Depends, HTTPException, status, Request, Form, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from starlette.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from passlib.context import CryptContext
import uvicorn
import json
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
import re
import torch
import warnings
warnings.filterwarnings('ignore') 
import nest_asyncio
from pyngrok import ngrok
from transformers import BitsAndBytesConfig

##### COSTANTI E VARIABILI #####
DB_FAISS_PATH = 'vectorstore/db_faiss'
chat_history = []

pre_prompt = """
[INST] <<SYS>>

Sei un veterinario molto bravo e preparato, pronto ad aiutare e comprendere meglio i problemi degli animali domestici.
Sei specializzato però in cani e gatti, quindi se ti viene richiesto qualcosa su altri animali, specifica che sei specializzato in cani e gatti
Aiuta chi ti fa le domande, risultando amichevole e socievole.
Spiega con parole semplici gli argomenti, in modo tale che le persone non avvezze al campo della veterinaria e della medicina, possano capire facilmente.
Dai risposte esaustive e ben argomentate.
Se una domanda sembra incoerente o priva di senso spiega che non è consona invece di dare una risposta errata.
Se invece una domanda è semplicemente discorsiva, cerca di mantenere il discorso.

Rispondi alle domande riguardanti i problemi degli animali, trovando risposta nei documenti forniti. 
I documenti sono prettamente specifici, tendi ad astrarre l'argomento.
Semplifica sempre le informazioni per chi non è esperto del settore.
Se la risposta proviene da documenti diversi, menziona tutte le possibilità ed utilizza i titoli per separare gli argomenti o i domini.

<</SYS>>

"""

MODEL_NAME = "swap-uniba/LLaMAntino-2-chat-7b-hf-UltraChat-ITA"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, device_map = 'cuda')
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_8bit=True, device_map = 'cuda')

generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
generation_config.max_new_tokens = 1024
generation_config.temperature = 0.0001
generation_config.top_p = 0.95
generation_config.do_sample = True
generation_config.repetition_penalty = 1.15

text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    generation_config=generation_config,
)



##### ATTIVAZIONE FASTAPI E GESTIONE TEMPLATES #####
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(SessionMiddleware, secret_key="your_secret_key")


##### GESTIONE UTENTI E VERIFICA PASSWD #####

fake_users_db = []
# classe utente
class User:
    def __init__(self, username: str, hashed_password: str):
        self.username = username
        self.hashed_password = hashed_password

# Gestione hashing password per salvataggio
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
hashed_password_user1 = pwd_context.hash("admin@123")
hashed_password_user2 = pwd_context.hash("user@123")

# Creazione utenti
fake_users_db.append(User(username="admin", hashed_password=hashed_password_user1))
fake_users_db.append(User(username="user", hashed_password=hashed_password_user2))

# Funzione che verifica la password inserita
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Funzione per ottenere un utente, dato l'username
def get_user(username: str):
    for user in fake_users_db:
        if user.username == username:
            return user



##### GESTIONE CHATBOT #####
# Recupero Chain
def retrieval_qa_chain(prompt, db):
    
    llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return qa_chain

# Creazione prompt template
def set_custom_prompt():
    prompt_template = pre_prompt + "CONTEXT:\n\n{context}\n" +"Question : {question}" + "[\INST]"

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return prompt

# Creazione Chain
def build_chain_custom():
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large",
                                       model_kwargs={'device': 'cuda'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(qa_prompt, db)
    return qa

# Avvio Chain
def run_chain(chain, prompt: str):    
    print("Chat History:")
    print(chat_history)
    return chain({"query": prompt, "chat_history": chat_history})

# Formattazione risposta
def process_answer(answer, prompt: str):
    final = answer['result'].split('[\\INST]')
    final_answer = final[1]
    chat_history.append((prompt, final_answer))
    return final_answer

##### CREAZIONE CHAIN LANGCHAIN #####
chain = build_chain_custom()

##### ENDPOINT LOGIN #####
@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    user = get_user(username)
    if user is None or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Username o password errati",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Salvataggio utente in sessione
    request.session['user'] = user.username
    response_data = jsonable_encoder(json.dumps({"msg": "Success",}))
    res = Response(response_data)
    return res

##### ENDPOINT LOGOUT #####
@app.get("/logout")
async def logout(request: Request):
    # Rimozione utente dalla sessione
    request.session.pop('user', None)
    return RedirectResponse(url="/")

##### ENDPOINT CHAT (PROTETTO AI LOGGATI) #####
@app.get("/chat")
async def chat(request: Request):
    # Controlla se l'utente è loggato tramite sessione
    user = request.session.get('user')
    if user is None:
        # Redirect al login se non loggato
        return RedirectResponse(url="/")
    return templates.TemplateResponse("chat.html", {"request": request, "user": user})

##### ENDPOINT INIZIALE (PROTETTO AI LOGGATI) #####
@app.get("/")
async def read_root(request: Request):
    # Controllo se l'utente è loggato
    user = request.session.get('user')
    if user is None:
        # Mostro pagina login se l'utente non è loggato
        return templates.TemplateResponse("login.html", {"request": request})
    else:
        # Faccio il reindirizzamento alle chat se l'utente è loggato
        return RedirectResponse(url="/chat")

##### ENDPOINT RISPOSTA DOMANDA #####
@app.post("/chat_response")
async def chat_resonse(request: Request, prompt: str = Form(...)):
    result = run_chain(chain=chain, prompt=prompt)
    print("Result:")
    print(result)
    answer = process_answer(result, prompt=prompt)
    source_documents = result['source_documents']
    source_documents_list = []
    page_number_list = []
    for doc in source_documents:
        source_doc = doc.metadata['source']
        page_number = doc.metadata['page']
        if source_doc not in source_documents_list:
            source_documents_list.append(source_doc)
            page_number_list.append(page_number)

    response_data = jsonable_encoder(json.dumps({"answer": answer, "source_documents_list": source_documents_list, "page_number_list": page_number_list}))
    res = Response(response_data)
    return res

if __name__ == "__main__":
  # Get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
  auth_token = "2b2Cx2IiSD458IgPw7Elkg00eKL_2jo9noNx56D7h5F1V1wcH"

  # Set the authtoken
  ngrok.set_auth_token(auth_token)

  # Connect to ngrok
  ngrok_tunnel = ngrok.connect(8000)

  # Print the public URL
  print('Public URL:', ngrok_tunnel.public_url)

  # Apply nest_asyncio
  nest_asyncio.apply()

  # Run the uvicorn server
  #uvicorn.run("app:app", port=8000, reload=True)
  uvicorn.run("app:app", port=8000)

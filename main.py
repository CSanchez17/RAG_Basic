import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Configuración del Modelo y Embeddings (Local con Ollama)
model_name = "deepseek-r1:32b"
embeddings = OllamaEmbeddings(model=model_name)
llm = ChatOllama(model=model_name)

CHROMA_DIR = "./chroma_db"
DATA_DIR = r'C:\Projects\InfoRetrieval\data'

# 2. Carga de documentos desde la carpeta
print("Cargando documentos...")
loader = DirectoryLoader(DATA_DIR, glob="./*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

# 3. División de texto (Chunking)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 4. Base de Datos Vectorial persistente (crea o actualiza)
if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
    print("Cargando base de datos vectorial existente...")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    # Detectar qué fuentes ya están indexadas
    existing_sources = set(
        m["source"] for m in vectorstore.get()["metadatas"] if m and "source" in m
    )
    new_splits = [s for s in splits if s.metadata.get("source") not in existing_sources]

    if new_splits:
        print(f"Añadiendo {len(new_splits)} fragmentos nuevos a la base de datos...")
        vectorstore.add_documents(new_splits)
    else:
        print("No hay documentos nuevos. Usando la base de datos existente.")
else:
    print(f"Creando nueva base de datos vectorial con {len(splits)} fragmentos...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

# 5. Configuración de la cadena de consulta (LCEL)
# Aumentamos 'k' para traer más fragmentos (por ejemplo, 10 o 15)
# 'fetch_k' ayuda a que Chroma busque en un pool más grande antes de filtrar
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)

# Refinamos el prompt para que el modelo sepa que debe hacer un análisis integral
prompt = ChatPromptTemplate.from_template("""
Eres un analista experto. Utiliza TODO el contexto proporcionado para generar una respuesta exhaustiva y detallada. 
Si el contexto abarca diferentes documentos, sintetiza la información de todos ellos.

Contexto:
{context}

Pregunta: {question}

Respuesta analítica:
""")

def format_docs(docs):
    # Añadimos una marca visual para que el modelo identifique dónde termina un fragmento y empieza otro
    return "\n--- NUEVO FRAGMENTO/DOCUMENTO ---\n".join(doc.page_content for doc in docs)

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Ejecución de la consulta
pregunta = "¿Cuál es el tema principal de los documentos cargados?"
print(f"\nPregunta: {pregunta}")

respuesta = qa_chain.invoke(pregunta)
print("\nRespuesta del LLM:")
print(respuesta)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from FlagEmbedding import BGEM3FlagModel
import uvicorn
import torch
import os
from contextlib import asynccontextmanager

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
def load_model():
    # Load BGE M3 Model at startup
    print('\nLoad BGE M3 Model ...')
    model_bgem3 = BGEM3FlagModel(
        'BAAI/bge-m3',
        use_fp16=True,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
        device='cuda:0'

    )
    # model_bgem3.model.to('cuda:0')

    return model_bgem3

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["answer_to_everything"] = load_model()
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

# Initialize FastAPI app
app = FastAPI(title="BGE M3 Score API", lifespan=lifespan , description="API to compute BGE M3 similarity scores between two sentences")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)








# Define request body structure using Pydantic
class SentencePair(BaseModel):
    sentence_1: str
    sentence_2: str

# Define the scoring function
def compute_bge_score(sentences_1: str, sentences_2: str, model_bgem3):
    sentence_pairs = [[sentences_1, sentences_2]]
    with torch.no_grad():  # Tắt gradient để tiết kiệm bộ nhớ
        bge_score = model_bgem3.compute_score(sentence_pairs, 
                                            max_passage_length=128,
                                            weights_for_different_modes=[1, 0.3, 1],
                                            )
    return bge_score['colbert+sparse+dense'][0]

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the BGE M3 Score API. Use POST /compute-score/ to calculate similarity scores."}

# Endpoint to compute BGE score
@app.post("/compute-score/")
async def get_bge_score(pair: SentencePair):
    try:
        # Extract sentences from request body
        sentence_1 = pair.sentence_1
        sentence_2 = pair.sentence_2
        
        # Validate input
        if not sentence_1 or not sentence_2:
            raise HTTPException(status_code=400, detail="Both sentences must be non-empty")
        
        # Compute score
        score = compute_bge_score(sentence_1, sentence_2, ml_models["answer_to_everything"])
        
        # Return result
        return {'score' : score}

    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing score: {str(e)}")

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
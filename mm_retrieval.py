import torch
from src.ir import Retriever

# Define a query and a list of passages
query = "Curiosity was launched to explore Mars"
images = [
    "/data/chenyuan/reproduce/VDR-main/examples/images/mars.png",
    "/data/chenyuan/reproduce/VDR-main/examples/images/moto.png"
]

# Initialize the retriever
vdr_cm = Retriever.from_pretrained("vsearch/vdr-cross-modal")
vdr_cm = vdr_cm.to("cuda")

# Embed the query and passages
q_emb = vdr_cm.encoder_q.embed(query)  # q for text; shape: [1, V]
p_emb = vdr_cm.encoder_p.embed(images)  # p for image; shape: [4, V]

# Query-passage Relevance
scores = q_emb @ p_emb.t()
print(scores)

# Output: 
# tensor([[0.2700, 0.0942]], device='cuda:0')
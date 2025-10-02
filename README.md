This project was a very short research sprint I did over the course of a day as a part of the ARENA Program Hackathon. 
The aim was to create a basis of vectors for the residual stream of gpt2-xl which is directly linked to the default representations of tokens within the residual stream.
This allows us to express outputs of different attention and mlp blocks in terms of tokens, and directly see semantic meaning added to the residual stream as it goes through the network.
Create_basis.py creates and stores the basis, while main.py allows us to perform tests on the output.

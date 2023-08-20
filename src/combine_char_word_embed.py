import torch.nn as nn
import torch

from transformers import logging
logging.set_verbosity_error()

from src.char_Embedding import char_embed
from src.modified_roberta_for_charEmb import Roberta_modified_for_charEmb


class combine_char_roberta(nn.Module):

    def __init__(self, config, device="cuda:0"):
        super().__init__()
        self.config=config
        self.device = device
        
        self.char_embed_combine=self.config.use_char_embed
        if self.char_embed_combine==True:

            print("char embedding initialized")

            self.in_features=768+config.char_hidden_size    
            self.out_features=768
            self.combine_layer=nn.Linear(self.in_features,self.out_features).to(device)
            self.char_embedding=char_embed(self.config,device=self.device).to(device)

        self.rob_embedding=Roberta_modified_for_charEmb(self.config).to(device)


    def combine_embed(self,sentences,nums):
     
        roberta_emb,roberta_input_len=self.rob_embedding(self.config,sentences,nums,self.device)
        # print(roberta_emb.size(),roberta_input_len)

        if self.char_embed_combine==True:
            
            char_emb,char_input_len=self.char_embedding.char_final_embed(sentences)
            input=torch.cat([roberta_emb,char_emb],dim=2).to(self.device)   
            # print("char_embeddig added") 
            return input, roberta_input_len

        return roberta_emb,roberta_input_len
        
    def forward(self,sentences,nums):
        input,input_len=self.combine_embed(sentences,nums)
        
        if self.char_embed_combine==True:
            final_embedding = self.combine_layer(input).to(self.device)
            return final_embedding,input_len
        #print(final_embedding)

        return input,input_len





# s1="Is CV num2 palakkad num3 of bbdfbbf mbfjkbnbj ?"
# s2="Maybe num1 not."
# s3="It's num1,Bro !"
# s4="just num4 "

# sentences=[s1,s2,s3,s4]
# embed=combine_char_roberta()
# temp1,temp2=embed(sentences)
#print(temp1.size(),temp2)




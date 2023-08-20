import numpy as np
import pandas as pd
import ampligraph
import tensorflow as tf
import os
import torch 
import torch.nn as nn

from ampligraph.latent_features import ScoringBasedEmbeddingModel
from ampligraph.utils import save_model
from ampligraph.utils import restore_model
from ampligraph.utils import create_tensorboard_visualizations

# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE


path=os.getcwd()+"/src/components/Custom_mawps_100/TransE_model"   # for final run 

class NumberEmbed(nn.Module):
    def __init__(self,config,path= os.getcwd()+"/src/components/TransE_model" ):
        super(NumberEmbed, self).__init__()
        self.config=config
        
        if self.config.use_layer_add_num==True:
            path=os.getcwd()+"/src/components/Custom_mawps_100/TransE_model" 
        else:
            path=os.getcwd()+"/src/components/CUSTOM_FOR_MAWPS/TransE_model"
        print("path: " , path)
        
        self.path=path 
        self.new_model = restore_model(model_name_path=path)
        

    def get_num_emb(self,number):
        
        number=str(round(float(number),5))
        embedding = self.new_model.get_embeddings([number], embedding_type='e')
        embedding = np.array([embedding])
        embedding = torch.FloatTensor(embedding)
        return embedding




# num_emb=NumberEmbed()
# res=num_emb.get_num_emb("0.0")
# print(res.shape)
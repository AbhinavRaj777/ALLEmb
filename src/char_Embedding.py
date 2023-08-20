import torch
import torch.nn as nn
import string 
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import AutoTokenizer


ls_vocab=["<c>"]+["<c_pad>"]+["</c>"]+["<s>"]+["<pad>"]+["</s>"]+["UNK_char"]+list(string.printable)

charVocab={}
Inv_charVocab={}
for idx,ele in enumerate(ls_vocab):
    charVocab[idx]=ele
    Inv_charVocab[ele]=float(idx)

len_of_char_vocab=len(charVocab)



class CharEncoder(nn.Module):

        def __init__(self, hidden_size=768,embedding_size = 512, cell_type='gru', nlayers=1, dropout=0.1, bidirectional=True):
                super(CharEncoder, self).__init__()
                self.hidden_size = hidden_size
                self.nlayers = nlayers
                self.dropout = dropout
                self.cell_type = cell_type
                self.embedding_size = embedding_size
                self.bidirectional = bidirectional

                if self.cell_type == 'lstm':
                        self.rnn = nn.LSTM(self.embedding_size, self.hidden_size,
                                                           num_layers=self.nlayers,
                                                           dropout=(0 if self.nlayers == 1 else dropout),
                                                           bidirectional=bidirectional)
         
                elif self.cell_type == 'gru':
                        self.rnn = nn.GRU(self.embedding_size, self.hidden_size,
                                                          num_layers=self.nlayers,
                                                          dropout=(0 if self.nlayers == 1 else dropout),
                                                          bidirectional=bidirectional)
                else:
                        self.rnn = nn.RNN(self.embedding_size, self.hidden_size,
                                                          num_layers=self.nlayers,
                                                          nonlinearity='tanh',                                                  # ['relu', 'tanh']
                                                          dropout=(0 if self.nlayers == 1 else dropout),
                                                          bidirectional=bidirectional)

        def forward(self, sorted_seqs, sorted_len, orig_idx, device="cuda:0" ,hidden=None):

                packed = torch.nn.utils.rnn.pack_padded_sequence(
                        sorted_seqs, sorted_len)
                outputs, hidden = self.rnn(packed, hidden)
                outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
                outputs)  # unpack (back to padded)

                outputs=outputs.to(device)
                orig_idx=orig_idx.to(device)

                #print(outputs.device,orig_idx.device)
                outputs = outputs.index_select(1, orig_idx)
                

                if self.bidirectional:
                        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs

                hidden=hidden[0,:,:]+hidden[1,:,:]  # concatinating the last hidden state of each word 
                return outputs, hidden

class char_embed(nn.Module):
    
    def __init__(self,config,device="cuda:0"):
        super().__init__()
        self.config=config
        self.roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.embed_dimension=self.config.char_embedding_size
        char_vocab_len=len(charVocab)
        self.device=device
    
        self.char_embedding = nn.Embedding(char_vocab_len,self.embed_dimension).to(device)
        self.char_encoder=CharEncoder(hidden_size=config.char_hidden_size,embedding_size = config.char_embedding_size, nlayers=config.char_nlayer, dropout=config.char_dropout).to(device)

    def remove_space_token(self,word):
        if word[0] == "Ġ":
            new_word = " " + word[1:]
            return new_word
        else:
            return word

    def Char_pre_processing(self,sentences):

        tokens=[]
        for sent in sentences:

            pre_token_with_idx=self.roberta_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sent)
            # print(pre_token_with_idx)
            pre_token=[pre_token_with_idx[i][0] for i in range(len(pre_token_with_idx))]
            # print(pre_token)
            token_for_char = ["<s>"]+[self.remove_space_token(pre_token[i]) for i in range(len(pre_token))]+["</s>"]     
            # print(token_for_char) 
            # tokens.append(token_for_char)

            temp_ls=[]
            #### combining "number" and "1" ,because it is splitted in pre-tokenize step.
            for i in range(len(token_for_char)):
                if token_for_char[i] in ["number"," number"]:
                    temp_ls.append(token_for_char[i] + token_for_char[i+1])
                elif token_for_char[i] in ["0","1","2","3","4","5","6"]:
                    pass
                else:
                    temp_ls.append(token_for_char[i])

            token_for_char=temp_ls
            tokens.append(token_for_char)

        # print(tokens)
        input_length=[len(token) for token in tokens]
        # print(input_length)

        max_length=max(input_length)
        pad_tokens=[]
        for token in tokens:
            pad_tokens.append(token + ["<pad>"]*(max_length-len(token)))

        return pad_tokens,input_length
        # print(pad_tokens)
    
    def cal_max_len_word(self,word_ls):
        max_len_word=0
        for word in word_ls:
            if word=="<s>" or word=="</s>" or word=="<pad>":
                continue
            else:
                if len(word)>max_len_word:
                    max_len_word=len(word)

        return max_len_word


    def sort_batch(self,batch_of_words_idx_form,words_len):
        orig_idx=list(range(len(words_len)))

        sorted_word_idx=sorted(orig_idx,key=lambda k: words_len[k], reverse=True) 

        # print("orig_idx",orig_idx,sorted_word_idx)
        orig_idx_retrieve= sorted(orig_idx, key=lambda k: sorted_word_idx[k])

        sorted_word_seq_vec=[]
        # print("sorted_word_idx",sorted_word_idx)
        # print("len of batch_of_words_idx_form",len(batch_of_words_idx_form))

        for idx in sorted_word_idx:
            sorted_word_seq_vec.append(batch_of_words_idx_form[idx])
        
    
        sorted_len=[]
        for i in sorted_word_idx:
            sorted_len.append(words_len[i])

        return sorted_word_seq_vec,sorted_len,orig_idx_retrieve


    def char_final_embed(self,sentences):
    
       # print("<<<<<<<<<<<<<<< Creating character Embedding >>>>>>>>>>>>>>>>>>>>>>>>>>")

        
        pad_tokens,input_length = self.Char_pre_processing(sentences)
        char_encoder_hidden_ls=[]

        for token in pad_tokens:
        # print(token)  
        
            words_batch=[]
            words_batch_idx=[]
            input_len=[]

            max_length_word=self.cal_max_len_word(token)+2  # for start and end char token 
            # input_len.append(max_length_word)
            for word in token:
                
                if word=="<s>" or word=="</s>" or word=="<pad>":
                    word=[word]
                    input_len.append(3)        

                else:
                    word=list(word) 
                    input_len.append(len(word)+2)        
                

                word=["<c>"]+word+["</c>"] + ["<c_pad>"]*(max_length_word-len(word)-2)
                # print(word) 
                # print(max_length_word)
                words_batch.append(word)
                
                temp_dict=[]
                for char in word:

                        if  char in  Inv_charVocab.keys():
                                temp_dict.append(Inv_charVocab[char])
                        else:
                                temp_dict.append(Inv_charVocab["UNK_char"])

                #words_batch_idx.append([Inv_charVocab[char] for char in word ])
                words_batch_idx.append(temp_dict)

            sorted_words_batch_idx,sorted_word_len,orig_idx_retrieve=self.sort_batch(words_batch_idx,input_len)

            # print(orig_idx_retrieve)
            # # print(len(sorted_words_batch_idx))
            sorted_words_batch_idx=Variable(torch.LongTensor(sorted_words_batch_idx))
            sorted_word_len=torch.LongTensor(sorted_word_len)
            orig_idx_retrieve=torch.LongTensor(orig_idx_retrieve)

            # print(input_len,sorted_word_len)
            # print(sorted_words_batch_idx.size())
            # print(words_batch_idx)
            sorted_words_batch_idx=sorted_words_batch_idx.to(self.device)
            sorted_word_seq_vec=self.char_embedding(sorted_words_batch_idx)

            sorted_word_seq_vec=sorted_word_seq_vec.transpose(0,1)  # max_length * batch size*embedding 
            # print(sorted_word_seq_vec.size())

            char_encoder_outputs, char_encoder_hidden = self.char_encoder(sorted_word_seq_vec,
                                                                    sorted_word_len,
                                                                    orig_idx_retrieve)


            char_encoder_hidden_ls.append(char_encoder_hidden)
            # print(char_encoder_hidden.size())
            # print("*" * 50)

        # print(len(char_encoder_hidden_ls))
        final_char_embedding = torch.stack(char_encoder_hidden_ls,dim=0).to(self.device)
        # print(final_char_embedding.size(),input_length)

        return final_char_embedding,input_length




# ############## Example ######## 
#s1="Is CV most palakkad Anaconda of bbdfbbf mbfjkbnbj ?"
#s2="Maybe or not."
#s3="It's NLP,Bro !"

#sentences=[s1,s2,s3]

#char_embedding=char_embed()
#final_char_emb,input_len=char_embedding.char_final_embed(sentences)
#print(final_char_emb.size(),input_len)




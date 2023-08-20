import torch.nn as nn
import torch
from transformers import AutoTokenizer,AutoModel
from src.components.number_embedding import NumberEmbed  #  when run from main file 
# from  components.number_embedding import NumberEmbed       #  when run from this file 

class Roberta_modified_for_charEmb(nn.Module):
    
    def __init__(self,config,roberta_model="roberta-base",device="cuda:0"):
        super().__init__()
        self.roberta_layer = AutoModel.from_pretrained("/home/abhiraj/AllEmb/offline_model/robert_layer_offline")
        self.roberta_tokenizer = AutoTokenizer.from_pretrained("/home/abhiraj/AllEmb/offline_model/robert_tokenizer_offline")
        self.device = device
        self.config=config
        self.number_embedding=self.config.use_num_embed
        
        self.num_dimension_input = 100+768
        self.num_out_features  = 768

        if config.use_layer_add_num==True:
            self.combine_layer_for_num =nn.Linear(self.num_dimension_input , self.num_out_features ).to(device)

        freeze_roberta=False
        if freeze_roberta:
            for p in self.roberta_layer.parameters():
                p.requires_grad = False

        if self.number_embedding==True:
            print("number embeddig initialized")
            self.n_emb=NumberEmbed(self.config)

    def replace_space_token_with_space(self,word):
        if word[0] == "Ġ":
            new_word = " "+word[1:]
            return new_word
        else:
            return word

    def replace_token_with_ids(self,sent_tokens):
        tokens_idx=[]
        # print(sent_tokens)
        for tokens in sent_tokens:
            ls=[]
            for token in tokens:
                # print(type(token))
                
                temp=self.roberta_tokenizer.encode(token)[1:-1]    
                if len(temp)==1:
                    ls.append(temp[0])
                else:
                    ls.append(temp)
            
            tokens_idx.append(ls)
        
        return tokens_idx 

    def keep_track_split(self,padded_tokens_idx):
        merge_list=[]
        temp1=[]


        for sent_idx , sent_tokens in enumerate(padded_tokens_idx):
            temp2=[]

            for token in sent_tokens:

                token_len = len(token) if isinstance(token, list) else 1

                if token_len > 1: 
                    merge_list.append([sent_idx , ( len(temp2) , len(temp2)+token_len-1 )])
                    temp2+=token
                else:
                    temp2.append(token)
            temp1.append(temp2)
        
        return temp1,merge_list

    def extra_padding(self,tokens):

        max_len=max([len(token) for token in tokens])
        return [token+[1] * (max_len-len(token)) for token in tokens]     ## "1" is padding_index(<PAD>)s in RoBerta 

    def count_dim_for_merge(self,merge_list,batch_size):
        temp={}
        for i in range(batch_size):
            temp[f"dim{i}_count"]=0
        # dim0_count=0
        # dim1_count=0
        # dim2_count=0
        # dim3_count=0

        for val in merge_list:
            # if val[0]==0: 
            #     dim0_count+=1
            # elif val[0]==1: 
            #     dim1_count+=1
            # elif val[0]==2:
            #     dim2_count+=1
            # else:
            #     dim3_count+=1
            
            j=val[0]
            temp[f"dim{j}_count"]+=1


        return tuple(temp.values())

    def merge_embedd(self,a,a_new,merge_list,count,count_dim,sent_idx,i):
        ## a is robert embedd matrix
    
        #### order matter here , we have to sort the merge_list accoding to sent_idx and index in tuple 

        start_idx=merge_list[i][1][0]
        end_idx=merge_list[i][1][1]+1 
        # print(a.size(),sent_idx,start_idx,end_idx)

        combine_subword=a[sent_idx,start_idx:end_idx,:].mean(0,keepdim=True)
        

        if count_dim[sent_idx]==0: return a_new
        if count_dim[sent_idx]==1: return torch.cat([a[sent_idx,:start_idx,:],combine_subword,a[sent_idx,end_idx:,:]])

        if count==1:
            # print(combine_subword.size())
            # print(a[sent_idx,:start_idx,:].size())
            a_new=torch.cat([a[sent_idx,:start_idx,:],combine_subword])
            # print(a_new.size())
        
        elif count== count_dim[sent_idx]: 
            a_new = torch.cat([a_new,a[sent_idx,merge_list[i-1][1][1]+1:merge_list[i][1][0],:],combine_subword,a[sent_idx,end_idx:,:]])
            # print(a_new.size(),2,sent_idx,i)
            # print(a_new.size(),combine_subword.size(),a[sent_idx,end_idx:,:].size())
            # print(start_idx,end_idx)

        else:
            a_new=torch.cat([a_new,a[sent_idx,merge_list[i-1][1][1]+1:merge_list[i][1][0],:],combine_subword])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
            # print(a_new.size(),3)     
        return a_new

    def combine_subword(self,robert_embed,merge_list):
        
        temp_dict={}
        temp_dict_new={}

        batch_size=robert_embed.size()[0]

        for i in range(batch_size):
            key_1="dim"+str(i)+"_count"
            temp_dict[key_1]=0

            key_2="a_new"+str(i)
            temp_dict_new[key_2]=robert_embed[i,:,:]
        
        # print(temp_dict["dim0_count"],temp_dict["a_new3"])
        # dim0_count=0
        # dim1_count=0
        # dim2_count=0
        # dim3_count=0

        # a_new0=robert_embed[0,:,:]
        # a_new1=robert_embed[1,:,:]
        # a_new2=robert_embed[2,:,:]
        # a_new3=robert_embed[3,:,:]

        

        a=robert_embed
        count_dim=self.count_dim_for_merge(merge_list,batch_size)
        # print(count_dim)
        # print(merge_list)
        
        for i in range(len(merge_list)):
            # print(i)
            j=merge_list[i][0]
            temp_dict[f"dim{j}_count"]+=1 
            # print(j)
            temp_dict_new[f"a_new{j}"]=self.merge_embedd(robert_embed ,temp_dict_new[f"a_new{j}"] , merge_list , temp_dict[f"dim{j}_count"], count_dim,j ,i)
            # print(temp_dict_new[f"a_new{j}"].size(),"temp_dict_new")
                # if merge_list[i][0]==1:
                #     dim1_count+=1
                #     a_new1=self.merge_embedd(robert_embed,a_new1,merge_list,dim1_count,count_dim,1 ,i)

                # if merge_list[i][0]==2:
                #     dim2_count+=1
                #     a_new2=self.merge_embedd(robert_embed,a_new2,merge_list,dim2_count,count_dim, 2 ,i)

                # if merge_list[i][0]==3:
                #     dim3_count+=1
                #     a_new3=self.merge_embedd(robert_embed,a_new3,merge_list,dim3_count,count_dim, 3 ,i)

        return temp_dict_new

    def robertify_input(self,sentences):    

        all_tokens=[]
        # print(sentences)
        for sent in sentences:
            
            pre_token_with_idx = self.roberta_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sent)  # it contain  positions index + tokens 
            # print(pre_token_with_idx)
            pre_token = [pre_token_with_idx[i][0] for i in range(len(pre_token_with_idx))]
            # print(pre_token)
            token_for_char = [self.replace_space_token_with_space(pre_token[i]) for i in range(len(pre_token))]
            # print(token_for_char)
            # print(token_for_char)
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
            # print(token_for_char)
            tokens = ['<s>'] + token_for_char + ['</s>']
            all_tokens.append(tokens)
        
        # print(all_tokens)

        
        num_embed_token=["number0","number1" , "number2" , "number3" , "number4" , "number5" , "number6"," number0", " number1" , " number2" , " number3" , " number4" , " number5" , " number6"]
        num_merge_list=[]

        for idx,tokens in enumerate(all_tokens):
            for i  in range(len(tokens)):
                if tokens[i] in num_embed_token:
                    num_merge_list.append((idx,i))


        # print(num_merge_list)
        input_lengths_without_word_split=[len(tokens) for tokens in all_tokens]   
        # print("input_lengths_without_word_split",input_lengths_without_word_split)
        max_length=max(input_lengths_without_word_split)
        padded_tokens=[tokens+["<pad>"]*(max_length-len(tokens)) for tokens in all_tokens]
        padded_tokens_idx=self.replace_token_with_ids(padded_tokens)
        # print(padded_tokens_idx)
        # attention_masks=self.calculate_attn_mask(padded_tokens_idx)
        # for token in padded_tokens_idx: print(len(token)) 

        padded_token_without_merge, merge_list= self.keep_track_split(padded_tokens_idx)
        # for token in padded_token_without_merge: print(len(token)) 
        # print(padded_token_without_merge)
        # print(merge_list)

        extra_padding_tokens=self.extra_padding(padded_token_without_merge)
        # for token in extra_padding_tokens: print(len(token)) 

        
        all_tokens_idx = torch.tensor(extra_padding_tokens)

        pad_token = self.roberta_tokenizer.convert_tokens_to_ids('<pad>')
        attn_masks = (all_tokens_idx != pad_token).long()
        
        input_len=[]
        for each_mask in attn_masks:  
            input_len.append(torch.sum((each_mask!=0)).item())
 
        # print(padded_tokens)
        # print(padded_tokens_idx)

        return all_tokens_idx ,all_tokens, attn_masks, input_lengths_without_word_split ,input_len ,merge_list ,num_merge_list

    def forward(self,config,sentence,nums,device="cuda:0"): 
        #print("<<<<<<<<<<<<<<<<<<<<<<<<< Creating Roberta Embedding >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        padded_token_without_merge ,all_tokens_string, attn_masks, input_lengths_without_subword,input_len_with_subword , merge_list ,num_merge_list = self.robertify_input(sentence)              
        # print(padded_token_without_merge.size() , input_lengths_without_subword, input_len_with_subword, merge_list ,num_merge_list)
        # print(all_tokens_string)
        padded_token_without_merge=padded_token_without_merge.to(device)
        attn_masks= attn_masks.to(device)
        self.roberta_layer= self.roberta_layer.to(device)

        #print(device)
        #print(self.roberta_layer.device,padded_token_witout_merge.device,attn_masks.device)
        cont_reps = self.roberta_layer(padded_token_without_merge, attention_mask = attn_masks)   

        roberta_emb=cont_reps.last_hidden_state.to(device)
        
        
        # print(roberta_emb.size(),"roberta_emb_size")
        #print(merge_list)
        

        temp=self.combine_subword(roberta_emb,merge_list)
        max_length=max(input_lengths_without_subword)


        # print(temp["a_new0"].size())
        # print(temp["a_new1"].size())
        # print(temp["a_new2"].size())

        # print(temp["a_new2"][7][:5],temp["a_new2"][-1][:5])
        # print(temp["a_new1"][5][:5],temp["a_new1"][-1][:5])


        combine_list=[x[0:max_length] for x in list(temp.values())]
        emb=torch.stack(combine_list).to(device)
        
        
        if self.number_embedding == True:
            numbers_string = [num.split() for num in nums]
            # print(numbers_string)
            alpha=config.num_alpha
            # print(alpha)
            for ls in num_merge_list:
            
                sent_idx=int(ls[0])
                word_idx=int(ls[1])

                number_string_idx=all_tokens_string[sent_idx][word_idx][-1]
                # print(number_string_idx)
                # print(sent_idx,word_idx,all_tokens_string)
                number=numbers_string[sent_idx][int(number_string_idx)]   ## number in string
                
                # try:
                #     num_emb=self.n_emb.get_num_emb(str(float(number)))
                # except:
                #     number=number[0:16]
                #     num_emb=self.n_emb.get_num_emb(str(float(number)))

                num_emb=self.n_emb.get_num_emb(number)
                num_emb=num_emb.to(device)

                # print(num_emb.size(),"num_emb")
                # print(emb[sent_idx,word_idx,:].size(),"emb_size")s
                
                if config.use_layer_add_num==True:
                    # print("entered into combining layer")
                    # print(emb[sent_idx,word_idx,:].size())
                    print(num_emb[0,0,:].size())
                    input=torch.cat([emb[sent_idx,word_idx,:],num_emb[0,0,:]],dim=0).to(self.device)
                    emb[sent_idx,word_idx,:] = self.combine_layer_for_num(input).to(self.device)
            
                else:
                    emb[sent_idx,word_idx,:]=alpha*emb[sent_idx,word_idx,:]+(1-alpha)*num_emb
                    # print("didn't entered")
                # print("number embedding added") 
                num_emb=num_emb.to("cpu")
    
        # print(emb.size())
        # print(input_lengths_without_subword,input_len_with_subword)
        return emb,input_lengths_without_subword


# s1="Is CV number0 palakkad Anaconda number1 number2 ggcdytfi ?"
# s2="number0 or number1."
# s3="It's NLP,Bro !"
# # s4="It's NLP,Bro !"
# # s5="num3 aise hi "
# # s6="irfgabnfkbnb"

# nums = ['55 10 13', '5.0 45.0', '']

# # data=['dave had number0 files and number1 apps on his phone . after deleting some apps and files he had number2 apps and number3 files left . how many apps did he delete ?', 'paul got a box of number0 crayons for his birthday . during the school year he gave number1 crayons to his friends while he lost number2 crayons . how many crayons had been lost or given away ?', 'edward spent $ number0 to buy books and $ number1 to buy pens . now he has $ number2 . how much more did edward spend on books than pens ?', 'at the arcade dave had won number0 tickets . if he used number1 to buy some toys and number2 more to buy some clothes how many more tickets did dave use to buy toys than he did to buy clothes ?']
# data=[s1,s2,s3]
# embedding=Roberta_modified_for_charEmb()
# emb,inp_len=embedding(data,nums)
# print(emb.size(),inp_len)


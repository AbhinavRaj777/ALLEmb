import os
import sys
import math 
import logging 
import pdb 
import random
import numpy as np 
from attrdict import AttrDict 
import torch 
from torch.utils.data import DataLoader

#from tensorboardX import SummaryWriter 
from collections import OrderedDict 
try:
        import cPickle as pickle
except ImportError:
        import pickle
import copy 




### uncomment it when you need to hyperparameter tunning #################
# ############## for hyperparameter tunning #######
# import ray
# from ray import tune
# from ray import air
# from ray.air import session
# from ray.air.checkpoint import Checkpoint
# from ray.tune.schedulers import ASHAScheduler
# from ray.tune.search.hyperopt import HyperOptSearch
# from ray.tune.search import ConcurrencyLimiter
# from ray.tune.search.hebo import HEBOSearch
# ##################################################

from src.args import build_parser
# from src.args import build_search_space      ## uncomment for hyperparam_tuning
from src.utils.helper import *
from src.utils.logger import get_logger, print_log, store_results, store_val_results
from src.dataloader import TextDataset
from src.modelv2 import build_model, train_model, run_validation, estimate_confidence
from src.confidence_estimation import *

global log_folder
global model_folder
global result_folder
global data_path
global board_path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
current_working_directory=os.getcwd()
log_folder =current_working_directory+ "/logs"
model_folder = current_working_directory+ "/models"
outputs_folder = current_working_directory + "/outputs"
result_folder = './out/'
data_path = './data/'
board_path = './runs/'


def load_data(config, logger):
        '''
                Loads the data from the datapath in torch dataset form

                Args:
                        config (dict) : configuration/args
                        logger (logger) : logger object for logging

                Returns:
                        dataloader(s)
        '''
        if config.mode == 'train':
                logger.debug('Loading Training Data...')

                '''Load Datasets'''
                train_set = TextDataset(data_path=data_path, dataset=config.dataset,
                                                                datatype='train', max_length=config.max_length, is_debug=config.debug)
                val_set = TextDataset(data_path=data_path, dataset=config.dataset, datatype='dev', max_length=config.max_length, 
                                                                is_debug=config.debug, grade_info=config.grade_disp, type_info=config.type_disp, 
                                                                challenge_info=config.challenge_disp)

                '''In case of sort by length, write a different case with shuffle=False '''
                train_dataloader = DataLoader(
                        train_set, batch_size=config.batch_size, shuffle=True, num_workers=5)
                val_dataloader = DataLoader(
                        val_set, batch_size=config.batch_size, shuffle=True, num_workers=5)

                train_size = len(train_dataloader) * config.batch_size
                val_size = len(val_dataloader)* config.batch_size
                #for data in val_dataloader: print(len(data["ques"]))
                msg = 'Training and Validation Data Loaded:\nTrain Size: {}\nVal Size: {}'.format(train_size, val_size)
                logger.info(msg)

                return train_dataloader, val_dataloader

        elif config.mode == 'test' or config.mode == 'conf':
                logger.debug('Loading Test Data...')

                test_set = TextDataset(data_path=data_path, dataset=config.dataset,
                                                           datatype='test', max_length=config.max_length, is_debug=config.debug)
                test_dataloader = DataLoader(
                        test_set, batch_size=config.batch_size, shuffle=True, num_workers=5)

                logger.info('Test Data Loaded...')
                return test_dataloader

        else:
                logger.critical('Invalid Mode Specified')
                raise Exception('{} is not a valid mode'.format(config.mode))

def main():
        '''read arguments'''
        # parser = build_parser()
        hyper_par=False
        if hyper_par==True:
                parser = build_search_space()
        else:
                parser = build_parser()

        args = parser.parse_args()
        config = args
        print(config)
        config.hyper_param=hyper_par

        mode = config.mode
        if mode == 'train':
                is_train = True
        else:
                is_train = False


        ''' Set seed for reproducibility'''
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        random.seed(config.seed)

        '''GPU initialization'''
        # if torch.cuda.is_available():

        device = gpu_init_pytorch(config.gpu)
        
        

        #print(device)
        if config.full_cv:
                global data_path
                data_name = config.dataset
                data_path = data_path + data_name + '/'
                config.val_result_path = os.path.join(result_folder, 'CV_results_{}.json'.format(data_name))
                fold_acc_score = 0.0
                folds_scores = []
                for z in range(5):
                        run_name = config.run_name + '_fold' + str(z)
                        config.dataset = 'fold' + str(z)
                        config.log_path = os.path.join(log_folder, run_name)
                        config.model_path = os.path.join(model_folder, run_name)
                        config.board_path = os.path.join(board_path, run_name)
                        config.outputs_path = os.path.join(outputs_folder, run_name)

                        vocab1_path = os.path.join(config.model_path, 'vocab1.p')
                        vocab2_path = os.path.join(config.model_path, 'vocab2.p')
                        config_file = os.path.join(config.model_path, 'config.p')
                        log_file = os.path.join(config.log_path, 'log.txt')

                        if config.results:
                                config.result_path = os.path.join(result_folder, 'val_results_{}_{}.json'.format(data_name, config.dataset))

                        if is_train:
                                create_save_directories(config.log_path)
                                create_save_directories(config.model_path)
                                create_save_directories(config.outputs_path)
                        else:
                                create_save_directories(config.log_path)
                                create_save_directories(config.result_path)

                        logger = get_logger(run_name, log_file, logging.DEBUG)
                        #writer = SummaryWriter(config.board_path)

                        logger.debug('Created Relevant Directories')
                        logger.info('Experiment Name: {}'.format(config.run_name))

                        '''Read Files and create/load Vocab'''
                        if is_train:
                                train_dataloader, val_dataloader = load_data(config, logger)

                                logger.debug('Creating Vocab...')

                                voc1 = Voc1()
                                voc1.create_vocab_dict(config, train_dataloader)

                                # To Do : Remove Later
                                voc1.add_to_vocab_dict(config, val_dataloader)

                                voc2 = Voc2(config)
                                voc2.create_vocab_dict(config, train_dataloader)

                                # To Do : Remove Later
                                voc2.add_to_vocab_dict(config, val_dataloader)

                                logger.info(
                                        'Vocab Created with number of words : {}'.format(voc1.nwords))

                                with open(vocab1_path, 'wb') as f:
                                        pickle.dump(voc1, f, protocol=pickle.HIGHEST_PROTOCOL)
                                with open(vocab2_path, 'wb') as f:
                                        pickle.dump(voc2, f, protocol=pickle.HIGHEST_PROTOCOL)

                                logger.info('Vocab saved at {}'.format(vocab1_path))

                        else:
                                test_dataloader = load_data(config, logger)
                                logger.info('Loading Vocab File...')

                                with open(vocab1_path, 'rb') as f:
                                        voc1 = pickle.load(f)
                                with open(vocab2_path, 'rb') as f:
                                        voc2 = pickle.load(f)

                                logger.info('Vocab Files loaded from {}\nNumber of Words: {}'.format(vocab1_path, voc1.nwords))

                        print('checkpoint process started')
                        checkpoint = get_latest_checkpoint(config.model_path, logger)
                        print('checkpoint process ended: checkpoint path:',checkpoint)


                        if is_train:
                                # print('1:1')
                                model = build_model(config=config, voc1=voc1, voc2=voc2, device=device, logger=logger, num_iters=len(train_dataloader))
                                # print('1:2')
                                logger.info('Initialized Model')

                                if checkpoint == None:
                                        min_val_loss = torch.tensor(float('inf')).item()
                                        min_train_loss = torch.tensor(float('inf')).item()
                                        max_val_bleu = 0.0
                                        max_val_acc = 0.0
                                        max_train_acc = 0.0
                                        best_epoch = 0
                                        epoch_offset = 0
                                else:
                                        epoch_offset, min_train_loss, min_val_loss, max_train_acc, max_val_acc, max_val_bleu, best_epoch, voc1, voc2 = load_checkpoint(config, model, config.mode, checkpoint, logger, device)

                                with open(config_file, 'wb') as f:
                                        pickle.dump(vars(config), f, protocol=pickle.HIGHEST_PROTOCOL)

                                logger.debug('Config File Saved')

                                logger.info('Starting Training Procedure')
                                max_val_acc = train_model(model, train_dataloader, val_dataloader, voc1, voc2,
                                                        device, config, logger, epoch_offset, min_val_loss, max_val_bleu, max_val_acc, min_train_loss, max_train_acc, best_epoch, writer)

                        else:
                                gpu = config.gpu

                                with open(config_file, 'rb') as f:
                                        config = AttrDict(pickle.load(f))
                                        config.gpu = gpu

                                model = build_model(config=config, voc1=voc1, voc2=voc2, device=device, logger=logger)

                                epoch_offset, min_train_loss, min_val_loss, max_train_acc, max_val_acc, max_val_bleu, best_epoch, voc1, voc2 = load_checkpoint(config, model, config.mode, checkpoint, logger, device)

                                logger.info('Prediction from')
                                od = OrderedDict()
                                od['epoch'] = epoch_offset
                                od['min_train_loss'] = min_train_loss
                                od['min_val_loss'] = min_val_loss
                                od['max_train_acc'] = max_train_acc
                                od['max_val_acc'] = max_val_acc
                                od['max_val_bleu'] = max_val_bleu
                                od['best_epoch'] = best_epoch
                                print_log(logger, od)

                                test_acc_epoch, test_loss_epoch = run_validation(config, model, test_dataloader, voc1, voc2, device, logger)
                                logger.info('Accuracy: {} \t Loss: {}'.format(test_acc_epoch, test_loss_epoch))

                        fold_acc_score += max_val_acc
                        folds_scores.append(max_val_acc)

                fold_acc_score = fold_acc_score/5
                store_val_results(config, fold_acc_score, folds_scores)
                logger.info('Final Val score: {}'.format(fold_acc_score))


        else:
                '''Run Config files/paths'''
                run_name = config.run_name
                config.log_path = os.path.join(log_folder, run_name)
                config.model_path = os.path.join(model_folder, run_name)
                config.board_path = os.path.join(board_path, run_name)
                config.outputs_path = os.path.join(outputs_folder, run_name)

                vocab1_path = os.path.join(config.model_path, 'vocab1.p')
                vocab2_path = os.path.join(config.model_path, 'vocab2.p')
                config_file = os.path.join(config.model_path, 'config.p')
                log_file = os.path.join(config.log_path, 'log.txt')


                if config.results:
                        config.result_path = os.path.join(result_folder, 'val_results_{}.json'.format(config.dataset))

                if is_train:
                        create_save_directories(config.log_path)
                        create_save_directories(config.model_path)
                        create_save_directories(config.outputs_path)
                else:
                        create_save_directories(config.log_path)
                        create_save_directories(config.result_path)

                logger = get_logger(run_name, log_file, logging.DEBUG)
#                writer = SummaryWriter(config.board_path)




                logger.debug('Created Relevant Directories')
                logger.info('Experiment Name: {}'.format(config.run_name))

                '''Read Files and create/load Vocab'''
                if is_train:
                        train_dataloader, val_dataloader = load_data(config, logger)
                        
                        logger.debug('Creating Vocab...')

                        voc1 = Voc1()
                        voc1.create_vocab_dict(config, train_dataloader)

                        # To Do : Remove Later
                        voc1.add_to_vocab_dict(config, val_dataloader)

                        voc2 = Voc2(config)
                        voc2.create_vocab_dict(config, train_dataloader)
                        # print(voc2.get_id("<s>"))
                        # To Do : Remove Later
                        voc2.add_to_vocab_dict(config, val_dataloader)

                        logger.info(
                                'Vocab Created with number of words : {}'.format(voc1.nwords))

                        with open(vocab1_path, 'wb') as f:
                                pickle.dump(voc1, f, protocol=pickle.HIGHEST_PROTOCOL)
                        with open(vocab2_path, 'wb') as f:
                                pickle.dump(voc2, f, protocol=pickle.HIGHEST_PROTOCOL)

                        logger.info('Vocab saved at {}'.format(vocab1_path))

                else:
                        test_dataloader = load_data(config, logger)
                        logger.info('Loading Vocab File...')

                        with open(vocab1_path, 'rb') as f:
                                voc1 = pickle.load(f)
                        with open(vocab2_path, 'rb') as f:
                                voc2 = pickle.load(f)

                        logger.info('Vocab Files loaded from {}\nNumber of Words: {}'.format(vocab1_path, voc1.nwords))

                if is_train:

                        with open(config_file, 'wb') as f:
                                pickle.dump(vars(config), f, protocol=pickle.HIGHEST_PROTOCOL)

                                logger.debug('Config File Saved')

                        if config.hyper_param:

                        # only for  testing hyper parameter tunning.
                        # train_model(model, train_dataloader, val_dataloader, voc1, voc2,
                                                # device, config, logger, epoch_offset, min_val_loss, max_val_bleu, max_val_acc, min_train_loss, max_train_acc, best_epoch, writer)
                                print(config.hyper_param)

                                ## PARAMETER OF TUNER

                                dict_config_temp = copy.deepcopy(vars(config))
                                dict_config_add  = copy.deepcopy(vars(config))
                                dict_config_inp={"hidden_size":dict_config_temp["hidden_size"],
                                                 "dropout":dict_config_temp["dropout"],
                                                 "lr":dict_config_temp["lr"],
                                                 "emb_lr":dict_config_temp["emb_lr"],
                                                 "char_hidden_size":dict_config_temp["char_hidden_size"],
                                                 "char_embedding_size":dict_config_temp["char_embedding_size"],
                                                 "char_nlayer":dict_config_temp["char_nlayer"],
                                                 "char_dropout":dict_config_temp["char_nlayer"],
                                                 "emb2_size":dict_config_temp["emb2_size"] }

                                dict_config={}
                                dict_config["hidden_size"]=256
                                dict_config["dropout"]=0.1
                                dict_config["lr"]=0.0005
                                #dict_config["batch_size"]=4
                                dict_config["emb_lr"]=1e-5
                                dict_config["char_hidden_size"]=768
                                dict_config["char_embedding_size"]=512
                                dict_config["char_nlayer"]=1
                                dict_config["char_dropout"]=0.1
                                dict_config["emb2_size"]=16
                                current_best_params = [dict_config]

                                print("*"*50)
                                print(current_best_params)
                                print("#"*50)
                                print(dict_config_inp)
                                print("$"*50)

                                num_samples=50
                                max_num_epochs=100
                                gpus_per_trial=1
                                cpu_per_trial=8
                                 ### converting namespace to dictionary because tuner require dictionary

                                #print(dict_config,type(dict_config))
                                #ray.init(runtime_env={"RAY_memory_usage_threshold":1} )
                                scheduler = ASHAScheduler(
                                        metric="val_acc_epoch",                                        
                                        mode="max",                                        
                                        max_t=max_num_epochs,
                                        grace_period=50,
                                        reduction_factor=2)


                                
                                #hyperopt_search = HyperOptSearch(points_to_evaluate=current_best_params)
                                #hyperopt_search = ConcurrencyLimiter(hyperopt_search, max_concurrent=1)
                                
                                 
                                search_type=HEBOSearch(points_to_evaluate=current_best_params)
                                search_type=ConcurrencyLimiter(search_type, max_concurrent=1)
                                

                                tuner = tune.Tuner(
                                                tune.with_resources(
                                                        tune.with_parameters(train_model,dict_config_add=dict_config_add,
                                                                                        train_dataloader=train_dataloader, 
                                                                                        val_dataloader=val_dataloader, 
                                                                                        voc1=voc1,
                                                                                        voc2=voc2,
                                                                                        device=device,
                                                                                        logger=logger,
                                                                                writer=None ),
                                                resources={"cpu":cpu_per_trial, "gpu": gpus_per_trial}),

                                                tune_config=tune.TuneConfig( metric="val_acc_epoch", mode="max",
                                                                                search_alg=search_type,
                                                                                num_samples=num_samples,
                                                                                ),
                                                run_config=air.RunConfig(local_dir="./tune_results", name="hpt_HEBO"),
                                                param_space=dict_config_inp
                                )

                                # don't  give writer  value to something because it will interfere with  ray tune 
                                
#                                trainable_with_params=tune.with_parameters(train_model,
#                                                                train_dataloader=train_dataloader, 
#                                                                 val_dataloader=val_dataloader, 
#                                                                 voc1=voc1,
#                                                                 voc2=voc2,
#                                                                 device=device,
#                                                                 logger=logger,
#                                                                 writer=None )
#
#                                tuner = tune.Tuner.restore(path="/home/mlg2/apply_charEmbedding/tune_results/hpt_HEBO",trainable=trainable_with_params,restart_errored = True )
                                
                                results = tuner.fit()

                                best_result = results.get_best_result("val_acc_epoch", "max")

                                print("Best trial config: {}".format(best_result.config))
                                print("Best trial final validation accuracy: {}".format(
                                        best_result.metrics["val_acc_epoch"])) 
                        
                                # print("Best trial final validation accuracy: {}".format(
                                #         best_result.metrics["accuracy"]))


                        else:
                                logger.debug("going to call train model") 
                                print("start")
                                train_model(config, train_dataloader, val_dataloader, voc1, voc2, device,logger, writer=None)

                else :
                        gpu = config.gpu
                        conf = config.conf
                        sim_criteria = config.sim_criteria
                        adv = config.adv
                        mode = config.mode
                        dataset = config.dataset
                        batch_size = config.batch_size
                        with open(config_file, 'rb') as f:
                                config = AttrDict(pickle.load(f))
                                config.gpu = gpu
                                config.conf = conf
                                config.sim_criteria = sim_criteria
                                config.adv = adv
                                config.mode = mode
                                config.dataset = dataset
                                config.batch_size = batch_size

                        model = build_model(config=config, voc1=voc1, voc2=voc2, device=device, logger=logger,num_iters=len(test_dataloader))

                        epoch_offset, min_train_loss, min_val_loss, max_train_acc, max_val_acc, max_val_bleu, best_epoch, voc1, voc2 = load_checkpoint(config, model, config.mode, checkpoint, logger, device)

                        logger.info('Prediction from')
                        od = OrderedDict()
                        od['epoch'] = epoch_offset
                        od['min_train_loss'] = min_train_loss
                        od['min_val_loss'] = min_val_loss
                        od['max_train_acc'] = max_train_acc
                        od['max_val_acc'] = max_val_acc
                        od['max_val_bleu'] = max_val_bleu
                        od['best_epoch'] = best_epoch
                        print_log(logger, od)

                        if config.mode == 'test':
                                test_acc_epoch = run_validation(config, model, test_dataloader, voc1, voc2, device, logger, 0)
                                logger.info('Accuracy: {}'.format(test_acc_epoch))
                        else:
                                estimate_confidence(config, model, test_dataloader, logger)



if __name__ == '__main__':

        main()


''' Just docstring format '''



#metric="val_acc_epoc>
#                                                                                mode="max",




# class Vehicles(object):
#       '''
#       The Vehicle object contains a lot of vehicles

#       Args:
#               arg (str): The arg is used for...
#               *args: The variable arguments are used for...
#               **kwargs: The keyword arguments are used for...

#       Attributes:
#               arg (str): This is where we store arg,
#       '''
#       def __init__(self, arg, *args, **kwargs):
#               self.arg = arg

#       def cars(self, distance,destination):
#               '''We can't travel distance in vehicles without fuels, so here is the fuels

#               Args:
#                       distance (int): The amount of distance traveled
#                       destination (bool): Should the fuels refilled to cover the distance?

#               Raises:
#                       RuntimeError: Out of fuel

#               Returns:
#                       cars: A car mileage
#               '''
#               pass

import torch
from src.model.LGI import LGI
from src.utils import net_utils
import numpy as np
from .parallel_apply import parallel_apply

class Parallel_LGI(nn.Module):
    """
    DataParallel class for Parallel LGI
    since the input length is not constant, 
    """
    def __init__(self,config, logger=None, verbose=True, device_ids=None, output_device=None):
        super(Parallel_LGI,self).__init__()
        self.module = LGI(config,logger=logger,verbose=verbose)
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.device_ids = device_ids
        self.output_device = output_device
    
    def scatter(self,batch):
        batch_size = len(batch)
        device_cnt = len(self.device_ids)
        new_batch = []
        # case: batch size is smaller than device count
        if batch_size < device_cnt:
            for i in range(batch_size):
                device_batch = [batch[i]]
                self.batch_to_device(device_batch,self.device_ids[i])
                new_batch.append(device_batch)
            return new_batch
        # else: batch size is bigger than device count
        each_size = batch_size / device_cnt
        idxs = [(round(each_size * i, each_size * (i+1)))for i in range(device_cnt)]
        for i, (st, ed) in enumerate(idxs):
            device_batch = {}
            for k in batch.keys():
                device_batch[k] = batch[k][st:ed].to(self.device_ids[i]) \
                    if isinstance(data, torch.Tensor) else batch[k][st:ed]
            new_batch.append(device_batch)
        return new_batch
    
    def replicate(self, module, device_ids):
        return torch.nn.parallel.replicate(module,device_ids)
    
    def gather(self,outs,output_device):
        out_batch = {}
        for k in outs[0].keys():
            datas = []
            for o in outs:
                datas.append(o[k])
            # tensor
            if isinstance(datas[0]),torch.Tensor):
                new_datas = [d.to(output_device) for d in datas]
                out_batch[k] = torch.cat(new_datas,dim=0)
            # np array
            elif isinstance(datas[0],np.array):
                out_batch[k] = np.concatenate(datas,axis=0)
            # list
            else:
                new_datas = []
                for d in datas:
                    new_datas += d
                out_batch[k] = new_datas
        return out_batch
            
    def parallel_apply(self,replicas,inputs):
        return parallel_apply(replicas,inputs)

    def forward(self, batch):
        # if device is cpu
        if not self.device_ids:
            return self.module(batch)
        replicas = self.replicate(self.module,self.device_ids)
        inputs = self.scatter(batch)
        outputs = self.parallel_apply(replicas,inputs)
        return self.gather(outputs,self.output_device)

    ###################################### LGI functions ################################################
    def forward_update(self,net_inps,gts):
        net_out = self.forward(net_inps)
        loss = self.module.loss_fn(net_out, gts, count_loss=True)
        self.module.update(loss)
        return {"loss":loss, "net_output":net_out}

    def visualize(self,*args,**kwargs):
        self.module.visualize(*args,**kwargs)
    
    def extract_output(self,*args,**kwargs):
        self.module.extract_output(*args,**kwargs)

    def prepare_batch(self,*args,**kwargs):
        return self.module.prepare_batch(*args,**kwargs)
    
    def reset_status(self,*args,**kwargs):
        self.module.reset_status(*args,**kwargs)

    def compute_status(self,*args,**kwargs):
        self.module.compute_status(*args,**kwargs)

    def save_results(self,*args,**kwargs):
        self.module.save_results(*args,**kwargs)

    def renew_best_score(self):
        return self.module.renew_best_score()

    def bring_dataset_info(*args,**kwargs):
        self.module.bring_dataset_info(*args,**kwargs)

    def model_specific_config_update(*args,**kwargs):
        return self.module.model_specific_config_update(*args,**kwargs)

    def dataset_specific_config_update(*args,**kwargs):
        return self.module.dataset_specific_config_update(*args,**kwargs)
    

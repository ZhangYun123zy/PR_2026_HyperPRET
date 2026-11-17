from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy

from .helper import *
from utils import *
from dataloader.data_utils import *
from torch.utils.data import DataLoader
from sklearn import metrics
def cal_kl_div(log_p, q, weights):
    '''
    log_p.shape = (B, n_way_all)
    q.shape = (B, n_way_all)
    '''
    log_q = torch.log(q)
    loss = q * (log_q - log_p)              # (B, n_way_all)
    loss = loss * weights.unsqueeze(-1)     # (B, n_way_all)
    loss = loss.sum()
    return loss

def top1_acc(pred, label):
    '''
    compute top1 accuarcy
    pred.shape  = (n_batch, n_class)
    label.shape = (n_batch,)
    '''
    _, pred_y = torch.max(pred, dim=1)
    acc = (pred_y == label).float().mean().item()
    return acc
class ExemplarSet(torch.utils.data.Dataset):

    def __init__(self, exemplar_path, exemplar_label, dataset='mini_imagenet', train=True):
        #self.dataset = dataset
        self.train = train  # training or testing stage
        self.return_idx = False
        self.update(exemplar_path, exemplar_label)
    
    def update(self, exemplar_path, exemplar_label):
        self.data = exemplar_path
        self.label = exemplar_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        
        data, label = self.data[i], self.label[i]
        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.tensor(label)
        #data= np.asarray(data, dtype='float32')
        #label = np.asarray(label, dtype='int64')
        if self.return_idx:
            return data, label, i
        else:
            #print(data.shape)
            #print(type(data),type(data))
            return data, label
        
def w_KDloss(input, target, tmp=4, weights=None, t_prob=False):
    log_p = torch.log_softmax(input/tmp, dim=1) 
    if (t_prob == True):
        q = target
    else:
        q = torch.softmax(target/tmp, dim=1)
    if weights is None:
        weights = torch.ones(log_p.shape[0]).cuda()
    loss = cal_kl_div(log_p, q, weights)
    loss = loss * (tmp**2) / weights.sum()
    return loss
class Batch_DataSampler():
    def __init__(self, label, n_batch, n_query, ep_per_batch=1):
        self.n_batch = n_batch  # number of episode
        self.n_query = n_query  # number of query samples to sample from base classes
        self.ep_per_batch = ep_per_batch
        self.label = np.array(label) 

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                l = np.random.choice(len(self.label), self.n_query, replace=True)
                batch.append(torch.from_numpy(l))
            batch = torch.stack(batch) # bs * n_query 
            yield batch.view(-1)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class BinaryNet(nn.Module):
    '''
    simple binary classification network with 1 hidden layer
    '''
    def __init__(self, input_dim, hidden_dim=64):
        super(BinaryNet, self).__init__()   
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim    

        self.fc_layer1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc_layer2 = nn.Linear(self.hidden_dim, 2)       
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.fc_layer1(x)
        x = self.relu(x)
        x = self.fc_layer2(x)
        return x
    
def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([140, 67, 46]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 100, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 200]) / 255.
        if item == 5:
            y[index] = np.array([164, 75, 155]) / 255.
        if item == 6:
            y[index] = np.array([101, 174, 255]) / 255.
        if item == 7:
            y[index] = np.array([216, 191, 216]) / 255.
        if item == 8:
            y[index] = np.array([60, 91, 112]) / 255.
        if item == 9:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 10:
            y[index] = np.array([255, 255, 255]) / 255.
        if item == 11:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 12:
            y[index] = np.array([100, 0, 255]) / 255.
        if item == 13:
            y[index] = np.array([0, 200, 254]) / 255.
        if item == 14:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 15:
            y[index] = np.array([171, 175, 80]) / 255.
        if item == 16:
            y[index] = np.array([101, 193, 60]) / 255.
        
    return y
def get_classification_map(y_pred, y):

    height = y.shape[0]
    width = y.shape[1]
    k = 0
    cls_labels = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            target = int(y[i, j])
            if target == 0:
                continue
            else:
                cls_labels[i][j] = y_pred[k]+1
                k += 1

    return  cls_labels

def classification_map(map, ground_truth, dpi, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1]*2.0/dpi, ground_truth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)

    return 0

def get_cls_map(y_pred,y,session):

    #y_pred, y_new = test(device, net, all_data_loader)
    print(y_pred.shape,y.shape)
    cls_labels = get_classification_map(y_pred, y)
    x = np.ravel(cls_labels)
    y_list = list_to_colormap(x)

    y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))
    classification_map(y_re, y, 300,
                       'classification_maps/' + 'HyperPRET_fused_session{}.png'.format(session))
    print('------Get classification maps successful-------')

def set_bn_state(model, set_eval=True):
    if set_eval:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    else:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()

class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        self.model = MYNET(self.args, mode=self.args.base_mode)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        self.feature_extractor=self.model.module.encoder
        
        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
            
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def get_optimizer_base(self):

        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        return optimizer, scheduler

    def get_dataloader(self, session,seed):
        if session == 0:
            trainset, trainloader, testloader,all_loader,y,y_all_seen = get_base_dataloader(self.args,seed)
        else:
            trainset, trainloader, testloader,all_loader,y,y_all_seen = get_new_dataloader(self.args, session,seed)
        return trainset, trainloader, testloader,all_loader,y,y_all_seen
    
    def get_logits(self, backbone,ft_weight, x, return_feature=False):
        '''
        get the output logits based on the given feature extractor, classification weights and scalar factor
        '''
        f0 = backbone.module.encode(x)  
        if 'cos' in backbone.module.mode:
            
            x = F.linear(F.normalize(f0, p=2, dim=-1), F.normalize(ft_weight.weight, p=2, dim=-1))
            pred = self.args.temperature * x
            
        elif 'dot' in backbone.module.mode:
            x = ft_weight(f0)
            pred = self.args.temperature * x                                                                            # (B, 3, H, W)

        if return_feature:
            return pred, f0
        else:
            return pred
        
    def save_all_samples_base(self, trainset, model, session=0, transform_test=None):
        '''
        save 5 exemplars per base class  
        '''
        assert session == 0
        self.class_max_sim  = [-1] * self.args.base_class
        self.exemplar_idxs  = [-1] * self.args.base_class
        self.exemplar_path  = ['None'] * self.args.base_class
        self.exemplar_label = list(range(len(self.exemplar_path)))
        #print(self.exemplar_label)
        batch_size_tmp = 256
        
        class_tmp = list(get_session_classes(self.args, session))
        trainset.return_idx = True
        if transform_test is not None:
            trainset.transform = transform_test
        trainloader = DataLoader(dataset=trainset, batch_size=batch_size_tmp, shuffle=False,
                                 num_workers=8, pin_memory=True)
        model.eval()
        with torch.no_grad():
            be=0
            for batch in trainloader:
                data, labels = [_.cuda() for _ in batch]      # (bs, 3, H, W), (bs, ), (bs, )
                #print(labels)
                pred = model(data.cuda())[:, :self.args.base_class]  # (bs, n_way_tmp)
                #print(pred.shape)
                idxs=torch.arange(be,be+labels.shape[0]).cuda()
                #print(idxs)
                for c in class_tmp:
                    pred_c = pred[labels == c]                      # (bs_c, n_way_tmp)
                    idxs_c = idxs[labels == c]                      # (bs_c, )
                    #print(idxs_c)
                    if pred_c.shape[0] > 0:
                        pred_c = pred_c[:, c]                       # (bs_c, )
                        sim_c, idx_c = torch.max(pred_c, dim=0)
                        #print(sim_c,idx_c)
                        if sim_c > self.class_max_sim[c]:
                            self.class_max_sim[c] = sim_c.item()#分类结果最大的值
                            self.exemplar_idxs[c] = idxs_c[idx_c] #分类结果最大值对应的位置
                            self.exemplar_path[c] = np.asarray(data[labels == c][idx_c].cpu())#分类结果最大值对应的数据
                be=be+labels.shape[0]
        trainset.return_idx = False

        # sample the remaining exemplars
        labels_all = [trainset[i][1] for i in range(len(trainset))]
        #print(labels_all)
        data_all = [np.array(trainset[i][0]) for i in range(len(trainset))]
        labels_all = np.array(labels_all)
        #print(data_all)
        data_all = np.array(data_all)
        #print(labels_all.shape,data_all.shape)
        
        idxs_all = np.arange(len(labels_all))
        n_exemplar_per_class = 5
        self.exemplar_path_all  = [] 
        self.exemplar_label_all = np.repeat(np.array(self.exemplar_label), n_exemplar_per_class)
        #print(self.exemplar_label_all)
        self.exemplar_label_all = self.exemplar_label_all.tolist() 
        for c in class_tmp:
            self.exemplar_path_all.append(self.exemplar_path[c])
            idxs_all_c = list(idxs_all[labels_all == c])#找到第c类对应的位置们
            #print(self.exemplar_idxs[c])
            assert self.exemplar_idxs[c] in idxs_all_c
            idxs_all_c.remove(self.exemplar_idxs[c])
            idxs_sampled_c = random.sample(idxs_all_c, n_exemplar_per_class - 1)
            for idx_tmp in idxs_sampled_c:
                self.exemplar_path_all.append(data_all[idx_tmp])
                #print(data.shape,data_all.shape)
        
        #for  x in self.exemplar_label_all:
        #    print(type(x))
        self.exemplar_path_all = [np.asarray(x) for x in self.exemplar_path_all]
        assert len(self.exemplar_path_all) == len(self.exemplar_label_all)
        self.base_exemplar_set  = ExemplarSet(deepcopy(self.exemplar_path_all), deepcopy(self.exemplar_label_all), dataset=self.args.dataset, train=True)      

    def train(self,seed):
        args = self.args
        t_start_time = time.time()
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Total:', total_num, 'Trainable:', trainable_num)
        # init train statistics
        result_list = [args]
        #gen_mask
        masknum=3
        mask=np.zeros((args.base_class,args.num_classes))
        for i in range(args.num_classes-args.base_class):
            picked_dummy=np.random.choice(args.base_class,masknum,replace=False)
            mask[:,i+args.base_class][picked_dummy]=1
        mask=torch.tensor(mask).cuda()

        for session in range(args.start_session, args.sessions):
            #print("*******************************************")
            train_set, trainloader, testloader,all_loader,_,y_all_seen = self.get_dataloader(session,seed)
            self.model.load_state_dict(self.best_model_dict)
            
            if session == 0:  # load base class train img label
                #print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()

                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # train base sess
                    tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, args,mask)
                    # test model with all seen class
                    tsl, tsa,aa,kappa = test(self.model, testloader, epoch, args, session,y_all=y_all_seen,all_loader=all_loader)

                    # save better model
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['aa'][session] = float('%.3f' % (aa * 100))
                        self.trlog['kappa'][session] = float('%.3f' % (kappa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                       self.trlog['max_acc'][session]))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = scheduler.get_last_lr()[0]
                    result_list.append(
                        'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)
                    self.model = replace_base_fc(train_set, None, self.model, args)
                    best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir)

                    self.model.module.mode = 'avg_cos'
                    tsl, tsa,aa,kappa = test(self.model, testloader, 0, args, session,y_all=y_all_seen,all_loader=all_loader)
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['aa'][session] = float('%.3f' % (aa * 100))
                        self.trlog['kappa'][session] = float('%.3f' % (kappa * 100))
                        print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))

                #save dummy classifiers
                self.dummy_classifiers=deepcopy(self.model.module.fc.weight.detach())
                
                self.dummy_classifiers=F.normalize(self.dummy_classifiers[self.args.base_class:,:],p=2,dim=-1)
                self.old_classifiers=self.dummy_classifiers[:self.args.base_class,:]
                self.save_all_samples_base(train_set, self.model, session)
            else:  # incremental learning sessions
                print("training session: [%d]" % session)

                self.model.module.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = None
                
                class_index = np.arange(args.base_class+session-1,args.base_class+session*args.way)
                print(class_index)
                self.model.module.update_fc(trainloader, np.unique(class_index), session)
                print("########################")
                _,_,_,_ = self.test_intergrate(self.model, testloader, epoch,args, session,validation=True,y_all=y_all_seen,all_loader=all_loader)
                _=self.update_model_ft(trainloader,session)
                tsa,aa,kappa = self.test_intergrate_attn(self.model,self.feature_fixed,testloader, 0,args, session,y_all=y_all_seen,all_loader=all_loader)
                
                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                self.trlog['aa'][session] = float('%.3f' % (aa * 100))
                self.trlog['kappa'][session] = float('%.3f' % (kappa * 100))

                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        result_list.append(self.trlog['aa'])
        result_list.append(self.trlog['kappa'])
        
        #self.trlog['one_seed']=self.trlog['max_acc']+self.trlog['aa']+self.trlog['kappa']
        result_list.append(self.trlog['one_seed'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)
        print(self.trlog['one_seed'])
        return self.trlog['one_seed_old'],self.trlog['one_seed']
    
    def update_model_ft(self, trainloader, session):
        '''
        update the feature extractor and the classification weights based on finetuning
        '''
        # define teacher/stduent feature and weights
        assert session >= 1
        EMA_prob = False
        tmp_all_way = self.args.num_classes#all classes
        tmp_base_way = tmp_all_way - self.args.way

        self.feature_fixed=MYNET(self.args, mode=self.model.module.mode)
        self.feature_fixed = nn.DataParallel(self.feature_fixed, list(range(self.args.num_gpu)))
        self.feature_fixed = self.feature_fixed.cuda()
        #frozen
        self.feature_fixed.load_state_dict(self.model.state_dict(), strict=True)
        for p in self.feature_fixed.parameters():
            p.requires_grad = False
        self.feature_fixed.eval()


        self.weights_fixed=nn.Linear(self.model.module.num_features, self.model.module.pre_allocate, bias=False).cuda()
        self.weights_fixed.load_state_dict(self.model.module.fc.state_dict(), strict=True)
        for p in self.weights_fixed.parameters():
            p.requires_grad = False

       
        ft_weight =nn.Linear(self.model.module.num_features, self.model.module.pre_allocate, bias=False).cuda()
        ft_weight.load_state_dict(self.model.module.fc.state_dict(), strict=True)
          # (tmp_all_way, C)

        for p in self.model.parameters():
            p.requires_grad = False
        
        params_list = [{'params': ft_weight.parameters(), 'lr': 0.01}#,{'params':self.model.module.encoder.layer3.parameters()}
                           ]
        
        self.binary_classifier = BinaryNet(2*tmp_all_way, 64).cuda()#this
     
        self.binary_classifier.train()
        params_list.append({'params': self.binary_classifier.parameters(), 'lr': 0.01})
        
        # init optimizer
        ft_optimizer = torch.optim.SGD(params_list, lr=0.01, 
                                           momentum=0.9, weight_decay=5e-4, nesterov=True)
       

        # loss recorders
        L_ce = AverageMeter()
        L_kd = AverageMeter()
        L_extra = AverageMeter()
        
        # define exemplar dataloader
        assert len(trainloader) == 1
        n_iters = 100
        tqdm_train = np.arange(n_iters)
        data, label = [], []
        batch_size_base = self.args.batch_size_exemplar#self.args.batch_size_exemplar
        # base class exemplars
        #print( np.asarray(self.base_exemplar_set.label).shape)
        print("base_exemplar number:",len(self.base_exemplar_set))
        base_exemplar_sampler  = Batch_DataSampler(self.base_exemplar_set.label, 
                                                           n_iters, batch_size_base) 
        base_exemplar_loader   =torch.utils.data.DataLoader(dataset=self.base_exemplar_set, batch_sampler=base_exemplar_sampler, num_workers=8, pin_memory=True)# torch.utils.data.DataLoader(dataset=self.base_exemplar_set, batch_size=batch_size_base, shuffle=True,
                                              #num_workers=8, pin_memory=True)
        base_exemplar_loader   = iter(base_exemplar_loader)
           
        
        # finetuning
        for i in tqdm_train:
            # load N * K training data
            for batch in trainloader:
                data_tmp, label_tmp = [_.cuda() for _ in batch]
                data.append(data_tmp)
                label.append(label_tmp)
            #print("the number of samples in session t",len(data),len(label))
            # formulate a batch
            batch_x = torch.cat(data, dim=0)
            label_x = torch.cat(label, dim=0)
            data = []
            label = []
                
            batch_e, label_e = next(base_exemplar_loader)

            batch_e = batch_e.cuda()
            label_e = label_e.long().cuda()
            bs_new = batch_x.shape[0]
            bs_old = batch_e.shape[0]
            batch_all = torch.cat([batch_x, batch_e], dim=0)
            label_all = torch.cat([label_x, label_e], dim=0)
               
                # forward data flow
                #print("base label preserved:",label_e)
            logits_all, f_n = self.get_logits(self.model,ft_weight, batch_all, return_feature=True)
            logits_x = logits_all[:bs_new]
            #print("bs_new:",bs_new)#5 shot
            #print("logits_all shape:",logits_all.shape)
            with torch.no_grad():
                assert self.weights_fixed.weight.shape[0] == tmp_all_way
                #print("the shape of tmp_all_way:",tmp_all_way)#old class+new class
                logits_t, f_b = self.get_logits(self.feature_fixed, self.weights_fixed, batch_all, return_feature=True)
    
            scores_b = logits_t.clone()#
            scores_n = logits_all.clone()#(novel+old, C)
                    #print(scores_n.shape,scores_b.shape)
            pred_b = torch.softmax(scores_b, dim=-1)                            # (bs, C)
            pred_n = torch.softmax(scores_n, dim=-1)                            # (bs, C)
            fc_input = torch.cat([pred_b, pred_n], dim=-1)                  # (bs, 2C)

            attn_logits  = self.binary_classifier(fc_input)                     # (bs, 2)
            attn_weights = torch.softmax(attn_logits, dim=-1)                   # (bs, 2)
                    #print(attn_weights.shape)
                
            scores_attn = attn_weights[:, 0:1] * scores_b + attn_weights[:, 1:2] * scores_n#fuse old and new branch
            loss_attn = self.args.w_BC_cls * F.cross_entropy(scores_attn, label_all, reduction='mean')
            label_binary_all = (label_all >= self.args.base_class).long()
                    #print(label_binary_all)
            loss_binary = self.args.w_BC_binary * F.cross_entropy(attn_logits, label_binary_all, reduction='mean')#the flexible adaption to novel data                   
                                 
            logits_t = logits_t[(bs_new - self.args.batch_size_current):]           # (bs_exe, base_way+n_way)

            loss_ce = F.cross_entropy(logits_x, label_x, reduction='mean')# classification loss of novel class
            loss_ce = self.args.w_cls * loss_ce
            L_ce.update(loss_ce.item())
                
            ### distillation loss
            logits_s = logits_all[(-1 * logits_t.shape[0]):]
            #print(logits_s.shape)
            assert logits_s.shape[0] == logits_t.shape[0]
            
            KD_weights = torch.ones(logits_s.shape[0]).cuda()
            if hasattr(self.args, 'ft_KD_all') and self.args.ft_KD_all:
                loss_kd = w_KDloss(logits_s[:, :tmp_all_way],  logits_t[:, :tmp_all_way],  self.args.ft_T, KD_weights, EMA_prob)
            else:
                loss_kd = w_KDloss(logits_s[:, :tmp_base_way], logits_t[:, :tmp_base_way], self.args.ft_T, KD_weights, EMA_prob)#prevent forgetting
            
            loss_kd = self.args.w_d * loss_kd
            L_kd.update(loss_kd.item())

            ### extra loss
            loss_extra = None
            if self.args.using_exemplars:
                loss_extra = self.args.w_e * F.cross_entropy(logits_all[bs_new:], label_e, reduction='mean')
                L_extra.update(loss_extra.item())
                    #print(loss_extra)

            if loss_extra is not None:
                loss = loss_ce + loss_kd + loss_extra#loss
            else:
                loss = loss_ce + loss_kd

            loss += (loss_attn + loss_binary)
                
                # update
            ft_optimizer.zero_grad()
            loss.backward()
            ft_optimizer.step()
            str_tqdm = 'Session %d finetuning iter %d/%d: loss_ce=%.4f loss_kd=%.4f loss_extra=%.4f'%(session, (i+1),
                            n_iters, loss_ce.item(), loss_kd.item(), (loss_extra.item() if loss_extra is not None else 0))
                
            if self.args.using_exemplars and ('attn' in self.args.merge_strategy):
                str_tqdm += ' loss_attn=%.4f loss_binary=%.4f'%(loss_attn.item(), loss_binary.item())


        self.model.module.fc.load_state_dict(ft_weight.state_dict())
        
        for p in self.model.parameters():
            p.requires_grad = False

        return [L_ce, L_kd, L_extra]
    
    def test_intergrate(self, model, testloader, epoch,args, session,validation=True,y_all=0,all_loader=0):
        test_class = args.base_class + session * args.way
        model = model.eval()
        vl = Averager()
        va = Averager()
        va5= Averager()
        lgt=torch.tensor([])
        lbs=torch.tensor([])

        proj_matrix=torch.mm(self.dummy_classifiers,F.normalize(torch.transpose(model.module.fc.weight[:test_class, :],1,0),p=2,dim=-1))
        #print(model.module.fc.weight[:test_class, :].shape)
        print("proj_matrix",proj_matrix.shape)
        eta=args.eta
        
        softmaxed_proj_matrix=F.softmax(proj_matrix,dim=1)
        real_y=[]
        pred_y=[]
        with torch.no_grad():
            total=0
            correct=0
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]
                
                emb=model.module.encode(data)

                proj=torch.mm(F.normalize(emb,p=2,dim=-1),torch.transpose(self.dummy_classifiers,1,0))
                topk, indices = torch.topk(proj, 2)
                res = (torch.zeros_like(proj))
                res_logit = res.scatter(1, indices, topk)
                logits1=torch.mm(res_logit,proj_matrix)
                logits2 = model.module.forpass_fc(data)[:, :test_class] 
                logits=eta*F.softmax(logits1,dim=1)+(1-eta)*F.softmax(logits2,dim=1)
                loss = F.cross_entropy(logits, test_label)
                acc = count_acc(logits, test_label)

                top5acc=count_acc_topk(logits, test_label)
                vl.add(loss.item())
                va.add(acc)
                va5.add(top5acc)
                lgt=torch.cat([lgt,logits.cpu()])
                lbs=torch.cat([lbs,test_label.cpu()])
                real_y.append(test_label.cpu().numpy())
                pred_y.append(torch.argmax(logits, dim=1).cpu().numpy())
                total += logits.size(0)
                correct += (torch.argmax(logits, dim=1).cpu().numpy() == test_label.cpu().numpy()).sum()

            vl = vl.item()
            va = va.item()
            va5= va5.item()
            real_y = np.concatenate(real_y, axis=0)
            pred_y = np.concatenate(pred_y, axis=0)
            acc_c=correct/total
            print(va,acc_c)
            C = metrics.confusion_matrix(np.array(real_y),np.array(pred_y))
            aa= np.diag(C) / np.sum(C, 1, dtype=np.float32)#
            print(aa)
            c_aa=aa
            aa=aa.mean(0)
            print(aa)
            kappa=metrics.cohen_kappa_score(np.array(real_y),np.array(pred_y))
            print('epo {}, test, loss={:.4f} acc={:.4f}, acc@5={:.4f},aa={:.4f},kappa={:.4f}'.format(epoch, vl, va,va5,aa,kappa))
        self.trlog['one_seed_old']=np.concatenate([c_aa,np.array([aa]),np.array([acc_c]),np.array([kappa])])
        return vl, va,aa,kappa
    
    def test_intergrate_attn(self, model,fixed_model, testloader, epoch,args, session,y_all=0,all_loader=0):
        test_class = args.base_class + session * args.way
        model = model.eval()
        fixed_model=fixed_model.eval()

        proj_matrix=torch.mm(self.dummy_classifiers,F.normalize(torch.transpose(model.module.fc.weight[:test_class, :],1,0),p=2,dim=-1))
        
        scores_org = []
        scores=[]
        y=[]
        with torch.no_grad():
            for i, batch in enumerate(testloader):
                data, test_label = [_.cuda() for _ in batch]
                pred, _ = self.get_logits(fixed_model, fixed_model.module.fc, data, return_feature=True)
                scores_org.append(pred)  
                pred, _ = self.get_logits(model, model.module.fc, data, return_feature=True)
                scores.append(pred)  
                y.append(test_label)     
        y = torch.cat(y, dim=0)
        scores_org = torch.cat(scores_org, dim=0)
        scores = torch.cat(scores, dim=0)
        acc = top1_acc(scores, y)#oa
        acc = top1_acc(scores_org, y)
        pred_b = torch.softmax(scores_org.clone(), dim=-1)                  # (bs, C)
        pred_n = torch.softmax(scores.clone(), dim=-1)                      # (bs, C)

        fc_input=torch.cat([pred_b, pred_n], dim=-1)    
        attn_logits  = self.binary_classifier(fc_input)                     # (bs, 2)
        attn_weights = torch.softmax(attn_logits, dim=-1)                   # (bs, 2)
        scores_b = scores_org.clone()
        scores_n = scores.clone()
        scores = attn_weights[:, 0:1] * scores_b + attn_weights[:, 1:2] * scores_n  
        
        acc = top1_acc(scores, y)#oa
        _, pred_y = torch.max(scores, dim=1)
        C = metrics.confusion_matrix(np.array(y.cpu()),np.array(pred_y.cpu()))
        aa= np.diag(C) / np.sum(C, 1, dtype=np.float32)#
        c_aa=aa
        aa=aa.mean(0)
        print(aa)
        kappa=metrics.cohen_kappa_score(np.array(y.cpu()),np.array(pred_y.cpu()))
        #get classification map

        y_all_pred = []
        y_all_pred_org = []
        with torch.no_grad():
            for i, batch in enumerate(all_loader):
                data, test_label = [_.cuda() for _ in batch]
                pred, _ = self.get_logits(fixed_model, fixed_model.module.fc, data, return_feature=True)
                y_all_pred_org.append(pred)  
                pred1, _ = self.get_logits(model, model.module.fc, data, return_feature=True)
                y_all_pred.append(pred1)  
        
        y_all_pred_org = torch.cat(y_all_pred_org, dim=0)
        y_all_pred = torch.cat(y_all_pred, dim=0)


        # Use the same softmax inputs as the OA/AA/Kappa evaluation above.
        fc_input=torch.cat([torch.softmax(y_all_pred_org, dim=-1),
                            torch.softmax(y_all_pred, dim=-1)], dim=-1)
        attn_logits  = self.binary_classifier(fc_input)                     # (bs, 2)
        attn_weights = torch.softmax(attn_logits, dim=-1)                   # (bs, 2)
        y_all_pred_org = y_all_pred_org.clone()
        y_all_pred = y_all_pred.clone()
        y_all_scores = attn_weights[:, 0:1] * y_all_pred_org + attn_weights[:, 1:2] * y_all_pred  
        _,y_all_pred=torch.max(y_all_scores,dim=1)
        #print("test shape",y.shape,scores.shape)
        get_cls_map(y_all_pred, y_all, session)
        print(c_aa.shape,np.array(aa).shape,np.array(kappa).shape)
        self.trlog['one_seed']=np.concatenate([c_aa,np.array([aa]),np.array([acc]),np.array([kappa])])
        return acc,aa,kappa

    def set_save_path(self):
        mode = self.args.base_mode + '-' + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
            self.args.save_path = self.args.save_path + 'Bal%.2f-LossIter%d' % (
                self.args.balance, self.args.loss_iter)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Cosine':
            self.args.save_path = self.args.save_path + 'Cosine-Epo_%d-Lr_%.4f' % (
                self.args.epochs_base, self.args.lr_base)
            self.args.save_path = self.args.save_path + 'Bal%.2f-LossIter%d' % (
                self.args.balance, self.args.loss_iter)

        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None
